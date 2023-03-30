import torch
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
from torch import nn
from data.data_utils import *


def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                 device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)


def severe_unbalance_domain(domains):
    num_domains = []
    for i in range(3):
        num_domains.append(torch.sum(torch.eq(domains, i)))
        if num_domains[i] <= 1:
            return True
    return False


def generate_vector():
    s = torch.rand(2)
    t = torch.rand(2)
    s_or_t = torch.maximum(s, t)
    s_and_t = torch.minimum(s, t)

    return s, t, s_or_t, s_and_t


def generate_meta_data(data, labels, domains, rand_domain):
    meta_train_ind = torch.nonzero(domains != rand_domain[2]).squeeze()
    meta_test_ind = torch.nonzero(domains == rand_domain[2]).squeeze()

    meta_train_data = data[meta_train_ind]
    meta_train_labels = labels[meta_train_ind]
    meta_train_domains = domains[meta_train_ind]
    meta_test_data = data[meta_test_ind]
    meta_test_labels = labels[meta_test_ind]

    return meta_train_data, meta_train_labels, meta_train_domains, meta_train_ind, meta_test_data, meta_test_labels, meta_test_ind


def filt_aug(data, labels, domains, rand_domain):
    idx = []
    for i in range(int(domains.size(0) / 2)):
        if domains[i] == rand_domain[2] or domains[i + int(domains.size(0) / 2)] == rand_domain[2]:
            continue
        idx.append(i)
        idx.append(i + int(domains.size(0) / 2))

    meta_train_data = data[idx]
    meta_train_labels = labels[idx]

    return meta_train_data, meta_train_labels, idx


def generate_meta_data_aug(data, labels):
    rand_ind = torch.multinomial(torch.ones_like(labels).float(), labels.size(0), replacement=False)
    meta_train_ind = rand_ind[:-int(labels.size(0) / 3)]
    meta_test_ind = rand_ind[-int(labels.size(0) / 3):]
    meta_train_data = data[meta_train_ind]
    meta_test_data = data[meta_test_ind]
    meta_train_labels = labels[meta_train_ind]
    meta_test_labels = labels[meta_test_ind]

    return meta_train_data, meta_train_labels, meta_train_ind, meta_test_data, meta_test_labels, meta_test_ind



def generate_meta_train_groups_with_vector(data_a, labels_a, data_b, labels_b, r):
    ind_a = torch.multinomial(torch.ones_like(labels_a).float(), int(r[0] * labels_a.size(0) + 1), replacement=False)
    ind_b = torch.multinomial(torch.ones_like(labels_b).float(), int(r[1] * labels_b.size(0) + 1), replacement=False)
    if ind_a.size(0) == 1:
        data_aa = data_a[ind_a.squeeze()].unsqueeze(0)
        labels_aa = labels_a[ind_a.squeeze()].unsqueeze(0)
    else:
        data_aa = data_a[ind_a.squeeze()]
        labels_aa = labels_a[ind_a.squeeze()]
    if ind_b.size(0) == 1:
        data_bb = data_b[ind_b.squeeze()].unsqueeze(0)
        labels_bb = labels_b[ind_b.squeeze()].unsqueeze(0)
    else:
        data_bb = data_b[ind_b.squeeze()]
        labels_bb = labels_b[ind_b.squeeze()]

    data = torch.cat((data_aa, data_bb), dim=0)
    labels = torch.cat((labels_aa, labels_bb), dim=0)

    return data, labels



def generate_meta_train_groups(data, labels, domains, type="sample"):
    if type == "sample":
        rand = torch.multinomial(torch.ones_like(labels).float(), labels.size(0), replacement=False)
        slice1 = int(rand.size(0) / 3) + 1
        slice2 = slice1 + int((rand.size(0) - slice1) / 2)

        data_s_and_t = data[rand[:slice1]]
        data_s = torch.cat((data[rand[:slice1]], data[rand[slice1:slice2]]), 0)
        data_t = torch.cat((data[rand[:slice1]], data[rand[slice2:]]), 0)
        data_s_or_t = data[rand[:]]

        labels_s_and_t = labels[rand[:slice1]]
        labels_s = torch.cat((labels[rand[:slice1]], labels[rand[slice1:slice2]]), 0)
        labels_t = torch.cat((labels[rand[:slice1]], labels[rand[slice2:]]), 0)
        labels_s_or_t = labels[rand[:]]
    elif type == "vector":
        s, t, s_or_t, s_and_t = generate_vector()
        ind_a = torch.nonzero(domains == domains[0])
        ind_b = torch.nonzero(domains != domains[0])
        data_a = data[ind_a.squeeze()]
        labels_a = labels[ind_a.squeeze()]
        data_b = data[ind_b.squeeze()]
        labels_b = labels[ind_b.squeeze()]

        data_s, labels_s = generate_meta_train_groups_with_vector(data_a, labels_a, data_b, labels_b, s)
        data_t, labels_t = generate_meta_train_groups_with_vector(data_a, labels_a, data_b, labels_b, t)
        data_s_and_t, labels_s_and_t = generate_meta_train_groups_with_vector(data_a, labels_a, data_b, labels_b,
                                                                              s_and_t)
        data_s_or_t, labels_s_or_t = generate_meta_train_groups_with_vector(data_a, labels_a, data_b, labels_b, s_or_t)

    return data_s, labels_s, data_t, labels_t, data_s_and_t, labels_s_and_t, data_s_or_t, labels_s_or_t



def generate_meta_train_groups_aug(data, labels, type="sample"):
    rand = torch.multinomial(torch.ones_like(labels).float(), labels.size(0), replacement=False)
    slice1 = int(rand.size(0) / 3)
    slice2 = int(rand.size(0) * 2 / 3)

    data_s_and_t = data[rand[:slice1]]
    data_s = torch.cat((data[rand[:slice1]], data[rand[slice1:slice2]]), 0)
    data_t = torch.cat((data[rand[:slice1]], data[rand[slice2:]]), 0)
    data_s_or_t = data[rand[:]]

    labels_s_and_t = labels[rand[:slice1]]
    labels_s = torch.cat((labels[rand[:slice1]], labels[rand[slice1:slice2]]), 0)
    labels_t = torch.cat((labels[rand[:slice1]], labels[rand[slice2:]]), 0)
    labels_s_or_t = labels[rand[:]]

    return data_s, labels_s, data_t, labels_t, data_s_and_t, labels_s_and_t, data_s_or_t, labels_s_or_t


def meta_loss(encoder, classifier, data, labels, test_data, test_labels):
    meta_encoder = encoder.clone()
    meta_classifier = classifier.clone()
    feature = meta_encoder(data)
    logit = meta_classifier(feature)
    loss = nn.CrossEntropyLoss(reduction='sum')(logit, labels)
    meta_classifier.adapt(loss)
    meta_encoder.adapt(loss)
    feature_test = meta_encoder(test_data)
    logit_test = meta_classifier(feature_test)
    loss_test = nn.CrossEntropyLoss()(logit_test, test_labels)

    return loss_test