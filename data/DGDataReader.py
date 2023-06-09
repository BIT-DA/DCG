from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from data.data_utils import *
import random
import cv2
import os
import torch
import torch.nn.functional as F

class DGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):
        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        if self.transformer is not None:
            img = self.transformer(img)
        label = self.labels[index]
        return img, label

class DC_DGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None,pre_transformer=None):
        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.pre_transformer = pre_transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        if self.pre_transformer is not None:
            pre_img = self.pre_transformer(img)
        if self.transformer is not None:
            img = self.transformer(img)
        label = self.labels[index]
        return pre_img, img, label


class FourierDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None, from_domain=None, alpha=1.0):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.from_domain = from_domain
        self.alpha = alpha
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)

        img_s, label_s, domain_s = self.sample_image(domain)
        img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
        img = [img_o, img_s, img_s2o, img_o2s]
        label = [label, label_s, label, label_s]
        domain = [domain, domain_s, domain_s, domain]
        return img, label, domain

    def sample_image(self, domain):
        if self.from_domain == 'all':
            domain_idx = random.randint(0, len(self.names)-1)
        elif self.from_domain == 'inter':
            domains = list(range(len(self.names)))
            domains.remove(domain)
            domain_idx = random.sample(domains, 1)[0]
        elif self.from_domain == 'intra':
            domain_idx = domain
        else:
            raise ValueError("Not implemented")
        img_idx = random.randint(0, len(self.names[domain_idx])-1)
        img_name_sampled = self.names[domain_idx][img_idx]
        img_name_sampled = os.path.join(self.args.input_dir, img_name_sampled)
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        label_sampled = self.labels[domain_idx][img_idx]
        return self.transformer(img_sampled), label_sampled, domain_idx



def get_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path)
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    return DGDataset(args, names, labels, img_transform)

def get_dc_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path)
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    img_pre_transform = get_pre_transform(image_size, crop, jitter)
    return DC_DGDataset(args, names, labels, img_transform, img_pre_transform)


def get_fourier_dataset(args, path, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
        from_domain = config["from_domain"]
        alpha = config["alpha"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return FourierDGDataset(args, names, labels, img_transform, from_domain, alpha)





