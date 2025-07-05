import os
import copy
import random
from tqdm import tqdm
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

from lib.dataset.objectnet_dataset import ObjectNetDataset
from lib.dataset.imagenet_v2 import ImageNetV2

import torch
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True

def reindex_dataset(dataset, mask):
    """
    Reindex dataset according to mask
    """
    temp = np.nonzero(mask)[0]
    converter = {}
    for i, t in enumerate(temp):
        converter[i] = t # convert imagenet-a/r class to imagenet class
    dataset.targets = [converter[t] for t in dataset.targets]
    for i, sample in enumerate(dataset.samples):
        dataset.samples[i] = (sample[0], dataset.targets[i])
    return dataset


def get_concated_imagenet_c(root, preprocess=None):
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    datasets = []
    print("Load imagenet-c")
    for corruption in tqdm(corruptions):
        for severity in range(1,6):
            path = os.path.join(root, 'imagenet-c', corruption, str(severity))
            dataset = ImageFolder(path, transform=preprocess)
            datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    return dataset

def get_dataset(args, preprocess):
    if args.dataset == 'objectnet':
        dataset = ObjectNetDataset(root=f"{args.root}/objectnet-1.0", transform=preprocess, reindex=True)
    elif args.dataset == 'imagenet-c':
        dataset = get_concated_imagenet_c(args.root, preprocess)
    elif args.dataset == 'imagenet-v2':
        dataset = ImageNetV2(os.path.join(args.root, args.dataset), transform=preprocess)
    else:
        dataset = ImageFolder(os.path.join(args.root, args.dataset), transform=preprocess)
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    val_dataset = copy.deepcopy(dataset)
    train_dataset = dataset
    test_dataset = ImageFolder(os.path.join(args.root, 'imagenet/val'), transform=preprocess)

    return train_dataset, val_dataset, test_dataset

