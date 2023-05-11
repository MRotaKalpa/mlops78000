import os
import pandas as pd
import torch
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import ai8x
import re

import torchvision
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

enabled_classes = ('negative', 'person',)
class_to_label = {
    k: i
    for i, k in enumerate(enabled_classes)
}

class TopViewedPeople(Dataset):

    def __init__(self, data_dir, train=True, transform=None):

        self.transform = transform
        self.train = train
        
        download = not os.path.exists(os.path.join(data_dir, 'CIFAR10'))
        
        if self.train:
            cifar_transform = transform['cifar10_train']
        else:
            cifar_transform = transform['cifar10_test']
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(data_dir, 'CIFAR10'),
            train=self.train, 
            download=download, 
            transform=cifar_transform)

        images = []
        labels = []

        img_folder = 'train_subset_canteen' if self.train else 'test_canteen'
        for img_path in (Path(data_dir) / img_folder).glob('*.png'):
            lbl = "person"
            if lbl in enabled_classes:
                images.append(img_path)
                labels.append(class_to_label[lbl])

        img_folder = 'train' if self.train else 'test'
        for img_path in (Path(data_dir) / img_folder).glob('*.jpg'):
            lbl = "person"
            if lbl in enabled_classes:
                images.append(img_path)
                labels.append(class_to_label[lbl])

        img_folder = 'train_subset_negatives' if self.train else 'test_negatives'
        for img_path in (Path(data_dir) / img_folder).glob('*.png'):
            lbl = "negative"
            if lbl in enabled_classes:
                images.append(img_path)
                labels.append(class_to_label[lbl])

        data = {'images': images, 'labels': labels}

        self.img_data = pd.DataFrame(data)

    def __len__(self):
        return len(self.cifar_dataset) + len(self.img_data)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if 0 <= index < len(self.cifar_dataset):
            return self.cifar_dataset[index]

        # shift due to the concatenation with cifar10
        index = index - len(self.cifar_dataset)
        img_name = os.path.join(self.img_data.loc[index, 'images'])

        image = Image.open(img_name)
        image = image.resize((40, 40))
        label = self.img_data.loc[index, 'labels']
        if self.transform is not None:
            if self.train:
                image = self.transform['train'](image)
            else:
                image = self.transform['test'](image)
        return image, label + 10  # + 10 is due to the concatenation with cifar10


def top_viewed_people_get_datasets(data, load_train=True, load_test=True):
    
    (data_dir, args) = data
    print(data_dir)
    
    train_dataset = None
    test_dataset = None
   
    data_transforms = {
        'cifar10_train': transforms.Compose([
             transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             ai8x.normalize(args=args)
        ]),
        'cifar10_test': transforms.Compose([
             transforms.ToTensor(),
             ai8x.normalize(args=args)
        ]),
        'train': transforms.Compose([
             transforms.RandomCrop(32),
             transforms.RandomAffine(degrees=10, translate=None, scale=None),
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             ai8x.normalize(args=args)
        ]),
        'test': transforms.Compose([
             transforms.CenterCrop((32, 32)),
             transforms.ToTensor(),
             ai8x.normalize(args=args)
        ])
    }
    
    if load_train:
        train_dataset = TopViewedPeople(data_dir + '/kalpa', train=True, transform=data_transforms)
    
    if load_test:
        test_dataset = TopViewedPeople(data_dir + '/kalpa', train=False, transform=data_transforms)
        
        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    
    return train_dataset, test_dataset


datasets = [
    {
        'name': 'top_viewed_people',
        'input': (3, 32, 32),
        'output': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                   'ship', 'truck') + enabled_classes,
        'loader': top_viewed_people_get_datasets,
    }
]
