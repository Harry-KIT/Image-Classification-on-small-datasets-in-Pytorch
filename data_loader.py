import random
import torch
from torchvision import datasets, transforms
import os


class RandomDiscreteRotation(object):
    def __init__(self, angles):
        self.discrete_angles = angles

    def __call__(self, img):
        angle = random.choice(self.discrete_angles)
        return transforms.functional.rotate(img, angle)


class Dataloader(object):
    def __init__(self, data_dir, image_size, batch_size, shuffle = True, number_workers = 0):
        # self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.number_workers = number_workers
        self.image_size = image_size

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomDiscreteRotation([0, 90, 180, 270]),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ])
        }

        self._init_data_sets()
    
    def _init_data_sets(self):
        self.data_sets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}

        self.data_loaders = {x: torch.utils.data.DataLoader(self.data_sets[x], batch_size=self.batch_size,
                                                            shuffle=self.shuffle, num_workers=self.number_workers)
                             for x in ['train', 'val']}
        self.data_sizes = {x: len(self.data_sets[x]) for x in ['train', 'val']}
        self.data_classes = self.data_sets['train'].classes

    def load_data(self, data_set='train'):
        return self.data_loaders[data_set]

       