from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import math
from csv import DictReader

class MyDataset(Dataset):
    def __init__(self, version='it6',split='train', transform=None, target_transform=None, url_csv_file=None, file_suffix=None):

        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.images = []
        self.targets = []        
        self.version = '_'+version if split == 'train' else ''
        
        # LOAD SPLIT CSV FILE
        
        self.root_dir = url_csv_file
        with open(self.root_dir + self.split + file_suffix + self.version+ '.csv') as f:
            csv_file = DictReader(f)
            for row in csv_file:
                self.images.append(row["image_urls"])
                self.targets.append(row["target_urls"])

        
            
            """
            self.root_dir = url_dataset
            #super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
            self.images_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
            self.targets_dir = os.path.join(self.root_dir, 'gtFine', self.split)
            self.images = []
            self.targets = []
            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    img_id = file_name.split('_leftImg8bit')[0]
                    target_name = '{}_{}_{}'.format(img_id,'gtFine', 'labelTrainIds.png')
                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(os.path.join(target_dir, target_name))
          """
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        image = Image.open(self.images[index]) #.convert('RGB')        
        target = Image.open(self.targets[index])
        #target2 = io.imread(self.targets[index])
         # Convertir la PIL image a un tensor manualmente para que no haga la normalizacion        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target = torch.from_numpy(np.array(target))
        return image, target