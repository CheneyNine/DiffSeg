import imgaug as ia
import os
import torch
from PIL import Image
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class MyDataset(Dataset):
    def __init__(self, scale, mode='train', data_type='unhealthy'):
        super().__init__()
        self.mode = mode
        self.data_type = data_type
        # Adjust paths based on the data type (healthy or unhealthy)
        if mode == 'train':
            if self.data_type == 'healthy':
                self.img_path = os.path.join(self.data_type, 'img')
                #self.img_path = os.path.join(self.dataset_path, mode, 'img')
                self.mask_path = os.path.join(self.data_type, mode, 'mask')
            elif self.data_type == 'healthy_unhealthy':
                self.img_path = os.path.join('healthy', 'img')
                self.mask_path = os.path.join('healthy', 'target')
            else: #train mode with unhealthy
                self.img_path = os.path.join(self.data_type, mode, 'img')
                self.mask_path = os.path.join(self.data_type, mode, 'mask')
        else:  # 'unhealthy' or other types
            self.img_path = os.path.join(self.data_type, self.mode, 'img')
            self.mask_path = os.path.join(self.data_type, self.mode, 'mask')

        self.image_lists, self.label_lists = self.read_list(self.img_path)


        # Data augmentation sequence
        self.flip = iaa.SomeOf((1, 4), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.1),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Affine(rotate=(-5, 5),
                       scale={"x": (0.9, 1.1), "y": (0.8, 1.2)}),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.ContrastNormalization((0.5, 1.5))], random_order=True)
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        self.to_tensor = transforms.ToTensor()

    def read_list(self, img_path):
        img_list = []
        label_list = []

        for img_file in sorted(os.listdir(img_path)):
            img_list.append(os.path.join(img_path, img_file))
            if self.mode != 'test':
                label_file = img_file.replace('.jpg', '_segmentation.png')
                label_list.append(os.path.join(self.mask_path, label_file))


        print('img_list_1',img_list[0])
        print('label_list_1',label_list[0])
        return img_list, label_list

    def __getitem__(self, index):
        img_cur = Image.open(self.image_lists[index]).convert('RGB')

        if self.mode == 'train':
            # Convert PIL image to numpy array for augmentation
            img_np = np.array(img_cur)
            # Apply augmentations
            img_np = self.flip.augment_image(img_np)
            # Convert back to PIL image
            img_cur = Image.fromarray(img_np)

        # Resize and convert to tensor
        img = self.resize_img(img_cur)
        img = self.to_tensor(img)


        if self.mode == 'train' and self.data_type == 'unhealthy':
            return img
        elif self.mode == 'train' and self.data_type == 'healthy_unhealthy':

            label_cur = Image.open(self.label_lists[index]).convert('RGB')
            # Resize and convert to tensor
            label = self.resize_label(label_cur)
            label = self.to_tensor(label)

            return img, label

    def __len__(self):
        return len(self.image_lists)
