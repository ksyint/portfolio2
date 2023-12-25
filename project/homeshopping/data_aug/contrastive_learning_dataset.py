import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=0.2):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.7,1.2)),
        #                                     #   transforms.RandomResizedCrop(size=size),
        #                                       transforms.RandomHorizontalFlip(),
        #                                       transforms.RandomApply([color_jitter], p=0.8),
        #                                     #   transforms.RandomGrayscale(p=0.2),
        #                                     #   GaussianBlur(kernel_size=int(0.1 * size)),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize(0.5,0.5)])
        
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                              transforms.RandomResizedCrop(size=size, scale=(0.5,1.2)),
                                            transforms.RandomAffine(5, translate=(0.2,0.2), shear=10),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            # GaussianBlur(kernel_size=int(0.1 * size)),
                                            GaussianBlur(kernel_size=20),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.5,0.5)])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                                                          
                            'custom': lambda: datasets.ImageFolder(self.root_folder, 
                                                                   transform=ContrastiveLearningViewGenerator(
                                                                       self.get_simclr_pipeline_transform(224),
                                                                       n_views)),
                                            
                            'custom_labeled': lambda: SimCLRCustom(self.root_folder,
                                                                   transform=self.get_simclr_pipeline_transform((320,180))),
                                                                   
                            'livecommerce': lambda: LiveCommerce(self.root_folder,
                                                                   transform=self.get_simclr_pipeline_transform((320,180)))}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class SimCLRCustom(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, base_path = None, transform = None, csi=False):

        self.base_path = base_path
        self.transform = transform

        self.image_paths = []
        self.image_label_dict = dict()        
        self.label_image_dict = dict()

        self.labelnames = sorted(os.listdir(base_path))
        for label in self.labelnames:
            folderpath = os.path.join(base_path, label)
            namelist = sorted(os.listdir(folderpath))
            imgpaths = []
            for name in namelist:
                imgpath = os.path.join(folderpath, name)
                imgpaths.append(imgpath)
                self.image_paths.append(imgpath)
                self.image_label_dict[imgpath] = label
            self.label_image_dict[label] = imgpaths
        
        self.csi = csi


    def __getitem__(self, index):
        label_index = index

        label = self.labelnames[label_index]
        imgpath_list = self.label_image_dict[label]
        num_img_choices = len(imgpath_list)
        idx_a = np.random.randint(num_img_choices)
        idx_b = np.random.randint(num_img_choices)
        imgpath_a = imgpath_list[idx_a]
        imgpath_b = imgpath_list[idx_b]

        img1 = Image.open(imgpath_a).convert('RGB')
        img2 = Image.open(imgpath_b).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.csi:
            if np.random.random()<0.5:
                hue = (0.4, 0.5)
            else:
                hue = (-0.5,-0.4)
            
            size=224

            negative_transform = transforms.Compose([transforms.ColorJitter(saturation=[2.0,3.0], hue=hue),
                                                     transforms.RandomResizedCrop(size=size, scale=(0.5,1.2)),
                                                     transforms.RandomAffine(5, translate=(0.2,0.2), shear=10),
                                                     transforms.RandomHorizontalFlip(),
                                                     GaussianBlur(kernel_size=int(0.1 * size)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize(0.5,0.5)])
            img_a = Image.open(imgpath_a).convert('RGB')
            img_b = Image.open(imgpath_b).convert('RGB')
            img1_neg = negative_transform(img_a)
            img2_neg = negative_transform(img_b)
            return [img1, img1_neg, img2, img2_neg], label_index

        return [img1, img2], label_index

    def __len__(self):
        return len(self.labelnames)
    

class LiveCommerce(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, base_path = None, transform = None):

        self.base_path = base_path
        self.transform = transform

        self.video_frame_dict = dict()
        self.frame_imgpath_dict = dict()        

        self.videonames = sorted(os.listdir(base_path))
        for videoname in self.videonames:
            folderpath = os.path.join(base_path, videoname)
            positive_namelist = sorted(os.listdir(folderpath))
            positives = []
            for name in positive_namelist:
                positive_data = os.path.join(folderpath, name)
                positive_images = sorted(os.listdir(positive_data))
                imgpaths = []
                for imgname in positive_images:
                    imgpath = os.path.join(positive_data, imgname)
                    imgpaths.append(imgpath)

                self.frame_imgpath_dict[positive_data] = imgpaths
                positives.append(positive_data)
            self.video_frame_dict[videoname] = positives

    def __getitem__(self, index):

        video = self.videonames[index]
        positives = self.video_frame_dict[video]
        positive = np.random.choice(positives)
        positive_images = self.frame_imgpath_dict[positive]
        num_img_choices = len(positive_images)

        idx_a = np.random.randint(num_img_choices)
        idx_b = np.random.randint(num_img_choices)
        # print(positive)
        # print(positive_images[idx_a])
        imgpath_a = positive_images[idx_a]
        imgpath_b = positive_images[idx_b]

        img1 = Image.open(imgpath_a).convert('RGB')
        img2 = Image.open(imgpath_b).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [img1, img2], index

    def __len__(self):
        return len(self.videonames)