import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import json
import collections
import PIL
from PIL import Image
import random
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torchvision
import random
# from utils.vis import *
from scipy import ndimage
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
import math
from PIL import Image, ImageOps


class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):

        # The input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        max_len = {k: -1 for k in keys}

        trailing_dims = {k: None for k in keys}

        # Make first pass to get shape info for padding
        for _, labels in batch:
            for k in keys:
                try:
                    max_len[k] = max(max_len[k], len(labels[k]))
                    if len(labels[k]) > 0:
                        trailing_dims[k] = labels[k].size()[1:]
                except TypeError:   # scalar
                    pass

        pad = MultiObjectDataLoader._pad_tensor
        for i in range(len(batch)):
            for k in keys:
                if trailing_dims[k] is None:
                    continue
                size = [max_len[k]] + list(trailing_dims[k])
                batch[i][1][k] = pad(batch[i][1][k], size)


        return default_collate(batch)

    @staticmethod
    def _pad_tensor(x, size, value=None):
        assert isinstance(x, torch.Tensor)
        input_size = len(x)
        if value is None:
            value = float('nan')

        # Copy input tensor into a tensor filled with specified value
        # Convert everything to float, not ideal but it's robust
        out = torch.zeros(*size, dtype=torch.float)
        out.fill_(value)
        if input_size > 0:  # only if at least one element in the sequence
            out[:input_size] = x.float()
        return out


class ImageNet_votecut_top1_Dataset(Dataset):
    def __init__(self, image_size, split='train', args=None):
        self.image_size = image_size
        self.args = args
        self.split = split

        self.image_folder = 'path to imagenet_trainset'
        self.mask_folder = 'path to masks_top1_single_component folder' ## parse with utils/preprocess_votecut.py


        self.fname_list = []
        for class_name in os.listdir(self.mask_folder):
            fnames = sorted(os.listdir(os.path.join(self.mask_folder, class_name)))
            fnames = [os.path.join(class_name, fname) for fname in fnames]
            self.fname_list.extend(fnames)

        if not args.eval_mode and split == 'train':
            random.shuffle(self.fname_list)


        self.image_resize = transforms.Resize((self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

        self.image_resize_before_crop = transforms.Resize((400, 400), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.mask_resize_before_crop = transforms.Resize((400, 400), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        

    def __getitem__(self, index):
        
        return self.get_single_item(index, random_crop=True)
    
    def get_single_item(self, index, random_crop=False):
        filename = self.fname_list[index]
        
        image = cv2.imread(os.path.join(self.image_folder, filename.replace('.png','.JPEG')))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_folder, filename.replace('.JPEG','.png')), cv2.IMREAD_GRAYSCALE)
        if mask.shape[1] != image.shape[1] or mask.shape[0] != image.shape[0]:
            mask_rotated = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            mask = mask_rotated 
        image = self.to_tensor(image)
        mask = self.to_tensor(mask) * 255

        if mask.max() > 128:
            mask = torch.where(mask>0, 1, 0)
        else:
            mask = torch.where(mask==1, 1.0, 0.0)
            

        mask = mask[0]
        mask = mask.to(torch.int)
        mask = torch.where(mask>0, 1, 0)
        y, x = torch.where(mask>0)
        if len(y) == 0 or len(x) == 0:
            image = self.image_resize(image)
            labels = {
                'center_field': torch.zeros((2, self.image_size, self.image_size)),
                'saliency_mask': torch.zeros((self.image_size, self.image_size)),
                'instance_mask': torch.zeros((self.image_size, self.image_size)),
                'object_center': torch.tensor([0, 0]),
                'sdf': torch.zeros((self.image_size, self.image_size)),
                }
            return image, labels

        image = self.image_resize_before_crop(image) 
        mask = self.mask_resize_before_crop(mask.unsqueeze(0))[0]
        y, x = torch.where(mask>0)
        if len(y) == 0 or len(x) == 0:
            image = self.image_resize(image)
            labels = {
                'center_field': torch.zeros((2, self.image_size, self.image_size)),
                'saliency_mask': torch.zeros((self.image_size, self.image_size)),
                'instance_mask': torch.zeros((self.image_size, self.image_size)),
                'object_center': torch.tensor([0, 0]),
                'sdf': torch.zeros((self.image_size, self.image_size)),
                }
            return image, labels
        
        obj_x_center = (torch.min(x) + torch.max(x)) / 2
        obj_y_center = (torch.min(y) + torch.max(y)) / 2

        if random_crop:
            sdf = cv2.distanceTransform(np.uint8(mask.numpy()), cv2.DIST_L2, 3)
            sdf = sdf / sdf.max() if sdf.max() > 0 else sdf
            sdf = torch.tensor(sdf)
            all_data = torch.cat((image, sdf.unsqueeze(0), mask.unsqueeze(0))) ## [3+1+1, H, W]

            crop = transforms.RandomResizedCrop(size=(self.image_size, self.image_size))
            params = crop.get_params(all_data, scale=(self.args.random_crop_scale_min, self.args.random_crop_scale_max), ratio=(0.75, 1.33)) ## [top, left, height, width]
            all_data = transforms.functional.crop(all_data, *params)

            image = all_data[0:image.shape[0]] ## [3, H, W]
            sdf = all_data[image.shape[0]:image.shape[0]+sdf.shape[0]] ## [1, H, W]
            mask = all_data[-1] ## [H, W]
            image = self.image_resize(image) 
            mask = self.mask_resize(mask.unsqueeze(0))[0]
            sdf = self.image_resize(sdf)[0]

            top, left, height, width = params

            crop_center_y = (obj_y_center - top) * (self.image_size/height)
            crop_center_x = (obj_x_center - left) * (self.image_size/width)
            object_center = torch.tensor([crop_center_x, crop_center_y])
        else:
            object_center = torch.tensor([obj_x_center*(self.image_size/mask.shape[1]), obj_y_center*(self.image_size/mask.shape[0])])
            image = self.image_resize(image) 
            mask = self.mask_resize(mask.unsqueeze(0))[0]
            sdf = cv2.distanceTransform(np.uint8(mask.numpy()), cv2.DIST_L2, 3)
            sdf = sdf / sdf.max() if sdf.max() > 0 else sdf
            sdf = torch.tensor(sdf)

        if self.args.use_bg_sdf:
            ## create negative sdf on the background
            bg_mask = torch.where(mask==0, 1, 0)
            bg_sdf = cv2.distanceTransform(np.uint8(bg_mask.numpy()), cv2.DIST_L2, 3)
            bg_sdf = bg_sdf / bg_sdf.max() if bg_sdf.max() > 0 else bg_sdf
            bg_sdf = torch.tensor(bg_sdf) * (-1)
            sdf = sdf + bg_sdf


        xv, yv = torch.meshgrid([torch.arange(mask.shape[0]),torch.arange(mask.shape[1])])
        grid = torch.stack((xv, yv), 2).view((1, mask.shape[0], mask.shape[1], 2)).float() ## [1, image_size, image_size, 2]
        grid = grid.squeeze(0).permute(2,0,1) ## [2, H, W]

        center_field = torch.zeros_like(grid) ## [2, H, W]

        object_center_field = grid - torch.tensor([object_center[1], object_center[0]]).unsqueeze(1).unsqueeze(1) ## [2, H, W]
        object_center_field = torch.nn.functional.normalize(object_center_field, dim=0)
        center_field += torch.where(mask>0, 1, 0) * object_center_field
        # if self.args.use_reverse_center_field:
        #     center_field -= torch.where(mask>0, 0, 1) * object_center_field
            
        mask = mask.to(torch.int)
        center_field = torch.nn.functional.normalize(center_field, dim=0)

            
        labels = {
                'center_field': center_field,
                'saliency_mask': torch.where(mask>0, 1, 0),
                'instance_mask': mask,
                'object_center': object_center,
                'sdf': sdf,
                } 

        return image, labels
      
    def __len__(self):
        return len(self.fname_list)

class ImageNet_votecut_top1_Loader:
    
    def __init__(self, args, cuda=torch.cuda.is_available()):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}

        train_set = ImageNet_votecut_top1_Dataset(image_size=args.image_size, split='train', args=args)
        test_set = ImageNet_votecut_top1_Dataset(image_size=args.image_size, split='test', args=args)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]


class ImageNet_votecut_labeled_classifier_Dataset(Dataset):
    def __init__(self, image_size, split='train', args=None):
        self.image_size = image_size
        self.args = args
        self.split = split

        self.image_folder = 'path to imagenet_trainset'
        self.mask_folder = 'path to masks_top1_single_component folder' ## parse with utils/preprocess_votecut.py
        self.full_mask_folder = 'path to votecut original masks' ## parse with utils/vis_votecut.py

        self.fname_list = []
        for class_name in os.listdir(self.mask_folder):
            fnames = sorted(os.listdir(os.path.join(self.mask_folder, class_name)))
            fnames = [os.path.join(class_name, fname) for fname in fnames]
            self.fname_list.extend(fnames)
        # self.fname_list = sorted(self.fname_list)
        if not args.eval_mode and split == 'train':
            random.shuffle(self.fname_list)


        self.image_resize = transforms.Resize((self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()



    def __getitem__(self, index):
        random_number = random.random()
        use_bg_image = False
        if random_number < 0.5:
            use_bg_image = True

        filename = self.fname_list[index]
        
        if use_bg_image:
            try:      
                image = cv2.imread(os.path.join(self.image_folder, filename.replace('.png','.JPEG')))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(os.path.join(self.full_mask_folder, filename.replace('.JPEG','.png')), cv2.IMREAD_GRAYSCALE)
                mask = np.array(mask>0).astype(np.uint8)

                if mask.shape[1] != image.shape[1] or mask.shape[0] != image.shape[0]:
                    mask_rotated = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                    mask = mask_rotated 
                
                bg_mask = 1 - mask
                paded_bg_mask = cv2.copyMakeBorder(bg_mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
                bg_sdf = cv2.distanceTransform(paded_bg_mask, cv2.DIST_L2, 3)
                bg_sdf = bg_sdf[10:-10, 10:-10] 
                y_center, x_center = np.unravel_index(bg_sdf.argmax(), bg_sdf.shape)
                center_radius = bg_sdf[y_center, x_center]
                x1 = int(x_center - center_radius)
                y1 = int(y_center - center_radius)
                x2 = int(x_center + center_radius)
                y2 = int(y_center + center_radius)


                image = image[y1:y2, x1:x2,:]

                image = self.to_tensor(image)
                image = self.image_resize(image)

                class_label = torch.tensor([0.0])
                
                return image, {'class_label': class_label}
            except Exception as e:
                pass

        image = cv2.imread(os.path.join(self.image_folder, filename.replace('.png','.JPEG')))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_folder, filename.replace('.JPEG','.png')), cv2.IMREAD_GRAYSCALE)

        if mask.shape[1] != image.shape[1] or mask.shape[0] != image.shape[0]:
            mask_rotated = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            mask = mask_rotated 

        image = self.to_tensor(image)
        mask = self.to_tensor(mask) 

        crop = transforms.RandomResizedCrop(size=(self.image_size, self.image_size))
        image_and_mask = crop(torch.cat([image, mask],dim=0))

        image = image_and_mask[0:3]
        mask = image_and_mask[3:]
        if mask.sum() > 1:
            class_label = torch.tensor([1.0])
        else:
            class_label = torch.tensor([0.0])

    
        return image, {'class_label': class_label}


    def __len__(self):
        return len(self.fname_list)

class ImageNet_votecut_labeled_classifier_Loader:
    
    def __init__(self, args, cuda=torch.cuda.is_available()):

        # Default arguments for dataloaders
        kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if cuda else {}
        # kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}

        train_set = ImageNet_votecut_labeled_classifier_Dataset(image_size=args.image_size, split='train', args=args)
        test_set = ImageNet_votecut_labeled_classifier_Dataset(image_size=args.image_size, split='test', args=args)

        # Dataloaders
        self.train_loader = MultiObjectDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.test_loader = MultiObjectDataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        self.data_shape = self.train_loader.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]

class COCO_Dataset(Dataset):
    def __init__(self, image_size, split, args=None):
        self.image_size = image_size
        self.args = args
        
        self.image_folder = 'path to train2017' if split=='train' else 'path to val2017'
        self.gt_annotations_root = 'path to instances_train2017.json' if split=='train' else 'path to instances_val2017.json'
        self.coco = COCO(self.gt_annotations_root)


        with open(self.gt_annotations_root) as f:
            self.gt_annotations = json.load(f)
        
        self.gt_instance_annotations = self.gt_annotations['annotations']

        self.image_resize = transforms.Resize((self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((self.image_size, self.image_size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

        ## formuate imageid - fname dictionary
        self.image_id_to_fname = {}
        self.fname_to_image_id = {}
        self.image_id_to_image_info = {}
        for image_info in self.gt_annotations['images']:
            self.image_id_to_fname[int(image_info['id'])] = image_info['file_name']
            self.fname_to_image_id[image_info['file_name'].replace('jpg', 'jpg')] = int(image_info['id'])
            self.image_id_to_image_info[int(image_info['id'])] = image_info

        self.fname_to_annotation_idx_dict = {}
        for ann_idx in range(0, len(self.gt_instance_annotations)):
            ann = self.gt_instance_annotations[ann_idx]
            if ann is None:
                continue
            category_id = ann['category_id']
            # class_name = self.CLASS_IDX_TO_CLASS_NAME_DICT[str(category_id)]
            # if class_name not in self.selected_classes:
            #     continue
            filename = self.image_id_to_fname[ann['image_id']].replace('jpg', 'jpg')
            if filename in self.fname_to_annotation_idx_dict.keys():
                self.fname_to_annotation_idx_dict[filename].append(ann_idx)
            else:
                self.fname_to_annotation_idx_dict[filename] = [ann_idx]

        self.image_filenames = list(set(os.listdir(self.image_folder)))
        self.image_filenames = sorted(self.image_filenames)
        
        
        if self.args is not None:
            if self.args.start_idx != -1 and self.args.end_idx != -1:
                print('select from', self.args.start_idx, 'to', self.args.end_idx)
                self.image_filenames = self.image_filenames[self.args.start_idx:self.args.end_idx]

        self.sample_idx_list = np.arange(len(self.image_filenames))



    def __len__(self):
        return len(self.image_filenames)
    
    def get_image_with_index(self, index):
        sample_idx = self.sample_idx_list[index]
        filename = self.image_filenames[sample_idx]
        image = Image.open(os.path.join(self.image_folder, filename)).convert('RGB') 
        image = self.to_tensor(image)

        labels = {
            'image_id': torch.tensor(self.fname_to_image_id[filename])
        }
        return image, labels
    
    def get_image_with_image_id(self, image_id):
        filename = self.image_id_to_fname[image_id]
        image = Image.open(os.path.join(self.image_folder, filename)).convert('RGB') 
        image = self.to_tensor(image)
        if self.args.normalize_image:
            image = self.normalize(image)
        labels = {
            'image_id': torch.tensor(image_id)
        }
        return image, labels
    
