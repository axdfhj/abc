import glob
import logging
import os
import random
import json

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset


from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset, \
    OurInpaintingDataset as OurInpaintingEvaluationDataset, ceil_modulo, InpaintingEvalOnlineDataset
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


# class InpaintingTrainDataset(Dataset):
#     def __init__(self, indir, mask_generator, transform):
#         self.in_files = list(glob.glob(os.path.join(
#             indir, '**', '*.jpg'), recursive=True))
#         self.mask_generator = mask_generator
#         self.transform = transform
#         self.iter_i = 0

#     def __len__(self):
#         return len(self.in_files)

#     def __getitem__(self, item):
#         path = self.in_files[item]
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = self.transform(image=img)['image']
#         img = np.transpose(img, (2, 0, 1))
#         # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
#         mask = self.mask_generator(img, iter_i=self.iter_i)  # 修改mask生成方式
#         self.iter_i += 1
#         return dict(image=img,
#                     mask=mask)


class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        super().__init__()
        self.in_files = list(glob.glob(os.path.join(
            indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.mask_transform = get_musk_transforms('default', 720)
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def bbox_index(self, mask):
        length = mask.shape[1]
        mask = np.reshape(mask, (-1))
        fir = np.argmax(mask)
        mask = mask[::-1]
        last = mask.shape[0]-np.argmax(mask)-1
        return fir//length, fir % length, last//length, last % length

    def modify_human(self, human, bbox1, bbox2, shape):
        print(bbox1)
        print(bbox2)
        delta_x1 = bbox1[2]-bbox1[0]+1
        delta_y1 = bbox1[3]-bbox1[1]+1
        delta_x2 = bbox2[2]-bbox2[0]+1
        delta_y2 = bbox2[3]-bbox2[1]+1
        ideal_x = bbox1[0]+delta_x1//2-delta_x2//2
        ideal_y = bbox1[1]+delta_y1//2-delta_y2//2
        x = max(ideal_x, 0)
        x = min(x, shape[0]-delta_x2)
        y = max(ideal_y, 0)
        y = min(y, shape[1]-delta_y2)
        modified_human = np.zeros(shape)[None, ...]
        modified_human = np.tile(modified_human, (3, 1, 1))
        modified_human[..., x:x+delta_x2, y:y+delta_y2] = human[...,
                                                                bbox2[0]:bbox2[0]+delta_x2, bbox2[1]:bbox2[1]+delta_y2]
        return modified_human

    def __getitem__(self, item1):
        item2 = random.randint(0, len(self.in_files)-1)
        ls = []
        for i in (item1, item2):
            path = self.in_files[i]
            path_json = path[:-4]+'.json'
            with open(path_json, 'r') as file:
                bbox = json.load(file)[0]
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            mask = self.mask_generator(img.shape[:-1], bbox)
            mask = np.tile(mask, (1, 1, 3)).astype(np.uint8)
            invalid = True
            times = 0
            while(invalid):
                seed = random.randint(0, 12345649879)
                random.seed(seed)
                mask_t = self.mask_transform(image=mask)['image']
                mask_t = np.transpose(mask_t, (2, 0, 1))
                mask_t = (mask_t > np.mean(mask_t))*1.0
                bbox = self.bbox_index(mask_t[0])
                if np.prod(bbox-np.array([0, 0, mask_t[0].shape[0], mask_t[0].shape[1]])) != 0 or times == 20:
                    mask = mask_t
                    invalid = False
                else:
                    times += 1
                    continue
            random.seed(seed)
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            if i == item1:
                ls.append(img)
                ls.append(mask)
                bbox1 = bbox
            else:
                ls.append(self.modify_human(
                    img*mask, bbox1, bbox, img[0].shape))
        #可视化图像
        # img = np.transpose(ls[0], (1, 2, 0))
        # mask = np.transpose(ls[1], (1, 2, 0))
        # human = np.transpose(ls[2], (1, 2, 0))
        # cv2.imwrite('./background.jpg', img*(1-mask)*255)
        # cv2.imwrite('./human.jpg', human*255)
        # set_trace()
        return dict(image=ls[0], mask=ls[1], human=ls[2])


class InpaintingTrainWebDataset(IterableDataset):
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = webdataset.Dataset(indir).shuffle(
            shuffle_buffer).decode('rgb').to_tuple('jpg')
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img,) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype('uint8')
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img,
                       mask=mask)


class ImgSegmentationDataset(Dataset):
    def __init__(self, indir, mask_generator, transform, out_size, segm_indir, semantic_seg_n_classes):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(glob.glob(os.path.join(
            indir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size))
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img)
        segm, segm_classes = self.load_semantic_segm(path)
        result = dict(image=img,
                      mask=mask,
                      segm=segm,
                      segm_classes=segm_classes)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(
            self.indir, self.segm_indir).replace(".jpg", ".png")
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int)-1, 0, None))
        # w x h x n_classes
        ohe = F.one_hot(tensor.long(), num_classes=self.semantic_seg_n_classes)
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()

        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def get_musk_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size,
                          border_mode=cv2.BORDER_CONSTANT, mask_value=0),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            # A.CLAHE(),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.2, contrast_limit=0.2),
            # A.HueSaturationValue(
            #     hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()

        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir, kind='default', out_size=736, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="default", dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    LOGGER.info(
        f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}')

    mask_generator = get_mask_generator(
        kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'default_web':
        dataset = InpaintingTrainWebDataset(indir=indir,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(
            dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, kind='default', out_size=512, transform_variant='default', **kwargs):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, out_size=out_size, transform_variant=transform_variant, **kwargs) for idir in indir
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get(
        "mask_generator_kind"), kwargs=kwargs.get("mask_gen_kwargs"))

    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'our_eval':
        dataset = OurInpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'online':
        dataset = InpaintingEvalOnlineDataset(indir=indir,
                                              mask_generator=mask_generator,
                                              transform=transform,
                                              out_size=out_size,
                                              **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(max_size, ceil_modulo(
            random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(
            area // out_height, round_to_mod))
    else:
        out_width = min(max_size, ceil_modulo(
            random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(
            area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)
