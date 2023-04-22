import os
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class IRSeg(data.Dataset):

    def __init__(self, cfg, mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing 数据集图片的预处理
        # 定于转换方式，将几个操作组合起来
        self.im_to_tensor = transforms.Compose([
            # 归一化到(0,1)
            transforms.ToTensor(),
            # nb到(-1,1)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        # 数据集所在目录
        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        # 裁切尺寸
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                # 亮度
                brightness=cfg['brightness'],
                # 对比度
                contrast=cfg['contrast'],
                # 饱和度
                saturation=cfg['saturation']),
            # 依据概率p对图片进行水平翻转
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            # 依据给定size随机剪裁
            RandomCrop(crop_size, pad_if_needed=True)
        ])


        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
            self.binary_class_weight = np.array([1.5121, 10.2388])
        # 解决图像分割中样本不平衡问题
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")

        # 读取训练集测试集目录
        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()


        image = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_rgb.png'))
        depth = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_th.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))
        bound = Image.open(os.path.join(self.root, 'bound', image_path+'.png'))
        binary_label = Image.open(os.path.join(self.root, 'binary_labels', image_path + '.png'))


        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            'bound': bound,
            'binary_label': binary_label,
            # 'attention_map': attention_map,
        }


        if self.mode in ['train', 'trainval'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)


        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()
        # sample['attention_map'] = torch.from_numpy(np.asarray(sample['attention_map'], dtype=np.int64) / 255.).long()

        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (64, 0, 128),  # car
            (64, 64, 0),  # person
            (0, 128, 192),  # bike
            (0, 0, 192),  # curve
            (128, 128, 0),  # car_stop
            (64, 64, 128),  # guardrail
            (192, 128, 128),  # color_cone
            (192, 64, 0),  # bump
        ]


if __name__ == '__main__':
    import json
    path = '/home/wby/Desktop/MGSGNet/configs/irseg.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    dataset = IRSeg(cfg, mode='test')
    print(len(dataset))
