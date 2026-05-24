import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ToTensor(object):
    def __call__(self, data):
        data['image'] = F.to_tensor(data['image'])
        data['label'] = F.to_tensor(data['label'])
        data['edges'] = F.to_tensor(data['edges'])
        return data

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data['image'] = F.resize(data['image'], self.size)
        data['label'] = F.resize(data['label'], self.size)
        data['edges'] = F.resize(data['edges'], self.size)
        return data

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data['image'] = F.hflip(data['image'])
            data['label'] = F.hflip(data['label'])
            data['edges'] = F.hflip(data['edges'])
        return data

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            data['image'] = F.vflip(data['image'])
            data['label'] = F.vflip(data['label'])
            data['edges'] = F.vflip(data['edges'])
        return data

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data['image'] = F.normalize(data['image'], self.mean, self.std)
        return data

class FullDataset(Dataset):
    def __init__(self, args, size, mode):
        image_root = os.path.join(args.train_path, "RGB\\")
        gt_root = os.path.join(args.train_path, "GT\\")
        edge_root = os.path.join(args.train_path, "edges\\")
        # edge_root = os.path.join(args.path, "binarized_depth_edges\\")
        # edge_root = os.path.join(args.path, "edges_old\\")
        # edge_root = os.path.join(args.path, "orig_edges\\")
        # edge_root = os.path.join(args.path, "depth\\")
        # edge_root = os.path.join(args.path, "monocular_depth\\")

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        edges = self.binary_loader(self.edges[idx])

        data = {'image': image, 'label': label, 'edges': edges}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, args, size):
        image_root = os.path.join(args.path, "RGB\\")
        gt_root = os.path.join(args.path, "GT\\")
        edge_root = os.path.join(args.path, "edges\\")
        # edge_root = os.path.join(args.path, "binarized_depth_edges\\")
        # edge_root = os.path.join(args.path, "edges_old\\")
        # edge_root = os.path.join(args.path, "orig_edges\\")
        # edge_root = os.path.join(args.path, "depth\\")
        # edge_root = os.path.join(args.path, "monocular_depth\\")


        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.edge_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        self.gt_transform = transforms.ToTensor()
        self.length = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        edge = self.binary_loader(self.edges[self.index])
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        image = self.transform(image).unsqueeze(0)
        edge = self.edge_transform(edge).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        self.index += 1

        return image, gt, edge, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
