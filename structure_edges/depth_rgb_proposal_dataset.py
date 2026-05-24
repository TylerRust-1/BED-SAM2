import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from saliency_depth import saliency_depth_calc
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


class FullDataset(Dataset):
    def __init__(self, args):
        image_root = os.path.join(args.path, "RGB\\")
        gt_root = os.path.join(args.path, "GT\\")
        depth_root = os.path.join(args.path, "monocular_depth\\")

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)

    def __getitem__(self, idx):
        edges_save_path = self.depths[idx].replace('monocular_depth', 'edges')  # or wherever you want
        os.makedirs(os.path.dirname(edges_save_path), exist_ok=True)

        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        depth = self.depth_loader(self.depths[idx])

        combined_edges = saliency_depth_calc(depth, image, label)
        # combined_edges.save(edges_save_path) 
        return 0, edges_save_path

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            rgb = img.convert('RGB')
            return rgb

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # img = PIL.ImageOps.invert(img)  # invert depth map
            return img.convert('L')



#### For edge generation, run this script to generate and save edge maps for all images in the dataset.
parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--num_workers", default=4, type=int)
args = parser.parse_args() 

def main(args):
    dataset = FullDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)
    
    for combined, path in tqdm(loader):
        continue
        # print(f"Saved: {path}")

if __name__ == "__main__":
    args.batch_size = 1
    # Edge generation is CPU-bound and independent per image, so fan out across
    # all cores. (OpenCV is pinned to 1 thread/worker in saliency_depth.py to
    # avoid oversubscription.) batch_size stays 1: images have varying sizes.
    args.num_workers = os.cpu_count() or 4

    ############### RGB SOD Datasets #################
    # args.path = "..\\datasets\\SOD\\DUTS-TE\\"

    args.path = "C:\\Projects\\datasets\\SOD\\DUTS-TR"

    # args.path = "..\\datasets\\SOD\\DUT-OMRON\\"

    # args.path = "..\\datasets\\SOD\\ECSSD\\"

    # args.path = "..\\datasets\\SOD\\HKU-IS\\"

    # args.path = "..\\datasets\\SOD\\PASCAL-S\\"

    # args.path = "..\\datasets\\SOD\\HRSOD-TR\\"

    # args.path = "..\\datasets\\SOD\\UHRSD-TR\\"

    # args.path = "..\\datasets\\SOD\\DAVIS-S\\"

    ############### RGB-D SOD Datasets #################

    # args.path = "..\\datasets\\RGB-D_SOD\\NJU2K_NLPR\\NJU2K_NLPR_Train\\"

    # args.path = "..\\datasets\\RGB-D_SOD\\NJU2K\\NJU2K_Test\\"

    # args.path = "..\\datasets\\RGB-D_SOD\\NLPR\\NLPR_Test\\"

    # args.path = "..\\datasets\\RGB-D_SOD\\SIP\\"

    # args.path = "..\\datasets\\RGB-D_SOD\\STERE\\"
 
    ############## COD Datasets ##################
    
    # args.path = "..\\datasets\\COD\\CAMO\\Test\\"

    # args.path = "..\\datasets\\COD\\CHAMELEON\\Test\\"

    # args.path = "..\\datasets\\COD\\COD10K\\Test\\"

    # args.path = "..\\datasets\\COD\\NC4K\\"

    main(args)