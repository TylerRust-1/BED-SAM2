import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from model import BED_SAM2, print_trainable_params
import time
import torch.optim as opt
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser("SAM2-DUNet")
parser.add_argument("--config", type=str, help="Path to a YAML training config (e.g. configs/sod.yaml)")
parser.add_argument("--hiera_path", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--path", type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--num_workers", default=4, type=int)
args = parser.parse_args() 

def load_config(path):
    if not path or not os.path.isfile(path):
        raise SystemExit(
            "Please pass a valid --config, e.g.\n"
            "  python train.py --config configs/sod.yaml\n"
            "  python train.py --config configs/rgbd_sod.yaml\n"
            "  python train.py --config configs/cod.yaml"
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.items():
        setattr(args, key, value)
    print(f"Loaded config from {path}")
    return args

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def main(args):
    start_time = time.time()
    device = torch.device("cuda")
    seed_torch()

    train_dataset = FullDataset(args, args.size, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)

    model = BED_SAM2(args.hiera_path)
    # Load pretrained weights if available
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint), strict=True)
    model.to(device)
    
    scaler = torch.GradScaler("cuda")
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-6)

    os.makedirs(args.save_path, exist_ok=True)

    model.train()

    for epoch in range(args.epoch):
        batch_loss = 0.0
        for i, batch in enumerate(train_loader):
            x = batch['image']
            target = batch['label'].to(device)
            edge = batch['edges']
            
            x = torch.cat((x, edge), dim=1)
            x = x.to(device)

            optimizer.zero_grad() 
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred0, pred1, pred2 = model(x)
                loss0 = structure_loss(pred0, target) 
                loss1 = structure_loss(pred1, target)
                loss2 = structure_loss(pred2, target)
                loss = loss0 + loss1 + loss2
                batch_loss += loss.item()
                
            # Scale gradients
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update Optimizer
            scaler.step(optimizer)
            scaler.update()
            
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{:.4f}".format(epoch + 1, i + 1, loss))

        batch_loss /= (i + 1)
        print(f"Epoch {epoch + 1} completed. Average Loss: {batch_loss:.4f}")
        
        # Update learning rate
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == args.epoch - 1 or epoch == 0:
            checkpoint_path = os.path.join(args.save_path, f'SAM2-UNet-{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Checkpoint Saved at {checkpoint_path}]")

        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    load_config(args.config)
    main(args)