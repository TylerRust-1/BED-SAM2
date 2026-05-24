import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from model import BED_SAM2, print_trainable_params
from dataset import TestDataset
from tqdm import tqdm
import py_sod_metrics
import time
from ptflops import get_model_complexity_info
import copy

torch.backends.cudnn.benchmark = True
evaluate = False

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--checkpoint", type=str, help="path to the checkpoint of sam2-unet")
parser.add_argument("--version", type=str, help="path to save the predicted masks")
parser.add_argument("--test_image_path", type=str, help="path to the image files for testing")
parser.add_argument("--test_depth_path", type=str, help="path to the depth files for testing")
parser.add_argument("--test_mask_path", type=str, help="path to the mask files for testing")
parser.add_argument("--save_path", type=str, help="path to save the predicted masks")
parser.add_argument("--save_images", type=bool, help="bool to save SOD masks")
args = parser.parse_args()

def load_model(checkpoint_path, device):
    model = BED_SAM2().to(device)
    model.load_state_dict(torch.load(checkpoint_path), strict=True)
    model.eval()
    return model.to(device)

def run_inference(model, test_loader, device):
    x, gt, edge, name = test_loader.load_data()
    x = torch.cat((x, edge), dim=1)
    x = x.to(device)
    res, _, _ = model(x)
    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = torch.sigmoid(res).cpu().squeeze().numpy()
    return res, gt, name

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = TestDataset(args, args.size)

    model = load_model(args.checkpoint, device)

    #########################################################
    ### Model Inference and Evaluation
    if evaluate:
        print_trainable_params(model)

        model_for_flops = copy.deepcopy(model)
        
        # 2. Run FLOPs on the COPY
        macs, params = get_model_complexity_info(
            model_for_flops, 
            (4, args.size, args.size), 
            as_strings=True, 
            verbose=False, 
            print_per_layer_stat=False
        )
        print(f"Computational complexity: {macs}, Parameters: {params}")
        
        # 3. Delete the copy to free up memory immediately
        del model_for_flops
        torch.cuda.empty_cache() 
        # --- END CHANGE ---

        dummy_input = torch.randn(1, 4, args.size, args.size).to(device)
        with torch.no_grad():
            # Warmup
            for _ in range(20): 
                _ = model(dummy_input)
            torch.cuda.synchronize()
            
            # Measurement
            start = time.time()
            for _ in range(100): 
                _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()

        total_time_seconds = end - start
        time_per_image_seconds = total_time_seconds / 100
        time_per_image_ms = time_per_image_seconds * 1000

        print(f"Runtime: {time_per_image_ms:.2f} ms per image")

    ########################################################

    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE_ = py_sod_metrics.MAE()  # Keep this separate
    WF = py_sod_metrics.WeightedFmeasure()

    # Setup FMv2 for curve-based metrics (F-measure, IoU)
    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "iou": py_sod_metrics.IOUHandler(**sample_gray),
        }
    )

    with torch.no_grad():
        with torch.autocast("cuda"):
            for i in tqdm(range(test_loader.length), desc="Evaluating"):
                pred, gt, name = run_inference(model, test_loader, device)
                
                ### FmeasureV2 needs float
                FMv2.step(pred=pred.astype(np.float64), gt=gt.astype(np.float64))

                pred = (pred * 255).astype(np.uint8)
                gt = gt.astype(np.uint8)

                # Metrics that can work with uint8
                SM.step(pred=pred, gt=gt)
                EM.step(pred=pred, gt=gt)
                MAE_.step(pred=pred, gt=gt)
                WF.step(pred=pred, gt=gt)

                if args.save_images == True:
                    last_token = os.path.basename(name)
                    filepath = os.path.join(args.save_path, last_token[:-4] + ".png")
                    imageio.imsave(filepath, pred)

    fmv2 = FMv2.get_results()

    results = {
        "mIoU": fmv2["iou"]["dynamic"].mean(),
        "Smeasure": SM.get_results()["sm"],
        "maxFm": fmv2["fm"]["dynamic"].max(),
        "w_Fm": WF.get_results()["wfm"],
        "meanEm": EM.get_results()["em"]["curve"].mean(),
        "MAE": MAE_.get_results()["mae"], 
    }

    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    args.save_images = False

    ################################ SOD Checkpoints ##################################
    args.size = 352
    args.checkpoint = ".\\checkpoints\\BED-SAM2-best.pth"

    ################################ Evaluation Datasets ##################################

    args.path = ".\\datasets\\COD\\CAMO\\Test\\"
    args.save_path = ".\\test\\CAMO\\"

    main(args)

    