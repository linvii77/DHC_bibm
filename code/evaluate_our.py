import os
import sys
import numpy as np
import argparse
import SimpleITK as sitk
from medpy import metric
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_list
from utils.config import Config

CLASS_NAMES = [
    "Aorta", "Gallbladder", "Spleen", "Left Kidney", "Right Kidney",
    "Liver", "Stomach", "Aorta2", "IVC", "PSV", "Pancreas", "RAG", "LAG",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="synapse")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--cps", type=str, default="AB")
    args = parser.parse_args()

    config = Config(args.task)
    test_cls = list(range(1, config.num_cls))  # 1..13
    ids_list = read_list("test", task=args.task)

    pred_dir = f"./logs/{args.exp}/predictions_{args.cps}"
    label_dir = os.path.join(config.save_dir, "npy")

    values = np.zeros((len(test_cls), 2))  # [DSC, HD95] per class

    for data_id in tqdm(ids_list, desc=args.exp):
        pred_path = os.path.join(pred_dir, f"{data_id}.nii.gz")
        label_path = os.path.join(label_dir, f"{data_id}_label.npy")

        if not os.path.exists(pred_path):
            print(f"[WARN] missing prediction: {pred_path}", file=sys.stderr)
            continue
        if not os.path.exists(label_path):
            print(f"[WARN] missing label: {label_path}", file=sys.stderr)
            continue

        itk_pred = sitk.ReadImage(pred_path)
        pred = sitk.GetArrayFromImage(itk_pred)   # (D, W, H) int
        label = np.load(label_path)                # (D, W, H) float32 0-13

        pred = pred.astype(np.int32)
        label = label.astype(np.int32)

        for i, cls_idx in enumerate(test_cls):
            pred_i = (pred == cls_idx)
            label_i = (label == cls_idx)
            if pred_i.sum() > 0 and label_i.sum() > 0:
                dsc = metric.binary.dc(pred_i, label_i) * 100
                hd = metric.binary.hd95(pred_i, label_i)
                values[i] += np.array([dsc, hd])
            elif pred_i.sum() == 0 and label_i.sum() == 0:
                values[i] += np.array([100.0, 0.0])
            # else: pred wrong (FP or FN) → 0 contribution, already 0

    values /= len(ids_list)

    print(f"\n====== {args.exp} ======")
    print(f"{'Class':<20} {'DSC':>8} {'HD95':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<18} {values[i,0]:>8.2f} {values[i,1]:>8.2f}")
    mean_dsc = np.mean(values[:, 0])
    mean_hd  = np.mean(values[:, 1])
    print(f"  {'MEAN':<18} {mean_dsc:>8.2f} {mean_hd:>8.2f}")
    print(f"DSC_ARRAY = {np.round(values[:,0], 2).tolist()}")
    print(f"HD95_ARRAY = {np.round(values[:,1], 2).tolist()}")

