import argparse
import os
import torch
import random
import numpy as np
from train import train_class
from test import test_class


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f">>> Random Seed set to: {seed}")


def parse_args():
    parser = argparse.ArgumentParser(description="ReMaskNet Training & Testing Framework")

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'all'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--classes', type=str, nargs='+', default=['bottle'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_root', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./MVTec_ad')
    parser.add_argument('--dtd_path', type=str, default='./dtd/images')
    parser.add_argument('--result_root', type=str, default='./results')

    parser.add_argument('--feature_layers', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--target_embed_dim', type=int, default=1536)
    parser.add_argument('--image_size', type=int, default=288)
    parser.add_argument('--resize', type=int, default=329)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--iteration', type=int, default=3)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--eval_iteration', type=int, default=3)

    args = parser.parse_args()
    return args


def run_training(args):
    os.makedirs(args.save_root, exist_ok=True)

    print(f"\n{'=' * 40}")
    print(f"STARTING TRAINING | Device: {args.device}")
    print(f"Classes: {args.classes}")
    print(f"{'=' * 40}")

    for class_name in args.classes:
        train_class(args, class_name)


def run_testing(args):
    os.makedirs(args.result_root, exist_ok=True)
    import csv

    csv_path = os.path.join(args.result_root, 'test_results_summary.csv')
    results = []

    print(f"\n{'=' * 40}")
    print(f"STARTING TESTING | Device: {args.device}")
    print(f"Classes: {args.classes}")
    print(f"{'=' * 40}")

    for class_name in args.classes:
        try:
            res = test_class(args, class_name)
            results.append(res)
        except Exception as e:
            print(f"Error testing {class_name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Img_AUROC', 'Img_AP', 'Pix_AUROC', 'Pix_AP'])
            avg_metrics = [0, 0, 0, 0]

            for r in results:
                writer.writerow([r['class'], f"{r['img_auroc']:.4f}", f"{r['img_ap']:.4f}", f"{r['pix_auroc']:.4f}",
                                 f"{r['pix_ap']:.4f}"])
                avg_metrics[0] += r['img_auroc']
                avg_metrics[1] += r['img_ap']
                avg_metrics[2] += r['pix_auroc']
                avg_metrics[3] += r['pix_ap']

            n = len(results)
            writer.writerow([])
            writer.writerow(
                ['AVERAGE', f"{avg_metrics[0] / n:.4f}", f"{avg_metrics[1] / n:.4f}", f"{avg_metrics[2] / n:.4f}",
                 f"{avg_metrics[3] / n:.4f}"])
            print(f"Test Summary saved to {csv_path}")


if __name__ == "__main__":
    args = parse_args()

    setup_seed(args.seed)

    if torch.cuda.is_available() and 'cuda' in args.device:
        torch.cuda.set_device(int(args.device.split(':')[-1]))

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'test':
        run_testing(args)
    elif args.mode == 'all':
        run_training(args)

        run_testing(args)
