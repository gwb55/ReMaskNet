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

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'all'],
                        help='运行模式: train(训练), test(测试), all(训练完直接测试)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子，默认0')
    parser.add_argument('--classes', type=str, nargs='+', default=['bottle'],
                        help='MVTecAD 类别列表，例如: --classes bottle cable screw')
    parser.add_argument('--device', type=str, default='cuda:0', help='计算设备 (e.g., cuda:0, cpu)')
    parser.add_argument('--save_root', type=str, default='./saved_models', help='模型保存路径')
    parser.add_argument('--data_root', type=str, default='/root/dfs/gwb/gwb/SimpleNetFi/data4/MVTec_ad',
                        help='MVTec数据集根目录')
    parser.add_argument('--dtd_path', type=str, default='/root/dfs/gwb/gwb/SimpleNetFi/data4/dtd/images',
                        help='DTD纹理数据集路径')
    parser.add_argument('--result_root', type=str, default='./results', help='测试结果/可视化保存路径')

    parser.add_argument('--feature_layers', type=int, nargs='+', default=[2, 3], help='特征提取层级')
    parser.add_argument('--target_embed_dim', type=int, default=1536, help='目标嵌入维度')
    parser.add_argument('--image_size', type=int, default=288, help='重心剪裁后输入图像尺寸')
    parser.add_argument('--resize', type=int, default=329, help='Resize尺寸')

    parser.add_argument('--epochs', type=int, default=200, help='训练总Epoch数')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--iteration', type=int, default=3, help='Phase 2 内部迭代次数')
    parser.add_argument('--eval_epoch', type=int, default=2, help='每多少个Epoch评估一次')
    parser.add_argument('--eval_iteration', type=int, default=3, help='评估时的迭代次数')

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