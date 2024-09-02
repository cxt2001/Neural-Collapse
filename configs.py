import argparse
import numpy as np
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help="rounds of training")
    parser.add_argument('--batch_size', type=int, default=32, help="rounds of training")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers for the data loading process")
    parser.add_argument('--gpu', type=str, default='0', help="gpu index selection")
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    args = parser.parse_args()
    return args
