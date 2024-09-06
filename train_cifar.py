import os
import logging
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from feature_extract import DataSaverHook
from neural_collapse import NeuralCollapseMetrics
import matplotlib.pyplot as plt
from models.vgg import vgg11_bn
from configs import args_parser


torch.set_num_threads(4)
torch.manual_seed(1523)
torch.cuda.manual_seed(1523)
np.random.seed(1523)

args = args_parser()
model_name = args.model

root_path = './results/'
task_name = 'dataset_%s_model_%s_epoch_%d_lr_%.4f' % (args.dataset, model_name, args.epoch, args.lr)
task_path = root_path + task_name + '/'
if os.path.exists(task_path) is False:
    os.makedirs(task_path)
sub_file_name = ['log/', 'model/', 'fig/', 'metrics/']
for sub_file in sub_file_name:
    if os.path.exists(task_path + sub_file) is False:
        os.makedirs(task_path + sub_file)

logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(task_path + 'log/%s.log' % task_name)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


if model_name == 'resnet18':
    model = models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    penultimate_layer = model.avgpool
    last_layer = model.fc
elif model_name == 'vgg11_bn':
    model = vgg11_bn()
    num_ftrs = model.classifier[6].in_features
    penultimate_layer = model.classifier[5]
    last_layer = model.classifier[6]
else:
    pass

if args.gpu == '-1':
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + args.gpu)

def plot_and_save(epochs, values, label, ylabel, save_path):
    """绘制并保存单个指标的图像"""
    plt.figure()
    plt.plot(epochs, values, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f'{label} vs Epochs')
    plt.savefig(f'{save_path}/{label}_epoch.png')
    plt.close()

def test(test_data_loader, model, device, logger):
    model.eval()
    correct = 0
    for data, label in test_data_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    acc = correct / len(test_data_loader.dataset)
    model.train()
    return acc

def train(train_data_loader, test_data_loader, model, criterion, optimizer, scheduler, device, epoch, logger, save_path):
    # 初始化数据存储容器
    nc1_values = []
    nc2_cosine_sim_values = []
    nc2_norm_var_values = []
    nc3_values = []
    nc4_values = []
    train_accuracy_values = []
    test_accuracy_values = []
    losses = []
    epochs = []

    model.train()
    loss = None
    data_hook = DataSaverHook(store_output=True)
    penultimate_layer.register_forward_hook(data_hook)
    metrics = NeuralCollapseMetrics(10, num_ftrs, device)
    best_acc = 0
    for iter in range(epoch):
        total_loss = 0
        metrics.reset()
        for data, label in train_data_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if iter % 5 == 0 or iter == epoch - 1:
                # get features
                features = data_hook.output_store.squeeze(-1).squeeze(-1)
                metrics.update(features, label)

        total_loss /= len(train_data_loader)

        if iter % 5 == 0 or iter == epoch - 1:
            # compute neural collapse metrics
            nc1 = metrics.compute_nc1()
            nc2_cosine_sim, nc2_norm_var = metrics.compute_nc2()
            weights = last_layer.weight.data
            nc3 = metrics.compute_nc3(weights)
            nc4 = 0
            for data, label in train_data_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                preds = output.argmax(dim=1, keepdim=True)
                features = data_hook.output_store.squeeze(-1).squeeze(-1)
                nc4 += metrics.compute_nc4(features, preds)
            nc4 = 1 - nc4 / len(train_data_loader.dataset)

            # test accuracy in train dataset
            train_acc = test(train_data_loader, model, device, logger)
            test_acc = test(test_data_loader, model, device, logger)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path + 'model/best_model.pth')

            # save metrics
            nc1_values.append(nc1)
            nc2_cosine_sim_values.append(nc2_cosine_sim)
            nc2_norm_var_values.append(nc2_norm_var)
            nc3_values.append(nc3)
            nc4_values.append(nc4)
            train_accuracy_values.append(train_acc)
            test_accuracy_values.append(test_acc)
            losses.append(total_loss)
            epochs.append(iter+1)

            # plot and save metrics
            plot_and_save(epochs, nc1_values, 'NC1', 'NC1', save_path + 'fig')
            plot_and_save(epochs, nc2_cosine_sim_values, 'NC2_cosine_sim', 'NC2_cosine_sim', save_path + 'fig')
            plot_and_save(epochs, nc2_norm_var_values, 'NC2_norm_var', 'NC2_norm_var', save_path + 'fig')
            plot_and_save(epochs, nc3_values, 'NC3', 'NC3', save_path + 'fig')
            plot_and_save(epochs, nc4_values, 'NC4', 'NC4', save_path + 'fig')
            plot_and_save(epochs, train_accuracy_values, 'Train_Accuracy', 'Train_Accuracy', save_path + 'fig')
            plot_and_save(epochs, test_accuracy_values, 'Test_Accuracy', 'Test_Accuracy', save_path + 'fig')
            plot_and_save(epochs, losses, 'Loss', 'Loss', save_path + 'fig')


            logger.info('Train Epoch: %d Loss: %.6f, NC1: %.6f, NC2_cosine_sim: %.6f, NC2_norm_var: %.6f, NC3: %.6f, NC4: %.6f, Train_acc: %.6f, Test_acc: %.6f' % (iter+1, total_loss, nc1, nc2_cosine_sim, nc2_norm_var, nc3, nc4, train_acc, test_acc))

    nc1_values = np.array(nc1_values)
    nc2_cosine_sim_values = np.array(nc2_cosine_sim_values)
    nc2_norm_var_values = np.array(nc2_norm_var_values)
    nc3_values = np.array(nc3_values)
    nc4_values = np.array(nc4_values)
    train_accuracy_values = np.array(train_accuracy_values)
    test_accuracy_values = np.array(test_accuracy_values)
    losses = np.array(losses)

    np.savez(save_path + 'metrics/' + 'metrics.npz', nc1=nc1_values, nc2_cosine_sim=nc2_cosine_sim_values, nc2_norm_var=nc2_norm_var_values, nc3=nc3_values, nc4=nc4_values, accuracy=train_accuracy_values, test_accuracy=test_accuracy_values, loss=losses)

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root='../data/cifar', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='../data/cifar', train=False, download=True, transform=test_transform)
    # train_size = int(0.8 * len(train_dataset))
    # valid_size = len(train_dataset) - train_size
    # train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train(train_loader, test_loader, model, criterion, optimizer, scheduler, device, args.epoch, logger, task_path)
