import torch
import sys
import copy
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import SVHN, USPS, MNIST
from torch.utils.data import DataLoader
import numpy as np
import random


def config_gpus_setup(model, gpus='all', print_stat=False):
    """
    :param model: pytorch model
    :param gpus: list of gpus ids to use or 'all' to use all of them (default)
    :param print_stat: if True - print the number of visible GPUs
    :return: model, device
    """

    ngpus = torch.cuda.device_count()

    if print_stat:
        print('Cuda see %s GPUs' % ngpus)

    if gpus == 'all':
        gpus = list(range(ngpus))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        if (ngpus > 1):
            model = torch.nn.DataParallel(model, device_ids=gpus)
    else:
        device = torch.device("cpu")

    return model, device


def set_random_seeds(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_model(nclasses=10):
    model = torchvision.models.resnet18()
    # Finetune Final few layers to adjust for tiny imagenet input
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nclasses)
    return model


class DuplicateToChannels:
    """Duplicate single channel 3 times"""

    def __init__(self):
        pass

    def __call__(self, x):
        return x.repeat((3,1,1))


def get_svhn(split='train', resize=(224, 224), batch_size=64):
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
    svhn = SVHN('../datasets/svhn', split=split, transform=transform)
    svhn_loader = DataLoader(svhn, batch_size=batch_size, shuffle=True, num_workers=20)
    return svhn_loader


def get_usps(split='train', resize=(224, 224), batch_size=64):
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), DuplicateToChannels()])
    usps = USPS('../datasets/usps', train=(split=='train'), transform=transform)
    usps_loader = DataLoader(usps, batch_size=batch_size, shuffle=True, num_workers=20)
    return usps_loader


def get_mnist(split='train', resize=(224, 224), batch_size=64):
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), DuplicateToChannels()])
    mnist = MNIST('../datasets/mnist', train=(split=='train'), transform=transform)
    mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=20)
    return mnist_loader


def get_device(model):
    try:
        device = model.device
    except:
        device = 'cuda'
    return device



def train_model(model, train_loader, criterion, optimizer,
                val_loader=None, scheduler=None, epochs=2, gpus='all',dataset_sizes=None, label_one_hot=False):

    def batch_step(model, X, y, optimizer, scheduler):
        optimizer.zero_grad()
        y_pred = model.forward(X)
        loss = criterion(y_pred, y)

        if model.training:
            loss.backward(retain_graph=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        return y_pred, loss, optimizer, scheduler

    def batch_metric_updates(X, y, y_pred, loss, running_loss, running_corrects, label_one_hot):
        _, y_scalar = torch.max(y_pred, 1)
        running_loss += loss.item() * X.size(0)

        if label_one_hot:
            _, y = torch.max(y, 1)
        running_corrects += torch.sum(y_scalar == y.data)
        return running_loss, running_corrects

    def epoch_step(model, loader, optimizer, scheduler, metrics, label_one_hot):
        running_loss = 0.0
        running_corrects = 0

        for i, (X, y) in enumerate(loader):
            device = get_device(model)
            X, y = X.to(device), y.to(device)

            y_pred, loss, optimizer, scheduler = batch_step(model, X, y, optimizer, scheduler)
            running_loss, running_corrects = batch_metric_updates(X, y, y_pred, loss, running_loss, running_corrects,
                                                                  label_one_hot)

        epoch_loss = running_loss / loader.dataset.data.shape[0]
        epoch_acc = running_corrects.double() / loader.dataset.data.shape[0]

        prefix = '' if model.training else 'val_'
        metrics['%sloss' % prefix].append(epoch_loss)
        metrics['%sacc' % prefix].append(epoch_acc)

        return model, optimizer, scheduler, metrics

    metrics = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    best_model = copy.deepcopy(model.state_dict())
    best_val_score = 0.0

    for epoch in range(epochs):
        model.train()
        model, optimizer, scheduler, metrics = epoch_step(model, train_loader, optimizer, scheduler, metrics,
                                                          label_one_hot)
        report = 'Epoch: %d, loss:%0.4f ,acc:%0.4f'%(epoch, metrics['loss'][-1],metrics['acc'][-1])

        if val_loader is not None:
            model.eval()
            model, optimizer, scheduler, metrics = epoch_step(model, val_loader, optimizer, scheduler, metrics,
                                                              label_one_hot)
            report = '%s ,val_loss:%0.4f ,val_acc:%0.4f'%(report, metrics['val_loss'][-1],metrics['val_acc'][-1])
            if metrics['val_acc'][-1] > best_val_score:
                best_val_score = metrics['val_acc'][-1]
                best_model = copy.deepcopy(model.state_dict())
                # model.load_state_dict(best_model_wts)

        print(report)

    return model, metrics, best_model


def print_and_log(txt, log_path):
    print(txt)
    f = open(log_path, 'a')
    f.write(txt+'\n')
    f.close()