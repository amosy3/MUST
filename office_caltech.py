import torch
from torch import nn
import torchvision
from torchvision import transforms
from collections import defaultdict
import pickle
import os
import numpy as np

import torch_utils
import torch.nn as nn
import torch.optim as optim
import copy
import random
import argparse
import datetime
from bn_per_domain_resnet import resnet101 as bnpd_resnet101

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=['amazon', 'caltech10', 'dslr', 'webcam'],
                        help="select target domain")
    parser.add_argument("--batch_size", type=int, default=32, help="All loaders batch size")
    parser.add_argument("--epochs", type=int, default=300, help="ping pong epochs")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of ping-pong loss")
    parser.add_argument("--pseudo_th", type=float, default=0.3, help="teacher threshold to give pseudo labels for student")
    parser.add_argument("--eval_freq", type=int, default=20, help="wait before eval and log models performance")
    parser.add_argument('--dgx3', action='store_true')
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument("--log", type=str, default=' ', help="set log file path")
    return parser.parse_args()


def get_model(nclasses=10, pretrained=True, domains=[]):
    if domains==[]:
        model = torchvision.models.resnet101(pretrained=pretrained)
    else:
        model = bnpd_resnet101(pretrained=pretrained, domains=domains)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, nclasses)
    return model


def get_bn_layer_params(bn_layer):
    layer_params = {'running_mean':bn_layer.running_mean.clone(),
                   'running_var':bn_layer.running_var.clone(),
                   'weight':bn_layer.weight.clone(),
                   'bias':bn_layer.bias.clone()}
    return layer_params


def set_bn_layer_params(layer, params):
    layer.running_mean = (params['running_mean'])
    layer.running_var = (params['running_var'])
    layer.weight = torch.nn.Parameter(params['weight'])
    layer.bias = torch.nn.Parameter(params['bias'])


def get_bn_params(model):
    bn_params = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    bn_params[0] = get_bn_layer_params(model.bn1)

    # copy all layer head
    for i, x in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        bn_params[i + 1][0][1] = get_bn_layer_params(x[0].bn1)
        bn_params[i + 1][0][2] = get_bn_layer_params(x[0].bn2)
        bn_params[i + 1][0][3] = get_bn_layer_params(x[0].bn3)
        bn_params[i + 1][0][4] = get_bn_layer_params(x[0].downsample[1])

    # layer1
    for i in range(2):
        bn_params[1][i + 1][1] = get_bn_layer_params(model.layer1[i].bn1)
        bn_params[1][i + 1][2] = get_bn_layer_params(model.layer1[i].bn2)
        bn_params[1][i + 1][3] = get_bn_layer_params(model.layer1[i].bn3)

    # layer2
    for i in range(7):
        bn_params[2][i + 1][1] = get_bn_layer_params(model.layer2[i].bn1)
        bn_params[2][i + 1][2] = get_bn_layer_params(model.layer2[i].bn2)
        bn_params[2][i + 1][3] = get_bn_layer_params(model.layer2[i].bn3)

    # layer3
    for i in range(35):
        bn_params[3][i + 1][1] = get_bn_layer_params(model.layer3[i].bn1)
        bn_params[3][i + 1][2] = get_bn_layer_params(model.layer3[i].bn2)
        bn_params[3][i + 1][3] = get_bn_layer_params(model.layer3[i].bn3)

    # layer4
    for i in range(2):
        bn_params[4][i + 1][1] = get_bn_layer_params(model.layer4[i].bn1)
        bn_params[4][i + 1][2] = get_bn_layer_params(model.layer4[i].bn2)
        bn_params[4][i + 1][3] = get_bn_layer_params(model.layer4[i].bn3)

    return bn_params


def inject_bn_params(model, bn_params):
    set_bn_layer_params(model.bn1, bn_params[0])

    for i, x in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        set_bn_layer_params(x[0].bn1, bn_params[i + 1][0][1])
        set_bn_layer_params(x[0].bn2, bn_params[i + 1][0][2])
        set_bn_layer_params(x[0].bn3, bn_params[i + 1][0][3])
        set_bn_layer_params(x[0].downsample[1], bn_params[i + 1][0][4])

        # layer1
    for i in range(2):
        set_bn_layer_params(model.layer1[i].bn1, bn_params[1][i + 1][1])
        set_bn_layer_params(model.layer1[i].bn2, bn_params[1][i + 1][2])
        set_bn_layer_params(model.layer1[i].bn3, bn_params[1][i + 1][3])

    # layer2
    for i in range(7):
        set_bn_layer_params(model.layer2[i].bn1, bn_params[2][i + 1][1])
        set_bn_layer_params(model.layer2[i].bn2, bn_params[2][i + 1][2])
        set_bn_layer_params(model.layer2[i].bn3, bn_params[2][i + 1][3])

    # layer3
    for i in range(35):
        set_bn_layer_params(model.layer3[i].bn1, bn_params[3][i + 1][1])
        set_bn_layer_params(model.layer3[i].bn2, bn_params[3][i + 1][2])
        set_bn_layer_params(model.layer3[i].bn3, bn_params[3][i + 1][3])

    # layer4
    for i in range(2):
        set_bn_layer_params(model.layer4[i].bn1, bn_params[4][i + 1][1])
        set_bn_layer_params(model.layer4[i].bn2, bn_params[4][i + 1][2])
        set_bn_layer_params(model.layer4[i].bn3, bn_params[4][i + 1][3])


def get_all_loaders(datasets, split='train'):
    data_path = '../datasets/office_caltech/%s/' % split
    loader = dict()
    iterator = dict()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for domain in datasets:
        data = torchvision.datasets.ImageFolder('%s/%s/' %(data_path, domain), transform=transform)
        loader[domain] = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        iterator[domain] = iter(loader[domain])
    return loader, iterator


def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def int2oh(y, nclass=10):
    y_oh = torch.zeros(y.shape[0], nclass).to(y.device)
    y_oh[torch.arange(y.shape[0]), y] = 1
    return y_oh


def differential_logits_xent(y,pred):
    return torch.mean(torch.sum(- y * nn.functional.log_softmax(pred, -1), dim=1),dim=0)


class MetaModel():
    def __init__(self, criterion='cross_entropy'):
        self.model = get_model(10)
        if criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            self.compare = 'int'
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
            self.compare = 'vec'
        if criterion == 'xent':
            self.criterion = differential_logits_xent
            self.compare = 'vec'
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.device = None
        self.domain2model_bns = dict()
        self.domain2metrics = dict()

    def set_gpus(self, device_ids='all', print_stat=False):
        ngpus = torch.cuda.device_count()
        if print_stat:
            print('Cuda see %s GPUs' % ngpus)

        if device_ids == 'all':
            device_ids = list(range(ngpus))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            if ngpus > 1:
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model = torch.nn.DataParallel(self.model.module, device_ids=device_ids)
                else:
                    self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        else:
            print('No GPU found - running on CPU')
            self.device = torch.device("cpu")

    def init_bn_for_all_domains(self, domains):
        for domain in domains:
            self.domain2model_bns[domain] = get_bn_params(self.model.module)

    def save_current_bn(self, domain):
        self.domain2model_bns[domain] = get_bn_params(self.model.module)

    def load_bn_to_model(self, domain):
        inject_bn_params(self.model.module, self.domain2model_bns[domain])

    def train_on_batch(self, X, y, retain_graph=False):
        self.optimizer.zero_grad()
        y_pred = self.model.forward(X)
        loss = self.criterion(y_pred, y)
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

        return y_pred, loss

    def eval_on_batch(self, X, y):
        y_pred = self.model.forward(X)
        loss = self.criterion(y_pred, y)
        return y_pred, loss

    def forward(self, X, domain):
        self.load_bn_to_model(domain)
        preds = self.model(X)
        self.save_current_bn(domain)
        return preds

    def init_metric_format(self, domains, format):
        for domain in domains:
            self.domain2metrics[domain] = copy.deepcopy(format)

    def eval_on_domain(self, domain, data_loader):
        running_loss, running_corrects, nsamples = 0.0, 0, 0
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            batch_pred = self.model(X)
            if self.compare == 'int':
                batch_loss = self.criterion(batch_pred, y)
            else:
                batch_loss = self.criterion(batch_pred, int2oh(y, nclass=10))

            running_loss += batch_loss.item() * X.size(0)
            batch_corrects = get_batch_corrects(batch_pred, y)
            running_corrects += batch_corrects
            nsamples += X.shape[0]

        return running_loss / nsamples, running_corrects.double().item() / nsamples


def get_batch_corrects(batch_pred, y):
    _, batch_preds_int = torch.max(batch_pred, 1)
    if len(y.shape) > 1:
        _, y = torch.max(y, 1)
    return torch.sum(batch_preds_int == y.data)


def print_and_log(txt, log_path):
    print(txt)
    f = open(log_path, 'a')
    f.write(txt+'\n')
    f.close()


def get_random_batch(domains, loader, iterator, p=None):
    domain = random.choice(domains) if p is None else random.choice(domains, p=p)
    try:
        X, y = iterator[domain].next()
    except:
        iterator[domain] = iter(loader[domain])
        X, y = iterator[domain].next()
    return X.to('cuda'),y.to('cuda'), domain


def eval_model(model, loader, log_prefix=''):
    model.eval()
    running_loss, running_corrects, nsamples = 0.0, 0, 0
    with torch.no_grad():
        for j, (X, y) in enumerate(loader):
            X, y = X.to('cuda'), y.to('cuda')
            pred = model(X)
            loss = ce(pred, y)
            running_loss += loss.item() * X.size(0)
            batch_corrects = get_batch_corrects(pred, y)
            running_corrects += batch_corrects
            nsamples += X.shape[0]
    eval_loss = running_loss / nsamples
    eval_acc = running_corrects.double().item() / nsamples
    print_and_log(log_prefix+"Target loss: %0.4f acc: %0.4f" % (eval_loss, eval_acc), log_folder + 'log')
    model.train()


def get_optimizer(model):
    learning_rate = 1e-4
    param_group = []
    for k, v in model.named_parameters():
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': learning_rate}]
        else:
            param_group += [{'params': v, 'lr': learning_rate * 10}]
    _optimizer = optim.SGD(param_group, momentum=0.9)
    return _optimizer


if __name__ == '__main__':

    args = get_args()
    log_folder = '../logs/office_caltech/%s_%s_%s/' % (args.target, args.log, datetime.datetime.now())
    log_path = log_folder + 'log'
    os.mkdir(log_folder)
    print_and_log('\n'.join(['%s=%s' % (k, v) for k, v in vars(args).items()]), log_folder + 'hyper_parameters')

    domains = ['amazon', 'dslr', 'webcam', 'caltech10'] #
    train_loader, train_iterator = get_all_loaders(domains, split='train')
    test_loader, test_iterator = get_all_loaders(domains, split='test')

    metrics_format = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    target = args.target
    sources = [x for x in domains if x != target]
    ws_path = '../logs/office_caltech/ws_teachers/%s.pt' % target

    if args.warm_start:
        ws_path = log_folder + 'ws_teacher.pt'
        model = get_model(nclasses=10, pretrained=True, domains=domains)
        model = model.to('cuda')
        model = torch.nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        model.train()

        for i in range(3000):
            X, y, domain = get_random_batch(sources, train_loader, train_iterator)
            model.module.set_bn_domain(domain=domain)
            optimizer.zero_grad()

            preds = model.forward(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                model.eval()
                running_loss, running_corrects, nsamples = 0.0, 0, 0
                for j, (X, y) in enumerate(test_loader[target]):
                    X, y = X.to('cuda'), y.to('cuda')
                    pred = model(X)
                    loss = criterion(pred, y)
                    running_loss += loss.item() * X.size(0)
                    batch_corrects = get_batch_corrects(pred, y)
                    running_corrects += batch_corrects
                    nsamples += X.shape[0]
                eval_loss = running_loss / nsamples
                eval_acc = running_corrects.double().item() / nsamples
                scheduler.step(eval_loss)
                print_and_log("%d - Target loss: %0.4f acc: %0.4f" %(i, eval_loss, eval_acc),log_folder+'log')
                model.train()

    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    teacher = get_model(nclasses=10, pretrained=True, domains=domains)
    teacher = torch.nn.DataParallel(teacher)
    teacher.load_state_dict(torch.load(ws_path))
    teacher = teacher.to('cuda')
    teacher_optimizer = get_optimizer(teacher)
    teacher_scheduler = optim.lr_scheduler.ReduceLROnPlateau(teacher_optimizer, 'min')
    teacher.module.set_bn_domain(domain=target)
    eval_model(teacher, test_loader[target], log_prefix='check: ')
    teacher.train()

    student = get_model(nclasses=10, pretrained=True, domains=domains)
    student = torch.nn.DataParallel(student)
    student.load_state_dict(torch.load(ws_path))
    student = student.to('cuda')
    student_optimizer = get_optimizer(teacher)
    student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(student_optimizer, 'min')
    student.train()
    student.module.set_bn_domain(domain=target)

    for i in range(args.epochs):
        Xs, ys, source = get_random_batch(sources, train_loader, train_iterator)
        teacher.module.set_bn_domain(domain=source)
        teacher_optimizer.zero_grad()
        preds = teacher.forward(Xs)
        src_loss = ce(preds, ys)
        src_loss.backward()

        # train student on teacher predictions
        teacher.module.set_bn_domain(domain=target)
        Xt, yt, _ = get_random_batch([target], train_loader, train_iterator)
        yt_teacher_preds = teacher.forward(Xt)
        # Thresholding
        probs = torch.max(nn.Softmax(dim=1)(yt_teacher_preds), 1)
        mask = probs[0] > args.pseudo_th
        if sum(mask) == 0:
            teacher_optimizer.step()

        student_optimizer.zero_grad()
        yt_student_preds = student.forward(Xt)
        student_loss = l1(yt_student_preds, yt_teacher_preds)
        student_loss.backward()
        student_optimizer.step()
        teacher_optimizer.step()

        #eval
        if i%args.eval_freq==0:
            eval_model(teacher, test_loader[target], log_prefix='Epoch %d Teacher - '%i)
            eval_model(student, test_loader[target], log_prefix='Epoch %d Student - '%i)