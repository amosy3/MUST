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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='real',
                        choices=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                        help="select target domain")
    parser.add_argument("--batch_size", type=int, default=256, help="All loaders batch size")
    parser.add_argument("--epochs", type=int, default=300, help="ping pong epochs")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of ping-pong loss")
    parser.add_argument("--pseudo_th", type=float, default=0.3, help="teacher threshold to give pseudo labels for student")
    parser.add_argument("--eval_freq", type=int, default=20, help="wait before eval and log models performance")
    parser.add_argument('--dgx3', action='store_true')
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument("--log", type=str, default=' ', help="set log file path")
    return parser.parse_args()


def get_model(nclasses=10, pretrained=True):
    model = torchvision.models.resnet152(pretrained=pretrained)
    # Finetune Final few layers to adjust for tiny imagenet input
    model.avgpool = nn.AdaptiveAvgPool2d(1)
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
    layer.weight =torch.nn.Parameter(params['weight'])
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
    data_path = 'data/DomainNet/%s/' % split if args.dgx3 else '../datasets/DomainNet/%s/' % split
    loader = dict()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    for domain in datasets:
        data = torchvision.datasets.ImageFolder('%s/%s' %(data_path, domain), transform=transform)
        loader[domain] = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=20)
    return loader


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
        self.model = get_model(345)
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
        self.scheduler = None
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
        if self.scheduler is not None:
            self.scheduler.step()
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
        self.load_bn_to_model(domain)
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            batch_pred = self.model(X)
            if self.compare == 'int':
                batch_loss = self.criterion(batch_pred, y)
            else:
                batch_loss = self.criterion(batch_pred, int2oh(y, nclass=345))

            running_loss += batch_loss.item() * X.size(0)
            batch_corrects = get_batch_corrects(batch_pred, y)
            running_corrects += batch_corrects
            nsamples += X.shape[0]
        self.save_current_bn(domain)

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


def remove_defaultdict_from_meta_model(meta):
    for k,v in meta.domain2model_bns.items():
        meta.domain2model_bns[k] = dict(v)
        for k2,v2 in v.items():
            meta.domain2model_bns[k][k2] = dict(v2)
            for k3,v3 in v2.items():
                if isinstance(v3, defaultdict):
                    meta.domain2model_bns[k][k2][k3] = dict(v3)


def get_random_batch(domains, loader):
    domain = random.choice(domains)
    domain_iterator = iter(loader[domain])
    X, y = domain_iterator.next()
    return X.to('cuda'),y.to('cuda'), domain




torch_utils.set_random_seeds()
if __name__ == '__main__':

    args = get_args()
    log_folder = '../logs/%s_%s/' % (args.log, datetime.datetime.now())
    log_path = log_folder + 'log'
    os.mkdir(log_folder)
    print_and_log('\n'.join(['%s=%s' % (k, v) for k, v in vars(args).items()]), log_folder + 'hyper_parameters')

    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    train_loader = get_all_loaders(domains, split='train')
    test_loader = get_all_loaders(domains, split='test')
    metrics_format = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    target = args.target
    sources = [x for x in domains if x != target]

    teacher = MetaModel(criterion='cross_entropy')
    teacher.model.train()
    teacher.set_gpus(print_stat=True)
    teacher.init_bn_for_all_domains(domains)
    teacher.init_metric_format(domains, metrics_format)

    student = MetaModel(criterion='l1')
    student.model.train()
    student.set_gpus()
    student.init_bn_for_all_domains(domains)
    student.init_metric_format(domains, metrics_format)

    if args.warm_start:
        path = log_folder+'/../ws_teachers/%s.pkl'%args.target
        if os.path.isfile(path) and False:
            teacher = load_object(path)
            teacher.set_gpus()
            teacher.init_metric_format(domains, metrics_format)

        else:
            print('Pretrained not found - start warm start for teacher model')
            for epoch in range(6):
                for source in sources:
                    running_loss, running_corrects, nsamples = 0.0, 0, 0
                    for i, (Xs, ys) in enumerate(train_loader[source]):
                        #train teacher on source data
                        teacher.load_bn_to_model(source)
                        Xs, ys = Xs.to(teacher.device), ys.to(teacher.device)
                        batch_pred, batch_loss = teacher.train_on_batch(Xs, ys)
                        teacher.save_current_bn(source)

                        running_loss += batch_loss.item() * Xs.size(0)
                        batch_corrects = get_batch_corrects(batch_pred, ys)
                        running_corrects += batch_corrects
                        nsamples += Xs.shape[0]

                    # Print teacher metrics on source data
                    val_loss, val_acc = teacher.eval_on_domain(source, test_loader[source])
                    print_params = (epoch, source, running_loss / nsamples, val_loss, running_corrects.double() / nsamples, val_acc)
                    report = 'Epoch: %d, Source: %s, loss:%0.4f (%0.4f),acc:%0.4f (%0.4f)' % print_params
                    print_and_log(report, log_folder+'warm_start')

                    # Print teacher metrics on target data
                    target_loss, target_acc = teacher.eval_on_domain(target, test_loader[target])
                    report = 'Epoch: %d, loss:%0.4f ,acc:%0.4f,' % (epoch, target_loss, target_acc)
                    print_and_log(report, log_folder+'warm_start')

            target_loss, target_acc = teacher.eval_on_domain(target, test_loader[target])
            print_and_log('Final acc:%0.4f' % target_acc, log_folder+'warm_start')
            remove_defaultdict_from_meta_model(teacher)
            save_object(teacher, log_folder+'ws_teacher.pkl')

        teacher_weights = copy.deepcopy(teacher.model.state_dict())
        student.model.load_state_dict(teacher_weights)
        student.model.domain2model_bns = teacher.domain2model_bns

    running_loss, running_corrects, nsamples = 0.0, 0, 0
    running_ping_pong, nping_pong = 0.0, 0

    for i in range(args.epochs):
        Xs, ys, source = get_random_batch(sources, train_loader)

        # train teacher on source data
        teacher.load_bn_to_model(source)
        teacher.optimizer.zero_grad()
        batch_pred = teacher.model.forward(Xs)
        batch_loss = teacher.criterion(batch_pred, ys)
        batch_loss.backward() 
        teacher.save_current_bn(source)

        # Log teacher scores on S
        running_loss += batch_loss.item() * Xs.size(0)
        batch_corrects = get_batch_corrects(batch_pred, ys)
        running_corrects += batch_corrects
        nsamples += Xs.shape[0]

        # train student on teacher predictions
        teacher.load_bn_to_model(target)
        student.load_bn_to_model(target)
        for j, (Xt, _) in enumerate(train_loader[target]):
            Xt = Xt.to(teacher.device)
            yt_teacher_estimate = teacher.model(Xt)
            #Thresholding
            probs = torch.max(nn.Softmax(dim=1)(yt_teacher_estimate), 1)
            mask = probs[0] > args.pseudo_th
            if sum(mask) == 0:
                continue
            batch_pred, batch_loss = student.train_on_batch(Xt[mask], yt_teacher_estimate[mask], retain_graph=True)
            running_ping_pong += batch_loss.item() * Xt.size(0)
            nping_pong += Xt.shape[0]
            if j > 1:
                break
        student.save_current_bn(target)

        student.load_bn_to_model(source)
        ys_student_estimate = student.model(Xs)
        student.save_current_bn(source)
        ping_pong_loss = args.alpha * nn.CrossEntropyLoss()(ys_student_estimate, ys)
        ping_pong_loss.backward()

        teacher.optimizer.step()

        if teacher.scheduler is not None:
            teacher.scheduler.step()
        teacher.save_current_bn(target)

        if i % args.eval_freq == 0:

            # Eval teacher on S
            report = 'Epoch: %d, Teacher on S - loss:%0.4f ,acc:%0.4f' % (
            i, running_loss / nsamples, running_corrects.double() / nsamples)
            print_and_log(report, log_folder + 'log')
            running_loss, running_corrects, nsamples = 0.0, 0, 0

            for source in sources:
                loss, acc = teacher.eval_on_domain(source, test_loader[source])
                teacher.domain2metrics[source]['val_loss'].append(loss)
                teacher.domain2metrics[source]['val_acc'].append(acc)
                report = 'Epoch %d, Teacher on Source: %s, val_loss:%0.4f, val_acc:%0.4f' % (i, source, loss, acc)
                print_and_log(report, log_folder + 'log')

            # Eval teacher on T 
            loss, acc = teacher.eval_on_domain(target, train_loader[target])
            val_loss, val_acc = teacher.eval_on_domain(target, test_loader[target])
            teacher.domain2metrics[target]['loss'].append(loss)
            teacher.domain2metrics[target]['acc'].append(acc)
            teacher.domain2metrics[target]['val_loss'].append(val_loss)
            teacher.domain2metrics[target]['val_acc'].append(val_acc)

            report = 'Epoch %d, Teacher on Target: %s, ,loss:%0.4f, acc:%0.4f, val_loss:%0.4f, val_acc:%0.4f' % (
            i, target, loss,
            acc, val_loss, val_acc)
            print_and_log(report, log_folder + 'log')

            # student teacher loss 
            report = 'Epoch: %d, Student on teacher loss:%0.4f' % (i, running_ping_pong / nping_pong)
            print_and_log(report, log_folder + 'log')
            running_ping_pong, nping_pong = 0.0, 0

            # Eval student on T 
            loss, acc = student.eval_on_domain(target, train_loader[target])
            val_loss, val_acc = student.eval_on_domain(target, test_loader[target])
            student.domain2metrics[target]['loss'].append(loss)
            student.domain2metrics[target]['acc'].append(acc)
            student.domain2metrics[target]['val_loss'].append(val_loss)
            student.domain2metrics[target]['val_acc'].append(val_acc)

            report = 'Epoch %d, Student on Target: %s, ,loss:%0.4f, acc:%0.4f, val_loss:%0.4f, val_acc:%0.4f' % (
            i, target, loss,
            acc, val_loss, val_acc)
            print_and_log(report, log_folder + 'log')

            # Eval student on S (reverse validation score - for early stopping and model selection)
            for source in sources:
                loss, acc = student.eval_on_domain(source, test_loader[source])
                student.domain2metrics[source]['val_loss'].append(loss)
                student.domain2metrics[source]['val_acc'].append(acc)
                report = 'Epoch %d, Student on Source: %s, val_loss:%0.4f, val_acc:%0.4f' % (i, source, loss, acc)
                print_and_log(report, log_folder + 'log')

            remove_defaultdict_from_meta_model(teacher)
            save_object(teacher, log_folder + 'teacher_checkpoint_%s.pkl'%i)
            remove_defaultdict_from_meta_model(student)
            save_object(student, log_folder + 'student_checkpoint_%s.pkl'%i)

    remove_defaultdict_from_meta_model(teacher)
    save_object(teacher, log_folder + 'final_teacher.pkl')
    remove_defaultdict_from_meta_model(student)
    save_object(student, log_folder + 'final_student.pkl')