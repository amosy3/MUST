import torch
from torch import nn
import torchvision
from torchvision import transforms
from collections import defaultdict
import pickle
import os
from collections import OrderedDict
import numpy as np

import torch_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
import argparse
import datetime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
                        help="select target domain")
    parser.add_argument("--batch_size", type=int, default=256, help="All loaders batch size")
    parser.add_argument("--epochs", type=int, default=250, help="ping pong epochs")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of ping-pong loss")
    parser.add_argument("--pseudo_th", type=float, default=0.95, help="teacher threshold to give pseudo labels for student")
    parser.add_argument("--eval_freq", type=int, default=20, help="wait before eval and log models performance")
    parser.add_argument('--dgx3', action='store_true')
    parser.add_argument('--warm_start', action='store_true')
    parser.add_argument("--log", type=str, default=' ', help="set log file path")
    # parser.add_argument("--source", type=str, default='mnist', choices=['mnist', 'usps', 'svhn'], help="select source domain")
    # parser.add_argument("--warm_start", type=str, choices=['cold', 'train', 'load'], default='load')
    return parser.parse_args()


def get_model(nclasses=10, pretrained=True):
    model = torchvision.models.resnet152(pretrained=pretrained)
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
    data_path = 'data/digit5' if args.dgx3 else '../datasets/digit5'
    loader = dict()
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    for domain in datasets:
        data = torchvision.datasets.ImageFolder('%s/%s/%s_images' %(data_path, domain, split), transform=transform)
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



def simulate_resnet152(x, weights, bn_params):
    x = F.conv2d(x, weights['conv1.weight'], stride=(2, 2), padding=(3, 3))
    x = F.batch_norm(x, bn_params[0]['running_mean'], bn_params[0]['running_var'],
                     weights['bn1.weight'], weights['bn1.bias'], training=True)
    x = F.relu(x, inplace=True)
    x = F.max_pool2d(x, kernel_size=3, stride=2)

    # layer 1
    for i in range(3):
        x = F.conv2d(x, weights['layer1.%d.conv1.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[1][i][1]['running_mean'], bn_params[1][i][1]['running_var'],
                         weights['layer1.%d.bn1.weight' % i], weights['layer1.%d.bn1.bias' % i], training=True)
        x = F.conv2d(x, weights['layer1.%d.conv2.weight' % i], stride=(1, 1), padding=(1, 1))
        x = F.batch_norm(x, bn_params[1][i][2]['running_mean'], bn_params[1][i][2]['running_var'],
                         weights['layer1.%d.bn2.weight' % i], weights['layer1.%d.bn2.bias' % i], training=True)
        x = F.conv2d(x, weights['layer1.%d.conv3.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[1][i][3]['running_mean'], bn_params[1][i][3]['running_var'],
                         weights['layer1.%d.bn3.weight' % i], weights['layer1.%d.bn3.bias' % i], training=True)
        x = F.relu(x, inplace=True)

        if i == 0:
            x = F.conv2d(x, weights['layer1.0.downsample.0.weight'], stride=(1, 1))
            x = F.batch_norm(x, bn_params[1][i][4]['running_mean'], bn_params[1][i][4]['running_var'],
                             weights['layer1.0.downsample.1.weight'], weights['layer1.0.downsample.1.bias'],
                             training=True)

    # layer 2
    for i in range(8):
        s = 2 if i == 0 else 1
        x = F.conv2d(x, weights['layer2.%d.conv1.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[2][i][1]['running_mean'], bn_params[2][i][1]['running_var'],
                         weights['layer2.%d.bn1.weight' % i], weights['layer2.%d.bn1.bias' % i], training=True)
        x = F.conv2d(x, weights['layer2.%d.conv2.weight' % i], stride=(s, s), padding=(1, 1))
        x = F.batch_norm(x, bn_params[2][i][2]['running_mean'], bn_params[2][i][2]['running_var'],
                         weights['layer2.%d.bn2.weight' % i], weights['layer2.%d.bn2.bias' % i], training=True)
        x = F.conv2d(x, weights['layer2.%d.conv3.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[2][i][3]['running_mean'], bn_params[2][i][3]['running_var'],
                         weights['layer2.%d.bn3.weight' % i], weights['layer2.%d.bn3.bias' % i], training=True)
        x = F.relu(x, inplace=True)

        if i == 0:
            x = F.conv2d(x, weights['layer2.0.downsample.0.weight'], stride=(2, 2))
            x = F.batch_norm(x, bn_params[2][i][4]['running_mean'], bn_params[2][i][4]['running_var'],
                             weights['layer2.0.downsample.1.weight'], weights['layer2.0.downsample.1.bias'],
                             training=True)

    # layer 3
    for i in range(36):
        s = 2 if i == 0 else 1
        x = F.conv2d(x, weights['layer3.%d.conv1.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[3][i][1]['running_mean'], bn_params[3][i][1]['running_var'],
                         weights['layer3.%d.bn1.weight' % i], weights['layer3.%d.bn1.bias' % i], training=True)
        x = F.conv2d(x, weights['layer3.%d.conv2.weight' % i], stride=(s, s), padding=(1, 1))
        x = F.batch_norm(x, bn_params[3][i][2]['running_mean'], bn_params[3][i][2]['running_var'],
                         weights['layer3.%d.bn2.weight' % i], weights['layer2.%d.bn2.bias' % i], training=True)
        x = F.conv2d(x, weights['layer3.%d.conv3.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[3][i][3]['running_mean'], bn_params[3][i][3]['running_var'],
                         weights['layer3.%d.bn3.weight' % i], weights['layer3.%d.bn3.bias' % i], training=True)
        x = F.relu(x, inplace=True)

        if i == 0:
            x = F.conv2d(x, weights['layer3.0.downsample.0.weight'], stride=(2, 2))
            x = F.batch_norm(x, bn_params[3][i][4]['running_mean'], bn_params[3][i][4]['running_var'],
                             weights['layer3.0.downsample.1.weight'], weights['layer3.0.downsample.1.bias'],
                             training=True)
    # layer 4
    for i in range(3):
        s = 2 if i == 0 else 1
        x = F.conv2d(x, weights['layer4.%d.conv1.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[4][i][1]['running_mean'], bn_params[4][i][1]['running_var'],
                         weights['layer4.%d.bn1.weight' % i], weights['layer4.%d.bn1.bias' % i], training=True)
        x = F.conv2d(x, weights['layer4.%d.conv2.weight' % i], stride=(s, s), padding=(1, 1))
        x = F.batch_norm(x, bn_params[4][i][2]['running_mean'], bn_params[4][i][2]['running_var'],
                         weights['layer4.%d.bn2.weight' % i], weights['layer4.%d.bn2.bias' % i], training=True)
        x = F.conv2d(x, weights['layer4.%d.conv3.weight' % i], stride=(1, 1))
        x = F.batch_norm(x, bn_params[4][i][3]['running_mean'], bn_params[4][i][3]['running_var'],
                         weights['layer4.%d.bn3.weight' % i], weights['layer4.%d.bn3.bias' % i], training=True)
        x = F.relu(x, inplace=True)

        if i == 0:
            x = F.conv2d(x, weights['layer4.0.downsample.0.weight'], stride=(2, 2))
            x = F.batch_norm(x, bn_params[4][i][4]['running_mean'], bn_params[4][i][4]['running_var'],
                             weights['layer4.0.downsample.1.weight'], weights['layer4.0.downsample.1.bias'],
                             training=True)

    x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
    x = F.linear(x, weights['fc.weight'], weights['fc.bias'])
    return x


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

    def eval_on_domain(self, domain, data_loader, get_ent=False):
        running_loss, running_corrects, nsamples = 0.0, 0, 0
        ent = np.array([])
        self.load_bn_to_model(domain)
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            batch_pred = self.model(X)
            ent = np.concatenate((ent, calc_predictions_entropy(batch_pred.detach())))
            if self.compare == 'int':
                batch_loss = self.criterion(batch_pred, y)
            else:
                batch_loss = self.criterion(batch_pred, int2oh(y, nclass=10))

            running_loss += batch_loss.item() * X.size(0)
            batch_corrects = get_batch_corrects(batch_pred, y)
            running_corrects += batch_corrects
            nsamples += X.shape[0]
        self.save_current_bn(domain)
        if get_ent:
            return running_loss / nsamples, running_corrects.double().item() / nsamples, ent
        else:
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


def calc_predictions_entropy(preds):
    preds = torch.nn.Softmax(dim=1)(preds)
    # preds = torch.nn.functional.normalize(preds, p=2, dim=1) ** 2
    h = torch.sum(-preds * torch.log(preds), dim=1)
    return h.cpu().numpy()

torch_utils.set_random_seeds()
if __name__ == '__main__':

    args = get_args()
    log_folder = '../logs/digit5/%s_%s_%s/' % (args.target, args.log, datetime.datetime.now())
    log_path = log_folder + 'log'
    os.mkdir(log_folder)
    print_and_log('\n'.join(['%s=%s' % (k, v) for k, v in vars(args).items()]), log_folder + 'hyper_parameters')

    domains = ['usps', 'mnist', 'mnist_m', 'svhn', 'syn'] # , 'usps'
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
        if os.path.isfile(path):
            teacher = load_object(path)
            teacher.set_gpus()
            teacher.init_metric_format(domains, metrics_format)

        else:
            print('Pretrained not found - start warm start for teacher model')
            for epoch in range(10):
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
                    report = 'Epoch: %d, Target - loss:%0.4f ,acc:%0.4f,' % (epoch, target_loss, target_acc)
                    print_and_log(report, log_folder+'warm_start')

            # save_meta_model(teacher, log_folder+'teacher_after_worm_start')
            target_loss, target_acc = teacher.eval_on_domain(target, test_loader[target])
            print_and_log('Final acc:%0.4f' % target_acc, log_folder+'warm_start')
            remove_defaultdict_from_meta_model(teacher)
            save_object(teacher, log_folder+'ws_teacher.pkl')

        teacher_weights = copy.deepcopy(teacher.model.state_dict())
        student.model.load_state_dict(teacher_weights)
        student.model.domain2model_bns = teacher.domain2model_bns

    running_loss, running_corrects, nsamples = 0.0, 0, 0
    running_ping_pong, nping_pong = 0.0, 0

    best_ping_pong_accs, min_source_loss, ent = 0.0, np.inf, (None,None)
    pp_acc_on_target, sl_acc_on_target = 0.0, 0.0

    for i in range(args.epochs):
        Xs, ys, source = get_random_batch(sources, train_loader)

        # train teacher on source data
        teacher.load_bn_to_model(source)
        teacher.optimizer.zero_grad()
        batch_pred = teacher.model.forward(Xs)
        batch_loss = teacher.criterion(batch_pred, ys)
        batch_loss.backward() #consider move to the end of the loop
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
            try:
                batch_pred, batch_loss = student.train_on_batch(Xt[mask], yt_teacher_estimate[mask], retain_graph=True)
            except:
                print('Error! - ', Xt.shape, sum(mask))
                continue
            running_ping_pong += batch_loss.item() * Xt.size(0)
            nping_pong += Xt.shape[0]
            if j > 1:
                break
        student.save_current_bn(target)

        teacher.optimizer.step()

        if teacher.scheduler is not None:
            teacher.scheduler.step()
        teacher.save_current_bn(target)

        if i % args.eval_freq == 0:

            # Eval teacher on S (teacher should be smart)
            report = 'Epoch: %d, Teacher on S - loss:%0.4f ,acc:%0.4f' % (
            i, running_loss / nsamples, running_corrects.double() / nsamples)
            print_and_log(report, log_folder + 'log')
            running_loss, running_corrects, nsamples = 0.0, 0, 0

            for source in sources:
                # check only on validation to save comutations
                loss, acc = teacher.eval_on_domain(source, test_loader[source])
                teacher.domain2metrics[source]['val_loss'].append(loss)
                teacher.domain2metrics[source]['val_acc'].append(acc)
                report = 'Epoch %d, Teacher on Source: %s, val_loss:%0.4f, val_acc:%0.4f' % (i, source, loss, acc)
                print_and_log(report, log_folder + 'log')

            # Eval teacher on T (teacher score as main classifier)
            loss, acc = teacher.eval_on_domain(target, train_loader[target])
            val_loss, teacher_target_acc, teacher_ent = teacher.eval_on_domain(target, test_loader[target], get_ent=True)
            teacher.domain2metrics[target]['loss'].append(loss)
            teacher.domain2metrics[target]['acc'].append(acc)
            teacher.domain2metrics[target]['val_loss'].append(val_loss)
            teacher.domain2metrics[target]['val_acc'].append(teacher_target_acc)
            report = 'Epoch %d, Teacher on Target: %s, ,loss:%0.4f, acc:%0.4f, val_loss:%0.4f, val_acc:%0.4f, ' \
                     'ent:%0.4f, ent_var:%0.4f' % \
                     (i, target, loss, acc, val_loss, teacher_target_acc, np.mean(teacher_ent), np.std(teacher_ent))
            np.save(log_folder + 'teacher_ent_%d.np'%i, teacher_ent)
            print_and_log(report, log_folder + 'log')

            # student teacher loss (student should learn to mimic the teacher)
            report = 'Epoch: %d, Student on teacher loss:%0.4f (ny=%d)' % (i, running_ping_pong / max(1,nping_pong), nping_pong)
            print_and_log(report, log_folder + 'log')
            running_ping_pong, nping_pong = 0.0, 0

            # Eval student on T (student score as main classifier)
            loss, acc = student.eval_on_domain(target, train_loader[target])
            val_loss, student_target_acc, student_ent = student.eval_on_domain(target, test_loader[target], get_ent=True)
            student.domain2metrics[target]['loss'].append(loss)
            student.domain2metrics[target]['acc'].append(acc)
            student.domain2metrics[target]['val_loss'].append(val_loss)
            student.domain2metrics[target]['val_acc'].append(student_target_acc)

            report = 'Epoch %d, Student on Target: %s, ,loss:%0.4f, acc:%0.4f, val_loss:%0.4f, val_acc:%0.4f, ' \
                     'ent:%0.4f, ent_var:%0.4f' % \
                     (i, target, loss, acc, val_loss, student_target_acc, np.mean(student_ent), np.std(student_ent))
            np.save(log_folder + 'student_ent_%d.np' % i, student_ent)
            print_and_log(report, log_folder + 'log')

            # Eval student on S (reverse validation score - for early stopping and model selection)
            ping_pong_accs = []
            for source in sources:
                loss, acc = student.eval_on_domain(source, test_loader[source])
                ping_pong_accs.append(acc)
                student.domain2metrics[source]['val_loss'].append(loss)
                student.domain2metrics[source]['val_acc'].append(acc)
                report = 'Epoch %d, Student on Source: %s, val_loss:%0.4f, val_acc:%0.4f' % (i, source, loss, acc)
                print_and_log(report, log_folder + 'log')


            if i==0:
                continue
            if best_ping_pong_accs < np.mean(ping_pong_accs):
                best_ping_pong_accs = np.mean(ping_pong_accs)
                pp_acc_on_target = teacher_target_acc, student_target_acc
                ent = teacher_ent, student_ent
                print_and_log('best_pp-%0.3f, Teacher:%0.3f, Student:%0.3f ' % (best_ping_pong_accs,
                                                                                pp_acc_on_target[0],
                                                                                pp_acc_on_target[1]),
                              log_folder + 'log')
            teacher.load_bn_to_model(target)
            torch.save(teacher.model.module, log_folder + 'teacher_checkpoint_%s.pkl'%i)
            student.load_bn_to_model(target)
            torch.save(student.model.module, log_folder + 'student_checkpoint_%s.pkl'%i)


    print_and_log('best_pp-%0.3f, Teacher:%0.3f, Student:%0.3f, teacher_ent: %0.3f, student_ent: %0.3f ' %
                  (best_ping_pong_accs, pp_acc_on_target[0], pp_acc_on_target[1], np.mean(teacher_ent),
                   np.mean(student_ent)),
                  log_folder + 'log')