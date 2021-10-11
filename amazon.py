import numpy as np
import time
import pickle
import copy
import torch
import torch.optim as optim
import os
import datetime
import torch.nn.functional as F
from scipy.sparse import coo_matrix
import random
import argparse
np.random.seed(111)
torch.manual_seed(111)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=["books", "dvd", "electronics", "kitchen"],
                        help="select target domain")
    parser.add_argument("--batch_size", type=int, default=32, help="All loaders batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="ping pong epochs")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of ping-pong loss")
    parser.add_argument("--pseudo_th", type=float, default=0.6, help="teacher threshold to give pseudo labels for student")
    parser.add_argument("--eval_freq", type=int, default=500, help="wait before eval and log models performance")
    parser.add_argument("--log", type=str, default=' ', help="set log file path")
    parser.add_argument('--only_bn', action='store_true')
    # hyperparameters from MDAN
    parser.add_argument("--dimension", type=int, default=5000, help="hyperparameters from MDAN")
    parser.add_argument("--num_trains", type=int, default=2000, help="hyperparameters from MDAN")
    return parser.parse_args()


def eval_on_domain(model, _X, _y, criterion=torch.nn.CrossEntropyLoss()):
    _pred = model(_X)
    _loss = criterion(_pred,_y)
    _, _preds_int = torch.max(_pred, 1)
    _ncorrect = torch.sum(_preds_int == _y.data)
    return _loss.item(), _ncorrect.float()/_X.shape[0]

def get_batch(_X, _y, batch_size = 32):
    rand_idx = np.arange(_X.shape[0])
    np.random.shuffle(rand_idx)
    rand_idx = rand_idx[:batch_size]
    return _X[rand_idx], _y[rand_idx]


def mdan_parsing():
    # dataset + parsing code is from https://github.com/KeiraZhao

    time_start = time.time()
    amazon = np.load("../datasets/amazon_product_reviews/amazon.npz")
    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :args.dimension]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    time_end = time.time()

    # print("Time used to process the Amazon data set = {} seconds.".format(time_end - time_start))
    # print("Number of training instances = {}, number of features = {}."
    #              .format(amazon_xx.shape[0], amazon_xx.shape[1]))
    # print("Number of nonzero elements = {}".format(amazon_xx.nnz))
    # print("amazon_xx shape = {}.".format(amazon_xx.shape))
    # print("amazon_yy shape = {}.".format(amazon_yy.shape))

    data_name = ["books", "dvd", "electronics", "kitchen"]
    num_data_sets = 4
    data_insts, data_labels, num_insts = [], [], []
    for i in range(num_data_sets):
        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i+1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
        # print("Length of the {} data set label list = {}, label values = {}, label balance = {}".format(
        #     data_name[i],
        #     amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :].shape[0],
        #     np.unique(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :]),
        #     np.sum(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
        # ))
        num_insts.append(amazon_offset[i+1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
    # print("Data sets: {}".format(data_name))
    # print("Number of total instances in the data sets: {}".format(num_insts))

    # Partition the data set into training and test parts, following the convention in the ICML-2012 paper, use a fixed
    # amount of instances as training and the rest as test.
    input_dim = amazon_xx.shape[1]

    return data_insts, data_labels


def get_bn_layer_params(bn_layer):
    layer_params = {'running_mean': bn_layer.running_mean.clone(),
                    'running_var': bn_layer.running_var.clone(),
                    'weight': bn_layer.weight.clone(),
                    'bias': bn_layer.bias.clone()}
    return layer_params


def set_bn_layer_params(layer, params):
    layer.running_mean = (params['running_mean'])
    layer.running_var = (params['running_var'])
    layer.weight = torch.nn.Parameter(params['weight'])
    layer.bias = torch.nn.Parameter(params['bias'])


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1=1000, n_hidden2=500, n_hidden3=100, n_output=2):
        super(Net, self).__init__()
        self.bn = torch.nn.BatchNorm1d(n_feature)
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)

        self.domain2model_bns = dict()

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)

        return x

    def save_current_bn(self, domain):
        self.domain2model_bns[domain] = get_bn_layer_params(self.bn)

    def load_bn_to_model(self, domain):
        set_bn_layer_params(self.bn, self.domain2model_bns[domain])

    def init_bn_for_all_domains(self, domains):
        for domain in domains:
            self.domain2model_bns[domain] = get_bn_layer_params(self.bn)


def print_and_log(txt, log_path):
    print(txt)
    f = open(log_path, 'a')
    f.write(txt+'\n')
    f.close()




if __name__ == '__main__':
    args = get_args()
    log_folder = '../logs/amazon/%s_%s_%s/' % (args.target, args.log, datetime.datetime.now())
    os.mkdir(log_folder)
    logfile = '../logs/amazon/%s%s_alpha_%0.1f_pseudo_th_%0.1f_%s.log' % (args.log, args.target, args.alpha,
                                                                          args.pseudo_th, datetime.datetime.now())

    logfile = logfile.replace(' ','_')
    data_insts, data_labels = mdan_parsing()
    domains = ["books", "dvd", "electronics", "kitchen"]
    X_train, X_test, y_train, y_test = dict(), dict(), dict(), dict()

    for i, domain in enumerate(domains):
        X_train[domain] = torch.tensor(data_insts[i][:args.num_trains, :].todense().astype(np.float32))
        y_train[domain] = torch.tensor(data_labels[i][:args.num_trains, :].ravel().astype(np.int64))
        X_test[domain] = torch.tensor(data_insts[i][args.num_trains:, :].todense().astype(np.float32))
        y_test[domain] = torch.tensor(data_labels[i][args.num_trains:, :].ravel().astype(np.int64))

    teacher = Net(args.dimension)
    teacher_optimizer = torch.optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9)
    teacher.init_bn_for_all_domains(domains)
    teacher.train()

    criterion = torch.nn.CrossEntropyLoss()

    student = Net(args.dimension)
    student_optimizer = torch.optim.SGD(student.parameters(), lr=0.001, momentum=0.9)
    student.init_bn_for_all_domains(domains)
    student.train()
    target = args.target
    sources = [x for x in domains if x != target]

    running_ncorrect, running_loss, ns_samples = 0.0, 0.0, 0
    running_student_loss, nt_samples = 0.0, 0
    running_ncorrect_ping_pong, running_ping_pong_loss = 0.0, 0.0

    best_ping_pong_accs, min_source_loss = 0.0, np.inf
    pp_acc_on_target, sl_acc_on_target = 0.0, 0.0

    n = 0

    for i in range(args.epochs):
        source = random.choice(sources)
        Xs, ys = get_batch(X_train[source], y_train[source], batch_size=args.batch_size)

        teacher.load_bn_to_model(source)
        prediction = teacher(Xs)
        loss = criterion(prediction, ys)
        teacher_optimizer.zero_grad()
        loss.backward()

        _, batch_preds_int = torch.max(prediction, 1)
        running_ncorrect += torch.sum(batch_preds_int == ys.data)
        running_loss += loss.item() * Xs.shape[0]
        ns_samples += Xs.shape[0]

        Xt, _ = get_batch(X_train[target], y_train[target], batch_size=args.batch_size)
        teacher.load_bn_to_model(target)
        pseudo_labels = teacher(Xt)

        probs = torch.max(torch.nn.Softmax(dim=1)(pseudo_labels), 1)
        mask = probs[0] > args.pseudo_th
        if sum(mask) == 0 or args.only_bn:
            n += 1
            teacher_optimizer.step()
        else:
            student.load_bn_to_model(target)
            stud_pred = student(Xt)
            student_loss = torch.nn.L1Loss()(stud_pred[mask], pseudo_labels[mask].detach())
            student_optimizer.zero_grad()
            student_loss.backward(retain_graph=True)
            student_optimizer.step()

            running_student_loss += student_loss.item() * Xt[mask].shape[0]
            nt_samples += Xt[mask].shape[0]

            student.load_bn_to_model(source)
            ys_student_est = student(Xs)
            ping_pong_loss = criterion(ys_student_est, ys)
            teacher_t_loss = args.alpha *torch.nn.L1Loss()(stud_pred[mask].detach(), pseudo_labels[mask])
            teacher_t_loss.backward(retain_graph=True)

            teacher_optimizer.step()
            _, batch_preds_int = torch.max(ys_student_est, 1)
            running_ncorrect_ping_pong += torch.sum(batch_preds_int == ys.data)
            running_ping_pong_loss += ping_pong_loss.item() * Xs.shape[0]

        if i % args.eval_freq == 0:
            source_loss = running_loss / ns_samples
            report = 'Epoch: %d, Teacher on S - loss:%0.4f ,acc:%0.4f' % (
                i, source_loss, running_ncorrect.double() / ns_samples)
            print_and_log(report, logfile)
            running_loss, running_ncorrect, ns_samples = 0.0, 0, 0

            for source in sources:
                teacher.load_bn_to_model(source)
                source_loss, source_acc = eval_on_domain(teacher, X_test[source], y_test[source])
                report = 'Epoch %d, Teacher on Source: %s, val_loss:%0.4f, val_acc:%0.4f' % (
                i, source, source_loss, source_acc)
                print_and_log(report, logfile)

            teacher.load_bn_to_model(target)
            teacher_target_loss, teacher_target_acc = eval_on_domain(teacher, X_test[target], y_test[target])
            report = 'Epoch %d, \t\tTeacher on Target: %s, val_loss:%0.4f, val_acc:%0.4f' % (
            i, target, teacher_target_loss,
            teacher_target_acc)
            print_and_log(report,logfile)

            report = 'Epoch: %d, Student on teacher loss:%0.4f, nt=%d' % (i, running_student_loss / max(nt_samples, 1),
                                                                          nt_samples)
            print_and_log(report,logfile)
            running_student_loss, nt_samples = 0.0, 0

            student.load_bn_to_model(target)
            student_target_loss, student_target_acc = eval_on_domain(student, X_test[target], y_test[target])
            report = 'Epoch %d, \t\tStudent on Target: %s, val_loss:%0.4f, val_acc:%0.4f' % (
            i, target, student_target_loss,
            student_target_acc)
            print_and_log(report, logfile)

            ping_pong_accs = []
            for source in sources:
                student.load_bn_to_model(source)
                source_loss, source_acc = eval_on_domain(student, X_test[source], y_test[source])
                ping_pong_accs.append(source_acc)
                report = 'Epoch %d, Student on Source: %s, val_loss:%0.4f, val_acc:%0.4f' % (
                i, source, source_loss, source_acc)
                print_and_log(report, logfile)


            if best_ping_pong_accs < np.mean(ping_pong_accs):
                best_ping_pong_accs = np.mean(ping_pong_accs)
                pp_acc_on_target = teacher_target_acc, student_target_acc

            teacher.load_bn_to_model(target)
            torch.save(teacher, log_folder + 'teacher_checkpoint_%s.pkl' % i)
            student.load_bn_to_model(target)
            torch.save(student, log_folder + 'student_checkpoint_%s.pkl' % i)

            if min_source_loss > source_loss:
                min_source_loss = source_loss
                sl_acc_on_target = teacher_target_acc, student_target_acc
    print_and_log('best_pp-%0.3f, Teacher:%0.3f, Student:%0.3f ' % (best_ping_pong_accs,
                                                                    pp_acc_on_target[0],
                                                                    pp_acc_on_target[1]),
                  logfile)
    print_and_log('best_sl-%0.3f, Teacher:%0.3f, Student:%0.3f ' % (min_source_loss,
                                                                    sl_acc_on_target[0],
                                                                    sl_acc_on_target[1]),
                  logfile)
