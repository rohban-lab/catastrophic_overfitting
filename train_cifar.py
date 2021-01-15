import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18

from utils import *

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def quantile_fgsm(model, X, y, epsilon, alpha, q_val, fgsm_init):
    delta = torch.zeros_like(X)
    if fgsm_init=='random':
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    output = model(normalize(X + delta))
    F.cross_entropy(output, y).backward()
    grad = delta.grad.detach()
    q_grad = torch.quantile(torch.abs(grad).view(grad.size(0), -1), q_val, dim=1)
    grad[torch.abs(grad) < q_grad.view(grad.size(0), 1, 1, 1)] = 0
    d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
    d = clamp(d, lower_limit - X, upper_limit - X)
    return d.detach()

def consensus_fgsm(model, X, y, epsilon, alpha, samples, zeroing_th=-1, parallel=True):
    if zeroing_th==-1:
        zeroing_th = samples
    if parallel:
        X_cat = torch.cat([X for i in range(samples)], dim=0)
        delta_cat = torch.zeros_like(X_cat)
        delta_cat.uniform_(-epsilon, epsilon)
        delta_cat = clamp(delta_cat, lower_limit - X_cat, upper_limit - X_cat)
        delta_cat.requires_grad = True
        y_cat = torch.cat([y for i in range(samples)], dim=0)
        output = model(normalize(X_cat + delta_cat))
        F.cross_entropy(output, y_cat).backward()
        grad_cat = delta_cat.grad.detach()
        grads = [grad_cat[i*X.size(0):(i+1)*X.size(0)] for i in range(samples)]
    else :
        grads = []
        for _ in range(samples):
            delta = torch.zeros_like(X)
            delta.uniform_(-epsilon, epsilon)
            delta = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(normalize(X + delta))
            F.cross_entropy(output, y).backward()
            grads += [torch.clone(delta.grad.detach())]
    g = sum([torch.sign(grads[i]) for i in range(samples)])
    grad = torch.where(torch.abs(g) < 
              (zeroing_th - (samples - zeroing_th)),
              torch.zeros_like(g), g)
    delta = torch.zeros_like(X).cuda() 
    d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
    d = clamp(d, lower_limit - X, upper_limit - X)
    return d.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, fgsm_init=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if attack_iters>1 or fgsm_init=='random': 
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
            max_delta[all_loss >= max_loss] = torch.clone(delta.detach()[all_loss >= max_loss])
            max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=52, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise'])
    parser.add_argument('--piecewise-lr-drop', default=50, type=int)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack', default='cfgsm', type=str, choices=['cfgsm', 'qfgsm', 'fgsm'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=2, type=float)
    parser.add_argument('--c-samps', default=3, type=int)
    parser.add_argument('--c-th', default=-1, type=int)
    parser.add_argument('--c-parallel', action='store_true')
    parser.add_argument('--q-val', default=0.4, type=float)
    parser.add_argument('--fgsm-init', default='random', type=str, choices=['zero', 'random'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--full-test', action='store_true')
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_set_x = Transform(test_set, [])
    test_batches = Batches(test_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).cuda()
    model.train()

    params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t < args.piecewise_lr_drop:
                return args.lr_max
            else:
                return args.lr_max / 10.

    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch-1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t Train Time \t Test Time \t LR \t \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc \t Defence Mean \t Attack Mean')
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        defence_mean = 0
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                delta = attack_pgd(model, X, y, epsilon, args.fgsm_alpha * epsilon, 1, 1, 'l_inf', fgsm_init=args.fgsm_init)
            elif args.attack == 'cfgsm':
                delta = consensus_fgsm(model, X, y, epsilon, args.fgsm_alpha * epsilon, args.c_samps, args.c_th, args.c_parallel)
            elif args.attack == 'qfgsm':
                delta = quantile_fgsm(model, X, y, epsilon, args.fgsm_alpha * epsilon, args.q_val, args.fgsm_init)
            delta = delta.detach()
            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            defence_mean += torch.mean(torch.abs(delta)) * y.size(0)

        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        attack_mean = 0
        for i, batch in enumerate(test_batches):
            if not epoch+1==epochs and not args.full_test and i > len(test_batches) / 10:
                break
            X, y = batch['input'], batch['target']

            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 10, 1, 'l_inf', early_stop=args.eval)
            delta = delta.detach()

            robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = criterion(robust_output, y)

            output = model(normalize(X))
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)
            attack_mean += torch.mean(torch.abs(delta)) * y.size(0)

        test_time = time.time()

        
        if not args.eval:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f',
                epoch, train_time - start_time, test_time - train_time, lr,
                train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n,
                defence_mean*255/train_n, attack_mean*255/test_n)

            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
                torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(args.fname, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n,
                -1, attack_mean*255/test_n)
                
        if args.eval or epoch+1 == epochs:
            start_test_time = time.time()
            test_loss = 0
            test_acc = 0
            test_robust_loss = 0
            test_robust_acc = 0
            test_n = 0
            for i, batch in enumerate(test_batches):
                X, y = batch['input'], batch['target']

                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 50, 10, 'l_inf', early_stop=True)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                test_robust_loss += robust_loss.item() * y.size(0)
                test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                test_loss += loss.item() * y.size(0)
                test_acc += (output.max(1)[1] == y).sum().item()
                test_n += y.size(0)
            
            logger.info('PGD50 \t time: %.1f,\t clean loss: %.4f,\t clean acc: %.4f,\t robust loss: %.4f,\t robust acc: %.4f',
                time.time() - start_test_time,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return
        


if __name__ == "__main__":
    main()
