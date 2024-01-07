# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from runtime.loggers import Logger

class BaseCallback(ABC):
    def on_fit_start(self, optimizer, args):
        pass

    def on_fit_end(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_start(self):
        pass

    def on_validation_step(self, input, target, pred):
        pass

    def on_validation_end(self, epoch=None):
        pass

    def on_checkpoint_load(self, checkpoint):
        pass

    def on_checkpoint_save(self, checkpoint):
        pass


class LRSchedulerCallback(BaseCallback):
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.scheduler = None

    @abstractmethod
    def get_scheduler(self, optimizer, args):
        pass

    def on_fit_start(self, optimizer, args):
        self.scheduler = self.get_scheduler(optimizer, args)

    def on_checkpoint_load(self, checkpoint):
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


    def on_checkpoint_save(self, checkpoint):
        checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

    def on_epoch_end(self,loss=None):
        if self.logger is not None:
            self.logger.log_metrics({'learning rate': self.scheduler.get_last_lr()[0]}, step=self.scheduler.last_epoch)
        if not loss==None:
            self.scheduler.step(metrics=loss)
        else:
            print ("scheduler step activate!!!")
            self.scheduler.step()

import math
from torch.optim.lr_scheduler import _LRScheduler,ReduceLROnPlateau

class tmp_add(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]
class tmp_rlp(_LRScheduler):
    def __init__(self, optimizer, multiplier=100, total_epoch=10):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler =tmp_add(optimizer,factor=0.75,patience=4,threshold=0.001,verbose=True) 
        self.finished = False
        super(tmp_rlp, self).__init__(optimizer)

    def get_last_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != tmp_add:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1,mode=None):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        self.mode=mode
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_last_lr(self):
        if self.mode=='finetune':
            print ("fine-tuning fixed lr")
            return [self.eta_max]
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None,loss=None,metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group['lr'] = lr

class QM9LRSchedulerCallback(LRSchedulerCallback):
    def __init__(self, logger, epochs):
        super().__init__(logger)
        self.epochs = epochs

    def get_scheduler(self, optimizer, args):
        min_lr = args.min_learning_rate if args.min_learning_rate else args.learning_rate / 100.0
        max_lr=args.max_learning_rate
        t_up=args.warm_up_cycle
        t_0=args.cycle_period
        print ("##########",max_lr)
        print ("@@@@@@@@@@",t_up)
        if args.sch_type=='rlp':
            return tmp_rlp(optimizer)
        if args.sch_type=='cosine':
            print ("!!!!!!!!!",t_0)
            return CosineAnnealingWarmUpRestarts(optimizer,T_0=t_0, eta_max=max_lr,T_up=t_up, gamma=0.50)
        elif args.sch_type=="finetune":
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            print ("Fine-tuning mode on")
            return CosineAnnealingWarmUpRestarts(optimizer,T_0=1000000, eta_max=0.0003,T_up=1, gamma=0.50,mode='finetune')
