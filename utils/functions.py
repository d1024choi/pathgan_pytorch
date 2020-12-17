import numpy as np
import math
import random
import sys
import pickle
import csv
import os
import time
import matplotlib.pyplot as plt
import cv2
import copy
import argparse
from skimage import transform
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


def get_dtypes(useGPU=True):

    if (useGPU):
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    else:
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor

    return long_dtype, float_dtype

def init_weights(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def from_list_to_tensor_to_cuda(x, dtype):

    '''
    x : a list of (seq_len x input_dim)
    '''

    # batch_size x seq_len x input_dim
    x = np.array(x)
    if (len(x.shape) == 2):
        x = np.expand_dims(x, axis=0)

    y = torch.from_numpy(x).permute(1, 0, 2)

    return y.type(dtype).cuda()

def cross_entropy_loss(logit, target):

    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

def l2_loss(pred_traj, pred_traj_gt):

    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt - pred_traj)**2

    return torch.sum(loss) / (seq_len * batch)

def bce_loss(input, target):

    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):

    y_fake = torch.ones_like(scores_fake)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):

    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.0)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real, loss_fake


def save_read_latest_checkpoint_num(path, val, isSave):

    file_name = path + '/checkpoint.txt'
    index = 0

    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index


def find_representative_drvint(input_ori):

    '''
    input: seq_len x 1
    '''

    input = np.copy(input_ori[0:5])

    drvint_codes = np.unique(input)
    num_unique_codes = drvint_codes.size
    drvint_codes = drvint_codes.reshape(num_unique_codes, 1)

    num_codes = []
    for i in range(num_unique_codes):
        cur_code = drvint_codes[i]
        num_codes.append(input[input[:] == cur_code].size)

    return int(drvint_codes[np.argsort(num_codes)[-1], 0])


def drvint_onehot_batch(drvint, dtype):

    '''
    drvint : batch_size x time x 2
    drvint[a, b, 0] : drive intention (0 ~ 6)
    drvint[a, b, 1] : initiation (-1000 or 1)
    '''

    batch_size = len(drvint)

    drvint_onehot = np.zeros(shape=(batch_size, 10))
    drvint_onehot_fake = np.zeros(shape=(batch_size, 10))
    for i in range(batch_size):

        # gt label
        index = find_representative_drvint(drvint[i][:, 0])
        drvint_onehot[i, index] = 1

        # TODO : how to set fake drvint?
        drvint_onehot_fake[i, 0] = 1

    y = torch.from_numpy(drvint_onehot)
    x = torch.from_numpy(drvint_onehot_fake)

    return y.type(dtype).cuda(), x.type(dtype).cuda()


def image_preprocessing(img_re, isAug):

    # TODO : how to set acceptance ratio ?

    if (isAug == 1):

        if (np.random.rand(1) < 0.5):

            # img blur
            if (np.random.rand(1) < 0.5):
                kernel = np.ones((5, 5), np.float32) / 25.0
                img_re_blur = cv2.filter2D(img_re, -1, kernel)

                weight = float(random.randint(0, 50)) / 100.0 # 0~0.5
                img_re = (weight)*img_re_blur + (1-weight)*img_re

            # gamma operation
            if (np.random.rand(1) < 0.5):

                if (np.random.rand(1) < 0.5):
                    gamma = 3
                else:
                    gamma = 0.33

                img_re_gamma = np.copy(img_re).astype('float') / 255.0 # 0~1
                img_re_gamma = np.power(img_re_gamma, gamma) * 255

                weight = float(random.randint(0, 50)) / 100.0 # 0 ~ 0.5
                img_re = (weight)*img_re_gamma + (1-weight)*img_re

                # cv2.imshow('test', img_re.astype('uint8'))
                # cv2.waitKey(0)


            # random channel swap
            if (np.random.rand(1) < 0.5):

                idx_list = [0, 1, 2]
                random.shuffle(idx_list)

                img_re_tmp = np.zeros_like(img_re)
                img_re_tmp[:, :, 0] = np.copy(img_re[:, :, idx_list[0]])
                img_re_tmp[:, :, 1] = np.copy(img_re[:, :, idx_list[1]])
                img_re_tmp[:, :, 2] = np.copy(img_re[:, :, idx_list[2]])
                img_re = np.copy(img_re_tmp)

                # cv2.imshow('test', img_re.astype('uint8'))
                # cv2.waitKey(0)


    return ((img_re.astype('float')/255.0) - 0.5) / 0.5


def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, max_num=5):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index


    num_remain = max_num - 1
    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)


def generate_drvint_text(drv_intention_code):

    text = '[error] no drvint code is available ...'
    if (drv_intention_code == 1):
        text = 'Go'
    elif (drv_intention_code == 2):
        text = 'Turn Left'
    elif (drv_intention_code == 3):
        text = 'Turn Right'
    elif (drv_intention_code == 4):
        text = 'U-turn'
    elif (drv_intention_code == 5):
        text = 'Left LC'
    elif (drv_intention_code == 6):
        text = 'Right LC'
    elif (drv_intention_code == 7):
        text = 'Avoidance'
    elif (drv_intention_code == 8):
        text = 'Left Way'
    elif (drv_intention_code == 9):
        text = 'Right Way'

    return text


# ---------------------------------------------------------------
# etc
# ---------------------------------------------------------------
def print_current_progress(e, b, num_batchs, time_spent):

    if b == num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r [Epoch %02d] %d / %d (%.4f sec/sample)' % (e, b, num_batchs, time_spent)),

    sys.stdout.flush()

