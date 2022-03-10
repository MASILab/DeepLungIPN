import argparse
import os, sys 
import time 
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, confusion_matrix
import pandas as pd
from loss_function import LossPool
from torch.utils import data

from tqdm import tqdm

from data_loader import *
from model import *

import yaml
import shutil
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--sess_csv', type=str, default='100004time1999',
                    help='sessions want to be tested')
parser.add_argument('--feat_root', type=str, default='/nfs/masi/gaor2/tmp/justtest/bbox',
                    help='the root for save feat data')
parser.add_argument('--save_csv_path', type=str, default='/nfs/masi/gaor2/tmp/justtest/prep',
                    help='the root for save result data')

args = parser.parse_args()

need_factor =  ['subjwithkaggle', 'subjwithfactor',  'norm_age', 'norm_bmi', 'phist',  'smo_status', 'norm_pky', 'norm_logsize',  'spic',  "upper",  'norm_logbmark', 'Mayo.Risk'] 

sess_mark_dict = {}

df = pd.read_csv(args.sess_csv)
sess_splits = df['id'].tolist()
testsplit = sess_splits

for i, item in df.iterrows():
    test_biomarker = np.zeros(12).astype('float32')
    for j in range(len(need_factor)):
        test_biomarker[j] = item[need_factor[j]]
    sess_mark_dict[item['id']] = test_biomarker

data_path = args.feat_root



self.model = MultipathModelBL(n_factor=10, dim = 5).to(self.device)

model_pth = './4_co_learning/trained_model.pth'

model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
#model.load_state_dict(torch.load(model_pth))

pred_list = []

for i in range(len(testsplit)):
    sess_id = testsplit[i]
    test_biomarker = sess_mark_dict[sess_id]
    
    test_imgfeat = np.load(data_path + '/' + sess_id + '.npy')
    test_biomarker = torch.from_numpy(test_biomarker).unsqueeze(0)
    test_imgfeat = torch.from_numpy(test_imgfeat).unsqueeze(0)
    imgPred, clicPred, bothImgPred, bothClicPred, bothPred = model(test_imgfeat, test_biomarker, test_imgfeat, test_biomarker)
    pred_list += list(bothPred.data.numpy())

data = pd.DataFrame()
data['id'] = testsplit
data['pred'] = pred_list

data.to_csv(args.save_csv_path, index = False)

print (pred_list)

