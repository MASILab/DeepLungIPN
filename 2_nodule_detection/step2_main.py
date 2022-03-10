#from config_submit import config as config_submit

import torch

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

#from layers import acc
from data_loader import DataBowl3Detector,collate
#from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas as pd
import pdb
import argparse


from detect_config import config

parser = argparse.ArgumentParser()

parser.add_argument('--sess_csv', type=str, default='test.csv',
                    help='sessions want to be tested')
parser.add_argument('--bbox_root', type=str, default='/nfs/masi/gaor2/tmp/justtest/bbox',
                    help='the root for save preprocessed data')
parser.add_argument('--prep_root', type=str, default='/nfs/masi/gaor2/tmp/justtest/prep',
                    help='the root for save preprocessed data')

args = parser.parse_args()
config['datadir'] = args.prep_root

sess_splits = pd.read_csv(args.sess_csv)['id'].tolist()
config['testsplit'] = sess_splits

nodmodel = import_module('net_detector')
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load('./2_nodule_detection/detector.ckpt')
nod_net.load_state_dict(checkpoint['state_dict'])

nod_net = nod_net

nod_net = nod_net

bbox_result_path = args.bbox_root
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)

split_comber = SplitComb(config1['sidelen'],config1['max_stride'],config1['stride'],config1['margin'],pad_value= config1['pad_value'])

dataset = DataBowl3Detector(config['testsplit'],config1,phase='test',split_comber=split_comber)
test_loader = DataLoader(dataset, batch_size = 1,
    shuffle = False, num_workers = 1, pin_memory=False, collate_fn =collate)

test_detect(test_loader, nod_net, get_pbb, bbox_result_path, config1)
