# import sys
# sys.path.append('../..')
# import func.models.crnn as crnn
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


# f = open('crossv.yaml', 'r').read()
# cfig = yaml.load(f)
# shutil.copyfile('./crossv.yaml', cfig['save_path'] + '/tmp_cfig.yaml')


def seed_everything(seed, cuda=True):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)



class Trainer(object):
    def __init__(self, cfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig

        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')


        if self.cfig['model_name'] == 'MultipathModel':
            self.model = MultipathModel(n_factor=self.cfig['n_factor']).to(self.device)
            
        if self.cfig['model_name'] == 'MultipathModelBL':
            self.model = MultipathModelBL(n_factor=self.cfig['n_factor'], dim = self.cfig['dim']).to(self.device)


        self.optim = torch.optim.SGD(self.model.parameters(), self.cfig['lr'],momentum = 0.9,weight_decay = self.cfig['weight_decay'])

        self.train_loader, self.eval_gen_dict = self.data_loader()

        self.lr = cfig['lr']


    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
                print('After modify, the learning rate is', param_group['lr'])

    def data_loader(self):
        
            df = pd.read_csv(self.cfig['label_csv'])
            print (self.cfig['label_csv'])
            df = df.query('gt_reg == gt_reg')
            
            df = df.loc[df['Mayo.Risk'] == df['Mayo.Risk']]
            #df = df.loc[df['Brock.Risk'] == df['Brock.Risk']]
            #df = df.loc[df['norm_logbmark'] == df['norm_logbmark']]
            
            print ('len df -----------', len(df))

            need_factor = ['subjwithkaggle', 'subjwithfactor',  'norm_age', 'norm_bmi', 'phist',  'smo_status', 'norm_pky', 'norm_logsize',  'spic',  "upper",  'norm_logbmark', 'Mayo.Risk']   # 'gender', complete_factor "GGO", 'partsolid', 'norm_race_mImput', 'Emphysema.Brock',
            
            
            list_IDs = list(set(df['usedsess'].tolist()))
            
            partition_IDs = {'train': [], 'validation': [], 'test': [], 'c3': [], 'c3_ipn': [], 'c2': [], 'c2_ipn': [], 'c4': [], 'c4_ipn': []}
            
            path_labels = {}
            path_factors = {}
            
            for i, item in df.iterrows():
                
                path = self.cfig['data_path']['mcl'] + '/'  + item['usedsess']  + '.npy'
                
                path_labels[path] = item['gt_reg']
                
                tmp_factors = []
                
                for name in need_factor: 
                    tmp_factors.append(float(item[name]))
                
                if 'default' not in path and not os.path.exists(path):
                    
                    if not os.path.exists(path):
                        print (path + ' is not existed')
                        path = path + 'default'  # change at 0909
                        
                path_factors[path] = tmp_factors
                
                if (item['Cohort'] == 3 and item['subjwithkaggle'] == 1 and item['Mayo.Risk'] == item['Mayo.Risk']):
                    partition_IDs['c3'].append(path)
                
                if (item['Cohort'] == 3 and item['Flag.IPN'] == 1 and item['subjwithkaggle'] == 1 and item['Mayo.Risk'] == item['Mayo.Risk']):
                    partition_IDs['c3_ipn'].append(path)

                if (item['Cohort'] == 2 and item['subjwithkaggle'] == 1 and item['Mayo.Risk'] == item['Mayo.Risk']):
                    partition_IDs['c2'].append(path)

                if (item['Cohort'] == 2 and item['Flag.IPN'] == 1 and item['subjwithkaggle'] == 1 and item['Mayo.Risk'] == item['Mayo.Risk']):
                    partition_IDs['c2_ipn'].append(path)
                    
                if (item['Cohort'] == 4 and item['subjwithkaggle'] == 1 and item['Mayo.Risk'] == item['Mayo.Risk']):
                    partition_IDs['c4'].append(path)

                if (item['Cohort'] == 4 and item['Flag.IPN'] == 1 and item['subjwithkaggle'] == 1 and item['Mayo.Risk'] == item['Mayo.Risk']):
                    partition_IDs['c4_ipn'].append(path)

                if ((item['Cohort'] in [1]) and item['phase'] == self.cfig['val_phase'] and item['Mayo.Risk'] == item['Mayo.Risk']): # 
                    partition_IDs['validation'].append(path)

                if (item['Cohort'] in [1] and item['phase'] != self.cfig['val_phase'] and item['Mayo.Risk'] == item['Mayo.Risk']): # 
                    partition_IDs['train'].append(path)
                    
            
            df = pd.read_csv(self.cfig['supplement_csv'])
            df = df.loc[df['phase'] != self.cfig['val_phase']]
            
            df = df.query('gt_reg == gt_reg')
            
            df = df.query('feat_exist == 1')
            
            cnt = 0
            
            for i, item in df.iterrows():
                
                path = self.cfig['data_path']['nlst'] + '/'  + item['usedsess']  + '.npy'
                
                if path in path_factors.keys():
                    continue
                cnt += 1
                if 'default' not in path and not os.path.exists(path):
        
                    if not os.path.exists(path):
                        print (path + ' is not existed')
                        path = path + 'default'
                        continue
                
                tmp_factors = [1, 0]

                partition_IDs['train'].append(path)
            
                for j in range(2, len(need_factor)):
                    tmp_factors.append(np.nan)
                
                #if path not in path_factors.keys():
                
                path_factors[path] = tmp_factors
                
                path_labels[path] = item['gt_reg']
                
            print ('number of samples added from supplement_csv: ', cnt)
            
            
            
            training_set = MultiPath_loaderv2(partition_IDs['train'],  path_factors, path_labels)
            validation_set = MultiPath_loaderv2(partition_IDs['validation'], path_factors, path_labels)

            c3_set = MultiPath_loaderv2(partition_IDs['c3'], path_factors, path_labels)
            c3_ipn_set = MultiPath_loaderv2(partition_IDs['c3_ipn'], path_factors, path_labels)
            c2_set = MultiPath_loaderv2(partition_IDs['c2'], path_factors, path_labels)
            c2_ipn_set = MultiPath_loaderv2(partition_IDs['c2_ipn'], path_factors, path_labels)  
            
            c4_set = MultiPath_loaderv2(partition_IDs['c4'], path_factors, path_labels)
            c4_ipn_set = MultiPath_loaderv2(partition_IDs['c4_ipn'], path_factors, path_labels)
                
            print('len of train set and val set', len(training_set), len(validation_set))
            
            paramstrain = {'shuffle': True,
                           'num_workers': 4,
                           'batch_size': self.cfig['batch_size']}
            paramstest = {'shuffle': False,
                          'num_workers': 4,
                          'batch_size': self.cfig['test_batch_size']}
            
            training_generator = data.DataLoader(training_set, **paramstrain)
            tr_generator = data.DataLoader(training_set, **paramstrain)
            validation_generator = data.DataLoader(validation_set, **paramstest)
            eval_gen_list = {}
            eval_gen_list['val'] = validation_generator
            eval_gen_list['train'] = tr_generator
            if self.cfig['external_test']:
                c3_generator = data.DataLoader(c3_set, **paramstest)
                c2_generator = data.DataLoader(c2_set, **paramstest)
                c3_ipn_generator = data.DataLoader(c3_ipn_set, **paramstest)
                c2_ipn_generator = data.DataLoader(c2_ipn_set, **paramstest)
                
                c4_generator = data.DataLoader(c4_set, **paramstest)
                c4_ipn_generator = data.DataLoader(c4_ipn_set, **paramstest)
                
                eval_gen_list['c3'] = c3_generator 
                eval_gen_list['c2'] = c2_generator
                eval_gen_list['c3_ipn'] = c3_ipn_generator
                eval_gen_list['c2_ipn'] = c2_ipn_generator
                
                eval_gen_list['c4'] = c4_generator
                eval_gen_list['c4_ipn'] = c4_ipn_generator
                
            else:    
                test_generator = data.DataLoader(test_set, **paramstest)
                eval_gen_list['test'] = test_generator


            return training_generator, eval_gen_list
            

    def train(self):
            for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
                if self.cfig['adjust_lr']:
                    self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
                print('lr: ', self.lr)
                model_root = os.path.join(self.cfig['save_path'], 'models')
                if not os.path.exists(model_root):
                    os.mkdir(model_root)
                model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
                if os.path.exists(model_pth) and self.cfig['use_exist_model']:
                    if self.device == 'cuda':  # there is a GPU device
                        self.model.load_state_dict(torch.load(model_pth))
                    else:
                        self.model.load_state_dict(
                            torch.load(model_pth, map_location=lambda storage, location: storage))
                else:
                    self.train_epoch(epoch)
                    torch.save(self.model.state_dict(), model_pth)
                    
                self.eval_epoch(epoch, 'val')
                
                self.eval_epoch(epoch, 'c2')
                self.eval_epoch(epoch, 'c2_ipn')
                self.eval_epoch(epoch, 'c3')
                self.eval_epoch(epoch, 'c3_ipn')
                self.eval_epoch(epoch, 'c4')
                self.eval_epoch(epoch, 'c4_ipn')
                

    def train_epoch(self, epoch):
            self.model.train()
            if not os.path.exists(self.csv_path):
                os.mkdir(self.csv_path)
            train_csv = os.path.join(self.csv_path, 'train.csv')
#             pred_list, target_list, loss_list, pos_list = [], [], [], []
#             t_loss, risk_loss = [], []
#             risk_list = []
            IMGPred, CLICPred, BOTHImgPred, BOTHClicPred, BOTHPred =  [], [], [], [], []
            IMGBi, CLICBi, BOTHImgBi, BOTHClicBi, BOTHBi =  [], [], [], [], []
            IMGTar, CLICTar, BOTHTar = [], [], []
            
            for batch_idx, data_tup in enumerate(self.train_loader):
                
                data,  factors, target, ID = data_tup
                data, target = data.to(self.device), target.to(self.device)
                factors = factors.to(self.device)
                imgOnly = data[(factors[:, 0] > 0.99).nonzero().squeeze()]
                clicOnly = factors[(factors[:, 1] > 0.99).nonzero().squeeze()]
                
                bothImg = data[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                bothFactor = factors[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                
                if len(imgOnly.shape) == 2: imgOnly = imgOnly.unsqueeze(0)
                if len(clicOnly.shape) == 1 and len(clicOnly) > 0: clicOnly = clicOnly.unsqueeze(0)
                if len(bothImg.shape) == 2 and len(bothImg) > 0: 
                    print (bothImg.shape)
                    bothImg = bothImg.unsqueeze(0)
                if len(bothFactor.shape) == 1 and len(bothFactor) > 0: 
                    print (bothFactor.shape)
                    bothFactor = bothFactor.unsqueeze(0)
                
                imgPred, clicPred, bothImgPred, bothClicPred, bothPred = self.model(imgOnly, clicOnly, bothImg, bothFactor)
               # pdb.set_trace()
                
                imgOnlyTar = target[(factors[:, 0] > 0.99).nonzero().squeeze()]
                clicOnlyTar = target[(factors[:, 1] > 0.99).nonzero().squeeze()]
                bothTar = target[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                self.cfig['bce_weight'] = [1.0, 1.0]
                imgloss = LossPool(imgPred , imgOnlyTar.float(), self.cfig, loss_name='bi_ce_loss').get_loss() 
                
                if clicPred.shape[0] == 0:
                    clicloss = 0
                else:
                    self.cfig['bce_weight'] = [1.0, 0.5]
                    clicloss = LossPool(clicPred,  clicOnlyTar.float(), self.cfig, loss_name='bi_ce_loss').get_loss() 
                
                if bothImgPred.shape[0] == 0:
                    bothimgloss = 0
                    bothclicloss = 0
                    lastloss = 0
                else:
                    self.cfig['bce_weight'] = [1.0, 0.5]
                    lastloss = LossPool(bothPred, bothTar.float(), self.cfig, loss_name=cfig['loss_name']).get_loss() 
                
                loss = self.cfig['img_alpha'] * imgloss + self.cfig['clic_alpha'] * clicloss + self.cfig['both_alpha']  * lastloss 
                #loss = imgloss #clicloss 
                
                self.optim.zero_grad()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optim.step()
                
                print_str = 'train epoch=%d, batch_idx=%d/%d\n' % (
                    epoch, batch_idx, len(self.train_loader))
                if batch_idx % 20 == 0: print(print_str)
                
                IMGPred += imgPred.data.cpu().numpy().tolist()
                CLICPred += clicPred.data.cpu().numpy().tolist()
                
                if bothImgPred.shape[0] > 0:
                    BOTHImgPred += bothImgPred.data.cpu().numpy().tolist()
                    BOTHClicPred += bothClicPred.data.cpu().numpy().tolist()
                    BOTHPred += bothPred.data.cpu().numpy().tolist()
                
                IMGBi += (imgPred > 0.5).tolist() 
                CLICBi +=  (clicPred > 0.5).tolist() 
                if bothImgPred.shape[0] > 0:
                    BOTHImgBi += (bothImgPred > 0.5).tolist() 
                    BOTHClicBi +=  (bothClicPred > 0.5).tolist() 
                    BOTHBi += (bothPred > 0.5).tolist() 
                    
                #pdb.set_trace()
                #print (imgOnlyTar)
                IMGTar += imgOnlyTar.data.cpu().numpy().tolist()
                
                if len(clicOnlyTar.shape) == 0:
                    CLICTar.append(clicOnlyTar.data.cpu().numpy().tolist())
                else:
                    CLICTar += clicOnlyTar.data.cpu().numpy().tolist()
                
                if len(bothTar.shape) == 0:
                    BOTHTar.append(bothTar.data.cpu().numpy().tolist())
                else:
                    BOTHTar += bothTar.data.cpu().numpy().tolist()
                

            #print(confusion_matrix(target_list, pred_list))

            fpr, tpr, threshold = metrics.roc_curve(IMGTar, IMGPred)
            IMGauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(CLICTar, CLICPred)
            CLICauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHImgPred)
            BOTHImgauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHClicPred)
            BOTHclicauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHPred)
            BOTHauc = metrics.auc(fpr, tpr)
            
            print ("Train: Length of IMGPred, CLICPred, BOTHImgPred, BOTHClicPred, BOTHPred")
            print (len(IMGPred), len(CLICPred), len(BOTHImgPred), len(BOTHClicPred), len(BOTHPred))
            
            print ('TRAIN: IMGauc, CLICauc, BOTHImgauc, BOTHclicauc, BOTHauc: ')
            
            print (round(IMGauc,3), round(CLICauc,3), round(BOTHImgauc,3), round(BOTHclicauc,3), round(BOTHauc, 3))
            
            
            # -------------------------save to csv -----------------------#
            if not os.path.exists(train_csv):
                csv_info = ['epoch', 'IMGauc', 'CLICauc', 'BOTHImgauc', 'BOTHclicauc', 'BOTHauc'] 
                
                init_csv = pd.DataFrame()
                for key in csv_info:
                    init_csv[key] = []
                init_csv.to_csv(train_csv)
            df = pd.read_csv(train_csv)
            data = pd.DataFrame()
            tmp_epoch = df['epoch'].tolist()
            tmp_epoch.append(epoch)

            print('------------------', tmp_epoch)
            
            tmp_IMGauc = df['IMGauc'].tolist()
            tmp_IMGauc.append(IMGauc)
            
            tmp_CLICauc = df['CLICauc'].tolist()
            tmp_CLICauc.append(CLICauc)
            
            tmp_BOTHImgauc = df['BOTHImgauc'].tolist()
            tmp_BOTHImgauc.append(BOTHImgauc)
            
            tmp_BOTHclicauc = df['BOTHclicauc'].tolist()
            tmp_BOTHclicauc.append(BOTHclicauc)
            
            tmp_BOTHauc = df['BOTHauc'].tolist()
            tmp_BOTHauc.append(BOTHauc)

            data['epoch'], data['IMGauc'], data['CLICauc'] = tmp_epoch, tmp_IMGauc, tmp_CLICauc

            data['BOTHImgauc'], data['BOTHclicauc'], data['BOTHauc'] = tmp_BOTHImgauc, tmp_BOTHclicauc, tmp_BOTHauc

            data.to_csv(train_csv)

    def eval_epoch(self, epoch, phase):
            self.model.eval()
            if not os.path.exists(self.csv_path):
                os.mkdir(self.csv_path)
            eval_csv = os.path.join(self.csv_path, phase + '.csv')
            IMGPred, CLICPred, BOTHImgPred, BOTHClicPred, BOTHPred =  [], [], [], [], []
            IMGBi, CLICBi, BOTHImgBi, BOTHClicBi, BOTHBi =  [], [], [], [], []
            IMGTar, CLICTar, BOTHTar = [], [], []
            loader = self.eval_gen_dict[phase]
#             if phase == 'eval':
#                 loader = self.val_loader
#             if phase == 'test':
#                 loader = self.test_loader

#             if phase == 'ext':
#                 loader = self.ext_loader
            with torch.no_grad():
                for batch_idx, data_tup in enumerate(loader):
                    
                    data,  factors, target, ID = data_tup

                    data, target = data.to(self.device), target.to(self.device)
                    factors = factors.to(self.device)
                    imgOnly = data[(factors[:, 0] > 0.99).nonzero().squeeze()]
                    clicOnly = factors[(factors[:, 1] > 0.99).nonzero().squeeze()]

                    bothImg = data[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                    bothFactor = factors[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]

                    imgPred, clicPred, bothImgPred, bothClicPred, bothPred = self.model(imgOnly, clicOnly, bothImg, bothFactor)
                    
                    imgOnlyTar = target[(factors[:, 0] > 0.99).nonzero().squeeze()]
                    clicOnlyTar = target[(factors[:, 1] > 0.99).nonzero().squeeze()]
                    bothTar = target[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]

                    IMGPred += imgPred.data.cpu().numpy().tolist()
                    CLICPred += clicPred.data.cpu().numpy().tolist()
                    BOTHImgPred += bothImgPred.data.cpu().numpy().tolist()
                    BOTHClicPred += bothClicPred.data.cpu().numpy().tolist()
                    BOTHPred += bothPred.data.cpu().numpy().tolist()

                    IMGBi += (imgPred > 0.5).tolist() 
                    CLICBi +=  (clicPred > 0.5).tolist() 
                    BOTHImgBi += (bothImgPred > 0.5).tolist() 
                    BOTHClicBi +=  (bothClicPred > 0.5).tolist() 
                    BOTHBi += (bothPred > 0.5).tolist() 

                    IMGTar += imgOnlyTar.data.cpu().numpy().tolist()
                    CLICTar += clicOnlyTar.data.cpu().numpy().tolist()
                    BOTHTar += bothTar.data.cpu().numpy().tolist()
                    
            #print(confusion_matrix(target_list, pred_list))

            fpr, tpr, threshold = metrics.roc_curve(IMGTar, IMGPred)
            IMGauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(CLICTar, CLICPred)
            CLICauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHImgPred)
            BOTHImgauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHClicPred)
            BOTHclicauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHPred)
            BOTHauc = metrics.auc(fpr, tpr)
            
            print (phase, ": Length of IMGPred, CLICPred, BOTHImgPred, BOTHClicPred, BOTHPred")
            print (len(IMGPred), len(CLICPred), len(BOTHImgPred), len(BOTHClicPred), len(BOTHPred))

            print (phase, ': IMGauc, CLICauc, BOTHImgauc, BOTHclicauc, BOTHauc: ')
            print (IMGauc, CLICauc, BOTHImgauc, BOTHclicauc, BOTHauc)
            
            if not os.path.exists(eval_csv):
                csv_info = ['epoch', 'IMGauc', 'CLICauc', 'BOTHImgauc', 'BOTHclicauc', 'BOTHauc'] 
                
                init_csv = pd.DataFrame()
                for key in csv_info:
                    init_csv[key] = []
                init_csv.to_csv(eval_csv)
            df = pd.read_csv(eval_csv)
            data = pd.DataFrame()
            tmp_epoch = df['epoch'].tolist()
            tmp_epoch.append(epoch)
            
            tmp_IMGauc = df['IMGauc'].tolist()
            tmp_IMGauc.append(IMGauc)
            
            tmp_CLICauc = df['CLICauc'].tolist()
            tmp_CLICauc.append(CLICauc)
            
            tmp_BOTHImgauc = df['BOTHImgauc'].tolist()
            tmp_BOTHImgauc.append(BOTHImgauc)
            
            tmp_BOTHclicauc = df['BOTHclicauc'].tolist()
            tmp_BOTHclicauc.append(BOTHclicauc)
            
            tmp_BOTHauc = df['BOTHauc'].tolist()
            tmp_BOTHauc.append(BOTHauc)

            data['epoch'], data['IMGauc'], data['CLICauc'] = tmp_epoch, tmp_IMGauc, tmp_CLICauc

            data['BOTHImgauc'], data['BOTHclicauc'], data['BOTHauc'] = tmp_BOTHImgauc, tmp_BOTHclicauc, tmp_BOTHauc

            data.to_csv(eval_csv)

    def test_epoch(self, model_pth, phase):
            self.model.eval()
#             model_root = os.path.join(self.cfig['save_path'], 'models')
#             model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            if self.device == 'cuda':  # there is a GPU device
                self.model.load_state_dict(torch.load(model_pth))
            else: 
                self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            IMGPred, CLICPred, BOTHImgPred, BOTHClicPred, BOTHPred =  [], [], [], [], []
            IMGBi, CLICBi, BOTHImgBi, BOTHClicBi, BOTHBi =  [], [], [], [], []
            IMGTar, CLICTar, BOTHTar = [], [], []  
            ID_list = []  
            loader = self.eval_gen_dict[phase]
#             if phase == 'eval': 
#                 loader = self.val_loader
#             if phase == 'test':  
#                 loader = self.test_loader
#             if phase == 'ext':  
#                 loader = self.ext_loader
            with torch.no_grad():
                for batch_idx, data_tup in enumerate(loader):
                    
                    data,  factors, target, ID = data_tup
                    #print (list(ID))
                    data, target = data.to(self.device), target.to(self.device)
                    factors = factors.to(self.device) 
                    imgOnly = data[(factors[:, 0] > 0.99).nonzero().squeeze()]
                    clicOnly = factors[(factors[:, 1] > 0.99).nonzero().squeeze()]

                    bothImg = data[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                    bothFactor = factors[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                    #pdb.set_trace()
                    imgPred, clicPred, bothImgPred, bothClicPred, bothPred = self.model(imgOnly, clicOnly, bothImg, bothFactor)
                    
                    factors = factors.cpu()
                    
                    imgOnlyTar = target[(factors[:, 0] > 0.99).nonzero().squeeze()]
                    clicOnlyTar = target[(factors[:, 1] > 0.99).nonzero().squeeze()]
                    bothTar = target[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                    
                    
                    
                    bothID = np.array(list(ID))[(factors[:, 0] * factors[:, 1] > 0.99).nonzero().squeeze()]
                    ImgID = np.array(list(ID))[(factors[:, 0]  > 0.99).nonzero().squeeze()]
                    clicID = np.array(list(ID))[(factors[:, 1]  > 0.99).nonzero().squeeze()]

                    IMGPred += imgPred.data.cpu().numpy().tolist()
                    CLICPred += clicPred.data.cpu().numpy().tolist()
                    BOTHImgPred += bothImgPred.data.cpu().numpy().tolist()
                    BOTHClicPred += bothClicPred.data.cpu().numpy().tolist()
                    BOTHPred += bothPred.data.cpu().numpy().tolist()

                    IMGBi += (imgPred > 0.5).tolist() 
                    CLICBi +=  (clicPred > 0.5).tolist() 
                    BOTHImgBi += (bothImgPred > 0.5).tolist() 
                    BOTHClicBi +=  (bothClicPred > 0.5).tolist() 
                    BOTHBi += (bothPred > 0.5).tolist() 

                    IMGTar += imgOnlyTar.data.cpu().numpy().tolist()
                    CLICTar += clicOnlyTar.data.cpu().numpy().tolist()
                    BOTHTar += bothTar.data.cpu().numpy().tolist()
                    ID_list += bothID.tolist()
                    
            #print(confusion_matrix(target_list, pred_list))

            fpr, tpr, threshold = metrics.roc_curve(IMGTar, IMGPred)
            IMGauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(CLICTar, CLICPred)
            CLICauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHImgPred)
            BOTHImgauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHClicPred)
            BOTHclicauc = metrics.auc(fpr, tpr)
            
            fpr, tpr, threshold = metrics.roc_curve(BOTHTar, BOTHPred)
            BOTHauc = metrics.auc(fpr, tpr)
            
            print ("Eval: Length of IMGPred, CLICPred, BOTHImgPred, BOTHClicPred, BOTHPred")
            print (len(IMGPred), len(CLICPred), len(BOTHImgPred), len(BOTHClicPred), len(BOTHPred))

            print ('EVAL: IMGauc, CLICauc, BOTHImgauc, BOTHclicauc, BOTHauc: ')
            print (IMGauc, CLICauc, BOTHImgauc, BOTHclicauc, BOTHauc)
            
            return ID_list, BOTHTar, BOTHPred, BOTHauc
            #return ID_list, BOTHTar, BOTHImgauc, BOTHclicauc, BOTHauc,  BOTHImgPred, BOTHClicPred, BOTHPred


if __name__ == '__main__':
    f = open('multipath_cfig.yaml', 'r').read()
    cfig = yaml.load(f)
    
    seed_everything(seed = 1)
    
    trainer = Trainer(cfig)
    
    shutil.copy(cfig['label_csv'], cfig['save_path'])
    codefile = os.listdir('./')
    for f in codefile:
        if not os.path.exists(cfig['save_path'] + '/code'):
            os.mkdir(cfig['save_path'] + '/code' )
        try:
            shutil.copy(f,os.path.join(cfig['save_path'] + '/code' ,f))
        except:
            print (f, ' copy failed')
    trainer.train()


#     ID_list, BothTar, BOTHImgauc, BOTHclicauc, BOTHauc,  BOTHImgPred, BOTHClicPred, BOTHPred = trainer.test_epoch(cfig['test_epoch'], cfig['test_phase'])
#     print ('BOTHImgauc, BOTHclicauc, BOTHauc', BOTHImgauc, BOTHclicauc, BOTHauc)
#     #print (ID_list)
#     data = pd.DataFrame()
    
#     data['ID_list'] = ID_list
#     data['BOTHImgPred'] = BOTHImgPred
#     data['BOTHClicPred'] = BOTHClicPred
#     data['BOTHPred'] = BOTHPred
#     data['BothTar'] = BothTar
    
#     save_root = '/nfs/masi/gaor2/saved_file/mcl2021/10factors_5dim/val1'
#     #phase = 'c2'
#     epoch = 72
#     for phase in ['val']: # ['c2', 'c2_ipn', 'c3', 'c3_ipn', 'c4', 'c4_ipn']:
#         data = pd.DataFrame()
#         ID_list, BOTHTar, BOTHPred, BOTHauc = trainer.test_epoch(save_root + '/models/model_epoch_{0:04d}.pth'.format(epoch), phase)
#         #pdb.set_trace()
#         data['ID_list'] = ID_list
#         data['BOTHPred'] = BOTHPred
#         data['BOTHTar'] = BOTHTar
#         print (phase, ' BOTHauc ', BOTHauc)

#         data.to_csv(save_root + '/' + phase + '_{:d}.csv'.format(epoch), index = False)
    
