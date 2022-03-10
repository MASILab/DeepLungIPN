import sys
sys.path.append('../..')
import func.models.crnn as crnn
import func.models.rnn as myrnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
import numpy as np
from torch.autograd import Variable

def seed_everything(seed, cuda=True):
  # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cuda:
        torch.cuda.manual_seed_all(seed)
    
seed_everything(seed = 0)

class MultipathModel(nn.Module):
    def __init__(self, n_factor):
        super(MultipathModel, self).__init__()
        self.n_class = 1
        
        self.fc1 = nn.Linear(128, 64)
        
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # consider to replace the attention layer with torch.nn.attention. 
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),  # 64,
            nn.Sigmoid()
        )
        
        self.cliclayer = nn.Sequential(
            nn.Linear(n_factor, 64), 
            nn.Tanh(),       # default Tanh
#             nn.Linear(64, 64),
#             nn.Tanh(),
            #self.dropout,
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        self.lastlayer = nn.Sequential(
            nn.Linear(4,8), 
            nn.LeakyReLU(),             
            nn.Linear(8, 1),
            nn.Sigmoid()
        )   
        
    def forward(self, imgOnly, factorsOnly, bothImg, bothFactor):

        if imgOnly.shape[0] > 0:
            
            b_size, n_nodule, c = imgOnly.size()
            centerFeat = imgOnly.view((b_size * n_nodule, c))
            out = self.dropout(centerFeat)
            out = nn.ReLU()(self.fc1(out))
            out = out.view(b_size, n_nodule, -1)

            A = self.attention(out)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=2)
            M = torch.bmm(A, out)
            M = M.squeeze(1)
            try:
                imgPred = self.classifier(M).squeeze(1)
            except:
                pdb.set_trace()
        
        if factorsOnly.shape[0] > 0:
            assert (factorsOnly[:, 1] == 1).all()
            clicPred = self.cliclayer(factorsOnly[:, 2:]).squeeze(1)
        else:
            clicPred = torch.zeros((0,1))
            
        if bothImg.shape[0] > 0:
            try:
                assert bothImg.shape[0] == bothFactor.shape[0]
                assert (bothFactor[:, 1] == 1).all()
                assert (bothFactor[:, 0] == 1).all()
            except:
                pdb.set_trace()
            b_size, n_nodule, c = bothImg.size()
            centerFeat = bothImg.view((b_size * n_nodule, c))
            out = self.dropout(centerFeat)
            out = nn.ReLU()(self.fc1(out))
            out = out.view(b_size, n_nodule, -1)

            A = self.attention(out)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=2)
            M = torch.bmm(A, out)
            
            M = M.squeeze(1)
            bothImgFeat = self.classifier(M)
            bothClicFeat = self.cliclayer(bothFactor[:, 2:])

            #bothPred = self.lastlayer(torch.cat((bothImgPred, bothClicPred), 1))
            #bothPred = self.lastlayer(torch.cat((bothImgPred, bothClicPred, bothFactor[:,-1].unsqueeze(1)), 1))
            bothPred = self.lastlayer(torch.cat((bothImgFeat, bothClicFeat, bothFactor[:,-2:]), 1))
            
            return imgPred, clicPred, bothImgFeat.squeeze(1), bothClicFeat.squeeze(1), bothPred.squeeze(1)
        else:
            return imgPred, clicPred, torch.zeros((0,1)), torch.zeros((0,1)), torch.zeros((0,1))

class MultipathModelBL(nn.Module):
    def __init__(self, n_factor, dim = 5):
        super(MultipathModelBL, self).__init__()
        self.n_class = 1
        
        self.fc1 = nn.Linear(128, 64)
        
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, dim),  # 64,
            nn.Sigmoid() #nn.LeakyReLU() #
        )
        self.imgonly_fc = nn.Sequential(
            nn.Linear(dim, 1), 
            nn.Sigmoid()
        )
        
        self.cliclayer = nn.Sequential(
            nn.Linear(n_factor, 64),  # 13, 10
            nn.Tanh(),       # default Tanh
            nn.Linear(64, dim),
            nn.Sigmoid(), # nn.LeakyReLU() #
        )
        
        self.cliconly_fc = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.lastlayer = nn.Sequential(
            nn.Linear(2 * dim + 2,8), 
            nn.LeakyReLU(),  # nn.ReLU() 0.919, nn.Tanh() 0.920, nn.PReLU 0.919, nn.ELU 0.918, nn.LeakyReLU 0.922
            nn.Linear(8, 1),
            nn.Sigmoid()
        )  
        
    def forward(self, imgOnly, factorsOnly, bothImg, bothFactor):

        if imgOnly.shape[0] > 0:
            
            b_size, n_nodule, c = imgOnly.size()
            
            centerFeat = imgOnly.view((b_size * n_nodule, c))
            out = self.dropout(centerFeat)
            out = nn.ReLU()(self.fc1(out))
            out = out.view(b_size, n_nodule, -1)

            A = self.attention(out)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=2)
            M = torch.bmm(A, out)
            M = M.squeeze(1)
            try:
                imgFeat = self.classifier(M)#.squeeze(1)
                imgPred = self.imgonly_fc(imgFeat).squeeze(1)
            except:
                pdb.set_trace()
        
        if factorsOnly.shape[0] > 0:
            assert (factorsOnly[:, 1] == 1).all()
            clicFeat = self.cliclayer(factorsOnly[:, 2:])#.squeeze(1)
            clicPred = self.cliconly_fc(clicFeat).squeeze(1)
        else:
            clicPred = torch.zeros((0,1))
            
        if bothImg.shape[0] > 0:
            try:
                assert bothImg.shape[0] == bothFactor.shape[0]
                assert (bothFactor[:, 1] == 1).all()
                assert (bothFactor[:, 0] == 1).all()
            except:
                pdb.set_trace()
                
            b_size, n_nodule, c = bothImg.size()
            centerFeat = bothImg.view((b_size * n_nodule, c))
            out = self.dropout(centerFeat)
            out = nn.ReLU()(self.fc1(out))
            out = out.view(b_size, n_nodule, -1)
            A = self.attention(out)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=2)
            M = torch.bmm(A, out)
            
            M = M.squeeze(1)
            bothImgFeat = self.classifier(M)
            bothClicFeat = self.cliclayer(bothFactor[:, 2:])
            bothImgPred = self.imgonly_fc(bothImgFeat).squeeze(1)
            bothClicPred = self.cliconly_fc(bothClicFeat).squeeze(1)
            #bothPred = self.lastlayer(torch.cat((bothImgPred, bothClicPred), 1))
            #bothPred = self.lastlayer(torch.cat((bothImgPred, bothClicPred, bothFactor[:,-1].unsqueeze(1)), 1))
            
            bothPred = self.lastlayer(torch.cat((bothImgFeat, bothClicFeat, bothFactor[:,-2:]), 1))
#             bothPred = self.lastlayer(torch.cat(((bothImgFeat + bothClicFeat) / 2, bothFactor[:,-2:]), 1))
            
            return imgPred, clicPred, bothImgPred, bothClicPred, bothPred.squeeze(1)
        else:
            return imgPred, clicPred, torch.zeros((0,1)), torch.zeros((0,1)), torch.zeros((0,1))
            

    



