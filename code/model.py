import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 128
        Co = args.kernel_num
        Ks = args.kernel_sizes


        # self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]

        # self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.conv1_1 = nn.Conv1d(Ci, Co, Ks)
        self.conv1_2 = nn.Conv1d(Ci, Co, Ks)

        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)


        self.fc1 = nn.Linear(2 * args.kernel_num, args.output_dim)
        self.fc2 = nn.Linear(args.output_dim, args.class_num)


    def forward(self, x1, x2):
        
        # if self.args.static:
        #     x1 = Variable(x1).double()
        #     x2 = Variable(x2).double()
        x1 = Variable(x1).float()
        x2 = Variable(x2).float()
        # x = x.unsqueeze(1)  # (N, Ci, W, D)
        x1 = x1.permute(0,2,1)
        x1 = self.conv1_1(x1)
        x1 = F.relu(x1)  #[B,Ks,W]
        x1 = F.max_pool1d(x1,x1.shape[2])
        x1 = x1.view(-1,x1.shape[1])
        x1 = self.dropout1(x1)  # (N, len(Ks)*Co)


        x2 = x2.permute(0,2,1)
        x2 = F.relu(self.conv1_1(x2))  #[B,Ks,W]
        x2 = F.max_pool1d(x2,x2.shape[2])
        x2 = x2.view(-1,x2.shape[1])
        x2 = self.dropout2(x2)  # (N, len(Ks)*Co)


        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        #
        x = torch.cat((x1,x2), 1)
        x = self.fc1(x)  # (N, C)
        x = self.dropout3(x)
        logit = self.fc2(x)
        return logit
