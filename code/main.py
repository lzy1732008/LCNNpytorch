#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import lightweight_convolution
import train
import mydatasets
import numpy as np
import model
# import tensorflow.contrib.keras as kr




parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-seq_length_1',type=int, default=30,help='initial first input length')
parser.add_argument('-seq_length_2',type=int, default=50,help='initial second input length')
parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=200, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=10,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=10, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=10, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot-lightcnn', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-con1-layers',type=int, default=5, help = 'number of convolution layers of the first input ')
parser.add_argument('-con2-layers',type=int, default=5, help = 'number of convolution layers of the first input ')
parser.add_argument('-dropout', type=float, default=0.8, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=256, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
parser.add_argument('--weight-softmax', default=True, type=bool)
parser.add_argument('--weight-dropout', type=float, default= 0.5,
                    help='dropout probability for conv weights')
parser.add_argument('-output-dim',type=int, default=128,help='the output dimension of fc1')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


def data_convert(vectors):
    ssls = list(filter(lambda x:x.strip() != '', vectors))
    n_vector = [list(map(float, list(filter(lambda x: x.strip() != '', ss.split('//'))))) for ss in ssls]
    return n_vector

def padding(sequence,length):
    if len(sequence) < length:
        sequence += list(np.zeros(shape=[length - len(sequence), 128]))
    else:
        sequence = sequence[:length]
    return sequence

def data_load(data_f,args):
    input_x1, input_x2,  input_y = [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        print('index:', i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 5:
            continue
        ssls = array[1].split(' ')
        ftzwls = array[2].split(' ')
        label = int(array[3].strip())
        ssls = padding(data_convert(ssls),args.seq_length_1)
        ftzwls = padding(data_convert(ftzwls),args.seq_length_2)
        input_x1.append(ssls)
        input_x2.append(ftzwls)
        input_y.append(label)
        # if label == 0:
        #     input_y.append([1, 0])
        # else:
        #     input_y.append([0, 1])

    return np.array(input_x1), np.array(input_x2),  np.array(input_y)


# load data
print("\nLoading data...")
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# print('train_iter',train_iter)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)
parent_dir = '/Users/wenny/PycharmProjects/wsfx_ks/wsfx2/source/set_4'
train_f = open(os.path.join(parent_dir,'train.txt'),'r',encoding='utf-8')
test_f = open(os.path.join(parent_dir,'test.txt'),'r',encoding='utf-8')
val_f = open(os.path.join(parent_dir,'val.txt'),'r',encoding='utf-8')
#
train_data = data_load(train_f,args)
train_f.close()

dev_data = data_load(val_f,args)
val_f.close()


# test_data = data_load(test_f,args)
# test_f.close()




# update args and print

args.embed_num = 128
args.class_num = 2
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')][0]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
input_size = 128
cnn = lightweight_convolution.CNN(args, args.embed_num, kernel_size=args.kernel_sizes,
                                                   padding_l=args.kernel_sizes-1, num_heads=4,
                 weight_dropout=args.weight_dropout, weight_softmax=args.weight_softmax, bias=False)

# cnn = model.CNN_Text(args)



# params = list(cnn.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))



if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()



if args.test:
    try:
        train.eval(test_data, cnn, args, flag=1)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_data, dev_data, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

