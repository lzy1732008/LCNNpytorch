import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

def batch_iter(x1, x2,  y, batch_size=128):
    """生成批次数据"""
    print('x1.shape:',np.array(x1).shape)
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    indices = np.random.permutation(np.arange(data_len)) #洗牌
    x1_shuffle = x1[indices]
    x2_shuffle = x2[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id],x2_shuffle[start_id:end_id],  y_shuffle[start_id:end_id]

def batch_iter_test(x1, x2,  y, batch_size=128):
    """生成批次数据"""
    print('x1.shape:',np.array(x1).shape)
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    x1_shuffle = x1
    x2_shuffle = x2
    y_shuffle = y

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id],x2_shuffle[start_id:end_id],  y_shuffle[start_id:end_id]

def train(train_data, dev_data, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        batch_train = batch_iter(train_data[0],train_data[1],train_data[2])
        flag = False
        for x1_batch, x2_batch, target in batch_train:
            optimizer.zero_grad()
            #padding

            x1_batch = torch.from_numpy(x1_batch)
            x2_batch = torch.from_numpy(x2_batch)
            target = torch.from_numpy(target)
            logit = model(x1_batch,x2_batch)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/len(x1_batch)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data,
                                                                             accuracy,
                                                                             corrects,
                                                                             len(x1_batch)))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_data, model, args, 0)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                        print('*')
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                        flag = True
                        break
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

        if flag == True:break


def eval(data, model, args,flag):
    model.eval()
    # model.load_state_dict(torch.load('/Users/wenny/nju/code/LightCNN/snapshot-cnn/2019-04-09_15-29-47/best_steps_1030.pt'))

    corrects, avg_loss = 0, 0
    if flag == 0:
        batch_eval = batch_iter(data[0], data[1], data[2])
    else:
        batch_eval = batch_iter_test(data[0], data[1], data[2])
    predict_pos = 0
    predict_true = 0
    allTrue = 0
    for x1_eval, x2_eval, target in batch_eval:
        x1_eval, x2_eval,target = torch.from_numpy(x1_eval), torch.from_numpy(x2_eval), torch.from_numpy(target)
        logit = model(x1_eval,x2_eval)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

        target = target.numpy()
        logit = torch.max(logit, 1)[1]
        logit = logit.numpy()

        #计算为1且预测正确的
        for i,j in zip(logit,target):
            if i == 1:
                predict_pos += 1
                if j == 1:
                    predict_true += 1
            if j == 1:
                allTrue += 1

    size = len(data[0])
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    #计算F1
    precision = predict_true / (predict_pos + 1)
    recall = predict_true / allTrue
    F1 = 2 * precision * recall / (precision + recall)
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) F1:{:.4f}% \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size,
                                                                       F1))
    return accuracy

def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
