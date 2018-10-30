import os
import sys
import time
import utils
import numpy as np
import torch as t
from config import DefaultCofig as cfg
from models.CSRNet import CSRNet
from torch.utils import data
import data_processing.dataset as dataset

#prefix_name
prefix_name = '_' + time.strftime('%m%d_%H:%M:%S')

#log
file = open(cfg.log_path + cfg.model + prefix_name + '_train.log', 'w')
sys.stdout = file


# load net
net = CSRNet(load_weights=True)
net.load_state_dict(t.load(cfg.saved_model_path + 'CSRNet_0716_15:48:37.pkl'))
if cfg.use_gpu:
    net.cuda()


# data preparation
train_data = dataset.ImgPklData(cfg.imagePath, cfg.pklPath, cfg.json_train)
train_dataLoader = data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)

val_data = dataset.ImgPklData(cfg.imagePath, cfg.pklPath, cfg.json_val)
val_dataLoader = data.DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)

# mae mse
best_mae = sys.maxsize
best_mse = sys.maxsize

# set loss and optimizer
# criterion = t.nn.MSELoss()
# if cfg.use_gpu:
#     criterion.cuda()
# optimizer = t.optim.Adam(net.parameters(), lr=cfg.lr)
criterion = t.nn.MSELoss(size_average=False)
if cfg.use_gpu:
    criterion.cuda()
optimizer = t.optim.SGD(net.parameters(), cfg.lr, momentum=cfg.momentum, weight_decay=cfg.decay)


# train
print('Train on {0} samples, validate on {1} samples'.format(train_data.__len__(), val_data.__len__()))
for epoch in range(cfg.epochs):

    utils.adjust_learning_rate(optimizer, epoch)

    print('Epoch {0}/{1}\n'.format(epoch + 1, cfg.epochs))
    epoch_start = time.time()
    for i, (x, y) in enumerate(train_dataLoader):

        step_start = time.time()

        x = t.autograd.Variable(x)
        y = t.autograd.Variable(y)
        xs = x.size()
        ys = y.size()
        if cfg.use_gpu:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        y_ = net(x)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()

        cur = time.time()

        print('{0:10d}/{1} {2} - USED: {3} - MSEloss: {4:.4f} - num: {5} - pnum: {6}'
              .format((i + 1) * cfg.batch_size,
                      train_data.__len__(),
                      utils.progress_bar((i + 1) * cfg.batch_size / train_data.__len__()),
                      utils.eta_format(cur - epoch_start),
                      loss,
                      int(np.round(y.sum())),
                      round(int(y_.cpu().detach().numpy().sum()))))

        del x, y, y_


    net.eval()

    mae = 0.0
    mse = 0.0
    for i, (x, y) in enumerate(val_dataLoader):

        x = t.autograd.Variable(x, volatile=True)
        y = t.autograd.Variable(y, volatile=True)
        if cfg.use_gpu:
            x = x.cuda()
            y = y.cuda()
        y_ = net(x)
        realCount = y.cpu().numpy().sum()
        outCount = y_.cpu().detach().numpy().sum()
        realCount = int(np.round(realCount))
        outCount = int(np.round(outCount))
        mae += np.abs(realCount - outCount)
        mse += ((realCount - outCount) * (realCount - outCount))

    mae = mae / val_data.__len__()
    mse = np.sqrt(mse / val_data.__len__())

    print('MAE: {0} - MSE: {1}'.format(mae, mse))

    if mae < best_mae:
        best_mae = mae
        best_mse = mse
        saved_model_name = cfg.saved_model_path + cfg.model + prefix_name + '.pkl'
        t.save(net.state_dict(), saved_model_name)
        print('saved model: ', saved_model_name)

    saved_model_epoch_name = cfg.saved_model_path + cfg.model + prefix_name + '_epoch_{0}.pkl'.format(epoch + 1)
    t.save(net.state_dict(), saved_model_epoch_name)
    print('saved model: ', saved_model_epoch_name)

    pre_saved_model_epoch_name = cfg.saved_model_path + cfg.model + prefix_name + '_epoch_{0}.pkl'.format(epoch)
    if os.path.exists(pre_saved_model_epoch_name):
        os.remove(pre_saved_model_epoch_name)

    print()

    net.train()