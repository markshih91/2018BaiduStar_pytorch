import sys
import numpy as np
import torch as t
from config import DefaultCofig as cfg
from models.CSRNet import CSRNet
from torch.utils import data
import data_processing.dataset as dataset

#prefix_name
prefix_name = '_' + cfg.test_time_str

# log
file = open(cfg.log_path + cfg.model + prefix_name + '_test.log', 'w')
sys.stdout = file

# load net
net = CSRNet()

net.load_state_dict(t.load(cfg.saved_model_path + cfg.model + prefix_name + '.pkl'))
if cfg.use_gpu:
    net.cuda()

test_data = dataset.ImgPklData(cfg.imagePath, cfg.pklPath, cfg.json_test)
test_dataLoader = data.DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

net.eval()

mae = 0.0
mse = 0.0
err_rate = 0.0
for i, (x, y) in enumerate(test_dataLoader):

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
    err_rate += np.abs(realCount - outCount)/realCount
    print('MAE: {0} - MSE: {1} - err_rate: {2}'.format(mae, mse, err_rate))

mae = mae/test_data.__len__()
mse = np.sqrt(mse/test_data.__len__())
err_rate /= test_data.__len__()

print('Final MAE: {0} - MSE: {1} - err_rate: {2:.4f}'.format(mae, mse, err_rate))