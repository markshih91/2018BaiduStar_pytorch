import sys
import numpy as np
import torch as t
from config import DefaultCofig as cfg
from models.ECSRNet import ECSRNet
from torch.utils import data
import data_processing.dataset as dataset

#prefix_name
prefix_name = '_' + cfg.test_time_str

# log
# file = open(cfg.log_path + cfg.model + prefix_name + '_result.log', 'w')
# sys.stdout = file

# load net
net = ECSRNet(load_weights=True)
net.load_state_dict(t.load(cfg.saved_model_path + cfg.model + prefix_name + '.pkl',
                           map_location=lambda storage, loc: storage))
if cfg.use_gpu:
    net.cuda()

pre_data = dataset.ImgData(cfg.predictImgPath, cfg.predictJson)
pre_dataLoader = data.DataLoader(pre_data, batch_size=1, shuffle=False)

net.eval()

for i, (id, img) in enumerate(pre_dataLoader):

    x = t.autograd.Variable(img, volatile=True)
    if cfg.use_gpu:
        x = x.cuda()
    y_ = net(x)
    outCount = y_.cpu().detach().numpy().sum()
    outCount = int(np.round(outCount))
    print('{0},{1}'.format(id.numpy()[0], outCount))
    # file.flush()