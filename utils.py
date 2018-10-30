import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from config import DefaultCofig as cfg

def progress_bar(rate, len=30):
    '''

    :param rate: progress rate
    :param len: length of bar
    :return:[============>.................]
    '''
    cur_len = int(rate * len)
    if cur_len == 0:
        bar = '[..............................]'
    elif cur_len < len:
        bar = '[' + ('=' * cur_len) + '>' + ('.' * (len - cur_len)) + ']'
    else:
        bar = '[==============================]'
    return bar


def eta_format(eta):
    if eta > 3600:
        eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
    elif eta > 60:
        eta_format = '%d:%02d' % (eta // 60, eta % 60)
    else:
        eta_format = '%ds' % eta
    return eta_format


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    cfg.lr = cfg.original_lr

    for i in range(len(cfg.steps)):

        scale = cfg.scales[i] if i < len(cfg.scales) else 1

        if epoch >= cfg.steps[i]:
            cfg.lr = cfg.lr * scale
            if epoch == cfg.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr


def gaussian_filter_density(shape, points):

    density = np.zeros(shape, dtype=np.float32)

    if len(points) == 0:
        return density

    tree = sp.spatial.KDTree(points.copy(), leafsize=2048)
    distances, locations = tree.query(points, k=4)

    for i, (x, y) in enumerate(points):
        pt2d = np.zeros(shape, dtype=np.float32)

        if y == pt2d.shape[0]:
            y -= 1
        if x == pt2d.shape[1]:
            x -= 1

        pt2d[y, x] = 1.
        if len(points) > 3:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        elif len(points) > 2:
            sigma = (distances[i][1] + distances[i][2]) * 0.1
        elif len(points) > 1:
            sigma = (distances[i][1]) * 0.1
        else:
            sigma = np.average(np.array(shape)) / 2. / 2.
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density