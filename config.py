
class DefaultCofig(object):

    # log info
    log_path = './logs/'

    # model info
    model = 'CSRNet'
    batch_size = 1
    use_gpu = True
    epochs = 800
    lr = 1e-6
    momentum = 0.9
    decay = 1e-4
    original_lr = 1e-6
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]
    workers = 4

    # path info
    json_train = './data/baidu_star_2018/annotation/train/train-5183.json'
    json_test = './data/baidu_star_2018/annotation/train/test-647.json'
    json_val = './data/baidu_star_2018/annotation/train/val-647.json'
    imagePath = './data/baidu_star_2018/image/'
    pklPath = './data/baidu_star_2018/CSRNet_heatmap_pkl/'
    saved_model_path = './checkpoints/'

    predictJson = './data/testSet/baidu_star_2018/annotation/annotation_test_stage1.json'
    predictImgPath = './data/testSet/baidu_star_2018/image/'

    #test info
    test_time_str = '0729_23:36:05'
