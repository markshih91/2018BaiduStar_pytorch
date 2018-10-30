import os
import random
import json
import pickle
import cv2
import numpy as np
import utils

jsonFile1 = '../data/baidu_star_2018/annotation/annotation_train_stage1.json'
jsonFile2 = '../data/baidu_star_2018/annotation/annotation_train_stage2.json'

new_jsonFile1 = '../data/baidu_star_2018/annotation/new_annotation_train_stage1.json'
new_jsonFile2 = '../data/baidu_star_2018/annotation/new_annotation_train_stage2.json'

imagePath = '../data/baidu_star_2018/image/'
pklPath = '../data/baidu_star_2018/CSRNet_heatmap_pkl/'

trainAnnPath = '../data/baidu_star_2018/annotation/train/'

# testJsonFile = '../data/testSet/baidu_star_2018/annotation/annotation_test_stage1.json'
# testImagePath = '../data/testSet/baidu_star_2018/image/'

def save_heatmap_pkl(jsonFile, pklPath, imagePath, save_new_json=False, new_jsonFile=None):
    with open(jsonFile, 'r') as jFile:
        data = json.load(jFile)
        annotations = data['annotations']
        new_annotations = []
        for case in annotations:

            # if case['id'] in [370, 572, 1353, 2146, 3090, 3327, 3532]:
            #     continue

            nameEnd = case['name'].find('.')
            case['heatmap_name'] = case['name'].replace(case['name'][nameEnd:], '.pkl')

            if case['type'] == 'bbox':
                case['old_annotation'] = case['annotation']

                annotation = []

                for region in case['old_annotation']:
                    dic = {'x':(region['x'] + round(region['w'] / 2)), 'y':(region['y'] + round(region['h'] / 2))}
                    annotation.append(dic)

                case['annotation'] = annotation

            new_annotations.append(case)

            img = cv2.imread(imagePath + case['name'], cv2.IMREAD_GRAYSCALE)

            row = img.shape[0]
            col = img.shape[1]

            if row >= col and row > 1080:
                rate = 1080.0 / row
                points = [[int(ann['x'] * rate / 16), int(ann['y'] * rate / 16)] for ann in case['annotation']]
                heatmap = utils.gaussian_filter_density(
                    (int(round(row * rate / 16)), int(round(col * rate / 16))), points)
            elif col >= row and col > 1080:
                rate = 1080.0 / col
                points = [[int(ann['x'] * rate / 16), int(ann['y'] * rate / 16)] for ann in case['annotation']]
                heatmap = utils.gaussian_filter_density(
                    (int(round(row * rate / 16)), int(round(col * rate / 16))), points)
            else:
                points = [[int(ann['x'] / 16), int(ann['y'] / 16)] for ann in case['annotation']]
                heatmap = utils.gaussian_filter_density((int(round(row / 16)), int(round(col / 16))), points)

            # points = [[int(ann['x'] / 16), int(ann['y'] / 16)] for ann in case['annotation']]
            # heatmap = utils.gaussian_filter_density((int(round(row / 16)), int(round(col / 16))), points)

            with open(pklPath + case['heatmap_name'], 'wb') as pkl:
                pickle.dump(heatmap, pkl)

            print('The', case['id'], 'th pkl file saved:', case['heatmap_name'])

        data['annotations'] = new_annotations

        if save_new_json == True:
            with open(new_jsonFile, 'w') as new_jFile:
                json.dump(data, new_jFile)


def cut_ignore_region(jsonFile, imagePath):
    with open(jsonFile, 'r') as jFile:
        data = json.load(jFile)
        annotations = data['annotations']

        for case in annotations:
            if len(case['ignore_region']) == 0:
                continue

            img = cv2.imread(imagePath + case['name'], cv2.IMREAD_COLOR)
            for points in case['ignore_region']:
                ps = [[point['x'], point['y']] for point in points]
                ps = np.array(ps)
                cv2.fillConvexPoly(img, ps, color=(0, 0, 0))
            cv2.imwrite(imagePath + case['name'], img)
            print(imagePath + case['name'])




def show_json(jsonFile, isNew=False):
    with open(jsonFile, 'r') as jFile:
        data = json.load(jFile)
        annotations = data['annotations']

        for case in annotations:
            print('id:',case['id'])
            print('name:',case['name'])
            if isNew:
                print('heatmap_name:',case['heatmap_name'])
            print('num:',case['num'])
            print('type:',case['type'])
            print('annotation:',case['annotation'])
            if isNew and case['type'] == 'bbox':
                print('old_annotation:', case['old_annotation'])
            print('ignore_region:',case['ignore_region'])
            print('----------------------------------------------------------------\n')
        print('total count :', len(annotations))


def show_test_json(jsonFile):
    with open(jsonFile, 'r') as jFile:
        data = json.load(jFile)
        annotations = data['annotations']

        for case in annotations:
            print('id:',case['id'])
            print('name:',case['name'])
            print('ignore_region:',case['ignore_region'])
            print('----------------------------------------------------------------\n')
        print('total count :', len(annotations))


def json_partition(new_jsonFileList, trainAnnPath, valRate, testRate):
    with open(new_jsonFileList[0], 'r') as new_jFile:
        data = json.load(new_jFile)
        annotations = data['annotations']


    with open(new_jsonFileList[1], 'r') as new_jFile:
        data = json.load(new_jFile)
        annotations += data['annotations']

    total_len = len(annotations)
    val_len = int(total_len * valRate)
    test_len = int(total_len * testRate)

    np.random.shuffle(annotations)

    val = {'annotations': annotations[:val_len]}
    test = {'annotations': annotations[val_len:val_len + test_len]}
    train = {'annotations': annotations[val_len + test_len:]}


    with open(trainAnnPath + 'val-{0}.json'.format(val_len), 'w') as jFile:
        json.dump(val, jFile)
        print('json validation file saved!')

    with open(trainAnnPath + 'test-{0}.json'.format(test_len), 'w') as jFile:
        json.dump(test, jFile)
        print('json test file saved!')

    with open(trainAnnPath + 'train-{0}.json'.format(total_len - val_len - test_len), 'w') as jFile:
        json.dump(train, jFile)
        print('json train file saved!')

def prepare_ssd_data(jsonFile, imagePath, test_rate = 0.2):
    with open(jsonFile, 'r') as jFile:
        data = json.load(jFile)
        annotations = data['annotations']

        train_txt = []

        for case in annotations:
            # if case['id'] in [370, 572, 1353, 2146, 3090, 3327, 3532]:
            #     continue

            if case['type'] == 'dot':
                continue


            train_txt.append(case['name'] + ' bbox_anns/ann_' + case['name'][13:-4] + '.pkl')
            with open(imagePath + 'bbox_anns/ann_' + case['name'][13:-4] + '.pkl', 'wb') as pkl:
                pickle.dump(case['annotation'], pkl)

        random.shuffle(train_txt)
        trainval_len = int(len(train_txt) * (1-test_rate))
        trainval_txt = train_txt[:trainval_len]
        test_txt = train_txt[trainval_len:]

        with open(imagePath + "trainval.txt", "w") as f:
            for line in trainval_txt:
                f.write(line)
                f.write('\n')

        with open(imagePath + "test.txt", "w") as f:
            for line in test_txt:
                f.write(line)
                f.write('\n')

if __name__=="__main__":

    # # create and save heatmap
    if not os.path.exists(pklPath + 'stage1/train/'):
        os.makedirs(pklPath + 'stage1/train/')

    save_heatmap_pkl(jsonFile1, pklPath, imagePath, save_new_json=False, new_jsonFile=new_jsonFile1)
    # show_json(new_jsonFile1, isNew=True)

    if not os.path.exists(pklPath + 'stage2/train/'):
        os.makedirs(pklPath + 'stage2/train/')

    save_heatmap_pkl(jsonFile2, pklPath, imagePath, save_new_json=False, new_jsonFile=new_jsonFile2)
    # show_json(new_jsonFile2, isNew=True)
    #
    #
    # # set ignore region value 0
    # cut_ignore_region(jsonFile1, imagePath)
    # cut_ignore_region(jsonFile2, imagePath)



    # # partition:train,val and test
    # if not os.path.exists(trainAnnPath):
    #     os.makedirs(trainAnnPath)
    #
    # json_partition([new_jsonFile1, new_jsonFile2], trainAnnPath, 0.1, 0.1)



    # show_test_json(testJsonFile)
    # cut_ignore_region(testJsonFile, testImagePath)

    # 60093051cfa045bbc6696daa5721b4be
    # 2fef13d20bf95efe0e9e1b4b02b71048
    # 180ed1dd12d000eb04eb1df4fc4c5617
    #
    # prepare_ssd_data(jsonFile, imagePath, test_rate=0.15)