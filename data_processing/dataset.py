import json
import pickle
import torch as t
from torch.utils import data
from PIL import Image
from torchvision import transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImgPklData(data.Dataset):

    def __init__(self, imgPath, pklPath, jsonFile):
        self.transform = transform
        with open(jsonFile, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            self.imgs = [imgPath + ann['name'] for ann in annotations]
            self.heatmaps = [pklPath + ann['heatmap_name'] for ann in annotations]

    def __getitem__(self, index):

        img = Image.open(self.imgs[index])
        # img = img.convert('L')
        img = img.convert('RGB')

        with open(self.heatmaps[index], 'rb') as pkl:
            heatmap = pickle.load(pkl)
            heatmap = heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])

        img = img.resize((heatmap.shape[2] * 8, heatmap.shape[1] * 8), Image.ANTIALIAS)
        img = self.transform(img)

        return t.Tensor(img), t.Tensor(heatmap)

    def __len__(self):
        return len(self.imgs)


class ImgData(data.Dataset):

    def __init__(self, imgPath, jsonFile):
        self.transform = transform
        with open(jsonFile, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            self.id_imgs = [[ann['id'], imgPath + ann['name']] for ann in annotations]

    def __getitem__(self, index):

        id_imgs = self.id_imgs[index]
        img = Image.open(id_imgs[1])
        img = img.convert('RGB')

        x = img.size[0]
        y = img.size[1]

        if x >= y and x > 1120:
            img = img.resize((1120, int(round(1120.0 * y / x))), Image.ANTIALIAS)
        elif y > x and y > 1120:
            img = img.resize((int(round(1120.0 * x / y)), 1120), Image.ANTIALIAS)


        img = self.transform(img)

        return id_imgs[0], t.Tensor(img)

    def __len__(self):
        return len(self.id_imgs)



if __name__=="__main__":

    dataSet = ImgPklData('../data/baidu_star_2018/image/', '../data/baidu_star_2018/heatmap_pkl/',
                         '../data/baidu_star_2018/annotation/stage1/train.json')
    loader = data.DataLoader(dataSet, batch_size=1)

    for i, (img, heatmap) in enumerate(loader):
        print(i)
        print(img)
        print(heatmap)