import json
import pathlib

import numpy as np
import timm
import torch
from data.test_dataset import TestDataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from tqdm.contrib import tenumerate

topk = 5
path_folder = '/home/li/main/dataset/wcp/annotation_json/evaluation.json'
eval_open = open(path_folder, 'r')
# 開いたファイルをJSONとして読み込む
evaluation = json.load(eval_open)

path_folder = "/home/li/main/dataset/shopping_mall/annotation" 

def my_model():
    model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
    model.head = nn.Linear(768, 1024, bias=True)

    return model

def test_dataloader(path):
    test_set = TestDataset(db_root=path)

    test_loader = DataLoader(test_set, batch_size=1, num_workers=16, shuffle=False)
    
    return test_loader

# def Accuracy(query, ans_img):
#     tolerance = "1m"
#     if len(ans_img) == 0:
#         return False
#     else:
#         # check = False
#         # for img in ans_img:
#         #     if img in evaluation[query][tolerance]:
#         #         check = True
#         #         break
#         return ans_img[0] in evaluation[query][tolerance]

def Accuracy(query, ans_img):
    if len(ans_img) == 0:
        return False
    else:
        path = "{0}/{1}.txt".format(path_folder, query)
        with open(path) as f:
            correct_list = [s.strip() for s in f.readlines()]
        # print(f'ans:{correct_list}')
        
        check = False
        for img in ans_img:
            if img in correct_list:
                check = True
                break
        
        return check

device = torch.device(type="cuda", index=0)

# query_path = '/home/li/main/dataset/wcp/query'
# reference_path = '/home/li/main/dataset/wcp/reference'
query_path = '/home/li/main/dataset/shopping_mall/query_cdm_resize_rename'
reference_path = '/home/li/main/dataset/shopping_mall/reference_resize_rename'

img_query = sorted(pathlib.Path(query_path).glob('*.jpg'))
img_reference = sorted(pathlib.Path(reference_path).glob('*.jpg'))

n_query = len(img_query)
n_reference = len(img_reference)

query_pred = np.zeros((n_query, 1024))
reference_pred = np.zeros((n_reference, 1024))

data_query = test_dataloader(query_path)
data_reference = test_dataloader(reference_path)

model = my_model().to(device=device)

with torch.no_grad():
    for i, data in tenumerate(data_query):
        data = data.to(device=device)
        pred = model(data).to('cpu').detach().numpy().copy().squeeze()
        
        query_pred[i, ...] = pred

    for i, data in tenumerate(data_reference):
        data = data.to(device=device)
        pred = model(data).to('cpu').detach().numpy().copy().squeeze()
        
        reference_pred[i, ...] = pred


n_query = 80
top1_acc = np.zeros(n_query)
for i in tqdm(range(n_query)):
    sim = np.dot(query_pred[i], reference_pred.T) / (np.linalg.norm(query_pred[i]) * np.linalg.norm(reference_pred,axis=1))
    idxs = np.argsort(sim)[::-1]
    
    top5_img = [img_reference[idx].stem[:4] for idx in idxs][:5]
    
    top1_acc[i] = Accuracy(img_query[i].stem[:4], top5_img)
    
    # print(f'{img_query[i].stem[:4]}: {top5_img} | {top1_acc[i]}')

print(np.mean(top1_acc))