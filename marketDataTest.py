# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import math
from tqdm import tqdm
## clip model
from transformers import CLIPProcessor, CLIPModel,CLIPVisionModel
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)
from transformers import AutoImageProcessor, Swinv2Model


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--usePretrain', default=False, type=bool)
parser.add_argument('--test_dir',default='./rawdata/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='clip-vit-base-patch16-finetuned', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--useSwin', default=False, type=bool, help='batchsize')
parser.add_argument('--linear_num', default=768, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
opt = parser.parse_args()


######################################################################
# Load model
#---------------------------
def load_network():
    save_path = os.path.join('./model',opt.name)   
    if opt.usePretrain:
            save_path = opt.name
    if not opt.useSwin:
        model = CLIPModel.from_pretrained(save_path)
        processor = CLIPProcessor.from_pretrained(save_path)
        vision_model = CLIPVisionModel.from_pretrained(save_path)
    else:
        if opt.usePretrain: 
            processor = AutoImageProcessor.from_pretrained(save_path)
            vision_model = Swinv2Model.from_pretrained(save_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            processor =  AutoImageProcessor.from_pretrained(save_path)
            processor_ = VisionTextDualEncoderProcessor(processor, tokenizer)
            Model = VisionTextDualEncoderModel.from_pretrained(save_path)
            vision_model = Model.vision_model
        
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
        vision_model = nn.DataParallel(vision_model)
    vision_model.to('cuda')

    return vision_model, processor


def extract_feature(model, processor,dataloaders):
    #features = torch.FloatTensor()
    # count = 0
    pbar = tqdm()
    for iter, data in enumerate(dataloaders):
        img, label = data
        n = len(img)
        pbar.update(n)
        
        inputs = processor(images=img, return_tensors="pt").to('cuda')
        outputs = model(**inputs)
        feat = outputs.pooler_output
        if iter == 0:
            features = feat.data.cpu()
        else:
            features = torch.cat((features,feat.data.cpu()), 0)
    pbar.close()
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


# 自定义集合函数
def my_collate_fn(batch):
    # batch 包含多个 tuple，每个 tuple 是 (image, label)
    images, labels = zip(*batch)
    # 现在 images 是一个包含多个 PIL 图像的元组
    # labels 是一个包含多个标签的元组
    return list(images), labels

data_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        #transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
])

######################################################################

    
def main():
 
    print('-------test-----------')

    name = opt.name
    data_dir = opt.test_dir

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x)) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                 shuffle=False, num_workers=16, collate_fn=my_collate_fn) for x in ['gallery','query']}


    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)


    model, processor = load_network()

    #print(model)
    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model, processor,dataloaders['gallery'])
        query_feature = extract_feature(model, processor,dataloaders['query'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    name.split('/')[-1]
    fileName = name.split('/')[-1]+'_pytorch_result.mat'
    scipy.io.savemat(os.path.join('market_result/', fileName), result)

    print(opt.name)
    
    result = './model/%s/result.txt'%opt.name
    #os.system('python evaluate_gpu.py | tee -a %s'%result)

    
if __name__ == "__main__":
    main()