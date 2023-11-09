import torch
import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import time
import timm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import snapshot_storing_n as ss



classes = {0:'ignored regions',1:'pedestrian',2:'people',3:'bicycle',4:'car',5:'van',6:'truck',7:'tricycle',8:'awning-tricycle',9:'bus',10:'motor',11:'others'}


class FasterRCNNModel(nn.Module):
    def __init__(self, model_name, pretrained = True, num_classes = len(classes)):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # Faster R-CNN은 객체 감지를 수행하므로 num_classes는 객체 클래스의 수여야 합니다.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        loss_dict = self.model(images, targets)
        return loss_dict



def extraction(annotations_dir, annotations, images_dir):
    data=[]
    for annotation in annotations:
        with open(os.path.join(annotations_dir ,annotation), 'r') as file:
            lines = file.readlines()
            file_name = annotation.rsplit('.txt',1)[0]
            for line in lines:
                boxes = line.strip().split(',')
                
                data.append({
                    'image_id' : file_name,
                    'image_path': os.path.join(images_dir, file_name + '.jpg'),
                    'x_min' : int(boxes[0]),
                    'y_max' : int(boxes[1]),
                    'x_max' : int(boxes[0])+int(boxes[2]),
                    'y_min' : int(boxes[1])+int(boxes[3]),
                    'score' : int(boxes[4]),
                    'class_label' : int(boxes[5]),
                    'truncation' : int(boxes[6]),
                    'occlusion' : int(boxes[7])
                    })

    return data

def unique(data_list):
    unique_data_dict = {}
    unique_data_list = []

    for record in data_list:
        image_id = record['image_id']
        if image_id not in unique_data_dict:
            unique_data_dict[image_id] = record
            unique_data_list.append(record)

    return unique_data_list

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_list, transform=None):

        self.unique_list = unique(data_list)
        self.data_list = data_list
        self.transform = transform
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_list = self.data_list
        sample = self.unique_list[idx]
        image_id = sample['image_id']
        
        image_path = sample['image_path']
        
        image = cv2.imread(image_path)


        matching_samples = [s for s in data_list if s['image_id'] == image_id]

        boxes = []  # List to store bounding boxes
        labels = []  # List to store class labels

        for match in matching_samples:
            x_min = match['x_min']
            y_min = match['y_min']
            x_max = match['x_max']
            y_max = match['y_max']
            class_label = match['class_label']

            boxes.append([x_min, y_max, x_max, y_min])
            labels.append(class_label)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),  # Convert boxes to a tensor
            'labels': torch.tensor(labels, dtype=torch.int64)  # Convert labels to a tensor
        }

        if self.transform:
            sample = {
                    'image' : image.astype(np.float32),
                    'bboxes' : target['boxes'],
                    'labels' : target['labels']
                    }
            
            sample = self.transform(**sample)
            image = sample['image']
            image = image.expand(3, -1, -1)

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)

        return image, target

def get_transform_train():
    return A.Compose([
#        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0)
    ], bbox_params={'format':'pascal_voc', 'label_fields': ['labels']})

def get_transform_valid():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))

script_start = time.time()
print(os.getpid())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


train_annotations_dir = "./data/DRONE/task1/VisDrone2019-DET-train/annotations/"

train_annotations = os.listdir(train_annotations_dir)

train_images_dir = "./data/DRONE/task1/VisDrone2019-DET-train/images/"


valid_annotations_dir = "./data/DRONE/task1/VisDrone2019-DET-val/annotations/"

valid_annotations = os.listdir(valid_annotations_dir)

valid_images_dir = "./data/DRONE/task1/VisDrone2019-DET-val/images/"


epochs = 5
BATCH_SIZE = 1
OUTPUT_PATH = './fio_test/drone_task1_checkpoint/'

#trainset
train_list = extraction(train_annotations_dir ,train_annotations ,train_images_dir)

train_dataset = ObjectDetectionDataset(train_list, transform=get_transform_train())

indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

#vaildset
valid_list = extraction(valid_annotations_dir ,valid_annotations ,valid_images_dir)

valid_dataset = ObjectDetectionDataset(valid_list, transform=get_transform_valid())

indices = torch.randperm(len(valid_dataset)).tolist()

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
    )


###############################################################3
#images, targets= next(iter(train_data_loader))
##print(targets)
##assert False
#
#images = list(image.to(device) for image in images)
#targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
##print(images)
#plt.figure(figsize=(20,20))
#for i, (image, target) in enumerate(zip(images, targets)):
#    plt.subplot(2,2, i+1)
#    boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#    sample = images[i].permute(1,2,0).cpu().numpy()
#    names = targets[i]['labels'].cpu().numpy().astype(np.int64)
##    print(i)
#
#    output_path = f"output_image_{i}.jpg"
#
#    #원본 이미지
#    #cv2.imwrite(output_path, sample)
#
##    print(sample)
#
#    names = targets[i]['labels'].cpu().numpy().astype(np.int64)
#    for i,box in enumerate(boxes):
#        cv2.rectangle(sample,
#                      (box[0], box[1]),
#                      (box[2],box[3]),
#                      (0, 0, 220), 2)
#        cv2.putText(sample, classes[names[i]], (box[0],box[1]+15),cv2.FONT_HERSHEY_COMPLEX ,0.5,(0,220,0),1,cv2.LINE_AA)
#
#    plt.axis('off')
#    output_path = f"output_image_{i}.jpg"
#
#    #mapping 이미지
#    cv2.imwrite(output_path, sample)
#
#####################################################################################
#
#
#assert False

num_classes = len(classes)
#model_resNet50.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model_resNet50 = FasterRCNNModel(num_classes, pretrained=True)
model_resNet50 = model_resNet50.to(device)

#criterion = nn.CrossEntropyLoss().to(device)
#regression_criterion = nn.SmoothL1Loss().to(device)

optimizer = torch.optim.Adam(model_resNet50.parameters(),lr = 1e-4)


checkpoint_list = []
epoch_list = []


start_model_time = time.time()

cost = 0
print("training start" + str(start_model_time - script_start ))
#manager = Manager()
#activate_snapshot = manager.Value('i', 0)
images, targets= next(iter(train_data_loader))
#print(images[0].size(0))
#assert False
def main():
    for epoch in range(epochs):
        avg_loss = 0
        start_epoch_time = time.time()
        for idx, (images, targets) in enumerate(train_data_loader):
            if  idx == 0:
                images = list(image.to(device) for image in images)

                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
              
                loss_dict = model_resNet50(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                               
              
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
                avg_loss += loss / len(train_data_loader)
                model_state = model_resNet50.state_dict()
                op_state = optimizer.state_dict()

                start_checkpoint_time = time.time()
                save = ss.async_checkpointing(model_state, op_state, epoch,OUTPUT_PATH)


                
                save.snapshot_storing(model_state, op_state, epoch,OUTPUT_PATH)


#                time.sleep(10)

                end_checkpoint_time = time.time()
                print("check"+str(end_checkpoint_time - script_start))
                checkpoint_delta = np.round(end_checkpoint_time - start_checkpoint_time, 5)
                checkpoint_list.append(checkpoint_delta)
            else:
                images = list(image.to(device) for image in images)

                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
              
                loss_dict = model_resNet50(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                               
              
                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
                avg_loss += loss / len(train_data_loader)

#
#    print('[Epoch: {:>4}) cost = {:>.9}'.format(epoch +1, avg_cost))
#    end_epoch_time = time.time()
#    print("epoch"+str(end_epoch_time - script_start))
#    epoch_delta = np.round(end_epoch_time - start_epoch_time, 5)
#    epoch_list.append(epoch_delta)
#
#
if __name__ == '__main__':
    main()
#    end_model_time = time.time()
#    model_delta = np.round(end_model_time - start_model_time, 5)
#    print(script_start - start_model_time)

def evaluate_model(model_resNet50, vaild_data_loader):
    model_resNet50.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(model_resNet50.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model_resNet50(images)

            # Store the predicted and true bounding boxes
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    return all_predictions, all_targets


valid_predictions, valid_targets = evaluate_model(model_resNet50, valid_data_loader)

#    print(model_delta)
#    print(epoch_list)
#    print(checkpoint_list)










