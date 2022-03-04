#!/usr/bin/env python
# coding: utf-8

# # Experiment A1: parse JSF
# Lets try to read one of the JSF files. Interpret them using this spec:
#   https://www.edgetech.com/wp-content/uploads/2019/07/0023492_Rev_E.pdf

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys
from urllib.parse import unquote
import copy

#%pip install opencv-python

import cv2

from tqdm import tqdm

from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import torchvision
import albumentations as A
import random
import math
import time

from torch import FloatTensor, LongTensor, nn, optim
from torchvision import datasets, models
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from engine import evaluate

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = "cpu"

def plot_loss(train_loss,val_loss):
    plt.clf()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.savefig('LossPlot.png', bbox_inches='tight')

def obj_collate_fn(batch):
    return tuple(zip(*batch))

def visualize_bbox(image,bbox_list,save=False):
    pic = image
    for bbox in bbox_list:
        pic = cv2.rectangle(pic, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
    if save: cv2.imwrite("bbox.png",pic)
    cv2.imshow("bboxes visualized", pic)
    cv2.waitKey(0) # this freezes and crashes for some reason

class SonarDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms):
        self.root = root
        self.transforms = transforms

        # Vott json vott format export is needed: vott-json-export folder with all the png grayscale images and one json file
        # that has the information of all the images included and the ones that have bounding boxes

        self.imgs = [s for s in os.listdir(root) if s.endswith('.png')]

        #self.imgs = self.imgs[0:int(len(self.imgs)*0.1)] # MAKE IT FAST FOR DEBUGGING TODO remove this urgently lol

        csv_boxes = pd.read_csv(csv_file)

        self.label2id = {
            "bike": 1,
            "debris": 2,
            "confirmed_body": 3,
            "anomaly": 4
        }

        img_name_to_box = {}
        for index, asset in csv_boxes.iterrows():
            img_filename = asset["image"]
            if img_filename not in self.imgs: continue  # if the image isnt in this folder but in the json skip it
            # this is so i can create a test train split

            if img_filename not in img_name_to_box:
                img_name_to_box[img_filename] = []
            img_name_to_box[img_filename].append(
                [asset["xmin"], asset["ymin"], asset["xmax"], asset["ymax"], self.label2id[asset["label"]]])

        self.img2boxes = img_name_to_box

    def get_image(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        image = cv2.imread(img_path)
        image = image.astype(np.uint8)
        return image

    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.root, self.imgs[idx])
        image = cv2.imread(img_path)
        image = image.astype(np.uint8)
        # pillow_image = Image.open(img_path)
        # image = np.array(pillow_image)

        boxes = []
        if self.imgs[idx] in self.img2boxes:
            boxes = self.img2boxes[self.imgs[idx]]

        num_objs = len(boxes)

        # convert everything into a torch.Tensor
        bboxes2 = []
        labels = []
        for box_label in boxes:
            bbox = box_label[:-1]
            label = box_label[-1]
            bboxes2.append(bbox)
            labels.append(label)

        boxes_tensor = torch.as_tensor(bboxes2, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = image_id

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=bboxes2, class_labels=labels)
            img = transformed['image']
            target["boxes"] = []
            if len(transformed['bboxes']) > 0:
                for bbox in transformed['bboxes']:
                    target["boxes"].append(torch.tensor(bbox))
                target["boxes"] = torch.stack(target["boxes"])
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

            target["labels"] = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)

        target["boxes"].to(device)
        target["labels"].to(device)
        target["image_id"].to(device)

        # The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image,
        # and should be in 0-1 range. Different images can have different sizes.

        # for rcnn : During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:
        # boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

        # labels (Int64Tensor[N]): the class label for each ground-truth box
        # get bounding box coordinates for each mask

        img = np.transpose(img, [2, 0, 1])
        img = torch.from_numpy(img)
        img.to(device)
        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":

    def get_class_stats(vott_csv):
        vott_csv = pd.read_csv(vott_csv)
        coverage_dict = dict()
        for index,something in vott_csv.iterrows():
            real_name = unquote(something["image"])
            part = real_name.split("_data_likely_containing_targets_")
            part = part[0]
            if "_data_likely_containing_targets_" not in real_name:
                part = real_name.split("_training_image_")
                part = part[0]
                part = part.split("_")[:-1]
                part = "_".join(part)
            if part not in coverage_dict:
                coverage_dict[part] = {
                    "bike" : 0,
                    "debris" : 0,
                    "confirmed_body": 0,
                    "anomaly": 0
                }
            coverage_dict[part][something["label"]] += 1
        return coverage_dict

    prefix = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/conversions/vott_to_vgg_proj/"
    target = "empty_vgg_json.json"
    source = "source_vott_csv.csv"

    target_folder = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/input"

    csv_proj_file = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/06_02_2022_BIG-export.csv"


    #vott_csv_to_vgg_proj_json(prefix+source,prefix+target)
    #mirror_starboard_images_into_port(target_folder)
    info_dict = get_class_stats(csv_proj_file)
    #pprint.pprint(info_dict)


    # In[21]:


    # go over all the confirmed bodies
    vott_csv = pd.read_csv(csv_proj_file)
    prefix = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/"

    you_want_to_do_this = False
    if you_want_to_do_this:
        for index,something in vott_csv.iterrows():
            real_name = something["image"]#unquote(something["image"])
            if not something["label"] == "confirmed_body": continue
            path = prefix+real_name
            im = Image.open(path)
            # This method will show image in any image viewer
            print(unquote(something["image"]))
            print("{},{}".format(something["xmin"],something["ymin"]))
            im.show()
            #bla_bla = input("Press enter to continue")


    # In[22]:



    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100
    c_bikes = 0
    c_debris = 0
    c_cnf_body = 0
    c_anomaly = 0

    train_set = [
        '03-01-2020 Hoogezand, winschoterdiep',
        '09-06-2020 Velden',
        '16-08-20 Reuver',
        '22-03-2020 weert_2',
        '22-03-2020 weert',
        '20-03-2020 weert_2',
        '20-03-2020 weert',
        '24-05-2020 Honselaarsplas',
        '29-06-2020 winterswijk',
        'maarseveen',
        '09-07-2020 Giesbeek',
        '30-06-2020 Giesbeek'
    ]
    test_set = [
        '06-12-2020 Burdaard',
        '09-11-2020 wemeldinge_2',
        '09-11-2020 wemeldinge',
        'brummen',
        'Coverage_12-11-2020 Brummen'
    ]
    dev_set = [
        '18-09-2020 stavoren',
        '27-03-2020 sluis Belfeld',
        '20-06-2020 Lemmer',
        '28-07-2020 Maastricht',
        '14-03-2020 breukelen'
    ]

    labels = ["train","test","dev"]
    bikes = [0,0,0]
    debris = [0,0,0]
    confirmed_body = [0,0,0]
    anomaly = [0,0,0]

    for key, value in info_dict.items():
        sample_index = 0 # training set
        if key in test_set:
            sample_index = 1
        if key in dev_set:
            sample_index = 2

        bikes[sample_index] += value["bike"]
        debris[sample_index] += value["debris"]
        confirmed_body[sample_index] += value["confirmed_body"]
        anomaly[sample_index] += value["anomaly"]


        c_bikes += value["bike"]
        c_debris += value["debris"]
        c_cnf_body += value["confirmed_body"]
        c_anomaly += value["anomaly"]


    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, bikes, width, label='Bikes')
    rects2 = ax.bar(x + width/2, debris, width, label='Debris')
    rects3 = ax.bar(x - width/4, confirmed_body, width, label='Confirmed bodies')
    rects4 = ax.bar(x + width/4, anomaly, width, label='Anomalies')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of created annotations for a class')
    ax.set_title('Grouping')
    ax.set_xticks(x) # values
    ax.set_xticklabels(labels) # labels
    plt.xticks(rotation=90)
    ax.legend()

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    #plt.show()
    plt.savefig('DatasetDistributions.png', bbox_inches='tight')

    print("Sum annotated bodies:{}, anomalies:{}, debris:{}, bikes:{}".format(c_cnf_body,c_anomaly,c_debris,c_bikes))


    #The input to the model is expected to be a list of tensors,
    #each of shape [C, H, W], one for each image, and should be in 0-1 range.
    #Different images can have different sizes.

    sonar_transform = A.Compose([ #rotations, strecthing, different intensities probably safe
        A.RandomCrop(width=1000, height=350), # TODO: figure out of this works, because the model isnt trained on this input
        A.VerticalFlip(p=0.5),                             # if not just resize to support size
        A.RandomBrightnessContrast(p=0.25),
        A.RandomGamma(p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), # can not visualize with this
    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))



    output_folder = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/"
    csv_file = output_folder+"06_02_2022_BIG-export.csv"
    train = output_folder+"train"
    dev = output_folder+"dev"
    test = output_folder+"test"

    train_dataset = SonarDataset(train,csv_file,sonar_transform)
    dev_dataset = SonarDataset(dev,csv_file,sonar_transform)
    test_dataset = SonarDataset(test,csv_file,sonar_transform)

    #visualize(train_dataset[0][0])
    #[x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
    print(train_dataset[21][0].shape)
    print(train_dataset[21][0].is_cuda)
    print(train_dataset[21][1])
    print(train_dataset[21][1]["boxes"].shape)
    print("Number of images train:{}, dev:{}, test:{}".format(len(train_dataset),len(dev_dataset),len(test_dataset)))

    torch.backends.cudnn.benchmark = True # speeds up training by some variable amount
    num_classes = 5  # debris + bike + anomaly + confirmed_victim + background

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,pretrained_backbone=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    print(device)
    print(torch.version.cuda)
    torch.cuda.empty_cache()

    def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, print_every = 50):
        model.train()

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        debug_counter = 0
        avg_loss_value = 0
        for images, targets in data_loader:
            #if debug_counter == 3: break
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets] # v.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            avg_loss_value = avg_loss_value + loss_value
            if debug_counter != 0 and debug_counter % print_every == 0:
                print_loss_value = avg_loss_value / debug_counter
                #print("Training progress {}/{}, cumulative avg loss: {}".format(epoch, debug_counter,len(data_loader),print_every,print_loss_value))

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            debug_counter += 1
        return avg_loss_value/debug_counter



    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # max batch for memory is 8
    # 0 workers= 1:14
    # 1 workers= 1:09
    # 2,3,4 = 1:10

    train_dataloader = DataLoader(train_dataset, batch_size=8,collate_fn=obj_collate_fn,pin_memory=True, shuffle=True,num_workers=1,drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4,collate_fn=obj_collate_fn,pin_memory=True, shuffle=True,num_workers=1,drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4,collate_fn=obj_collate_fn,pin_memory=True, shuffle=True,num_workers=1,drop_last=True)

    visualize_test_boxes = [[621.2121, 231.6017, 699.1342, 300.8658],
                  [505.4113, 751.0823, 596.3203, 808.4416],
                  [428.5714, 616.8831, 484.8485, 636.3636],
                  [444.8052, 216.4502, 531.3853, 239.1775],
                  [215.3680, 361.4719, 258.6580, 397.1861]]

    # keep in mind boxes returned from the dataset using the traditional indexing get item method are returned transformed,
    # while images returned using get_image are not

    #visualize_bbox(train_dataset.get_image(21), visualize_test_boxes, save=True)


    num_epochs = 50

    best_positive = 0
    best_false = np.inf
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_dataloader, device, epoch,print_every=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_eval_obj, stats = evaluate(model, dev_dataloader, device=device)
        val_loss = stats["val_loss"]
        positives= stats["TP"]
        falses = stats["FP"] + stats["FN"]
        if positives > best_positive and falses < best_false:
            print("Improvement, saved model!")
            torch.save(model.state_dict(), "saved_model.pt")
            best_positive = positives
            best_false = falses
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        plot_loss(train_loss_list,val_loss_list)
        print(stats)
        print("{}# epoch done, Train loss: {}, Validation loss: {}".format(epoch+1,train_loss,val_loss))

    print("That's it!")

