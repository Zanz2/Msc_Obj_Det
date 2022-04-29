#!/usr/bin/env python
# coding: utf-8

# # Experiment A1: parse JSF
# Lets try to read one of the JSF files. Interpret them using this spec:
#   https://www.edgetech.com/wp-content/uploads/2019/07/0023492_Rev_E.pdf


import os
import statistics
import sys
from urllib.parse import unquote
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import albumentations as A
import utils2
import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn

from torchvision.ops import nms
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from torch import FloatTensor, LongTensor, nn, optim
from torchvision import datasets, models
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchsummary import summary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = "cpu"
global_eval_current_model_path = ""

def fasterrcnn_resnet18(num_classes=91, pretrained_backbone=True,trainable_bb_layers=None, **kwargs):
    print("using resnet18 (shallower)")
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18',pretrained=pretrained_backbone,trainable_layers=trainable_bb_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model

def plot_x_y(train_loss,val_loss=[],mode="",path=""):
    plt.clf()
    lim = 0.5
    if max(train_loss) > 0.5: lim = 1
    if mode == "loss_plot":
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title("Train is blue, eval is orange")
        if os.path.isfile("LossPlot.png"): os.remove("LossPlot.png")
        plt.savefig("LossPlot.png", bbox_inches='tight')
    if mode == "accuracy_plot":
        plt.plot(train_loss)
        plt.ylim([0, lim])
        plt.title("Evaluation accuracy")
        if os.path.isfile("{}EvalAccuracyPlot.png".format(path)): os.remove("{}EvalAccuracyPlot.png".format(path))
        plt.savefig("{}EvalAccuracyPlot.png".format(path), bbox_inches='tight')
    if mode == "precision_recall_curve":
        xlim = 0.5
        if max(val_loss) > 0.5: xlim = 1
        plt.plot(val_loss)
        plt.plot(train_loss)
        plt.ylim([0, lim])
        plt.xlim([0, xlim])
        plt.title("Precision is orange, recall is blue")
        if os.path.isfile("{}PRPlot.png".format(path)): os.remove("{}PRPlot.png".format(path))
        plt.savefig("{}PRPlot.png".format(path), bbox_inches='tight')
    if mode == "precision_recall":
        xlim = 0.5
        if max(val_loss) > 0.5: xlim = 1
        plt.ylim([0, lim])
        plt.xlim([0, xlim])
        plt.plot(val_loss, train_loss)
        plt.title("Precision is Y, recall is X")
        if os.path.isfile("{}PRCurve_real.png".format(path)): os.remove("{}PRCurve_real.png".format(path))
        plt.savefig("{}PRCurve_real.png".format(path), bbox_inches='tight')

def obj_collate_fn(batch):
    return tuple(zip(*batch))

def round_down(x, a):
    return math.floor(x / a) * a

class SonarDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms, type="normal"):
        self.type = type
        self.root = root
        self.transforms = transforms
        self.transformed_images = {}

        # Vott json vott format export is needed: vott-json-export folder with all the png grayscale images and one json file
        # that has the information of all the images included and the ones that have bounding boxes

        self.imgs = [s for s in os.listdir(root) if s.endswith('.png')]

        #self.imgs = self.imgs[0:int(len(self.imgs)*0.1)] # MAKE IT FAST FOR DEBUGGING

        csv_boxes = pd.read_csv(csv_file)

        self.label2id = {
            "bike": 1,
            "confirmed_body": 3,
            "anomaly": 2,
            "debris": 4
        }
        reduced_set = []
        img_name_to_box = {}
        for index, asset in csv_boxes.iterrows():
            img_filename = asset["image"]

            if img_filename not in self.imgs or asset["label"] == "debris": continue  # if the image isnt in this folder but in the json skip it
            # this is so i can create a test train split

            if img_filename not in img_name_to_box:
                img_name_to_box[img_filename] = []
                if self.type == "reduced":
                    reduced_set.append(img_filename)
                    if asset["label"] == "confirmed_body":
                        for x in range(3): reduced_set.append(img_filename)
            img_name_to_box[img_filename].append(
                [asset["xmin"], asset["ymin"], asset["xmax"], asset["ymax"], self.label2id[asset["label"]]])
            if self.type == "oversampled" and asset["label"] == "confirmed_body":
                for x in range(3): self.imgs.append(img_filename)

        self.img2boxes = img_name_to_box
        if self.type == "reduced": self.imgs = reduced_set

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
        img = img / 255  # 0 to 1 range
        img = torch.from_numpy(img)
        img = img.float()
        img.to(device)
        return img, target

    def __len__(self):
        return len(self.imgs)

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


def visualize_bbox(image,bbox_list,gt_list=[],vis_pred_labels=[],gt_labels=[],vis_pred_scores=[],save=True,save_name=""):
    if save and save_name=="": save_name = "image"
    if len(bbox_list) == 0:
        return
    id2label = {
        1: "bike",
        3: "confirmed_body",
        2: "anomaly",
        4: "debris"
    }
    pic = image # vis_pred_labels=vis_pred_labels,gt_labels=gt_labels
    for index, gt in enumerate(gt_list):
        pic = cv2.rectangle(pic, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0,255,0), 1) # blue green red
        if len(gt_labels) > 0: pic = cv2.putText(pic, id2label[gt_labels[index]], (int(gt[0])+10, int(gt[1])+10), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,255,0), 1, cv2.LINE_AA)
    for index, bbox in enumerate(bbox_list):
        pic = cv2.rectangle(pic, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1) # blue green red
        label_score_string = " "
        if len(vis_pred_labels) > 0: label_score_string += id2label[vis_pred_labels[index]]
        if len(vis_pred_scores) > 0: label_score_string += " "+str(vis_pred_scores[index])[0:4]
        pic = cv2.putText(pic,label_score_string, (int(bbox[0]) + 10, int(bbox[1]) + 10),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    if save:
        if len(gt_list) > 0:
            save_name = "{}_GT".format(save_name)
        else:
            save_name = "{}_FP".format(save_name)
        cv2.imwrite("{}.jpg".format(save_name),pic,[int(cv2.IMWRITE_JPEG_QUALITY), 96])
        #print("{}.jpg".format(save_name))
    else:
        cv2.imshow("bboxes visualized", pic)
        cv2.waitKey(0) # this freezes and crashes for some reason

def tensor_to_img(tensor_array):
    tensor_array = tensor_array.cpu().detach().numpy()
    tensor_array = tensor_array * 255
    # h, w, c = img.shape cv2
    # c, h, w = shape for the model
    img = np.transpose(tensor_array, [1, 2, 0])
    img = img.astype(np.uint8)
    return img

def get_loss(data_loader,model,device):
    eval_loss = 0
    with torch.no_grad():
        model.train()
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # v.to(device)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            eval_loss += losses.item()
    model.eval()
    return eval_loss

def ruin_my_life(original_dict,score,type):
    if type == "TP":
        dict_score_min = int(score * 10)
        for key, value in original_dict["confidences"].items():
            if dict_score_min >= key:
                original_dict["confidences"][key][type] += 1
            else:
                original_dict["confidences"][key]["FN"] += 1
    if type == "FN":
        for key, value in original_dict["confidences"].items():
                original_dict["confidences"][key]["FN"] += 1
    if type == "FP":
        scores = score[0]
        used_indexes = score[1]
        labels = score[2]
        for index, score_val in enumerate(scores):
            if index in used_indexes or labels[index] != 3: continue
            dict_score_min = int(score_val*10)
            for key, value in original_dict["confidences"].items():
                if dict_score_min >= key:
                    original_dict["confidences"][key]["FP"] += 1
    return original_dict

def custom_evaluate(res_dict,targets,current_dict,images=[],visualize=False,IOU_TRESHOLD = 0.5,SCORE_TRESHOLD = 0.25,MAX_NUM_DET=300000):
    current_dict["iou_treshold"] = IOU_TRESHOLD
    current_dict["confidence_treshold"] = SCORE_TRESHOLD
    current_dict["max_num_det"] = MAX_NUM_DET
    for img_index,gt_target in enumerate(targets):
        dict_for_img = res_dict[gt_target["image_id"].item()]

        gt_boxes = gt_target["boxes"].tolist()
        gt_labels = gt_target["labels"].tolist()
        for label in gt_labels:
            current_dict["correct_max"][label] += 1
        current_dict["gt_total"] += len(gt_boxes)

        pred_scores = dict_for_img["scores"]
        if True: # apply nms or not (mainly for debug)
            #print("Scores before nms {}".format(len(pred_scores)))
            pred_boxes_mask = nms(boxes=dict_for_img["boxes"], scores=dict_for_img["scores"], iou_threshold=IOU_TRESHOLD)
            pred_boxes = dict_for_img["boxes"][pred_boxes_mask].tolist()
            pred_labels = dict_for_img["labels"][pred_boxes_mask].tolist()
            pred_scores = pred_scores[pred_boxes_mask].tolist()
            #print("Scores after nms {}".format(len(pred_scores)))
        else:
            pred_boxes = dict_for_img["boxes"].tolist()
            pred_labels = dict_for_img["labels"].tolist()
            pred_scores = pred_scores.tolist()

        num_predictions = len(pred_boxes)

        if num_predictions == 0:
            current_dict["FN"] += len(gt_boxes)
            for unp_label in gt_labels:
                if unp_label == 3: current_dict = ruin_my_life(current_dict, 0, "FN")
            continue

        if num_predictions < MAX_NUM_DET:  # if it made less predictions than our max, use how many it made
            MAX_DET = num_predictions
        else:
            MAX_DET = MAX_NUM_DET
        total_pred = 0
        for index in range(MAX_DET):
            if pred_scores[index] >= SCORE_TRESHOLD:
                current_dict["pred_total"][pred_labels[index]] += 1
                total_pred += 1
        used_indexes = []
        for gt_index in range(len(gt_boxes)):
            detected = False
            gt_box = gt_boxes[gt_index]
            gt_label = gt_labels[gt_index]
            best_IOU = 0
            pred_label = ""
            pred_index = None
            for index in range(MAX_DET):
                if pred_scores[index] < SCORE_TRESHOLD or index in used_indexes: continue
                box = pred_boxes[index]
                label = pred_labels[index]
                bb_gt = {
                    'x1': gt_box[0], 'x2': gt_box[2], 'y1': gt_box[1], 'y2': gt_box[3]
                }
                bb_pred = {
                    'x1': box[0], 'x2': box[2], 'y1': box[1], 'y2': box[3]
                }
                iou_val = get_iou(bb_gt, bb_pred)
                if iou_val > best_IOU:
                    best_IOU = iou_val
                    pred_label = label
                    pred_index = index
            if best_IOU > IOU_TRESHOLD: # 6: {"TP": 0, "FP": 0, "FN": 0},
                if pred_label != gt_label:
                    current_dict["missclassifications"] += 1
                else:
                    current_dict["TP"] += 1
                    current_dict["correct_total"][label] += 1
                if gt_label == 3: current_dict = ruin_my_life(current_dict, pred_scores[pred_index], "TP")
                detected = True
                used_indexes.append(pred_index)
            if not detected:
                current_dict["FN"] += 1
                if gt_label == 3: current_dict = ruin_my_life(current_dict, 0, "FN")

        current_dict = ruin_my_life(current_dict, [pred_scores,used_indexes,pred_labels], "FP")
        current_dict["FP"] += (total_pred - len(used_indexes))
        if visualize:
            vis_pred_boxes = [box for index, box in enumerate(pred_boxes) if pred_scores[index] > SCORE_TRESHOLD]
            vis_pred_labels = [pred_labels[index] for index, _ in enumerate(pred_boxes) if pred_scores[index] > SCORE_TRESHOLD]
            vis_pred_scores = [pred_scores[index] for index, _ in enumerate(pred_boxes) if pred_scores[index] > SCORE_TRESHOLD]

            numpy_image = tensor_to_img(images[img_index])
            if global_eval_current_model_path != "":
                visualize_bbox(numpy_image,vis_pred_boxes,gt_boxes,vis_pred_labels=vis_pred_labels,gt_labels=gt_labels,vis_pred_scores=vis_pred_scores,save_name="{}/predictions/img_id{}".format(global_eval_current_model_path,gt_target["image_id"].item()))
            else:
                visualize_bbox(numpy_image,vis_pred_boxes,gt_boxes,vis_pred_labels=vis_pred_labels,gt_labels=gt_labels,vis_pred_scores=vis_pred_scores,save_name="F:/projekti/msc_sonar_models/visualizations/predictions/img_id{}".format(gt_target["image_id"].item()))

    return current_dict

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, print_every=50):
    model.train()
    metric_logger = utils2.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils2.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    cumulative_stats_dict = {}

    lr_scheduler2 = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        #lr_scheduler2 = torch.optim.lr_scheduler.LinearLR(
        #    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        #)

    img_counter = 0
    avg_loss_value = 0
    for images, targets in metric_logger.log_every(data_loader, print_every, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # v.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils2.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        avg_loss_value = avg_loss_value + loss_value

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler2 is not None:
            lr_scheduler2.step()

        img_counter += 1
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    cumulative_stats_dict["loss"] = avg_loss_value / img_counter

    return metric_logger, cumulative_stats_dict

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, eval_visualize=False, score_threshold = 0):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils2.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    cumulative_stats_dict = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "missclassifications": 0,
        "gt_total": 0,
        "pred_total": [0, 0, 0, 0, 0],
        "correct_total": [0, 0, 0, 0, 0],
        "correct_max": [0, 0, 0, 0, 0],
        "confidences": {
            0: {"TP": 0, "FP": 0, "FN": 0},
            1: {"TP": 0, "FP": 0, "FN": 0},
            2: {"TP": 0, "FP": 0, "FN": 0},
            3: {"TP": 0, "FP": 0, "FN": 0},
            4: {"TP": 0, "FP": 0, "FN": 0},
            5: {"TP": 0, "FP": 0, "FN": 0},
            6: {"TP": 0, "FP": 0, "FN": 0},
            7: {"TP": 0, "FP": 0, "FN": 0},
            8: {"TP": 0, "FP": 0, "FN": 0},
            9: {"TP": 0, "FP": 0, "FN": 0}
        }
    }

    for images, targets in metric_logger.log_every(data_loader, 220, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        cumulative_stats_dict = custom_evaluate(res,targets,cumulative_stats_dict,images=images,visualize=eval_visualize,SCORE_TRESHOLD=score_threshold)

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    #eval_loss = get_loss(data_loader,model,device)
    #eval_loss = eval_loss / len(data_loader)
    eval_loss = 0.01
    cumulative_stats_dict["loss"] = eval_loss

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # the average precision (map) is calculated based on the precisions per category of image (labels)
    # max dets denotes the maximum number of proposals per image (sorted by best score descending)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #coco_evaluator.coco_eval["bbox"].analyze()

    torch.set_num_threads(n_threads)
    return coco_evaluator, cumulative_stats_dict

if __name__ == "__main__":
    # laptop = "C:/Users/zanza/Desktop/MSC_work/Msc_Obj_Det/"
    # desktop = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/"
    prefix1 = "C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/"
    target = "conversions/vott_to_vgg_proj/empty_vgg_json.json"
    source = "conversions/vott_to_vgg_proj/source_vott_csv.csv"
    target_folder = prefix1+"data/vott/run3_big/input"
    csv_proj_file = prefix1+"data/vott/run3_big/output/vott-csv-export/06_02_2022_BIG-export.csv"

    info_dict = get_class_stats(csv_proj_file)

    # go over all the confirmed bodies
    vott_csv = pd.read_csv(csv_proj_file)
    prefix = prefix1+"data/vott/run3_big/output/vott-csv-export/"

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
    dev_set = [
        '06-12-2020 Burdaard',
        '09-11-2020 wemeldinge_2',
        '09-11-2020 wemeldinge',
        'brummen',
        'Coverage_12-11-2020 Brummen'
    ]
    test_set = [
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
    fig.tight_layout()
    #plt.show()
    plt.savefig('DatasetDistributions.png', bbox_inches='tight')

    sonar_transform = A.Compose([ # strecthing, different intensities probably safe
        A.RandomCrop(width=1000, height=400),
        #A.RandomResizedCrop(350),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
        #A.Equalize(p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))

    sonar_eval_transform = A.Compose([  # strecthing, different intensities probably safe
        #A.RandomCrop(width=1000, height=350),
        #A.Equalize(p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    output_folder = prefix1+"data/vott/run3_big/output/vott-csv-export/"
    csv_file = output_folder+"06_02_2022_BIG-export.csv"
    train = output_folder+"train"
    test = output_folder+"dev" # switched because /test has more files for evaluation
    dev = output_folder+"test"

    train_dataset = SonarDataset(train,csv_file,sonar_transform,type="oversampled")
    dev_dataset = SonarDataset(dev,csv_file,sonar_eval_transform)
    test_dataset = SonarDataset(test,csv_file,sonar_eval_transform)

    #visualize(train_dataset[0][0])
    #[x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
    # h, w, c = img.shape

    pretrain_coco = False # mutually exclusive
    pretrain_imagenet = True # mutually exclusive
    weight_decay_val = 0.0000 # 0.00005
    bb_train_val = 5
    num_classes = 4  # bike + anomaly + confirmed_victim + background (debris is not used anymore)
    lr_val = 0.0001 # 0.00005

    print("Sum annotated bodies:{}, anomalies:{}, debris:{}, bikes:{}".format(c_cnf_body, c_anomaly, c_debris, c_bikes))
    print("Original shape:{}, new transformed shape:{}".format(train_dataset.get_image(21).shape,train_dataset[21][0].shape))
    print("{} to {}".format(torch.min(train_dataset[21][0]),torch.max(train_dataset[21][0])))
    print(train_dataset[21][0].is_cuda)
    print(train_dataset[21][1])
    print(train_dataset[21][1]["boxes"].shape)
    print("Number of images train:{}, dev:{}, test:{}".format(len(train_dataset),len(dev_dataset),len(test_dataset)))
    print(device)
    print(torch.version.cuda)
    print("Pretrained on coco:{}, pretrained on imagenet:{}".format(pretrain_coco,pretrain_imagenet))
    print("Weight decay:{}, trainable bb layers (5 is all):{}".format(weight_decay_val,bb_train_val))

    torch.backends.cudnn.benchmark = True
    anchor_sizes = ((32,), (64,), (128,), (256,),(512,)) # original
    #anchor_sizes = ((20,), (40,), (80,), (120,), (280,))
    aspect_ratios = ((0.15, 0.33, 0.5, 0.66, 1.0, 1.5, 2),) * len(anchor_sizes) # height / width
    rpn_sonar_anchor_gen = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    '''
    model = fasterrcnn_resnet18(
        pretrained_backbone=pretrain_imagenet,
        trainable_bb_layers=bb_train_val, # 5 is all (none are frozen)
        rpn_anchor_generator=rpn_sonar_anchor_gen,
        box_detections_per_img=280,
    )
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained_backbone=pretrain_imagenet,
        trainable_backbone_layers=bb_train_val, # 5 is all (none are frozen)
        rpn_anchor_generator=rpn_sonar_anchor_gen,
        rpn_pre_nms_top_n_train=8000, rpn_pre_nms_top_n_test=8000, # 10 000, 5000 was overkill
        rpn_post_nms_top_n_train=4000, rpn_post_nms_top_n_test=4000, # 5 000 2500 overkill

        box_detections_per_img=500, # test only (eval mode)
    )
    '''
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
        greater than rpn_score_thresh
        
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    '''
    # only pretrained on coco 2017, if pretrained_backbone = True AND pretrained=False then it uses a backbone pretrained on imagenet
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    torch.cuda.empty_cache()

    #model_name = "F:/projekti/msc_sonar_models/saved_model_epoch30.pt"
    #model = torch.load(model_name))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr_val, weight_decay=weight_decay_val)  # Trying ADAM here

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, # reduced step size, not using rn
                                                   step_size=25,
                                                   gamma=0.5)

    # max batch is 6 with resnet50 | 10 resnet18 (usually train is 2x dev or test)
    train_dataloader = DataLoader(train_dataset, batch_size=4,collate_fn=obj_collate_fn,shuffle=True, num_workers=1, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4,collate_fn=obj_collate_fn,pin_memory=True, shuffle=True, num_workers=1, drop_last=True) # To make dev validation loss more stable with small sample sizes data augmentation can be used
    test_dataloader = DataLoader(test_dataset, batch_size=4,collate_fn=obj_collate_fn,pin_memory=True, shuffle=True, num_workers=1, drop_last=True)

    num_epochs = 1000
    best_eval_loss = 1
    train_loss_list = []
    val_loss_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    best_recall = 0
    best_precision = 0

    eval_test = [ # in each of these folders there should be atleast 1 model that ends with .pt, and a folder of the same name (without .pt)
        "F:/projekti/msc_sonar_models/imagenet/",
        "F:/projekti/msc_sonar_models/coco/",
        "F:/projekti/msc_sonar_models/r18_imagenet/",
        "F:/projekti/msc_sonar_models/scratch/"
    ]
    if True:
        for epoch in range(num_epochs):
            logger, train_stats = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_every=250)
            train_loss = train_stats["loss"]

            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            coco_eval_obj, eval_stats = evaluate(model, dev_dataloader, device=device,eval_visualize=False)
            val_loss = eval_stats["loss"]

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            total = sum(eval_stats["pred_total"])
            class_index = 3  # 3 is confirmed body, 1 is bike, 2 is anomaly
            if eval_stats["TP"] != 0 and eval_stats["correct_total"][class_index] != 0:
                accuracy = (eval_stats["TP"] + eval_stats["TN"]) / total
                #precision = eval_stats["TP"] / (eval_stats["TP"] + eval_stats["FP"]) # macro all classes
                #recall = eval_stats["TP"] / (eval_stats["TP"] + eval_stats["FN"]) # macro all classes
                tp = eval_stats["correct_total"][class_index]
                fp = eval_stats["pred_total"][class_index] - eval_stats["correct_total"][class_index]
                fn = eval_stats["correct_max"][class_index] - eval_stats["correct_total"][class_index]
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)

            else:
                accuracy = 0
                precision = 0
                recall = 0
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            if (recall+precision) > (best_recall+best_precision):
                print("Improvement, saved model!")
                torch.save(model, "saved_model.pt")
                best_recall = recall
                best_precision = precision

            if epoch % 10 == 0:
                torch.save(model, "F:/projekti/msc_sonar_models/saved_model_epoch{}.pt".format(epoch))

            plot_x_y(train_loss_list, val_loss_list, mode="loss_plot")
            plot_x_y(precision_list, recall_list, mode="precision_recall_curve")
            plot_x_y(accuracy_list, mode="accuracy_plot")
            print("Eval custom metrics:")
            print("Total and correct labels to indexes {}".format(dev_dataset.label2id))
            print(eval_stats)
            print("class:{} precision:{} recall:{}".format(class_index,precision,recall))
            print("{}# epoch done, Train loss: {}, Validation loss: {}".format(epoch+1,train_loss,val_loss))
            print("---------------------------------------------------------------------")
    else:
        for model_dir in eval_test:
            for file in os.listdir(model_dir):
                if file.endswith(".pt"):
                    model_name = os.path.join(model_dir, file)
                    global_eval_current_model_path = os.path.join(model_dir, os.path.splitext(file)[0])
                    print("---------------------------- now working on -------------------------------")
                    print(model_name)
                    print(global_eval_current_model_path)

                    torch.cuda.empty_cache()
                    model = torch.load(model_name)
                    model.eval()
                    coco_eval_obj, eval_stats = evaluate(model,test_dataloader,device=device,eval_visualize = True,score_threshold = 0)

                    precision_list = []
                    recall_list = []
                    for score_step in range(0,10):
                        if eval_stats["confidences"][score_step]["TP"] != 0:
                            tp = eval_stats["confidences"][score_step]["TP"]
                            fp = eval_stats["confidences"][score_step]["FP"]
                            fn = eval_stats["confidences"][score_step]["FN"]
                            precision = tp / (tp + fp)
                            recall = tp / (tp + fn)
                        else:
                            precision = 0
                            recall = 0
                        precision_list.append(precision)
                        recall_list.append(recall)
                    print("labels to indexes {}".format(dev_dataset.label2id))
                    print(eval_stats)
                    print(recall_list)
                    print(precision_list)
                    plot_x_y(precision_list,recall_list, mode="precision_recall",path=global_eval_current_model_path+"/")

    print("That's it!")


