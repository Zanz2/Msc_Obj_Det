import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from torchvision.ops import nms
import utils2
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

def get_loss(data_loader,model,device):
    eval_loss = 0
    model.train()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # v.to(device)
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            eval_loss += losses.item()
    model.eval()
    return eval_loss

def custom_evaluate(res_dict,targets,current_dict,IOU_TRESHOLD = 0.5,SCORE_TRESHOLD = 0.5,MAX_NUM_DET=50):
    current_dict["iou_treshold"] = IOU_TRESHOLD
    current_dict["confidence_treshold"] = SCORE_TRESHOLD
    current_dict["max_num_det"] = MAX_NUM_DET
    for gt_target in targets:
        dict_for_img = res_dict[gt_target["image_id"].item()]

        gt_boxes = gt_target["boxes"].tolist()
        gt_labels = gt_target["labels"].tolist()
        current_dict["gt_total"] += len(gt_boxes)

        pred_labels = dict_for_img["labels"].tolist()
        pred_scores = dict_for_img["scores"]

        pred_boxes_mask = nms(boxes=dict_for_img["boxes"], scores=dict_for_img["scores"], iou_threshold=IOU_TRESHOLD)
        pred_boxes = dict_for_img["boxes"][pred_boxes_mask].tolist()
        pred_scores = pred_scores[pred_boxes_mask].tolist()

        num_predictions = len(pred_boxes)

        if num_predictions == 0:
            current_dict["FN"] += len(gt_boxes)
            continue

        if num_predictions < MAX_NUM_DET:  # if it made less predictions than our max, use how many it made
            MAX_DET = num_predictions
        else:
            MAX_DET = MAX_NUM_DET
        total_pred = 0
        for index in range(MAX_DET):
            if pred_scores[index] > SCORE_TRESHOLD:
                current_dict["pred_total_b_d_cb_a"][pred_labels[index]] += 1
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
            if best_IOU > IOU_TRESHOLD:
                if pred_label != gt_label:
                    current_dict["missclassifications"] += 1
                else:
                    current_dict["TP"] += 1
                detected = True
                used_indexes.append(pred_index)
            if not detected: current_dict["FN"] += 1
        current_dict["FP"] += (total_pred - len(used_indexes))

    total = sum(current_dict["pred_total_b_d_cb_a"])
    if current_dict["TP"] != 0:
        accuracy = (current_dict["TP"] + current_dict["TN"])/total
        precision = current_dict["TP"]/(current_dict["TP"]+current_dict["FP"])
        recall = current_dict["TP"]/(current_dict["TP"]+current_dict["FN"])
        current_dict["accuracy"] = accuracy
        current_dict["precision"] = precision
        current_dict["recall"] = recall
    return current_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, print_every=50,do_eval_metrics=False):
    model.train()
    metric_logger = utils2.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils2.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    cumulative_stats_dict = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0,
        "missclassifications": 0,
        "gt_total": 0,
        "pred_total_b_d_cb_a": [0, 0, 0, 0, 0],
        "recall": 0,
        "precision": 0,
        "accuracy": 0
    }

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        #lr_scheduler = torch.optim.lr_scheduler.LinearLR(
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

        if lr_scheduler is not None:
            lr_scheduler.step()

        img_counter += 1
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    cumulative_stats_dict["loss"] = avg_loss_value / img_counter

    if do_eval_metrics:
        print("Eval metrics of train set:")
        cpu_device = torch.device("cpu")
        model.eval()
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            cumulative_stats_dict = custom_evaluate(res, targets, cumulative_stats_dict)
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()
        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

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
def evaluate(model, data_loader, device):
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
        "pred_total_b_d_cb_a": [0,0,0,0,0],
        "recall": 0,
        "precision": 0,
        "accuracy": 0
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
        cumulative_stats_dict = custom_evaluate(res,targets,cumulative_stats_dict)

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    eval_loss = get_loss(data_loader,model,device)
    eval_loss = eval_loss / len(data_loader)
    cumulative_stats_dict["loss"] = eval_loss

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #coco_evaluator.coco_eval["bbox"].analyze()

    torch.set_num_threads(n_threads)
    return coco_evaluator, cumulative_stats_dict