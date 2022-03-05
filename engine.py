import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils2
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import pprint


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils2.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils2.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils2.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

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

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
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

    IOU_TRESHOLD = 0.5
    MAX_DET = 50
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    missclassifications = 0
    c_gt_boxes = 0
    c_pred_boxes = 0

    for images, targets in metric_logger.log_every(data_loader, 220, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        score_sorted_res = res # it was already sorted by score lol

        for target in targets:
            c_gt_boxes += len(target["boxes"])
            if len(score_sorted_res.items()) == 0:
                false_negatives += len(target["boxes"])

        for image_id,proposed_detections in score_sorted_res.items():
            c_pred_boxes += len(proposed_detections["boxes"])
            if len(proposed_detections["scores"]) < MAX_DET: # if it made less predictions than our max, use how many it made
                MAX_DET = len(proposed_detections["scores"])
            for index in range(MAX_DET):
                box = proposed_detections["boxes"][index]
                label = proposed_detections["labels"][index]
                for gt_target in targets:
                    #print("Proposal on IMG:{}, GT IMG: {}".format(image_id,gt_target["image_id"].item()))
                    if image_id == gt_target["image_id"].item():
                        for gt_index in range(len(gt_target["boxes"])):
                            gt_box = gt_target["boxes"][gt_index]
                            gt_label = gt_target["labels"][gt_index]
                            bb_gt = {
                                'x1': gt_box[0],
                                'x2': gt_box[2],
                                'y1': gt_box[1],
                                'y2': gt_box[3]
                            }
                            bb_pred = {
                                'x1': box[0],
                                'x2': box[2],
                                'y1': box[1],
                                'y2': box[3]
                            }
                            iou_val = get_iou(bb_gt,bb_pred)
                            print("---------------------")
                            print("Image ID: {}".format(image_id))
                            print("predicted label {} this is gt label {}".format(label,gt_label))
                            print("IOU value {}, gt box below, below that is pred box".format(iou_val))
                            print(bb_gt)
                            print(bb_pred)
                            print("---------------------")
                            if iou_val > IOU_TRESHOLD:
                                if label == gt_label:
                                    true_positives += 1
                                if label != gt_label:
                                    missclassifications += 1
                            elif iou_val != 0:
                                false_positives += 1
                            else:
                                false_negatives += 1

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    eval_loss = 0
    model.train()
    for images, targets in metric_logger.log_every(data_loader, 220, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # v.to(device)
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            eval_loss += losses.item()
    model.eval()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    eval_loss = eval_loss / len(data_loader)
    custom_stats = {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "missclassifications": missclassifications,
        "gt_total": c_gt_boxes,
        "pred_total": c_pred_boxes,
        "val_loss": eval_loss
    }
    return coco_evaluator, custom_stats