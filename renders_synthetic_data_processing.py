import os
import csv
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import math
import torch

'''
Author: Žan Žagar (zanz2 on github)
Authors readme with some cool lore: This work was done in the scope of my masters thesis AI internship at the Politie. The work took place over 7 months.
Below are the relevant files and how i used them (you might find your own solutions or improve the ones present, or even scrap them)

This was enough time to answer the research questions posed, but not to deliver a polished solution.
The main tasks were:
- Given raw sonar data .jsf files, create a solution that is able to parse these files into postprocessed images, that corrects
    the sonar noise that is present and occurs due to the nature of sonar imaging. Account for the changing resolutions, of the scans,
    equalize the "bright" areas at the left area of the sonar scan and "dark" areas at the rightmost areas.
    !!!File that does this: parse_and_preprocess_jsf.ipynb
    From sonar .jsf files normalized, padded and stitched 1000x1000 images are created, these were then annotated for bounding box object detection in VOTT
- The created dataset was then trained using pytorch, you should really scrap this file alltogether and probably use tensorflow,
    I have found the object detection evaluation tools in pytorch VERY lacking, and the documentation very lacking too, there are object detection evaluation libraries missing,
    so you have to use ones that work for linux and port them to windows.
    I had to write my own evaluation script just to find out how many TP FP and FNs the object was making because the ported library had no documentation and was
    not being maintained, visualizing bounding boxes also wasnt there so i had to write this, and furthemore the library itself was based on a 
     older version of numpy, so i had to fix some type casting errors to make it work, I then decided to just include the library in the git
     so i wouldnt have to do this everytime i cloned the project (the library is pycocotools).
    
    TLDR: skip this file, use tensorflow, some findings: pretrained was never better for me than models from scratch even when freezing various number of backbone layers
    I used oversampling and data augmentation to alleviate the class imbalance, use a custom anchor generator to generate more anchors at common sonar body aspect ratios,
    use sonar image mean and std (all 3 img channels are the same in grayscale data), 
    !!!File that does this: file train_model.py
- Next followed a lot of 3D modelling work, to create a scene to emulate the sonar environment in blender, this was very time intense and probably 70% of the time spent in this project.
    Using the python blender scripting engine, most aspects of the scene (such as sonar angle, body positions, limb variations etc) were randomly varied using a script i created
    to allow for highly varied and extendable generation of realistic fake bodies.
    !!!File that does this: blender_generate_renders.py
- Next followed a lot of postprocessing, to superimpose these varied fake bodies onto existing backgrounds, this was done using intense image manipulation,
    bitwise masking to allow me to seperate the background, the body and the shadow, then apllying different noise profiles to each, different transparency values to each,
    pixelation to each.
    !!! File that does this: renders_synthetic_data_processing.py

- Final notes: Each of these files can be improved significantly(except train_model.py, id scrap it and use tensorflow). Due to me wanting to round up
    my studies and find a job this was not possible in the given time frame, but most of the files have comments marked to-do (without the hyphen) with some
    of my ideas, you will probably come up with your own improvements. Out of these 4 files the blender renders file and the synthetic data processing could have
    been greatly extended if i was not pressed for time and money, i believe given enough time they could generate samples that are indistinguishable from the real data.
'''

class GlobalBackgrounds():
    def __init__(self):
        self.applicable_backgrounds = []
    def set_backgrounds(self, bgs_list):
        self.applicable_backgrounds = bgs_list
    def get_backgrounds(self):
        return self.applicable_backgrounds
    def get_background(self):
        return self.applicable_backgrounds[random.randint(0,len(self.applicable_backgrounds)-1)]

def anchor_box_analyze(anchor_box_file_path,print_positives_list = False):
    csv_boxes = pd.read_csv(anchor_box_file_path)
    ratios = {0.06: 0,0.24: 0,0.34: 0,0.55: 0,0.78:0,1: 0,1.5:0,2.2: 0}
    # anchor_sizes = ((25,), (75,), (150,), (300,),(400,))
    sizes = {20: 0,42: 0,62: 0,100: 0,280: 0}
    used,used_s,total = 0,0,0
    search_string = ""
    max_width,max_height,avg_width,avg_height = 0,0,0,0
    min_width,min_height = 1000,1000
    ratios_list = []
    for index, asset in csv_boxes.iterrows():
        if asset["label"] == "confirmed_body":
            search_string = "{}\n{}".format(search_string,asset["image"])
            width = asset["xmax"] - asset["xmin"]
            height = asset["ymax"] - asset["ymin"]
            ratio = height/width
            total += 1

            ratio = round(ratio, 2)
            width = round(width, 2)
            height = round(height, 2)
            ratios_list.append(ratio)

            if height > max_height: max_height = height
            if height < min_height: min_height = height
            avg_height += height

            if width > max_width: max_width = width
            if width < min_width: min_width = width
            avg_width += width

            for key,value in ratios.items():
                if ratio < key:
                    ratios[key] += 1
                    used += 1
                    break
            for key, value in sizes.items():
                if width < key:
                    sizes[key] += 1
                    used_s += 1
                    break
            for key, value in sizes.items():
                if height < key:
                    sizes[key] += 1
                    used_s += 1
                    break
    avg_width = round(avg_width/total,2)
    avg_height = round(avg_height/total,2)
    print(ratios)
    print(sizes)
    used_s = used_s/2

    print("Samples that are in ratios range {}/{}, samples that are in sizes range {}/{}".format(used,total,used_s,total))
    print("widths min:{} max:{} avg:{}".format(min_width,max_width,avg_width))
    print("heights min:{} max:{} avg:{}".format(min_height,max_height,avg_height))
    print("All ratios: {}".format(sorted(ratios_list)))
    if print_positives_list: print(search_string)

def remove_images_from_folder(image_folder,remove_list_file,dry_run=False):
    lines = []
    counter = 0
    prompt = input("Deleting images from folder {}, are you sure? y/n: ".format(image_folder))
    if prompt != "y": return
    with open(remove_list_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    for image_file in os.listdir(image_folder):
        if image_file.endswith(".png") and image_file in lines:
            image_file_path = os.path.join(image_folder,image_file)
            if not dry_run: os.remove(image_file_path)
            counter += 1

    print("{} out of {} deleted".format(counter,len(lines)))
    return counter > 0

def bool_to_str(boolean_val):
    if boolean_val:
        return "1"
    else:
        return "0"

def bb_one_time_fix(bbox_file,outputs_folder):
    valid_names = []
    for fg_file in os.listdir(outputs_folder):
        if not fg_file.endswith("_bg.png"): continue
        valid_names.append(fg_file)

    with open(bbox_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[1:]
    with open(bbox_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL)
        csv_writer.writerow(["image", "xmin", "ymin", "xmax", "ymax", "label"])
        for index, bbox_arr in enumerate(data):
            filename = valid_names[index]
            xmin = bbox_arr[1]
            ymin = bbox_arr[2]
            xmax = bbox_arr[3]
            ymax = bbox_arr[4]
            label = bbox_arr[5]
            if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1: continue
            csv_writer.writerow([filename, xmin, ymin, xmax, ymax, label])
    print("Bounding box csv file fixing finished")

def noisy(image,mask,type): # TODO WIP seperate random noise profiles depending on certain body factors
    output = image.copy()
    dist_size = 300
    if type == "shadow":
        low, mid, high = 2, 45, 70
        ret = np.round(np.random.normal(mid, 5, dist_size))
    if type == "body":
        low, mid, high = 60, 100, 160
        ret = np.round(np.random.normal(mid, 50, dist_size))
    for i in range(mask.shape[0]):
        if np.any(mask[i, ] > 0):
            for j in range(mask.shape[1]):
                if mask[i][j] != 0:
                    index = np.random.randint(0,dist_size)
                    while ret[index] < low or ret[index] > high:
                        index = np.random.randint(0, dist_size)
                    if index % 2 == 0:
                        output[i][j][0] = ret[index] # add some noise to B G and R
                        output[i][j][1] = ret[index]
                        output[i][j][2] = ret[index]
    return output

def apply_blur(image_name): # TODO WIP apply blur
    pass

def apply_pixelation(image_name, downsample_size): # downsample size = (w,h), it takes in grayscale images
    new_img = image_name
    # new_img = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX)
    # desired output size
    height, width = new_img.shape
    # Desired "pixelated" size
    w, h = downsample_size
    # Resize input to "pixelated" size
    temp = cv2.resize(new_img, (w, h), interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    new_img = cv2.resize(temp, (width, height), interpolation=cv2.INTER_CUBIC)
    # INTER_LINEAR(low-med outline artifacts), INTER_AREA(low-med artifacts), INTER_NEAREST(high outline artifacts), INTER_CUBIC(low outline artifacts), INTER_LANCZOS4 (low outline artifacts)
    return new_img

def superimpose(input_background,input_foreground,visible_part_alpha=0.5):
    # the chunk below superimposes our transparent image onto the background
    B, G, R, A = cv2.split(input_foreground) # first convert white to alpha
    h = input_foreground.shape[0]
    w = input_foreground.shape[1]
    alpha_channel = np.zeros((1000, 1000), np.uint8)
    for y in range(0, h):
        for x in range(0, w):
            if B[x,y] != 255 and G[x,y] != 255 and R[x,y] != 255:
                alpha_channel[x,y] = int(visible_part_alpha * 255)

    input_foreground = cv2.merge((B, G, R, alpha_channel))

    alpha_background = input_background[:, :, 3] / 255.0
    alpha_foreground = input_foreground[:, :, 3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        input_background[:, :, color] = alpha_foreground * input_foreground[:, :, color] + \
                                  alpha_background * input_background[:, :, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    input_background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return input_background

def populate_backgrounds(backgrounds_folder, bg_number):
    bg_object = GlobalBackgrounds()
    applicable_backgrounds = bg_object.get_backgrounds()
    if len(applicable_backgrounds) == bg_number: return
    backgrounds_bag = []
    for bg_file in os.listdir(backgrounds_folder):
        if bg_file.endswith(".png"):
            background_file = os.path.join(backgrounds_folder, bg_file)
            backgrounds_bag.append(background_file)
    while(len(applicable_backgrounds) < bg_number):
        random_index = random.randint(0,len(backgrounds_bag)-1)

        applicable_backgrounds.append(backgrounds_bag[random_index])
    random.shuffle(backgrounds_bag)
    bg_object.set_backgrounds(backgrounds_bag)
    return bg_object

def render_to_background(renders_folder,backgrounds_object,n_neg_per_pos=2,config_dict=None, samples_counter=0,output_root=None):
    # n_neg_per_pos is every positive sample also causes 2 negative samples, so the ratios is calculated as n_neg_per_pos+1 (default 1 in 3 pos to neg ratio)
    list_of_anchors = [] # bbox info goes here (fully automated annotation)
    if output_root is None:
        output_root = "{}/../outputs".format(renders_folder)

    debug_mode = False
    poses_in_seperate_folders = False
    save_img = True
    show_image_mode = False

    if config_dict is None: # if noises to apply arent specified, apply all
        do_pixelation = True
        do_salt_and_pepper_noise = True
        do_alpha_blending = True
    else:
        do_pixelation = config_dict["do_pixelation"]
        do_salt_and_pepper_noise = config_dict["do_salt_and_pepper_noise"]
        do_alpha_blending = config_dict["do_alpha_blending"]

    samples_len = len(os.listdir(renders_folder)[samples_counter:]) # the program never crashed, but if it did, you can resume by specifying a sample counter value for the render where it left off
    print_len = samples_len+samples_counter # just to accurately print progress if the program crashed and you specified a sample counter to resume at
    for fg_file in os.listdir(renders_folder)[samples_counter:]:
        if not fg_file.endswith(".png"): continue

        print("Processing {}/{}".format(samples_counter, print_len))
        render_file = os.path.join(renders_folder, fg_file)
        current_render = render_file
        current_bg = backgrounds_object.get_background() # from the list of supplied backgrounds with NO positive samples (only negatives) get a random one
        background = cv2.imread(current_bg)

        foreground_bbox = cv2.imread(current_render, cv2.IMREAD_UNCHANGED) # read the render image into an opencv matrix with the alpha channel

        B, G, R, A = cv2.split(foreground_bbox)   # makes transparency white, for automatic bbox detection
        alpha = A / 255
        R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)
        foreground_whitealpha = cv2.merge((B, G, R))
        foreground_whitealpha = cv2.cvtColor(foreground_whitealpha, cv2.COLOR_BGR2GRAY)
        #print("Unique gray values in this image: {}".format(np.unique(foreground_whitealpha))) # useful for finding background - body - shadow tresholding limits

        grey_tresh_min = 50  # below this pixel value is shadow
        grey_tresh_max = 200  # above this pixel value is background
        ret_bm, foreground_only_shadow_mask = cv2.threshold(foreground_whitealpha, grey_tresh_min, 255, cv2.THRESH_BINARY_INV)  # only shadow is white
        ret_bm, foreground_mask = cv2.threshold(foreground_whitealpha, grey_tresh_max, 255, cv2.THRESH_BINARY_INV)  # entire body is white
        foreground_only_body_mask = cv2.bitwise_xor(foreground_only_shadow_mask,foreground_whitealpha) # extract out of the above only body by doing a xor
        ret_bm, foreground_only_body_mask = cv2.threshold(foreground_only_body_mask, grey_tresh_max, 255, cv2.THRESH_BINARY_INV)
        h = foreground_mask.shape[0]
        w = foreground_mask.shape[1]
        # loop over the image, pixel by pixel
        ymin, ymax, xmin, xmax = -1, -1, -1, -1
        for y in range(0, h): # bounding box calculation
            row = foreground_mask[y,:]
            mask = row > 200
            if np.any(mask):
                if ymin == -1: ymin = y
                ymax = y
        for x in range(0, w):
            column = foreground_mask[:,x]
            mask = column > 200
            if np.any(mask):
                if xmin == -1: xmin = x
                xmax = x

        foreground_pixelated = foreground_whitealpha
        if do_pixelation:
            foreground_pixelated = apply_pixelation(foreground_whitealpha, (700, 700)) # apply pixelation, for a more rough looking body

        body_locs = np.where(foreground_only_body_mask != 0)
        shadow_locs = np.where(foreground_only_shadow_mask != 0)
        white_image = np.ones((1000,1000), np.uint8)
        white_image = white_image * 255
        foreground_body = white_image.copy()
        foreground_shadow = white_image.copy()
        foreground_body[body_locs[0], body_locs[1]] = foreground_pixelated[body_locs[0], body_locs[1]] # its named pixelated even though pixelation isnt necessarily applied in different noise profiles
        foreground_shadow[shadow_locs[0], shadow_locs[1]] = foreground_pixelated[shadow_locs[0], shadow_locs[1]]

        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGRA)
        foreground_body = cv2.cvtColor(foreground_body, cv2.COLOR_GRAY2BGRA)
        foreground_shadow = cv2.cvtColor(foreground_shadow, cv2.COLOR_GRAY2BGRA)

        if do_salt_and_pepper_noise: # input has to be BGRA
            background = noisy(background, mask=foreground_only_shadow_mask, type="shadow")
            background = noisy(background, mask=foreground_only_body_mask, type="body")

        if do_alpha_blending: # input has to be BGRA
            background = superimpose(background, foreground_body, visible_part_alpha=0.3)
            background = superimpose(background, foreground_shadow, visible_part_alpha=0.5)
        else:
            background = superimpose(background, foreground_body, visible_part_alpha=1)
            background = superimpose(background, foreground_shadow, visible_part_alpha=1)

        # background = cv2.rectangle(background, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=1) # bbox visualization
        background = cv2.cvtColor(background, cv2.COLOR_BGRA2GRAY)

        fg_prefix = fg_file.replace(".png", "")
        fg_prefix = "{}_pix{}_alpha{}_spnoise{}".format(fg_prefix,bool_to_str(do_pixelation),bool_to_str(do_alpha_blending),bool_to_str(do_salt_and_pepper_noise))
        fg_prefix = "{:04d}_{}".format(samples_counter,fg_prefix)

        img_name = "{}_bg.png".format(fg_prefix)
        if poses_in_seperate_folders: img_loc = "{}/{}/{}".format(output_root,fg_prefix,img_name)
        img_loc_all = "{}/all/{}".format(output_root,img_name)

        if poses_in_seperate_folders and not os.path.exists("{}/{}/".format(output_root,fg_prefix)):
            os.makedirs("{}/{}/".format(output_root,fg_prefix))
        if not os.path.exists("{}/all/".format(output_root)):
            os.makedirs("{}/all/".format(output_root))

        if save_img and poses_in_seperate_folders: cv2.imwrite(img_loc, background,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        if save_img:
            pos_sample = background
            cv2.imwrite(img_loc_all, pos_sample,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            for x in range(n_neg_per_pos):
                neg_sample = "{:04d}_pose_neg_sample_{}.png".format(samples_counter,x+1)
                neg_sample_loc_all = "{}/all/{}".format(output_root, neg_sample)
                rand_bg = backgrounds_object.get_background()
                rand_bg = cv2.imread(rand_bg)
                rand_bg = cv2.cvtColor(rand_bg, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(neg_sample_loc_all,rand_bg,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        print("{} Done".format(img_loc_all))
        bbox_dict = {
            "image": img_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": "confirmed_body"
        }
        list_of_anchors.append(bbox_dict)
        samples_counter += 1
        if show_image_mode or (debug_mode and (xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1)):
            # display the image
            if show_image_mode:
                cv2.imshow("Transparent to white image", foreground_whitealpha)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow("Background with pixelation({}) and body and shadow alpha({}) and salt and pepper noise({})".format(do_pixelation,do_alpha_blending,do_salt_and_pepper_noise), background)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print("Synthetic data generation finished {} samples generated".format(samples_counter))

    bbox_file = "{}/bounding_boxes.csv".format(output_root)
    if not os.path.isfile(bbox_file):
        print("Created bbox file since it didnt exist")
        file = open(bbox_file, "a+")
        file.close()  # just create the file if it doenst exist
    with open(bbox_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',quotechar='\"', quoting=csv.QUOTE_ALL)
        csv_writer.writerow(["image","xmin","ymin","xmax","ymax","label"])
        for bbox in list_of_anchors:
            filename = bbox["image"]
            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]
            label = bbox["label"]
            if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1: continue
            csv_writer.writerow([filename,xmin,ymin,xmax,ymax,label])
    print("Bounding box csv file generation finished")

def visualize_bboxfile(bbox_file_path,image_file_folder):
    with open(bbox_file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[1:]
    eligible_img_names = []
    for img_file in os.listdir(image_file_folder):
        if not img_file.endswith(".png"): continue
        eligible_img_names.append(img_file)
    for line in data:
        if line[0] not in eligible_img_names: continue
        img_path = "{}/{}".format(image_file_folder, line[0])
        pic = cv2.imread(img_path)
        pic = cv2.rectangle(pic, (int(float(line[1])), int(float(line[2]))), (int(float(line[3])), int(float(line[4]))), (0,0,255), 1) # blue green red
        cv2.imshow("bboxes visualized", pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def calculate_norm_and_std(image_folder_path):
    matrix_mean = []
    for index, img_file in enumerate(os.listdir(image_folder_path)):
        if not img_file.endswith(".png"): continue
        if index % 800 == 0: print(index)
        img_file_read = os.path.join(image_folder_path, img_file)

        image = cv2.imread(img_file_read)
        image = image.astype(np.uint8)
        img = np.transpose(image, [2, 0, 1])
        img = img / 255  # 0 to 1 range
        matrix_mean.append(np.mean(np.array(img[0]))) # its grayscale so all channels are the same
    print("Images mean: {}".format(np.mean(matrix_mean))) # for 500 its 0.2253309, for full train set its: 0.19964
    print("Images std: {}".format(np.std(matrix_mean)))   # for 500 its 0.1186108, for full train set its: 0.0582186

def logit2prob(logit):
    odds = math.exp(logit)
    prob = odds / (1 + odds)
    return prob


def prepare_ablation_folders():
    root_folder = 'F:/projekti/msc_sonar_models/synthetic_datasets/train_synth/outputs_base_train/'
    dev_root_folder = 'F:/projekti/msc_sonar_models/synthetic_datasets/dev_synth/outputs_base_dev/'

    synth_fol = root_folder+"all/" # base noise samples
    synth_dev_fol = dev_root_folder+"all/"
    synth_pos_filt_keyword = "_pix0_alpha1_spnoise1_bg.png" # base noise samples

    real_fol = 'C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/train/'
    real_dev_fol = 'C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/dev/'
    real_pos_list = 'C:/Users/Moji podatki/Desktop/github/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/imgs_containing_bodies_list.txt'

    real20synth80 = root_folder+"20real80synth/"
    real50synth50 = root_folder+"50real50synth/"
    real80synth20 = root_folder+"80real20synth/"
    extra_real50synth50 = root_folder+"extra_real50synth50/"
    dev_real20synth80 = dev_root_folder+"20real80synth/"
    dev_real50synth50 = dev_root_folder+"50real50synth/"
    dev_real80synth20 = dev_root_folder+"80real20synth/"
    eight = root_folder+"8000/"
    four = root_folder+"4000/"
    two = root_folder+"2000/"
    one = root_folder+"1000/"
    limb_novar = root_folder+"limb_limbvarxnone/"
    pose5 = root_folder+"pose_pose5/"
    sonar_blunt = root_folder+"sonar_blunt/"

    def ablation_data_ratio(synth_folder,synth_pos_keyword,real_folder,real_pos_list,ratio, output_folder, mode=""):
        synth_list = []
        for synth_file in os.listdir(synth_folder):
            if not synth_file.endswith(synth_pos_keyword): continue
            image_file_path = os.path.join(synth_folder, synth_file)
            synth_list.append(image_file_path)

        real_list = []
        a_file = open(real_pos_list)
        file_contents = a_file.read()
        real_pos_list = file_contents.splitlines()
        for real_file in os.listdir(real_folder):
            if real_file not in real_pos_list: continue
            image_file_path = os.path.join(real_folder, real_file)
            real_list.append(image_file_path)


        real_n = int(len(real_list)*ratio)
        synth_n = int(len(real_list)*(1-ratio))
        total = len(real_list)
        random.shuffle(real_list)
        random.shuffle(synth_list)
        if mode == "extend":
            real_n = int(len(real_list))
            print("Notice: Extended dataset, not for ablation study")

        while real_n + synth_n < total:
            if real_n < synth_n: real_n += 1
            if synth_n < real_n: synth_n += 1

        print("Real {} Synth {}, total {}".format(real_n,synth_n,total))
        new_list = real_list[:real_n] + synth_list[:synth_n]
        print(len(new_list))
        for file_path in new_list:
            file_name = os.path.basename(file_path)
            destination = output_folder+file_name
            shutil.copyfile(file_path, destination)

    def ablation_data_size(synth_folder, size, output_folder):
        synth_list = []
        for synth_file in os.listdir(synth_folder):
            image_file_path = os.path.join(synth_folder, synth_file)
            synth_list.append(image_file_path)
        random.shuffle(synth_list)
        synth_list = synth_list[:size]
        for file_path in synth_list:
            file_name = os.path.basename(file_path)
            destination = output_folder + file_name
            shutil.copyfile(file_path, destination)

    def ablation_data_filter(synth_folder, filename_include_keyword, output_folder):
        synth_list = []
        for synth_file in os.listdir(synth_folder):
            if filename_include_keyword not in synth_file: continue
            image_file_path = os.path.join(synth_folder, synth_file)
            synth_list.append(image_file_path)
        for file_path in synth_list:
            file_name = os.path.basename(file_path)
            destination = output_folder + file_name
            shutil.copyfile(file_path, destination)

    #ablation_data_ratio(synth_dev_fol, synth_pos_filt_keyword, real_dev_fol, real_pos_list, 0.2, dev_real20synth80)
    #ablation_data_ratio(synth_dev_fol, synth_pos_filt_keyword, real_dev_fol, real_pos_list, 0.5, dev_real50synth50)
    #ablation_data_ratio(synth_dev_fol, synth_pos_filt_keyword, real_dev_fol, real_pos_list, 0.8, dev_real80synth20)
    #ablation_data_ratio(synth_fol, synth_pos_filt_keyword, real_fol, real_pos_list, 0.2, real20synth80)
    #ablation_data_ratio(synth_fol, synth_pos_filt_keyword, real_fol, real_pos_list, 0.5, real50synth50)
    #ablation_data_ratio(synth_fol, synth_pos_filt_keyword, real_fol, real_pos_list, 0.8, real80synth20)
    #ablation_data_ratio(synth_fol, synth_pos_filt_keyword, real_fol, real_pos_list, 0, extra_real50synth50, mode="extend")

    #ablation_data_size(synth_fol, 8000, eight)
    #ablation_data_size(synth_fol, 4000, four)
    #ablation_data_size(synth_fol, 2000, two)
    #ablation_data_size(synth_fol, 1000, one)

    #ablation_data_filter(synth_fol, "_limbvarxnone_", limb_novar)
    #ablation_data_filter(synth_fol, "_pose5_", pose5)
    #ablation_data_filter(synth_fol, "_blunt_", sonar_blunt)
    # after this fill the folders up with negatives from real train


prepare_ablation_folders() # then fill the relevant folders with negativves
#anchor_box_analyze("C:/Users/zanza/Desktop/MSC_work/Msc_Obj_Det/data/vott/run3_big/output/vott-csv-export/06_02_2022_BIG-export_before_synth_added.csv")
#print(logit2prob(-37.4874))
#print(torch.sigmoid(torch.tensor(-37.4874,dtype=torch.float)))

# ['09-06-2020%20Velden_data_likely_containing_targets_port_training_image_1000x1000_203.png',
# '09-06-2020%20Velden_data_likely_containing_targets_starboard_training_image_1000x1000_201.png',
# '09-06-2020%20Velden_data_likely_containing_targets_starboard_training_image_1000x1000_73.png',
# '29-06-2020%20winterswijk_data_likely_containing_targets_port_training_image_1000x1000_150.png',
# 'maarseveen_data_likely_containing_targets_starboard_training_image_1000x1000_29.png',
# 'maarseveen_data_likely_containing_targets_starboard_training_image_1000x1000_29.png']

