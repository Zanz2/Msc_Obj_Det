import copy
import os
import csv
import cv2
import numpy as np
import pandas as pd

def anchor_box_analyze():
    csv_boxes = pd.read_csv("data/vott/06_02_2022_BIG-export.csv")
    ratios = {
        0.15: 0,
        0.24: 0,
        0.33: 0,
        0.5: 0,
        0.66: 0,
        1: 0,
        1.5: 0,
        2: 0,
    }
    # anchor_sizes = ((25,), (75,), (150,), (300,),(400,))
    sizes = {
        20: 0,
        40: 0,
        60: 0,
        90: 0,
        280: 0,
    }
    used = 0
    used_s = 0
    total = 0
    search_string = ""

    max_width = 0
    max_height = 0
    min_width = 1000
    min_height = 1000
    avg_width = 0
    avg_height = 0
    for index, asset in csv_boxes.iterrows():
        if asset["label"] == "confirmed_body":
            search_string = "{}\n{}".format(search_string,asset["image"])
            width = asset["xmax"] - asset["xmin"]
            height = asset["ymax"] - asset["ymin"]
            ratio = height/width
            total += 1

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
    avg_width = avg_width/total
    avg_height = avg_height/total
    print(ratios)
    print(used)
    print(total)
    print(sizes)
    used_s = used_s/2
    print(used_s)
    print("widths min:{} max:{} avg:{}".format(min_width,max_width,avg_width))
    print("heights min:{} max:{} avg:{}".format(min_height,max_height,avg_height))
    print(search_string)

def render_to_background(renders_folder,backgrounds_folder):
    background_images = []
    render_images = []
    list_of_anchors = []
    output_root = "{}/../outputs".format(renders_folder)
    for bg_file in os.listdir(backgrounds_folder):
        if bg_file.endswith(".png"):
            background_file = os.path.join(backgrounds_folder, bg_file)
            current_bg = background_file
            for fg_file in os.listdir(renders_folder):
                if fg_file.endswith(".png"):
                    render_file = os.path.join(renders_folder, fg_file)
                    current_render = render_file

                    background = cv2.imread(current_bg)
                    foreground = cv2.imread(current_render, cv2.IMREAD_UNCHANGED)
                    foreground_bbox = cv2.imread(current_render, cv2.IMREAD_UNCHANGED)

                    B, G, R, A = cv2.split(foreground_bbox)
                    alpha = A / 255
                    R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
                    G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
                    B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)
                    foreground_bbox = cv2.merge((B, G, R))
                    #ret_bm, foreground_bbox = cv2.threshold(foreground_bbox, 90, 255,cv2.THRESH_BINARY)  # bodies should have more alpha blend in this method than shadows, this is why things get tricky
                    #foreground_bbox = cv2.cvtColor(foreground_bbox, cv2.COLOR_BGR2GRAY)
                    h = foreground_bbox.shape[0]
                    w = foreground_bbox.shape[1]
                    # loop over the image, pixel by pixel
                    ymin, ymax, xmin, xmax = -1, -1, -1, -1
                    for y in range(0, h): # bounding box calculation
                        row = foreground_bbox[y,:]
                        mask = row < 127
                        if np.any(mask):
                            if ymin == -1: ymin = y
                            ymax = y
                    for x in range(0, w):
                        column = foreground_bbox[:,x]
                        mask = column < 127
                        if np.any(mask):
                            if xmin == -1: xmin = x
                            xmax = x

                    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGRA)
                    background_original = copy.deepcopy(background)
                    foreground = cv2.cvtColor(foreground, cv2.COLOR_RGBA2BGRA)

                    # normalize alpha channels from 0-255 to 0-1
                    alpha_background = background[:, :, 3] / 255.0
                    alpha_foreground = foreground[:, :, 3] / 255.0

                    # set adjusted colors
                    for color in range(0, 3):
                        background[:, :, color] = alpha_foreground * foreground[:, :, color] + \
                                                  alpha_background * background[:, :, color] * (1 - alpha_foreground)

                    # set adjusted alpha and denormalize back to 0-255
                    background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
                    alpha_blend = 0.5
                    red = (0, 0, 255)
                    background = cv2.addWeighted(background,alpha_blend,background_original,1-alpha_blend,0)
                    #background = cv2.rectangle(background, (xmin,ymin), (xmax,ymax), red, thickness=2) # bbox visualization
                    background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)


                    fg_prefix = fg_file.replace(".png","")
                    bg_prefix = bg_file.split("_")[0]

                    img_loc = "{}/{}/{}_bg_{}.png".format(output_root,fg_prefix,fg_prefix,bg_prefix)
                    img_loc_all = "{}/all/{}_bg_{}.png".format(output_root,fg_prefix,bg_prefix)
                    img_name = "{}_bg_{}.png".format(fg_prefix,bg_prefix)
                    # if not os.path.exists("{}/{}/".format(output_root,fg_prefix)):
                    #    os.makedirs("{}/{}/".format(output_root,fg_prefix))
                    if not os.path.exists("{}/all/".format(output_root)):
                        os.makedirs("{}/all/".format(output_root))
                    #cv2.imwrite(img_loc, background,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    cv2.imwrite(img_loc_all, background,[cv2.IMWRITE_PNG_COMPRESSION, 0])


                    bbox_dict = {
                        "image": img_name,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "label": "confirmed_body"
                    }
                    list_of_anchors.append(bbox_dict)
                    if True or xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                        # display the image
                        print(background.shape)
                        cv2.imshow("Composited image", foreground)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image", foreground_bbox)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image", background)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        #sys.exit(0)
            print("Synthetic data for background file {} finished".format(bg_file))

    bbox_file = "{}/bounding_boxes.csv".format(output_root)
    if not os.path.isfile(bbox_file):
        print("Created bbox file since it didnt exist")
        file = open(bbox_file, "a+")
        file.write('"image","xmin","ymin","xmax","ymax","label"\n')
        file.close()  # just create the file if it doenst exist
    with open(bbox_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',quotechar='\"', quoting=csv.QUOTE_ALL)
        for bbox in list_of_anchors:
            filename = bbox["image"]
            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]
            label = bbox["label"]
            if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1: continue
            csv_writer.writerow([filename,xmin,ymin,xmax,ymax,label])


#anchor_box_analyze()
render_to_background("../../predictions/renders/generated/transparent_bg/renders/","../../predictions/renders/generated/transparent_bg/bg")
