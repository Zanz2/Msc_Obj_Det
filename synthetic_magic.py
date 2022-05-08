import copy
import os
import csv
import cv2
import numpy as np
import pandas as pd
import sys

def anchor_box_analyze():
    csv_boxes = pd.read_csv("data/vott/06_02_2022_BIG-export.csv")
    ratios = {0.15: 0,0.24: 0,0.33: 0,0.5: 0,0.66: 0,1: 0,1.5: 0,2: 0}
    # anchor_sizes = ((25,), (75,), (150,), (300,),(400,))
    sizes = {20: 0,40: 0,60: 0,90: 0,280: 0}
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
    #print(search_string)

def noisy(image):
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.1
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out

def apply_noise(image_name, display=False):
    file_path_dest = 'C:/Users/zanza/Desktop/predictions/renders/'
    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)

    noise_images = []
    noise_images.append([img, (50, 50)])
    noise_images.append([img, (100, 100)])
    noise_images.append([img, (150, 150)])
    noise_images.append([img, (200, 200)])

    for noise_image, down_res in noise_images:
        new_img = noise_image
        target_avg_val = 100  # same values used in original jsf to image preprocessing

        # new_img = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX)
        new_img = new_img.astype(np.uint8)

        # desired output size
        new_img = noisy(new_img)

        height, width = (200, 200)
        # Desired "pixelated" size
        w, h = down_res
        # Resize input to "pixelated" size
        temp = cv2.resize(new_img, (w, h), interpolation=cv2.INTER_LINEAR)
        # Initialize output image
        new_img = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        if display: cv2.imshow("Noisy image", new_img)
        if display: cv2.waitKey(0)  # this freezes and crashes for some reason
        cv2.imwrite("{}hi_avg_syn_img_{}x{}.png".format(file_path_dest, w, h), new_img)

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

def render_to_background(renders_folder,backgrounds_folder):
    background_images = []
    render_images = []
    list_of_anchors = []
    output_root = "{}/../outputs".format(renders_folder)
    counter = 0
    for bg_file in os.listdir(backgrounds_folder):
        if bg_file.endswith(".png"):
            background_file = os.path.join(backgrounds_folder, bg_file)
            current_bg = background_file
            for fg_file in os.listdir(renders_folder):
                if fg_file.endswith(".png"):

                    if counter not in [1,4,5,6,8]: # TODO DEBUG REMOVE good visual examples: 1,4,5,6,8
                        counter += 1
                        continue

                    render_file = os.path.join(renders_folder, fg_file)
                    current_render = render_file

                    background = cv2.imread(current_bg)
                    foreground_original = cv2.imread(current_render, cv2.IMREAD_UNCHANGED)
                    foreground_bbox = cv2.imread(current_render, cv2.IMREAD_UNCHANGED)

                    B, G, R, A = cv2.split(foreground_bbox)   # makes transparency white, for easier bbox detection
                    alpha = A / 255
                    R = (255 * (1 - alpha) + R * alpha).astype(np.uint8)
                    G = (255 * (1 - alpha) + G * alpha).astype(np.uint8)
                    B = (255 * (1 - alpha) + B * alpha).astype(np.uint8)
                    foreground_whitealpha = cv2.merge((B, G, R))
                    foreground_whitealpha = cv2.cvtColor(foreground_whitealpha, cv2.COLOR_BGR2GRAY)


                    grey_tresh_min = 100  # below this pixel value is shadow
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


                    body_locs = np.where(foreground_only_body_mask != 0)  # Get the non-zero mask locations, less noise prone
                    shadow_locs = np.where(foreground_only_shadow_mask != 0)
                    white_image = np.ones((1000,1000), np.uint8)
                    white_image = white_image * 255
                    foreground_body = white_image.copy()
                    foreground_shadow = white_image.copy()
                    foreground_body[body_locs[0], body_locs[1]] = foreground_whitealpha[body_locs[0], body_locs[1]]
                    foreground_shadow[shadow_locs[0], shadow_locs[1]] = foreground_whitealpha[shadow_locs[0], shadow_locs[1]]

                    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGRA)
                    foreground_body = cv2.cvtColor(foreground_body, cv2.COLOR_GRAY2BGRA)
                    foreground_shadow = cv2.cvtColor(foreground_shadow, cv2.COLOR_GRAY2BGRA)

                    background = superimpose(background, foreground_body,visible_part_alpha=0.2)
                    background = superimpose(background, foreground_shadow, visible_part_alpha=0.6)

                    background = cv2.rectangle(background, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=1) # bbox visualization
                    background = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)


                    fg_prefix = fg_file.replace(".png","")
                    bg_prefix = bg_file.split("_")[0]

                    format_for_an_email_attachment = False
                    save_img = False

                    if format_for_an_email_attachment: img_loc = "{}/{}/{}_bg_{}.png".format(output_root,fg_prefix,fg_prefix,bg_prefix)
                    img_loc_all = "{}/all/{}_bg_{}.png".format(output_root,fg_prefix,bg_prefix)
                    img_name = "{}_bg_{}.png".format(fg_prefix,bg_prefix)
                    if format_for_an_email_attachment and not os.path.exists("{}/{}/".format(output_root,fg_prefix)):
                       os.makedirs("{}/{}/".format(output_root,fg_prefix))
                    if not os.path.exists("{}/all/".format(output_root)):
                        os.makedirs("{}/all/".format(output_root))
                    if save_img and format_for_an_email_attachment: cv2.imwrite(img_loc, background,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    if save_img: cv2.imwrite(img_loc_all, background,[cv2.IMWRITE_PNG_COMPRESSION, 0])


                    bbox_dict = {
                        "image": img_name,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "label": "confirmed_body"
                    }
                    list_of_anchors.append(bbox_dict)
                    if True or (xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1):
                        # display the image
                        '''
                        cv2.imshow("Composited image", foreground_original)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image", foreground_whitealpha)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image", foreground_only_shadow_mask)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image",foreground_only_body_mask)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image", foreground_body)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        cv2.imshow("Composited image", foreground_shadow)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        '''
                        cv2.imshow("Composited image", background)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        norm = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        norm = clahe.apply(norm)
                        cv2.imshow("Composited image", norm)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        norm = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(150, 150))  # INJECTING NOISE WORKS (sort of)
                        norm = clahe.apply(norm)
                        cv2.imshow("Composited image", norm)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        sys.exit(0)
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
