import bpy
import os
import numpy as np
import math
import mathutils
import cv2
import subprocess
import sys
import random


sys.path.append( 'C:/Users/zanza/Desktop/MSC_work/Msc_Obj_Det/' )
import renders_synthetic_data_processing as rsdp

def set_pose(pose_index):
    deselect_all()
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    hbp = bpy.data.objects['metarig'].pose # human body pose (hbp)
    # populate moveable body part bones
    
    # ALL BONES (too many)
    hips = hbp.bones['spine']
    spine = hbp.bones['spine.001']
    l_thigh = hbp.bones['thigh.L']
    r_thigh = hbp.bones['thigh.R']
    low_chest = hbp.bones['spine.002']
    up_chest = hbp.bones['spine.003']
    l_thigh_low = hbp.bones['shin.L']
    r_thigh_low = hbp.bones['shin.R']
    l_foot = hbp.bones['foot.L']
    r_foot = hbp.bones['foot.R']
    neck = hbp.bones['spine.004']  # this is useful for rotations of the neck
    head = hbp.bones['spine.005']
    l_shoulder = hbp.bones['shoulder.L']
    r_shoulder = hbp.bones['shoulder.R']
    l_up_arm = hbp.bones['upper_arm.L']
    r_up_arm = hbp.bones['upper_arm.R']
    l_forearm = hbp.bones['forearm.L']
    r_forearm = hbp.bones['forearm.R']
    l_hand = hbp.bones['hand.L']
    r_hand = hbp.bones['hand.R']
    
    # IK BONES (or more useful bones) #
    big_bone = hbp.bones['Bone'] # anchor to move the entire body and everything globally (ignores ik constraints)
    neck = hbp.bones['spine.004']  # this is useful for rotations of the neck
    spine = hbp.bones['spine'] # base body, parent
    ik_r_heel = hbp.bones['IK.heel.R']
    ik_l_heel = hbp.bones['IK.heel.L']
    ik_r_hand = hbp.bones['IK.hand.R']
    ik_l_hand = hbp.bones['IK.hand.L']
    
    #IK ANCHORS # 
    ik_r_heel_anch = hbp.bones['IK.knee.R']
    ik_l_heel_anch = hbp.bones['IK.knee.L']
    ik_r_hand_anch = hbp.bones['IK.elbow.R']
    ik_l_hand_anch = hbp.bones['IK.elbow.L']
    
    # main poses: 
    # 1.lying on back 
    # 2.lying on stomach
    # 3.lying on right side
    # 4.lying on left side 
    
    # 5.suspended in water face down (most common, due to stomach bloating the body turns so that it is facing down)
    # almost no air in body (check nzdl site for images)
    # some air in body
    # air in body 
    
    # 6.suspended in water face up (rare irl)
    
    # clothing varieties (sonar reflective/non reflective)
    # "noise" into positioning cases and movements, to produce even more slightly different readings
    
    # relevant read: http://www.nzdl.org/cgi-bin/library?e=d-00000-00---off-0aedl--00-0----0-10-0---0---0direct-10---4-------0-1l--11-en-50---20-about---00-0-1-00-0--4----0-0-11-10-0utfZz-8-00&cl=CL1.2&d=HASHb1391626134e8593d86a.14.5&gt=1
    
    reset_body_pos()
    
    main_pose = pose_index
    base_name = "pose{}_".format(main_pose)
    '''
    if main_pose == -1: # POSE NOT APPLICABLE; SKIP
        case = random.randint(1,3)
        if case == 1: rotate_obj(neck,(-6.88033,-0.147892,0.363392)) # head facing up
        if case == 2: rotate_obj(neck,(-13.516,84.3889,-7.39118)) # head facing left
        if case == 3: rotate_obj(neck,(38.8165,-98.2523,-46.5987)) # head facing right
        base_name = "{}head{}_".format(base_name,case)
        case = random.randint(1,3)
        if case == 1: pass # body is making normal contact with ground
        if case == 2: move_obj(big_bone,(0.0,0.001331,-0.034491)) # body is deeper inside floor
        if case == 3: move_obj(big_bone,(0.0,0.002889,-0.074915)) # body is very deep inside floor
        if case == 4: move_obj(big_bone,(0.0,0.004458,-0.11559)) # body is almost covered
        base_name = "{}groundcntct{}_".format(base_name,case)
        case = random.randint(1,4)
        if case == 1: pass # there is no bloating
        if case == 2: 
            move_obj(spine,(0,-0.014362,0.070285),mode="increment") # bloating is occuring
            rotate_obj(spine,(-7.18552,0.000005,-0.000003))
        if case == 3: 
            move_obj(spine,(-1.1493e-08,0.002004,0.145679),mode="increment") # very noticeable bloating is occuring
            rotate_obj(spine,(-10.6366,0.000008,-0.000004))
        if case == 4:
            move_obj(spine,(-3.06414e-08,0.029272,0.271292),mode="increment") # significant bloating is occuring
            rotate_obj(spine,(-19.8858,0.000015,-0.000008))
        base_name = "{}bloating{}_".format(base_name,case)
            
    if main_pose == -1: # POSE NOT APPLICABLE; SKIP
        rotate_obj(big_bone,(-6.66301,180,0)) # put body on stomach
        move_obj(ik_l_hand,(-0.002707,-0.130974,-0.107922))
        move_obj(ik_r_hand,(0.000987,-0.127867,-0.108482))
        case = random.randint(1,3)
        if case == 1: rotate_obj(neck,(8.2245,0.2116,-0.316992)) # head facing down
        if case == 2: rotate_obj(neck,(30.959,58.6218,26.5363)) # head facing left
        if case == 3: rotate_obj(neck,(28,-60.8972,-25.2778)) # head facing right
        base_name = "{}head{}_".format(base_name,case)
        case = random.randint(1,3)
        if case == 1: pass # body is making normal contact with ground
        if case == 2: move_obj(big_bone,(0.0,0.001331,-0.034491)) # body is deeper inside floor
        if case == 3: move_obj(big_bone,(0.0,0.002889,-0.074915)) # body is very deep inside floor
        if case == 4: move_obj(big_bone,(0.0,0.004458,-0.11559)) # body is almost covered
        base_name = "{}groundcntct{}_".format(base_name,case)
        case = random.randint(1,4)
        if case == 1: pass # there is no bloating
        if case == 2: 
            move_obj(spine,(0,-0.014362,-0.070285),mode="increment") # bloating is occuring
            rotate_obj(spine,(7.18552,0.000005,-0.000003))
        if case == 3: 
            move_obj(spine,(-1.1493e-08,0.002004,-0.145679),mode="increment") # very noticeable bloating is occuring
            rotate_obj(spine,(10.6366,0.000008,-0.000004))
        if case == 4:
            move_obj(spine,(-3.06414e-08,0.029272,-0.271292),mode="increment") # significant bloating is occuring
            rotate_obj(spine,(19.8858,0.000015,-0.000008))
        base_name = "{}bloating{}_".format(base_name,case)
    '''
    if main_pose == 1:
        rotate_obj(big_bone,(0,-90,0)) # put body on right side
        move_obj(big_bone,(0,-0.006503,0.168619))
        
        move_obj(ik_r_heel,(0.038994,0,0.092217))
        move_obj(ik_l_heel,(-0.340266,0.149321,0.249908))
        
        move_obj(ik_r_hand,(-0.086516,-0.445278,-0.062204))
        move_obj(ik_l_hand,(0.323569,-0.424115,-0.372801))
        
        move_obj(spine,(-0.007817,0.012015,-0.005908))
        rotate_obj(neck,(0.846842,-1.49348,16.4443))
        
    if main_pose == 2:
        rotate_obj(big_bone,(0,90,0)) # put body on left side
        move_obj(big_bone,(0,-0.006295,0.163217))
        
        move_obj(ik_l_heel,(-0.011615,0,0.093706))
        move_obj(ik_r_heel,(0.358405,0.149321,0.252102))
        
        move_obj(ik_l_hand,(0.125075,-0.38039,-0.112793))
        move_obj(ik_r_hand,(-0.299793,-0.424115,-0.371667))
        
        move_obj(spine,(-0.006417,-0.00024,0))
        rotate_obj(neck,(0,1.34564,-13.0188))
    
        
    if main_pose == 5 or main_pose == 4 or main_pose == 3:
        # 1 almost no air in body  (most common)
        # 2 some air in body
        # 3 air in body (rare, possible when someone is weighed down)
        
        rotate_obj(big_bone,(0,180,0)) # put body face down
        move_obj(big_bone,(0,-0.024475,0.634623))
        
        move_obj(ik_l_heel,(0.094336,-0.58818,0.475656))
        move_obj(ik_r_heel,(-0.04154,-0.552429,0.519427))
        move_obj(ik_l_heel_anch,(0.266225,0.564204,-0.631638))
        move_obj(ik_r_heel_anch,(-0.098696,0.609837,-0.657878))
        
        move_obj(ik_l_hand,(-0.07491,-0.603619,-0.648358))
        move_obj(ik_r_hand,(-0.080987,-0.594779,-0.677487))
        move_obj(ik_l_hand_anch,(0.315841,-0.005707,0.385657))
        move_obj(ik_r_hand_anch,(-0.561492,-0.00071,0.227132488))
        
        move_obj(spine,(0.095211,-0.154213,0.2357))
        rotate_obj(spine,(8.41567,-0.000007,0.000003))
        rotate_obj(neck,(33.2038,1.26253,-1.15656))
        
    if main_pose == 4 or main_pose == 3:
        move_obj(spine,(0.095211,-0.029933,0.130611))
        rotate_obj(spine,(-29.1303,0.000022,-0.000019))
        move_obj(ik_l_hand_anch,(-0.085021,0.0321,0.511855))
        move_obj(ik_r_hand_anch,(-0.222251,0.048074,0.414634))
        
    if main_pose == 3:
        move_obj(spine,(0.095211,-0.570773,-0.315036))
        rotate_obj(spine,(-61.9236,0.000032,-0.000044))
        move_obj(ik_l_hand,(-0.134185,-0.45768,-0.15469))
        move_obj(ik_r_hand,(0.045181,-0.480865,-0.202725))
    
    case = random.randint(1,4)
    if case == 1: pass # body is making normal contact with ground
    if case == 2: move_obj(big_bone,(0,0.001331,-0.05),mode="increment") # body is deeper inside floor
    if case == 3: move_obj(big_bone,(0,0.003889,-0.1),mode="increment") # body is very deep inside floor
    if case == 4: move_obj(big_bone,(0,0.005458,-0.2),mode="increment") # even deeper
    base_name = "{}submrgd{}_".format(base_name,case)
    
    # 6.suspended in water face up is rare in real life, because of limb weight, so it is not implemented
    
    # arms have a max range of 0.6
    # legs have a max range of 0.8 
    
    ik_r_heel = hbp.bones['IK.heel.R']
    ik_l_heel = hbp.bones['IK.heel.L']
    ik_r_hand = hbp.bones['IK.hand.R']
    ik_l_hand = hbp.bones['IK.hand.L']
    limb_array = [(ik_r_heel,0.6,0.15,"rleg"),(ik_l_heel,0.6,0.15,"lleg"),(ik_r_hand,0.4,0.15,"rhand"),(ik_l_hand,0.4,0.15,"lhand")]
    # each tuple = (limb object,max range of limb,decimal chance to occur,string name)
    
    base_name = "{}limbvar".format(base_name) # base_name = "{}limbvar0_".format(base_name)
    made_limb_var = False
    for limb_tuple in limb_array:
        limb_chance = random.uniform(0, 1)
        if limb_chance < limb_tuple[2]:
            limb_obj = limb_tuple[0]
            limb_str = limb_tuple[3]
            min_range,max_range = -limb_tuple[1],limb_tuple[1]
            x_loc = random.uniform(min_range,max_range)
            z_loc = random.uniform(min_range,max_range) # y in global is z in this local coordinate system, because its relative to the pose, and its flipped
            move_obj(limb_obj,(x_loc,0,z_loc),mode="increment")
            made_limb_var = True
            base_name = "{}x{}".format(base_name,limb_str)
    if not made_limb_var:
        base_name = "{}xnone_".format(base_name)
    else:
        base_name = "{}_".format(base_name)
            
        
    return base_name

    
def move_obj(obj,xyz_tuple,mode="absolute"): # (x,y,z) to be incremented 
    deselect_all()
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    #obj.select_set(True)
    # store the current location
    loc = obj.location

    # adding adjustment values to the property
    if mode == "absolute":
        obj.location = mathutils.Vector(xyz_tuple)
    elif mode =="increment":
        obj.location = loc + mathutils.Vector(xyz_tuple)
    
def rotate_obj(obj,xyz_tuple,mode="absolute"): # (x,y,z) rotations in degrees to be incremented
    deselect_all()
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    #obj.select_set(True)
    
    obj.rotation_mode = "XYZ"
    rot = obj.rotation_euler
    for coordinate in range(3):
        if xyz_tuple[coordinate] == False: continue
        if mode == "absolute":
            obj.rotation_euler[coordinate] = math.radians(xyz_tuple[coordinate])
        elif mode =="increment":
            obj.rotation_euler[coordinate] = rot[coordinate] + math.radians(xyz_tuple[coordinate])

def reset_body_pos(): # only in object mode, bones can not be reset in this way
    deselect_all()
    bones = bpy.data.objects['metarig'].pose.bones
    hbp = bones['Bone']
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    for obj in bones:
        obj.bone.select = True
    
    bpy.ops.pose.transforms_clear()

def deselect_all():
    for obj in bpy.data.objects:
        obj.select_set(False)

def randomize_sonar_angle(default=False): #effectively moves the light source up and down the z axis, which is a proxy for boat distance
    deselect_all()
    spot = bpy.data.objects['Spot']
    bpy.data.objects['Spot'].select_set(True)
    bpy.ops.object.mode_set(mode="OBJECT")
    # C.object.data.cutoff_distance = 100 # fixes cycles render engine light distance
    # C.object.data.falloff_type = "CONSTANT"
    angle = ""
    distance = random.randint(1,4)
    if distance == 4:
        z_pos = random.uniform(50, 80)
        angle = "vblunt"
    elif distance == 3:
        z_pos = random.uniform(30,50)
        angle = "blunt"
    elif distance == 2:
        z_pos = random.uniform(20,30)
        angle = "sharp"
    else:
        z_pos = random.uniform(15,20)
        angle = "vsharp"
    #body_y = bpy.data.objects['LightTrackObj'].location[1] # so that sonar rays are always perpendicular to body, like in real images
    #print(body_y) # i added a constraint trough the GUI, so the 0 y here gets ignored anyway, and the Y of the body is copied
    if not default:
        spot.location = mathutils.Vector((34.3238,0,z_pos))
        return angle
    if default: 
        spot.location = mathutils.Vector((34.3238,0,35)) # 35 z is default
        return "blunt"
    
def randomize_body_rotation():
    deselect_all()
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    hbp = bpy.data.objects['metarig'].pose
    big_bone = hbp.bones['Bone']
    rand_degrees = random.randint(0,360)
    rotate_obj(big_bone,(False,False,rand_degrees))
    return "zdeg{}_".format(rand_degrees)

def randomize_body_position():
    deselect_all()
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    hbp = bpy.data.objects['metarig'].pose
    big_bone = hbp.bones['Bone']
    rand_x = random.uniform(-7, 7)
    rand_y = random.uniform(-7, 8) # -7 = -0.2z | 8 = 0.38z
    yRange = 8 + 7
    zRange = 0.38 + 0.2  
    newZ = (((rand_y + 7) * zRange) / yRange) - 0.2 # this fixes a bug where
    # even though the only movement axis was y, z is modified anyway
    # this offset fixes that
    current_z = big_bone.location[2] + newZ
    print("Move: y:{}, calc z:{}".format(rand_y,newZ))
    move_obj(big_bone,(rand_x,rand_y,current_z))
    ret_string = "x{}_y{}_".format(round(rand_x, 2),round(rand_y, 2))
    ret_string = ret_string.replace("-","neg").replace(".","dot")
    return ret_string

def randomize_body_scaling():
    if random.randint(0,1) == 0:
        return "scale1_"
    scale = random.uniform(0.5, 1.1)
    # 0.5 to 1.1 range in regard to original
    deselect_all()
    bpy.data.objects['metarig'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.transform.resize(value=(scale, scale, scale))
    deselect_all()
    ret_str = "scale{}_".format(round(scale,2)).replace(".","dot")
    return ret_str

def render_scene(image_name,debug=False,frame=1):
    output_file = image_name
    output_file = "{}.png".format(output_file)
    if debug:
        bpy.context.scene.camera = bpy.context.scene.objects["SideViewDebugCamera"]
    else:
        bpy.context.scene.camera = bpy.context.scene.objects["Camera"]
    bpy.context.scene.render.filepath = output_file
    bpy.context.scene.render.resolution_x = 1000 #perhaps set resolution in code
    bpy.context.scene.render.resolution_y = 1000

    scene = bpy.context.scene
    scene.frame_set(frame)
    bpy.ops.render.render(write_still=True)

def generate_renders(output_folder,debug_folder):
    #widths of gt boxes min:23.80952380952381 max:603.3549783549784 avg:127.32661145509879
    #heights of gt boxes min:17.991004497751078 max:231.60173160173156 avg:50.13107902766274
    print("Start script")
    existing_images_list = []
    randomize_sonar_angle(default=True)
    do_debug_render = False
    
    do_render = True
    dynamic_sonar_angle = True
    dynamic_body_rotation = True
    dynamic_body_position = True
    dynamic_body_scaling = True
    n_images_to_generate = 869 #10000 for final
    
    # pos 5 is most common irl
    for x in range(n_images_to_generate): 
        print("Started {}/{}...".format(x,n_images_to_generate))
        pose_chance = random.uniform(0, 1)
        if pose_chance < 0.40:
            random_pose = 5
        elif pose_chance < 0.60:
            random_pose = 4
        elif pose_chance < 0.80:
            random_pose = 3
        elif pose_chance < 0.90:
            random_pose = 2 
        elif pose_chance < 1:
            random_pose = 1
        # 5 is most likely, 4 and 3 is less, 1 and 2 are rare irl
        output_name = set_pose(random_pose)
        debug_name = output_name
        
        if dynamic_body_rotation:
            degrees = randomize_body_rotation()
            output_name = "{}{}".format(output_name,degrees)
            debug_name = output_name
        if dynamic_body_position:
            x_y = randomize_body_position()
            output_name = "{}{}".format(output_name,x_y)
        if dynamic_body_scaling:
            scale_str = randomize_body_scaling()
            output_name = "{}{}".format(output_name,scale_str)
        if dynamic_sonar_angle: 
            angle = randomize_sonar_angle()
            output_name = "{}{}".format(output_name,angle)
            
        if do_render and output_name not in existing_images_list:
            output_file = "{}{}".format(output_folder,output_name)
            if do_debug_render and debug_name not in '\t'.join(existing_images_list):
                output_file_debug = "{}{}".format(debug_folder,debug_name)
                render_scene(output_file_debug,debug=True,frame=50)
            render_scene(output_file,frame=50)
            existing_images_list.append(output_name)
        print("DONE: {}".format(output_name))


save_folder = 'C:/Users/zanza/Desktop/predictions/renders/generated/transparent_bg/renders/'
save_debug_folder = 'C:/Users/zanza/Desktop/predictions/renders/generated/transparent_bg/debug/'
bg_folder = 'C:/Users/zanza/Desktop/predictions/renders/generated/transparent_bg/bg_train_synthetic/'
#bg_folder = 'C:/Users/zanza/Desktop/predictions/renders/generated/transparent_bg/bg/' # for testing

print("---------Running new version with bg object-----------")
#generate_renders(save_folder,save_debug_folder)


if True: # to debug set this to false

    config_dict_1 = {
        "do_pixelation": False,
        "do_salt_and_pepper_noise": True,
        "do_alpha_blending": True,
    }
    config_dict_2 = {
        "do_pixelation": True,
        "do_salt_and_pepper_noise": True,
        "do_alpha_blending": True,
    }
    config_dict_3 = {
        "do_pixelation": False,
        "do_salt_and_pepper_noise": False,
        "do_alpha_blending": True,
    }

    bg_object = rsdp.populate_backgrounds(bg_folder,5001)
    # render_to_background("renders_folder",background_object)

    # first base is alpha -> s&p, second is pixel -> alpha -> s&p, third is just alpha
    rsdp.render_to_background(save_folder,bg_object,output_root="D:/generated_transparent_bg/outputs_first",config_dict=config_dict_1)

    #rsdp.render_to_background(save_folder,bg_object,output_root="D:/generated_transparent_bg/outputs_second",config_dict=config_dict_2)

    #rsdp.render_to_background(save_folder,bg_object,output_root="D:/generated_transparent_bg/outputs_third",config_dict=config_dict_3)