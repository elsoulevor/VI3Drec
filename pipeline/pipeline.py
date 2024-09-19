import math
import numpy as np
import sys
import os
import argparse
import PIL.Image as pil
import glob
import matplotlib.pyplot as plt
import pyvista
import fast_simplification


sys.path.append("/home/vinhsp/3drecpipeline/SRD-Depth")
import utils
# import test_simple_copy
import torch
from torchvision import transforms, datasets
from stl import mesh

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    #parser.add_argument('--model_name', type=str,
    #                    help='name of a pretrained model to use')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--colormap",
                        help='Enter the depthmap s colormap',
                        default = "magma",
                        choices=[
                            "Spectral",
                            "magma",
                            "gist_gray",
                            "gray"])
    return parser.parse_args()

######### SUPPOSED TO BE THE PART FROM IMG TO DEPTH MAP ##########

def load_image(args):
    img = pil.open(args.image_path).convert('RGB')
    og_width, og_height = img.size
    return img

def load_model(args):
    utils.download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    model = None
    return model

def depth_est(paths, output_directory, model, args):
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                continue
            
            input_image = load_image(image_path)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            
            input_image = input_image.to(device)
    return

def set_dir_path(args):
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.jpg'))
        output_directory = args.image_path
    else:
        raise Exception("Can't find image or path : {}".format(args.image_path))
    return paths, output_directory

######### DEPTH MAP TO 3D ##########

def bw_to_depth(depth_map):
    bw_map = pil.open(depth_map)
    bw_map = np.array(bw_map)
    bw_map = bw_map / 255.0
    return bw_map

def depthmap_to_actualdepth(spectral_map, colormap):
    spectral_map = pil.open(spectral_map)
    spectral_map = np.array(spectral_map)
    if colormap == "gist_gray":
        #spectral_map = spectral_map / 65535.0
        spectral_map = spectral_map / 16581375.0
    else :
        spectral_map = spectral_map / 255.0
    cmap = plt.get_cmap(colormap)
    
    img = spectral_map.reshape(-1, spectral_map.shape[-1])
    unique_colors = np.unique(img, axis=0)
    
    sampled_colors = cmap(np.linspace(0, 1, unique_colors.shape[0]))[:, :3]
    
    normalized_depth = np.zeros(spectral_map.shape[:2])
    for i in range(spectral_map.shape[0]):
        for j in range(spectral_map.shape[1]):
            pixel_color = spectral_map[i, j]
            differences = np.linalg.norm(sampled_colors - pixel_color, axis=1)
            closest_color_index = np.argmin(differences)
            normalized_depth[i, j] = closest_color_index / (unique_colors.shape[0] - 1)
            #normalized_depth[i, j] = normalized_depth[i, j]/(1 + math.exp(.5-5*normalized_depth[i, j]))
            
            # If values are inverted (far is white), else comment
            normalized_depth[i, j] = 1 - normalized_depth[i, j]
                
    return normalized_depth

def scaling_factor(source_image):
    # target dimensions x = 21cm, y = 29.7cm, z = 10cm
    x, y = source_image.shape
    print(x,y)
    # if x < y: # Landscape
    #     if x/210 < y/297:
    #         target_dim = np.array([0.297, 0.297, 0.05])
    #         scaling_factor = target_dim / np.array([y, y, 1])
    #     else:
    #         target_dim = np.array([0.21, 0.21, 0.05])
    #         scaling_factor = target_dim / np.array([x, x, 1])
    # else: # Portrait
    #     if x/210 < y/297:
    #         target_dim = np.array([0.21, 0.21, 0.05])
    #         scaling_factor = target_dim / np.array([y, y, 1])
    #     else:
    #         target_dim = np.array([0.297, 0.297, 0.05])
    #         scaling_factor = target_dim / np.array([x, x, 1])
            
    if x < y: # Landscape
        if y/x > 297/210:
            target_dim = np.array([0.297, 0.297, 0.05])
            scaling_factor = target_dim / np.array([y, y, 1])
        else:
            target_dim = np.array([0.21, 0.21, 0.05])
            scaling_factor = target_dim / np.array([x, x, 1])
    else: # Portrait
        if x/y > 297/210:
            target_dim = np.array([0.21, 0.21, 0.05])
            scaling_factor = target_dim / np.array([y, y, 1])
        else:
            target_dim = np.array([0.297, 0.297, 0.05])
            scaling_factor = target_dim / np.array([x, x, 1])
            
    return scaling_factor

def generate_3d(int_depth_map):
    # input_img = pil.open(depth_map)
    img = np.array(int_depth_map)
    (x, y) = img.shape
    
    vertices = np.zeros((x*y, 3))
    for i in range(x):
        for j in range(y):
            vertices[(i*y+j)] = [i, j, img[i][j]]
    
    faces = []
    for i in range(x-1):
        for j in range(y-1):
            faces.append([j + i*y, j+1 + i*y, j+(i+1)*y])
            faces.append([j+1 + i*y, j + (i+1)*y, j+1 +(i+1)*y])
    faces = np.array(faces)
    
    mesh_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype = mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_3d.vectors[i][j] = vertices[f[j], :]
            
    # Scaling the object so that it fits within an A4 paper with 5cm thickness
    mesh_3d.vectors *= scaling_factor(int_depth_map)
    return mesh_3d

def simp_3d(path3d):
    mesh3d = pyvista.read(path3d)
    out = fast_simplification.simplify_mesh(mesh3d, target_reduction = .9)
    return out

def save_3d(mesh, path, output_dir):
    output_name = os.path.splitext(os.path.basename(path))[0]
    save_dest = os.path.join(output_dir, "{}.stl".format(output_name))
    mesh.save(save_dest)
    # simpled_3d = simp_3d(save_dest)
    # simpled_3d.save(save_dest)
    print("-> 3D model generated!")
    
import time

def pipeline(args):
    # img = load_image(args.image_path)
    
    # model = load_model()
    # depth_map = depth_est(model, img)
    start_time = time.time()
    paths, output_dir = set_dir_path(args)
    print(paths, output_dir)
    
    for img in paths:  
        int_depth_map = depthmap_to_actualdepth(paths[0], args.colormap)
        # int_depth_map = bw_to_depth(paths[0])
        
        mesh = generate_3d(int_depth_map)
        save_3d(mesh, img, output_dir)
    print("--- %s seconds ---" % (time.time() - start_time))
    return

if __name__ == '__main__':
    args = parse_args()
    pipeline(args)
