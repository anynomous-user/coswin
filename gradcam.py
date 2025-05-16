from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image,  preprocess_image
import cv2
import torch
import argparse
import numpy as np
import glob
import logging as log
from util.datahandler import datainfo, dataload
from model.model_builder import build_model
import os

def parser():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--model", type=str, default="coswin", help="Model name", choices =['vit','coswin','deit','swin'])
    parser.add_argument('--model_path', default='./saved/coswin/CIFAR10/model/checkpoint_coswin_CIFAR10.pt', type=str, help='checkpoint path', choices=['CIFAR10', 'CIFAR100','MNIST','TINY-IMAGENET'])
    parser.add_argument('--dir_name',default="cifar10", help="Image folder name",choices=['cifar10', 'cifar100','mnist','tiny_imagenet'])
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100','MNIST','TINY-IMAGENET'], type=str, help='path')
    return parser

def reshape_transform(tensor, height=4, width=4):

    argsparser = parser()
    args = argsparser.parse_args() 

    if args.model == "coswin": 
      result = tensor.reshape(tensor.size(0),
          height, width, tensor.size(2))
      
      result = result.transpose(2, 3).transpose(1, 2)
      return result
    elif args.model == "vit":
      height = 8  #32//4 or 64//8   
      width = 8
      result = tensor[:, 1 :  , :].reshape(tensor.size(0),
          height, width, tensor.size(2))
  
      result = result.transpose(2, 3).transpose(1, 2)
      return result

def gradCam(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_info = datainfo(args)
    
    IMG_SIZE = data_info['img_size']
    NUM_CLASSES = data_info['n_classes']
   

    model = build_model(IMG_SIZE, NUM_CLASSES, args)
    save = torch.load(args.model_path)
    model.load_state_dict(save['model_state_dict'])
    
    model.to(device) 
    
    if args.model == "vit":
      target_layer = [model.transformer.layers[-1][0].norm] 
    if args.model =="swin":
       target_layer =[model.layers[-1].blocks[-1].norm1]
    elif args.model=="deit" :
      target_layer = [model.blocks[-1].norm1]
    elif args.model =="coswin" : 
      target_layer =[model.layers[-1].blocks[-1].norm1]
      
    cam = GradCAMPlusPlus(model=model, 
                      target_layers=target_layer, 
                      reshape_transform=reshape_transform) 
                      

    dir = f"./images/{args.dir_name}/raw/*"
    list = glob.glob(dir, recursive=True)

    if (len(list)==0):
       print(f"Images not found in the directory: {dir}")
       return

    for img_path in list:
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
        rgb_img = np.float32(rgb_img) / 255
        
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor, targets=None,)

        grayscale_cam = grayscale_cam[0, :]
        img_path = img_path.replace("\\","/")

        img_name = img_path.split('/')[-1]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        save_dir= f'./images/{args.dir_name}/gradcam/{args.model}'
        os.makedirs(save_dir, exist_ok ="True")
        cv2.imwrite(f"{save_dir}/gradcam_{args.model}_{img_name}", cam_image)
    print("Grad-Cam images saved.")

if __name__== "__main__" :
    argsparser = parser()
    args = argsparser.parse_args()    
    gradCam(args)

        