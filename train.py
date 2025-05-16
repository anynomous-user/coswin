import torch
import torch.utils.data.dataloader
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from util.datahandler import datainfo, dataload
from model.model_builder import build_model
from util.ls_crossentropy import LabelSmoothingCrossEntropy
from util.autoaugment import CIFAR10Policy, ImageNetPolicy, SVHNPolicy
from util.random_erasing import RandomErasing
from util.scheduler import build_scheduler
from util.mixup import cutmix_data, mixup_data, mixup_criterion
from util.rasampler import RASampler
import argparse
from tqdm import tqdm
import os

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default=100, help ="Total epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--workers", type=int, default=2, help="Total worker" )
    parser.add_argument("--weight_decay", type=float, default=0.05, help='Weight decay')
    parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
    parser.add_argument('--warmup', default=10, type=int, help='Warmup epoch')
    parser.add_argument("--dataset", type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100', 'MNIST', 'SVHN', 'TINY-IMAGENET'], help="dataset")
    parser.add_argument('--beta', default=1.0, type=float,help='hyperparameter: beta')
    parser.add_argument('--alpha', default=1.0, type=float, help='mixup interpolation coefficient')  
    parser.add_argument('--model', type=str, default='coswin',choices=['coswin', 'vit', 'swin', 'resnet18', 'resnet56', 'mobilenetv2','deit', 'pvt', 't2t'])
    parser.add_argument('--output', type = str, default ="./saved")
    return parser

def train(args, model, criterion, optimizer, scheduler, train_loader, val_loader, model_name, device):
    name = f"{model_name}_{args.dataset}"
    writer = SummaryWriter(f"{args.output}/{args.model}/{args.dataset}/runs_{name}")
    lr = optimizer.param_groups[0]["lr"]

    for epoch in range(args.epochs):     
        print(f"Epoch {epoch+1}/{args.epochs}")     
        model.train()   
        
        train_loss = 0
        train_sample = 0
        train_bar = tqdm(train_loader, desc="Training")

        for i, data in enumerate(train_bar):
            img, label = data
            img = img.to(device)
            label = label.to(device)
    
            random = np.random.rand(1)
            mixup_probabilty = 0.5
            if random < mixup_probabilty:
                switching_prob = np.random.rand(1)         
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(img, label, args)
                    img[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(img)         
                    loss =  mixup_criterion(criterion, output, y_a, y_b, lam)    
                else:
                    img, y_a, y_b, lam = mixup_data(img, label, args)
                    output = model(img)
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)       
            else:
                output = model(img)       
                loss = criterion(output, label)  

            train_sample += img.size(0)
            train_loss += float(loss.item() * img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            avg_train_loss= train_loss / train_sample
            train_bar.set_postfix(train_loss = avg_train_loss)

        model.eval()

        val_loss = 0
        val_acc = 0
        val_sample = 0
        val_bar = tqdm(val_loader, desc="validation")

        with torch.no_grad():
            for i, data in enumerate(val_bar):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                
                output = model(img)
                loss = criterion(output, label)

                predicted = output.argmax(dim=1)
                correct = predicted.eq(label)  
                correct_count = correct.sum().item()
                accuracy = (correct_count / label.size(0)) * 100   

                val_sample += img.size(0)
                val_loss += float(loss.item() * img.size(0))
                val_acc += float(accuracy * img.size(0))
                
                avg_val_loss = val_loss/val_sample 
                avg_val_acc= val_acc/val_sample
                val_bar.set_postfix(val_loss = avg_val_loss, val_acc = avg_val_acc)

        writer.add_scalar("Loss/train", avg_train_loss, epoch )
        writer.add_scalar("Loss/val", avg_val_loss, epoch )
        writer.add_scalar("Accuracy/val", avg_val_acc, epoch)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        os.makedirs(f"{args.output}/{args.model}/{args.dataset}/model", exist_ok=True)
        torch.save(checkpoint, f'{args.output}/{args.model}/{args.dataset}/model/checkpoint_{name}.pt')

    writer.flush()
    writer.close()   

if __name__ == '__main__':
    argsparser = parser()
    args = argsparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_info = datainfo(args)
    model = build_model(data_info['img_size'], data_info['n_classes'], args)
    model.to(device)
    model_name = args.model

    transform = [                
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(data_info['img_size'], padding=4)
    ]
         
    if 'CIFAR' in args.dataset:
            transform += [ CIFAR10Policy()]   
    elif 'SVHN' in args.dataset:   
        transform += [SVHNPolicy()]            
    elif 'IMAGENET' in args.dataset:
            transform += [ImageNetPolicy()]
            
    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]    
    transform += [transforms.ToTensor(), *normalize]  
    transform += [RandomErasing(probability = 0.25, sh = 0.4, r1 = 0.3, mean=data_info['stat'][0]) ]
    transform = transforms.Compose(transform)
      
    train_dataset, val_dataset = dataload(args, transform, normalize, data_info)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, 3, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    criterion = LabelSmoothingCrossEntropy()    
    criterion = criterion.to(device)   
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    train(args, model, criterion, optimizer, scheduler,  train_loader, val_loader, model_name, device)

    
    


 
