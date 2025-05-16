from colorama import Fore, Style
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .tiny_dataset import TinyImageNet

def datainfo(args):
    if args.dataset == 'CIFAR10':
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32        
        
    elif args.dataset == 'CIFAR100':
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762) 
        img_size = 32        
        
    elif args.dataset == 'TINY-IMAGENET':
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64

    elif args.dataset == "MNIST":
        n_classes = 10
        img_mean, img_std = (0.1307,), (0.3081)
        img_size = 32

    elif args.dataset == 'SVHN':
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970) 
        img_size = 32
     
    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size    
    return data_info


def dataload(args, augmentations, normalize, data_info):
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=augmentations)
        val_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
        
    elif args.dataset == 'TINY-IMAGENET':
        train_dataset = TinyImageNet('./dataset', split='train', download=True, transform=augmentations)
        val_dataset = TinyImageNet('./dataset', split='val', download=True, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))
   
    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(data_info['img_size']),
                transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            *normalize])
                 
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    
    elif args.dataset == 'SVHN':
        train_dataset = datasets.SVHN(root=args.data_path, split='train', download=True, transform=augmentations)
        val_dataset = datasets.SVHN(root=args.data_path, split='test', download=True, transform=transforms.Compose([transforms.Resize(data_info['img_size']),transforms.ToTensor(),*normalize]))
        
    return train_dataset, val_dataset