from __future__ import print_function
import copy
import os
from re import A

import json

import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler 
from torch.utils.data import DataLoader

from utils.utils import save_checkpoint, cosine_lr, convert_models_to_fp32, refine_classname, get_model
from utils.lora import convert2lora, mark_only_lora_as_trainable
from utils.data_utils import get_dataset, set_seed

from lib.dataset.imagenetr_utils import imagenet_r_mask, imagenet_a_mask
from lib.dataset.objectnet_dataset import objectnet_mask
from lib.regularization import LearningWithoutForgetting, ewc_penalty
from lib.argument import parse_option
from lib.experiment import train, validate
from lib.evaluate import evaluations
from lib.prompter import PadPrompter

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_soup(args, model):
    # PRE, FT, EWC, LwF
    reg = args.regularization
    model_names = reg.split('-')[1:]
    params = []

    for model_name in model_names:
        if 'PRE' in model_name:
            params.append({'state_dict': model.state_dict()})
        else:
            # check whether path exists
            _path = args.filename.replace(reg, model_name)
            path = os.path.join(args.model_dir, _path, 'checkpoint.pth.tar')
            if not os.path.exists(path):
                print(path, "Not Exists")
                raise Exception
            print("Load ", path)
            params.append(torch.load(path))
            if params[-1]['epoch'] != args.epochs: # check fully-trained?
                print("Need to run more")
                raise Exception()
    
    # average all parameters in params to load.
    for key in params[0]['state_dict'].keys():
        params[0]['state_dict'][key] = params[0]['state_dict'][key].cuda()
        for i in range(1, len(params)):
            params[0]['state_dict'][key] += params[i]['state_dict'][key].cuda()
        params[0]['state_dict'][key] = params[0]['state_dict'][key] / len(params)
    model.load_state_dict(params[0]['state_dict'])

    model.eval()
    return model

def main():
    global best_acc1, device

    args = parse_option()
    print(args)

    if args.seed is not None:
        set_seed(args.seed)
    num_classes = 200   
    model, preprocess, tokenizer = get_model(args.model, num_classes, args.patch_size, device, arch=args.arch, d_pre=args.d_pre, pretrained=True)

    # Use LoRA
    if args.regularization == 'lora':
        model = convert2lora(model)
        mark_only_lora_as_trainable(model.blocks)
        for name, param in model.patch_embed.named_parameters():
            param.requires_grad = False
        for name, param in model.head.named_parameters():
            param.requires_grad = False
    elif args.regularization == 'HeadOnly':
        if args.model == 'vit':
            for name, param in model.patch_embed.named_parameters():
                param.requires_grad = False
            for name, param in model.blocks.named_parameters():
                param.requires_grad = False
        else: # resnet
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.fc.named_parameters():
                param.requires_grad = True
    elif args.regularization == 'Prompter':
        model.prompter = PadPrompter(args).to(device)
    print(model)

    if args.load is not None:
        state_dict = torch.load(args.load)
        model.load_state_dict(state_dict)
    

    convert_models_to_fp32(model)
    model.eval()

    # define criterion and optimizer
    if args.regularization == 'Prompter':
        target_param = model.prompter.parameters()
    else:
        target_param = model.parameters()
    optimizer = torch.optim.SGD(target_param,
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    args.model_folder = os.path.join(args.model_dir, args.filename)
    args.resume = os.path.join(args.model_folder, 'checkpoint.pth.tar')
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint.get('best_acc1', 0)
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # change the step.
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')
    train_dataset, val_dataset, test_dataset = get_dataset(args, preprocess)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True)


    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, pin_memory=True,
                            num_workers=args.num_workers, shuffle=False)
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
    else:
        class_names = [str(i) for i in range(1000)]

    class_names = refine_classname(class_names)
    texts = [template.format(label) for label in class_names]

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder, exist_ok=True)

    # wandb
    if args.use_wandb:
        wandb.init(project=args.project_name,
                   config=args,
                   name=args.filename,
                   dir=args.model_folder)
    
    if 'Soup' in args.regularization:
        model = get_soup(args, model)
    if args.regularization == 'LPFT':
        # load FT model.
        reg = args.regularization
        is_loaded = False
        # check whether path exists
        _path = args.filename.replace(reg, 'HeadOnly')
        path = os.path.join(args.model_dir, _path, 'checkpoint.pth.tar')
        if not os.path.exists(path):
            print(path, "Not Exists")
            raise Exception
        print("Load ", path)
        data = torch.load(path)
        if data['epoch'] == args.epochs: # check fully-trained?
            model.load_state_dict(data['state_dict'])
            is_loaded = True
        if not is_loaded:
            raise Exception("Not fully trained model")

    if args.regularization == 'PRE' or 'Soup' in args.regularization:
        evaluations(args, model, preprocess, texts, tokenizer, criterion)
        if args.use_wandb:
            wandb.run.finish()
        exit()

    imagenets = ['imagenet-r', 'imagenet-a', 'objectnet-v2']
    masks = [imagenet_r_mask, imagenet_a_mask, objectnet_mask]
    dataset2mask = {imagenets[i]: masks[i] for i in range(len(imagenets))}
    mask = dataset2mask.get(args.dataset)
    old_model = copy.deepcopy(model)  # evaluate the difference between shared tokens and unique tokens using the old model.

    # Get regularizer
    if args.regularization== 'lwf':
        lwf = LearningWithoutForgetting()
        lwf.prev_model = old_model
        lwf.mask = mask
        regularizer = lambda images, outputs, model: lwf(images, outputs)
    elif args.regularization == 'ewc':
        regularizer = lambda images, outputs, model: ewc_penalty(model, old_model.state_dict())        
    else:
        regularizer = None

    if args.state_dict:
        state_dict = torch.load(args.state_dict)
        model.load_state_dict(state_dict)
        model.eval()

    checkpoint_path = os.path.join(args.model_folder, 'checkpoint.pth.tar')
    if args.evaluate and os.path.exists(checkpoint_path): # and args.regularization != 'lora':
        # load model
        state_dict = torch.load(checkpoint_path)
        if state_dict['epoch'] == args.epochs:
            model.load_state_dict(state_dict['state_dict'])
            evaluations(args, model, preprocess, texts, tokenizer, criterion)
            if args.use_wandb:
                wandb.run.finish()
            exit()

    train_accs = []
    return_attention = False

    for epoch in range(args.epochs):
        # train for one epoch
        train_loss, train_acc = train(train_loader, texts, model, tokenizer, optimizer, scheduler, criterion, scaler, epoch, args, return_attention=return_attention, mask=mask, old_model=old_model, regularizer=regularizer)[:2]

        train_accs.append(train_acc)
        if args.use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                 }, step=epoch)

        if epoch % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args)

        if epoch == args.epochs - 1:
            val_acc1, val_loss = validate(val_loader, texts, model, tokenizer, criterion, args, return_attention=return_attention, mask=mask)[:2] #, old_model=old_model)
            train_acc1 = val_acc1
            train_loss = val_loss
            if args.use_wandb:
                wandb.log({
                    'val_loss': val_loss,
                    'val_acc': val_acc1,
                    'val_train_loss': train_loss,
                    'val_train_acc': train_acc1,
                }, step=epoch)

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, args, is_best=is_best)


    # evaluate on ImageNet-RIB
    results = evaluations(args, model, preprocess, texts, tokenizer, criterion)

    with open(f'{args.model_folder}/results.json', 'w') as f:
        json.dump(results, f)

    if args.use_wandb:
        wandb.run.finish()


if __name__ == '__main__':
    main()
