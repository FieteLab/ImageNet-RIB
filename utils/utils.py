from re import A
import shutil
import os
import torch
import numpy as np
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision import transforms


def get_model(model_name, num_classes, patch_size, device, arch='base', d_pre='imagenet', pretrained=True):
    preprocess = None
    tokenizer = None
    if model_name == 'open_clip':
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(f'ViT-B-{patch_size}', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer(f'ViT-B-{patch_size}')
    elif 'resnet' in model_name:
        if model_name == 'resnet18':
            model = resnet18(pretrained=True)
        elif model_name == 'resnet34':
            model = resnet34(pretrained=True)
        if model_name == 'resnet50':
            from torchvision.models import ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            from torchvision.models import ResNet101_Weights
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif model_name == 'resnet152':
            from torchvision.models import ResNet152_Weights
            model = resnet152(weights=ResNet152_Weights.DEFAULT)
        # imagenet preprocess
        model = model.to(device)
        preprocess = transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    elif model_name == 'clip':
        import clip
        model, preprocess = clip.load(f'ViT-B/{patch_size}', device, jit=False)
        tokenizer = clip.tokenize
    elif model_name == 'vit':
        import timm
        if d_pre == 'imagenet':
            arch = f'vit_{arch}_patch{patch_size}_224.augreg_in21k_ft_in1k'
        elif d_pre == 'orig':
            arch = f'vit_{arch}_patch{patch_size}_224.orig_in21k_ft_in1k'
        elif d_pre == 'in1k':
            arch = f'vit_{arch}_patch{patch_size}_224.augreg_in1k'
        elif d_pre == 'laion':
            arch = f'vit_{arch}_patch{patch_size}_clip_224.laion2b_ft_in1k'
        elif d_pre == 'openai':
            arch = f'vit_{arch}_patch{patch_size}_clip_224.openai_ft_in1k'
        elif d_pre == 'laion_only':
            arch = f'vit_{arch}_patch{patch_size}_clip_224.laion2b'
        elif d_pre == 'openai_only':
            arch = f'vit_{arch}_patch{patch_size}_clip_224.openai'
        model = timm.create_model(arch, pretrained=pretrained)
        model = model.to(device)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
    return model, preprocess, tokenizer


"""
    Copy from another
"""

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
