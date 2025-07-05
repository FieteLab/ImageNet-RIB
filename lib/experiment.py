import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from utils.utils import accuracy, AverageMeter, ProgressMeter


def get_image_logits(image_features, text_features, logit_scale, **kwargs):
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logits_per_image = logit_scale * image_features @ text_features.t()
    return logits_per_image


def model_forward(model, images, texts=None, tokenizer=None, model_name='clip', dataset='cifar100', patch_size=32, visualize=False, freeze_backbone=False, target=None, attr=None, mask=None, return_features=False):
    if 'clip' in model_name:
        text_tokens = tokenizer(texts).to(images.device)
        output = model(images, text_tokens)
        if len(output) == 2: # clip
            output = output[0]
        else: # open_clip
            output = get_image_logits(*output)
    elif 'resnet' in model_name:
        if hasattr(model, 'prompter'):
            images = model.prompter(images)
        output = model(images)
        attentions, cross_attentions, normed_cross_attentions, indices = None, None, None, None
        log_prob = None
        feature = None
    else: # normal vit
        if freeze_backbone:
            with torch.no_grad():
                output = model.forward_features(images, return_attention=True)
        else:
            output = model.forward_features(images, return_attention=True)
        if type(output) == tuple:
            output, attentions, cross_attentions, normed_cross_attentions, indices, _, log_prob = output
        else:
            attentions, cross_attentions, normed_cross_attentions, indices = None, None, None, None
            log_prob = None
        if return_features:
            feature = model.forward_head(output, pre_logits=True).detach().cpu()
        else:
            feature = None
        output = model.forward_head(output)
        
    if True: #'imagenet' in dataset and 'clip' not in model_name:
        if mask is not None:
            output = output[:, mask]
    return output, attentions, cross_attentions, normed_cross_attentions, indices, log_prob, feature



def train(train_loader, texts, model, tokenizer, optimizer, scheduler, criterion, scaler, epoch, args, return_attention=False, mask=None, old_model=None, regularizer=None):
    """
    Run one train epoch
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if hasattr(model, 'prompter'):
        model.eval()
        model.prompter.train()
    else:
        model.train()

    attentions = []
    attr_ids = []
    indices = []


    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for i, data in enumerate(tqdm(train_loader)):
        if len(data) == 4:
            images, target, attr, idx = data
        elif len(data) == 3:
            images, target, attr = data
            idx = None
        else:
            images, target = data
            attr = None
            idx = None
        # measure data loading time
        data_time.update(time.time() - end)
        attr_ids.append(attr)
#        indices.append(idx)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        if type(attr) == torch.Tensor:
            attr = attr.to(device)

        # with automatic mixed precision
        with autocast():
            output = model_forward(model, images, texts, tokenizer, args.model, args.dataset, patch_size=args.patch_size, freeze_backbone=args.freeze_backbone, target=target, attr=attr, mask=mask)

            if type(output) is tuple:
                output, atts, cross_attentions, normed_cross_attentions, indices, log_prob, feature = output
            if log_prob is not None:
                loss = nn.CrossEntropyLoss(reduction='none')(output, target)
                loss = loss + loss.detach() * log_prob.mean(-1)
                loss = loss.mean()
            else:
                loss = criterion(output, target)
            
            if regularizer is not None:
                loss += regularizer(images, output, model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()
        
        if 'clip' in args.model:
            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


    print(f' * Training Acc@1 {top1.avg:.3f}\t Loss {losses.avg:.3f}')

    if return_attention:
        if len(attentions) > 0 and attentions[0] is not None:
            attentions = torch.cat(attentions, 1).detach().cpu()
            attr_ids = torch.cat(attr_ids, 0)
        return losses.avg, top1.avg, attentions, attr_ids
    return losses.avg, top1.avg


def validate(val_loader, texts, model, tokenizer, criterion, args, prefix='', visualize=False, return_attention=False, mask=None, old_model=None, epoch=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1 = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1],
        prefix='Validate: ')

    # switch to evaluation mode
    model.eval()
    failure_samples = []
    failure_targets = []

    predictions = []
    ground_truths = []
    attentions = []
    attr_ids = []
    indices = []
    key_probs = []
    outputs = []
    targets = []
    features = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(tqdm(val_loader)):
            if len(data) == 4:
                images, target, attr, idx = data
                attr_ids.append(attr) #[:5])
                indices.append(idx)
            elif len(data) == 3:
                images, target, attr = data
                attr_ids.append(attr) #[:5])
                idx = None
            else:
                images, target = data
                attr = None

            images = images.to(device)
            target = target.to(device)
            if type(attr) == torch.Tensor:
                attr = attr.to(device)

#            output = model_forward(model, images, texts, tokenizer, args.drop_tokens, args.model, args.dataset, patch_size=args.patch_size, target=target, visualize=args.visualize, attr=attr, mask=mask, return_features=args.collect_features)
            output = model_forward(model, images, texts, tokenizer, args.model, args.dataset, patch_size=args.patch_size, target=target, visualize=args.visualize, attr=attr, mask=mask, return_features=args.collect_features)
            if type(output) is tuple:
                output, atts, catts, normed_cross_attentions, indices, log_prob, feature = output
            loss = criterion(output, target)
            if args.collect_failure:
                failure = output.argmax(dim=1) != target
                failure_samples.append(images[failure])
                failure_targets.append(target[failure])
            if args.collect_features:
                features.append(feature)

            targets.append(target)
            outputs.append(output)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        
        print(' * Original Acc@1 {top1_org.avg:.3f} Fine-Tuning Acc@1 {top1.avg:.3f}'
              .format(top1=top1, top1_org=top1_org))
        
        if len(attentions) > 0 and attentions[0] is not None:
            attentions = torch.cat(attentions, 1).cpu()
    analyzer.get_summary()
    if args.collect_failure:
        predictions = torch.cat(predictions)
        ground_truths = torch.cat(ground_truths)
        failure_samples = torch.cat(failure_samples, dim=0)
        failure_targets = torch.cat(failure_targets, dim=0)
    if args.collect_features:
        features = torch.cat(features, dim=0)
    

    if return_attention:
        if len(attr_ids) > 0:
            attr_ids = torch.cat(attr_ids)
        key_probs = torch.cat(key_probs, 1)
        if len(attentions) > 0:
            attentions = torch.cat(attentions, 1)
        return top1.avg, losses.avg, (failure_samples, failure_targets), attentions, attr_ids, key_probs, features
    return top1.avg, losses.avg, (failure_samples, failure_targets), features
