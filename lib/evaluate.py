import glob
import os
import numpy as np 

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from lib.dataset.imagenetr_utils import imagenet_r_mask, imagenet_a_mask
from lib.dataset.objectnet_dataset import objectnet_mask, ObjectNetDataset
from lib.dataset.imagenet_v2 import ImageNetV2
from lib.experiment import validate
import wandb
import json

def load_results(args):
    results = {}
    json_path = f'{args.model_folder}/results.json'
    if os.path.exists(json_path) and not args.force and not args.collect_features:
        with open(json_path, 'r') as f:
            results = json.load(f)
        for i in range(100):
            json_path = f'{args.model_folder}/results_{i}.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results.update(json.load(f))
    print("Loaded results", results)
    return results


def evaluations(args, model, preprocess, text, tokenizer, criterion, old_model=None, evaluation_datasets=['imagenet_variants', 'imagenet-c']):
    """
    Evaluate the model on various datasets.
    Args:
        args: arguments
        model: model
        preprocess: preprocess function
        text: text
        tokenizer: tokenizer
        criterion: criterion
        old_model: old model for ewc
    Returns:
        outputs
    """
    
    results = load_results(args)
    output = {}

    if 'imagenet_variants' in evaluation_datasets:
        _output = evaluate_imagenet_variants(args, model, preprocess, text, tokenizer, criterion, old_model=old_model, results=results)
        output.update(_output)

    json_path = f'{args.model_folder}/results.json'
    if os.path.exists(json_path) and not args.force and not args.collect_features:
        try:
            output.update(json.load(open(json_path, 'r')))
        except:
            pass
    with open(json_path, 'w') as f:
        json.dump(output, f)

    if 'imagenet-c' in evaluation_datasets:
        _output = evaluate_imagenet_c(args, model, preprocess, text, tokenizer, criterion, results=results, json_path=json_path)
        output.update(_output)
    if args.use_wandb:
        wandb.log(output)
    if os.path.exists(json_path):
        output.update(json.load(open(json_path, 'r')))
    with open(json_path, 'w') as f:
        json.dump(output, f)
    return output


def evaluate_imagenet_variants(args, model, preprocess, texts, tokenizer, criterion, old_model=None, results={}):
    """
    Evaluate the model on various imagenet variants.
    Args:
        args: arguments
        model: model
        preprocess: preprocess function
        texts: list of texts
        tokenizer: tokenizer
        criterion: criterion
        old_model: old model for ewc
    Returns:
        mean accuracy
    """
    dataset_names = ['imagenet/val',  'imagenet-a', 'imagenet-r', 'imagenet-sketch', 'objectnet-1.0', 'imagenet-cartoon', 'imagenet-drawing', 'imagenet-v2', 'objectnet-v2']
    masks = [None, imagenet_a_mask, imagenet_r_mask, None, None, None, None, None]
    masks = [None, imagenet_a_mask, imagenet_r_mask, None, None, None, None, None, objectnet_mask]
    acc1s = []
    for dataset_name, mask in zip(dataset_names, masks):
        if dataset_name in results and not args.collect_features: 
            acc1s.append(results[dataset_name])
            continue
        print("Evaluating", dataset_name)
        path = os.path.join(args.root, dataset_name)
        if dataset_name == 'objectnet-1.0':
            dataset = ObjectNetDataset(root=path, transform=preprocess)
        elif dataset_name == 'objectnet-v2':
            dataset = ObjectNetDataset(root=os.path.join(args.root, 'objectnet-1.0'), transform=preprocess, reindex=True)
        elif dataset_name == 'imagenet-v2':
            dataset = ImageNetV2(path, transform=preprocess)
        else:
            dataset = ImageFolder(path, transform=preprocess)
        data_loader = DataLoader(dataset,
                                batch_size=args.batch_size*2, pin_memory=True,
                                num_workers=args.num_workers, shuffle=False)
        prefix = dataset.root.split('/')[-1]
        acc1, loss, failures, features = validate(data_loader, texts, model, tokenizer, criterion, args, prefix=prefix, return_attention=False, mask=mask) #, old_model=old_model)
        acc1s.append(acc1)
        print(dataset_name, acc1)
        if args.collect_features:
            save_path = os.path.join(args.model_dir, args.filename, f"{dataset_name.replace('/','_')}_features.pth")
            torch.save(features, save_path)
            print(f'saved features to {save_path}')
        # save features


    print(dataset_names)
    print(acc1s)
    print(np.mean(acc1s))
    outputs = {name: acc for name, acc in zip(dataset_names, acc1s)}
    return outputs


def evaluate_imagenet_c(args, model, preprocess, texts, tokenizer, criterion, results={}, json_path=None):
    paths = glob.glob(os.path.join(args.root, 'imagenet-c/*'))
    imagenet_c = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    paths = [os.path.join(args.root, 'imagenet-c', name) for name in imagenet_c]
    acc1s = []
    outputs = {}
    json_output = {}
    if json_path is not None and os.path.exists(json_path):
        json_output = json.load(open(json_path, 'r'))
    for path in paths:
        if not os.path.isdir(path):
            continue
        name = path.replace(args.root, '')
        if name in results and not args.collect_features:
            acc1s.append(results[name])
            continue
        _acc = []
        for severity in range(1,6):
            _path = os.path.join(path, str(severity))
            name = _path.replace(args.root, '')
            if name in json_output and not args.collect_features:
                outputs[name] = json_output[name]
                continue
            dataset = ImageFolder(_path, transform=preprocess)
            data_loader = DataLoader(dataset,
                                    batch_size=args.batch_size*2, pin_memory=True,
                                    num_workers=args.num_workers, shuffle=False)
            prefix = dataset.root.split('/')[-1]
            print(prefix)
            acc1, loss, failures, features = validate(data_loader, texts, model, tokenizer, criterion, args, prefix=prefix, return_attention=False, mask=None)
            print(_path, acc1)
            outputs[name] = acc1
            json_output[name] = acc1
            _acc.append(acc1)
            if args.collect_features:
                dataset_name = _path.split('/')[-2] + '_' + _path.split('/')[-1]
                save_path = os.path.join(args.model_dir, args.filename, f"{dataset_name}_features.pth")
                torch.save(features, save_path)
                print(f'saved features to {save_path}')
        
        with open(json_path, 'w') as f:
            json.dump(json_output, f)

        acc1 = np.mean(_acc)
        acc1s.append(acc1)
        print(path, acc1)
    print(acc1s)
    print(np.mean(acc1s))
    outputs.update({name.replace(args.root, ''): acc for name, acc in zip(paths, acc1s)})
    return outputs


