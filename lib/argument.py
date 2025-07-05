# The code is based on https://github.com/hjbahng/visual_prompting

import argparse

def parse_option():
    parser = argparse.ArgumentParser('ImageNet-RIB')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=196,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='base')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--d_pre', type=str, default='imagenet')

    # Prompter 
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)

    parser.add_argument('--load', type=str, default=None,
                        help='load model')
    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--force', default=False,
                        action="store_true",
                        help='force evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    parser.add_argument('--project_name', default='Transfer', type=str)

    parser.add_argument('--collect_failure', default=False,
                        action="store_true")
    parser.add_argument('--state_dict', type=str, default='',
                        help='path to store state_dict')

    parser.add_argument('--collect_path_idx', type=int, default=-1)

    parser.add_argument('--regularization', type=str, default='',
                        help='Method')
    parser.add_argument('--no_split', action='store_true', default=False)
    parser.add_argument('--collect_features', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_b{}E{}_warmup_{}_trial_{}'. \
        format(args.regularization, args.dataset, args.model, args.arch, args.patch_size, args.d_pre,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.epochs, args.warmup, args.trial)
    if args.no_split:
        args.filename += '_no_split'
    return args


