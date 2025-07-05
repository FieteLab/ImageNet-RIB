# https://github.com/abarbu/objectnet-template-pytorch/blob/master/objectnet_pytorch_dataloader.py
import os
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import json

manual_converter = {
        'Band Aid': 'Band-Aid',
        'Coffee/French press': 'espresso machine',
        'Dress shoe (men)': 'slip-on shoe',
        'Bottle cap': 'bottle cap',
        'Match': 'match',
        'Bread loaf': 'baguette',
        }

objclasses = [409, 414, 418, 419, 434, 440, 455, 457, 462, 463, 470, 473, 487,
       499, 504, 507, 508, 531, 533, 539, 543, 545, 549, 550, 560, 567,
       578, 587, 588, 589, 610, 618, 619, 620, 623, 626, 629, 630, 632,
       644, 647, 651, 658, 659, 664, 671, 673, 677, 679, 694, 695, 696,
       700, 703, 720, 721, 725, 728, 729, 731, 732, 737, 740, 742, 752,
       761, 769, 770, 772, 773, 774, 783, 790, 792, 797, 804, 806, 809,
       811, 813, 828, 834, 837, 841, 846, 849, 850, 851, 859, 868, 879,
       882, 883, 893, 898, 902, 907, 909, 923, 930, 950, 951, 954, 968,
       999]

objectnet_mask = np.zeros(1000).astype(bool)
objectnet_mask[objclasses] = True

class ObjectNetDataset(VisionDataset):
    """
    ObjectNet dataset.

    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'transforms.ToTensor'
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        img_format (string): jpg
                             png - the original ObjectNet images are in png format
    """

    def __init__(self, root, transform=None, target_transform=None, transforms=None, img_format="png", imagenet_overlap_only=True, reindex=True):
        """Init ObjectNet pytorch dataloader."""
        super(ObjectNetDataset, self).__init__(root, transforms, transform, target_transform)

        self.loader = self.pil_loader
        self.img_format = img_format
        files = glob.glob(root+"/**/*."+img_format, recursive=True)
        self.pathDict = {}
        self.imagenet_overlap_only = imagenet_overlap_only
        for f in files:
            self.pathDict[f.split("/")[-1]] = (f, f.split("/")[-2])
        self.imgs = list(self.pathDict.keys())
        self.prepare_labels(root)

        self.reindex = reindex
        if self.reindex:
            unique_labels = np.unique(self.targets)
            self.label2reindex = {label: i for i, label in enumerate(unique_labels)}

    def prepare_labels(self, root):
        # read folder_to_object_label.json
        path = os.path.join(root, "mappings", "folder_to_objectnet_label.json")
        # read json
        # return dict
        with open(path, 'r') as f:
            self.folder_to_label =json.load(f)
        path = os.path.join(root, "mappings",  "objectnet_to_imagenet_1k.json")
        with open(path, 'r') as f:
            self.label_to_imgnet_label = json.load(f)
        # filter based on this
        overlapped_class = []
        for key in self.folder_to_label.keys():
            if self.folder_to_label[key] in self.label_to_imgnet_label:
                overlapped_class.append(key)
        # change the label to class_id
        if self.imagenet_overlap_only:
            valid_keys = []
            for key in self.imgs:
                folder_name = self.pathDict[key][1]
                if self.folder_to_label[folder_name] in self.label_to_imgnet_label:
                    valid_keys.append(key)
            self.imgs = valid_keys

            with open("lib/dataset/imagenet_class.json", 'r') as f:
                self.imgnet_class_to_label = json.load(f) # list   
            self.imagenet_label_to_class = {e:i for i, e in enumerate(self.imgnet_class_to_label)}

            self.folder_to_class = {}
            self.class_to_label = {}
            for folder_name, label in self.folder_to_label.items():
                if label in self.label_to_imgnet_label:
                    imagenet_label = self.label_to_imgnet_label[label]
                    if ',' in imagenet_label:
                        for _imagenet_label in imagenet_label.split(', '):
                            if _imagenet_label in self.imagenet_label_to_class:
                                class_id = self.imagenet_label_to_class[_imagenet_label]
                                imagenet_label = _imagenet_label
                                break
                    elif ';' in imagenet_label:
                        for _imagenet_label in imagenet_label.split('; '):
                            if _imagenet_label in self.imagenet_label_to_class:
                                class_id = self.imagenet_label_to_class[_imagenet_label]
                                imagenet_label = _imagenet_label
                                break
                    elif imagenet_label in self.imagenet_label_to_class:
                        class_id = self.imagenet_label_to_class[imagenet_label]
                    else:
                        imagenet_label = manual_converter[label]
                        class_id = self.imagenet_label_to_class[imagenet_label]
                    self.folder_to_class[folder_name] = class_id
                    self.class_to_label[class_id] = imagenet_label
            used_classes = np.zeros((1000,))
            used_classes[np.asarray(list(self.class_to_label.keys()))] = 1
            used_classes = used_classes.astype(bool)
            converter = {k: i for i, k in enumerate(np.where(used_classes)[0])}

            self.objectnet_mask = used_classes
        else:
            # allocate each object net label to class 
            self.folder_to_class = {e: i for i, e in enumerate(self.folder_to_label.keys())}
            self.class_to_label = {v: self.folder_to_label[k] for k, v in self.folder_to_class.items()}
            self.class_to_label = [self.class_to_label[i] for i in range(len(self.class_to_label))]

        # from self.imgs, extract target_list
        self.targets = [self.folder_to_class[self.pathDict[key][1]] for key in self.imgs]
        self.classes = [self.class_to_label[e] for e in sorted(self.class_to_label.keys())]


    def __getitem__(self, index):
        """
        Get an image and its label.

        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the image file name
        """
        img, target = self.getImage(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        target = self.folder_to_class[target]
        if self.reindex:
            target = self.label2reindex[target]
        return img, target

    def getImage(self, index):
        """
        Load the image and its label.

        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the image file name
        """
#        key, target 
        path, target = self.pathDict[self.imgs[index]]
        img = self.loader(path)

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width-2, height-2)
        img = img.crop(cropArea)
        return (img, target)

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.imgs)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class data_transform:
    def __init__(self):
        self.model_pretrain_params = {}
        self.model_pretrain_params['input_size'] = [3, 224, 224]
        self.model_pretrain_params['mean'] = [0.485, 0.456, 0.406]
        self.model_pretrain_params['std'] = [0.229, 0.224, 0.225]
        self.resize_dim = self.model_pretrain_params['input_size'][1]

    def getTransform(self):
        trans = transforms.Compose([transforms.Resize(self.resize_dim),
                                    transforms.CenterCrop(self.model_pretrain_params['input_size'][1:3]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.model_pretrain_params['mean'],
                                                         std=self.model_pretrain_params['std'])
                                    ])
        return trans


if __name__ == '__main__':
    dataset = ObjectNetDataset(root="datasets/objectnet-1.0", img_format="png")
    print(dataset[0])
    breakpoint()
