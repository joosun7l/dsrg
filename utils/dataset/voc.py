import os
import cv2
import pickle
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from torchvision.utils import save_image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tool import imutils
MODE = ['train', 'val', 'test']
IMG_FOLDER_NAME = "JPEGImages"
# Color mean in order RGB
IMAGE_MEAN_VALUE = [123.0, 104.0, 117.0]


def normalize(img):

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)
    proc_img[..., 0] = (imgarr[..., 0] / 255. - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] / 255. - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] / 255. - mean[2]) / std[2]
    return proc_img

transform = transforms.Compose([
                imutils.RandomResizeLong(321, 321),
                # transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # np.asarray,
                normalize,
                # imutils.RandomCrop(321),
                imutils.HWC_to_CHW,
                torch.from_numpy
            ])

class VOCDataset(Dataset):

    def __init__(self, root=None, gt_root=None, datalist=None, debugflag=False, resize_size=321, num_category=21):
        self.root = root
        self.resize_size = resize_size
        self.debugflag = debugflag

        self.image_names = [img_gt_name.split()[0][:-4]
                            for img_gt_name in open(datalist).read().splitlines()]

        if self.debugflag:
            self.image_names = self.image_names[:8]

        self.cue_data = pickle.load(open(gt_root, 'rb'))
        self.num_category = num_category

    def __len__(self):
        return len(self.image_names)
    
    def getfilename(self, idx):
        return self.image_names[idx]
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        image_path = os.path.join(self.root, IMG_FOLDER_NAME, name + '.jpg')

        # Load Image
        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = preprocess(img, self.resize_size, IMAGE_MEAN_VALUE).astype(np.float32)
        img_p = PIL.Image.open(image_path).convert("RGB")
        img_p = transform(img_p)

        # Load true GT image for debugging
        gt_image_path = os.path.join(self.root, "JPEGImages", name + '.jpg')
        true_gt_img = cv2.cvtColor(cv2.imread(gt_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        true_gt_img = preprocess(true_gt_img, self.resize_size, [0.0, 0.0, 0.0]).astype(np.float32)
        true_gt_img = torch.from_numpy(true_gt_img)    
        
        img = torch.from_numpy(img)
        label, gt_map = self.get_seed(idx)
        label = torch.from_numpy(label)
        gt_map = torch.from_numpy(gt_map.astype(np.float32))

        return img_p, label, gt_map, true_gt_img,img
    
    def get_seed(self, id_slice):
        label_seed = np.zeros((1, 1, self.num_category))
        label_seed[0, 0, self.cue_data["%s_labels" % id_slice]] = 1.
        cues = np.zeros([21, 41, 41])
        cues_i = self.cue_data["%s_cues" % id_slice]
        cues[cues_i[0], cues_i[1], cues_i[2]] = 1.
        return label_seed.astype(np.float32), cues.astype(np.float32)

def preprocess(image, size, mean_pixel):
    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]), size / float(image.shape[1]), 1.0),
                    order=1)
    image = image - np.array(mean_pixel)
    image = image.transpose([2, 0, 1])
    return image