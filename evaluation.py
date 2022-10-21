import os
import cv2
import argparse
import numpy as np
import types
import copyreg
from multiprocessing import Pool

CATEGORY_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

colors = np.array([(128, 0, 0), (0, 128,0 ), (128, 128, 0),
                  (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                  (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                  (0, 64, 128)]) # using palette for pascal voc

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0
            else:
                print("what'wrong", gt[i])

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate segmentation result')
    parser.add_argument('--pred-path', default='./result/1/label_mask', type=str, help='prediction result dir')
    parser.add_argument('--class-num', default=21, type=int,  help='class number include bg')
    parser.add_argument('--gt-path', default='./dataset/PascalVOC/VOCdevkit/VOC2012/SegmentationClassAug', type=str, help='ground truth dir')
    parser.add_argument('--image-list', default='./datalist/PascalVOC/val_id.txt', type=str, help='test ids file path')
    parser.add_argument('--save-name', default='./result/1/evaluation_result_direct', type=str, help='result file path')
    parser.add_argument('--color-mask', default=0, type=int, help='is it the colored result?')
    args = parser.parse_args()
    return args

def semantic2mask(img):
    result = np.zeros((img.shape[0], img.shape[1]))
    for index, color in enumerate(colors):
        class_location = [img==color][0]
        loc = np.logical_and(np.logical_and(class_location[:,:,0], \
                                            class_location[:,:,1]), \
                             class_location[:,:,2])
        result[loc] = index+1
    return result.astype(dtype=int)

if __name__ == '__main__':
    args = parse_args()
    print('Evaluation  is executed.')
    iter = 0
    m_list = []
    data_list = []
    image_list = [i.strip() for i in open(args.image_list) if not i.strip() == '']
    for index, img_id in enumerate(image_list):
        gt_img_path = os.path.join(args.gt_path, img_id + '.png')
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        
        if (args.color_mask):
            pred_img_path = os.path.join(args.pred_path, img_id + '_pred.png')
            pred = cv2.imread(pred_img_path)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            pred = semantic2mask(pred)
        else:         
            label_mask_img_path = os.path.join(args.pred_path, img_id + '_labelmask.png')
            pred = cv2.imread(label_mask_img_path, cv2.IMREAD_GRAYSCALE)
        data_list.append([gt.flatten(), pred.flatten()])
    print('All images are loaded')

    ConfM = ConfusionMatrix(args.class_num)
    f = ConfM.generateM
    pool = Pool()    
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    with open(args.save_name, 'w') as f:
        print('{0:12s}: {1:.4f}'.format('meanIOU', aveJ * 100))
        print('=' * 21)
        f.write('{0:12s}: {1:.4f}\n'.format('meanIOU', aveJ * 100))
        f.write('=' * 21)
        f.write('\n')
        for i, j in enumerate(j_list):
            print("{0:12s}: {1:.4f}".format(CATEGORY_LIST[i], j * 100))
            f.write("{0:12s}: {1:.4f}\n".format(CATEGORY_LIST[i], j * 100))

        f.write('Raw Result:\n')
        f.write('meanIOU: ' + str(aveJ) + '\n')
        f.write(str(j_list) + '\n')
        f.write(str(M) + '\n')
