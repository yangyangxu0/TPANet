import os
import numpy as np
from PIL import Image
import torch

import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random



class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


@torch.no_grad()
def save_predictions(task, preds, meta, save_dir):
    if task in ['edge', 'sal']:
        preds = 255 * torch.sigmoid(preds.squeeze(1))
    elif task in ['semseg', 'human_parts']:
        preds = torch.argmax(preds, dim=1)
    elif task == 'normals':
        norm = torch.norm(preds, p='fro', dim=1, keepdim=True).expand_as(preds)
        preds = 255 * (preds.div(norm) + 1.0) / 2.0
        preds[norm == 0] = 0
    elif task == 'depth':
        pass
    else:
        raise ValueError

    for idx, pred in enumerate(preds):
        im_height = meta['im_size'][0][idx]
        im_width = meta['im_size'][1][idx]
        im_name = meta['image'][idx]

        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[-2:]:
            delta_height = max(pred.shape[-2] - im_height, 0)
            delta_width = max(pred.shape[-1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                height_location = [delta_height // 2,
                                   (delta_height // 2) + im_height]
                width_location = [delta_width // 2,
                                  (delta_width // 2) + im_width]
                pred = pred[..., height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[-2:] == (im_height, im_width)
        if pred.ndim == 3:
            pred = pred.transpose(1, 2, 0)
        arr = pred.cpu().numpy()
        if task == 'depth':
            np.save(os.path.join(save_dir, '{}.npy'.format(im_name)), arr)
        else:
            image = Image.fromarray(arr.astype(np.uint8))
            image.save(os.path.join(save_dir, '{}.png'.format(im_name)))





def visulizeFeatureMapPCA(feature, label):
    random.seed(0)
    feature = feature.squeeze(0).data.cpu().numpy()

    # img_out = np.mean(feature, axis=0)
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    if c==1:
        pca = PCA(n_components=1) #'mle'
        pca.fit(img_out)
        img_out_pca = pca.transform(img_out)
        img_out_pca = img_out_pca.transpose(1, 0).reshape(1, h, w).transpose(1, 2, 0)
        img_out_pca = cv2.cvtColor(img_out_pca, cv2.COLOR_GRAY2RGB)
        #img_out_pca = np.stack((img_out_pca,) * 3, axis=-1).squeeze()
    else:
        pca = PCA(n_components=3)
        pca.fit(img_out)
        img_out_pca = pca.transform(img_out)
        img_out_pca = img_out_pca.transpose(1, 0).reshape(3, h, w).transpose(1, 2, 0)

    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(img_out_pca, (8 * w, 8 * h), interpolation=cv2.INTER_LINEAR)
    img_out = np.array(img_out_pca, dtype=np.uint8)
    # img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)

    #plt.title(label)
    plt.axis('off')
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.imshow(img_out)
    plt.savefig("/data/project/MDPT-main_0619_add_vitea_cul_gflop_swinlarge/visual/"+label+str(random.random())+".png",
                format='png',bbox_inches = 'tight', transparent=True, dpi=300, pad_inches = 0)
    #plt.savefig("/public/data2/users/xuyangyang36/project/MDPT_checkpoints/saveImage/test.png")
    plt.show()

    return img_out








