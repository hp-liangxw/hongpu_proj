# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import glob
import cv2
import json
import csv
import time
import logging

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class COCODemo(object):
    def __init__(
            self,
            cfg,
            show_mask_heatmaps=False,
            min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.show_mask_heatmaps = show_mask_heatmaps

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction


from maskrcnn_benchmark.config import cfg


def infer_simple(img_folder, output_folder, model_dir, threshold, logger):
    config_file = os.path.join(model_dir, 'configs/e2e_faster_rcnn_R_50_FPN_1x.yaml')
    weights = os.path.join(model_dir, 'output/model_final.pth')
    # read labels name
    with open(os.path.join(model_dir, 'output/labels.json'), 'r') as f:
        labels_datasets = json.load(f)
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    cfg.merge_from_list(["MODEL.WEIGHT", weights])
    coco_demo = COCODemo(
        cfg,
        min_image_size=500,
    )

    if len(labels_datasets) == len(threshold):
        if os.path.isdir(img_folder):
            im_list = glob.iglob(img_folder + '/*.jpg')
        else:
            im_list = [img_folder]
        im_len = len(glob.glob(img_folder + '/*.jpg'))

        output_csv = os.path.join(output_folder, 'output_{}.csv'.format(labels_datasets['1']))
        with open(output_csv, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['pic_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score'])

            for i, im_name in enumerate(im_list):

                logger.info('[{}/{}]  start {} image predict '.format(i + 1, im_len, os.path.basename(im_name)))
                im = cv2.imread(im_name)
                t = time.time()
                predictions = coco_demo.compute_prediction(im)

                logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                bbox = predictions.bbox.numpy()
                labels = predictions.get_field('labels').numpy()
                scores = predictions.get_field('scores').numpy()

                if i == 0:
                    logger.info(
                        ' \ Note: inference on the first image will be slower than the '
                        'rest (caches and auto-tuning need to warm up)'
                    )
                if len(bbox) == 0:
                    continue

                if max(scores) < min(threshold.values()):
                    continue

                for i in range(len(bbox)):
                    class_text = labels_datasets[str(labels[i])]
                    if scores[i] < threshold[class_text]:
                        continue
                    writer.writerow(
                        [os.path.basename(im_name).split('.')[0], bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3],
                         class_text, scores[i]])
    else:
        logger.info('--------------> ERROR predict ------------->')
