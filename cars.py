"""
Mask R-CNN
Configurations and data loading code for the cars dataset.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
import pdb
import skimage

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils

import zipfile
import shutil

from config import Config
import utils
import model as modellib
import video_dataset
import pickle

import sys
sys.path.insert(0, "/staging/leuven/stg_00027/imob/car_detection")
import tools

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "dataset/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configurations
############################################################


class CarsConfig(Config):
    """Configuration for training on cars.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cars"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes
    NUM_CLASSES = 1 + 1  # Car and background

    NUM_KEYPOINTS = 1
    MASK_SHAPE = [28, 28]
    KEYPOINT_MASK_SHAPE = [56,56]
    # DETECTION_MAX_INSTANCES = 50
    TRAIN_ROIS_PER_IMAGE = 100
    MAX_GT_INSTANCES = 128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    USE_MINI_MASK = True
    MASK_POOL_SIZE = 14
    KEYPOINT_MASK_POOL_SIZE = 7
    LEARNING_RATE = 0.002
    STEPS_PER_EPOCH = 1000
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.005

    # PART_STR = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
    #             "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    #             "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    PART_STR = ["center"]
    # LIMBS = [0,-1,-1,5,-1,6,5,7,6,8,7,9,8,10,11,13,12,14,13,15,14,16]

Person_ID = 1


############################################################
#  Dataset
############################################################

class CarsDataset(utils.Dataset):
    def __init__(self, task_type= "instances",class_map = None):
        assert task_type in ["instances", "car_keypoints"]
        self.task_type = task_type
        # the connection between 2 close keypoints
        # self._skeleton = []
        # keypoint names
        # ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
        # "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        # "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        # self._keypoint_names = []
        super().__init__(class_map)
    def load_cars(self, dataset_dir, subset, class_ids=None):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        """

        trajs = sum(pickle.load(open('all_results.p', 'rb')), [])

        # f = open(os.path.join(dataset_dir, subset + '.txt'), 'r')
        # video_files = [x.strip() for x in f.readlines()]
        # dataset = video_dataset.Dataset_from_videos(video_files)

        # Add classes
        self.add_class("car", 1, "car")

        # Add images

        for j, (annot_id, traj) in enumerate(trajs):
            if traj.class_name == "car":
                if traj.manual_annotations is not None:
                    shape = traj.video.get_frame_shape()
                    for i in range(traj.manual_annotations.shape[0]):
                        # TODO: Create function for this
                        # im = traj.video.get_frame(i + traj.manual_annotations_start, side=traj.side)
                        pos = np.hstack([traj.manual_annotations[i, :].reshape((-1, 2)), np.zeros((1, 1))])
                        points_2d = np.dot(traj.projection_matrix, np.vstack([pos.T, np.ones((1, 1))])).T
                        points_2d[:, :2] = points_2d[:, :2] / points_2d[:, 2].reshape((-1, 1))
                        keypoint = points_2d.astype('uint16').ravel()[:2]
                        contours = traj.contours[traj.local_frame(i + traj.manual_annotations_start)]

                        self.add_image(
                                "car",
                                image_id = '%i_%i' % (j, i),
                                path = traj.video.avi_list[0],
                                side = traj.side,
                                width = shape[0],
                                height = shape[1],
                                frame = i,
                                annotations = [{'keypoints': keypoint, 'contours': contours}]
                                )

                # while metadata is not None:
                #     self.add_image(
                #         "car",
                #         image_id=metadata['frame_id'],
                #         # path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                #         width=metadata['frame_width'],
                #         height=metadata['frame_height'],
                #         annotations={'keypoints': 1})
                #         # annotations=coco.loadAnns(coco.getAnnIds(
                #             # imgIds=[i], catIds=class_ids, iscrowd=None)))
                #     metadata = dataset.get_next_frame(return_frame=False)


        # if(self.task_type == "person_keypoints"):
        #     #the connection between 2 close keypoints
        #     self._skeleton = coco.loadCats(Person_ID)[0]["skeleton"]
        #     #keypoint names
        #     # ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
        #     # "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        #     # "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        #     self._keypoint_names = coco.loadCats(Person_ID)[0]["keypoints"]

        #     self._skeleton = np.array(self._skeleton,dtype=np.int32)

        #     print("Skeleton:",np.shape(self._skeleton))
        #     print("Keypoint names:",np.shape(self._keypoint_names))
        # if return_coco:
        #     return coco

    @property
    def skeleton(self):
        return self._skeleton
    @property
    def keypoint_names(self):
        return self._keypoint_names


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['annotations'])], dtype=np.uint8)

        for i, annot in enumerate(info['annotations']):
            for ctr in annot['contours']:
                rr, cc = skimage.draw.polygon(ctr[:, 1], ctr[:, 0])
                mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

        ## # If not a COCO image, delegate to parent class.
        ## image_info = self.image_info[image_id]
        ## if image_info["source"] != "coco":
        ##     return super(CocoDataset, self).load_mask(image_id)

        ## instance_masks = []
        ## class_ids = []
        ## annotations = self.image_info[image_id]["annotations"]
        ## # Build mask of shape [height, width, instance_count] and list
        ## # of class IDs that correspond to each channel of the mask.
        ## for annotation in annotations:
        ##     class_id = self.map_source_class_id(
        ##         "coco.{}".format(annotation['category_id']))
        ##     if class_id:
        ##         m = self.annToMask(annotation, image_info["height"],
        ##                            image_info["width"])
        ##         # Some objects are so small that they're less than 1 pixel area
        ##         # and end up rounded out. Skip those objects.
        ##         if m.max() < 1:
        ##             continue
        ##         # Is it a crowd? If so, use a negative class ID.
        ##         if annotation['iscrowd']:
        ##             # Use negative class ID for crowds
        ##             class_id *= -1
        ##             # For crowd masks, annToMask() sometimes returns a mask
        ##             # smaller than the given dimensions. If so, resize it.
        ##             if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
        ##                 m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
        ##         instance_masks.append(m)
        ##         class_ids.append(class_id)

        ## # Pack instance masks into an array
        ## if class_ids:
        ##     mask = np.stack(instance_masks, axis=2)
        ##     class_ids = np.array(class_ids, dtype=np.int32)
        ##     return mask, class_ids
        ## else:
        ##     # Call super class to return an empty mask
        ##     return super(CocoDataset, self).load_mask(image_id)


    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """
        info = self.image_info[image_id]

        mask = np.zeros((info['height'], info['width'], len(info['annotations'])), dtype=np.uint8)
        for i, annot in enumerate(info['annotations']):
            center = annot['keypoints'].ravel()
            mask[center[0], center[1], :] = 1

        keypoints = np.zeros((len(info['annotations']), 1, 3))
        for i, annot in enumerate(info['annotations']):
            keypoints[i, 0, :] = np.hstack([annot['keypoints'].reshape((1, 2)), np.ones((1, 1))])

        return keypoints, mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)



        ### # If not a COCO image, delegate to parent class.
        ### image_info = self.image_info[image_id]
        ### if image_info["source"] != "coco":
        ###     return super(CocoDataset, self).load_mask(image_id)

        ### keypoints = []
        ### class_ids = []
        ### instance_masks = []
        ### annotations = self.image_info[image_id]["annotations"]
        ### # Build mask of shape [height, width, instance_count] and list
        ### # of class IDs that correspond to each channel of the mask.
        ### for annotation in annotations:
        ###     class_id = self.map_source_class_id(
        ###         "coco.{}".format(annotation['category_id']))
        ###     assert class_id == 1
        ###     if class_id:

        ###         #load masks
        ###         m = self.annToMask(annotation, image_info["height"],
        ###                            image_info["width"])
        ###         # Some objects are so small that they're less than 1 pixel area
        ###         # and end up rounded out. Skip those objects.
        ###         if m.max() < 1:
        ###             continue
        ###         # Is it a crowd? If so, use a negative class ID.
        ###         if annotation['iscrowd']:
        ###             # Use negative class ID for crowds
        ###             class_id *= -1
        ###             # For crowd masks, annToMask() sometimes returns a mask
        ###             # smaller than the given dimensions. If so, resize it.
        ###             if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
        ###                 m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
        ###         instance_masks.append(m)
        ###         #load keypoints
        ###         keypoint = annotation["keypoints"]
        ###         keypoint = np.reshape(keypoint,(-1,3))

        ###         keypoints.append(keypoint)
        ###         class_ids.append(class_id)

        ### # Pack instance masks into an array
        ### if class_ids:
        ###     keypoints = np.array(keypoints,dtype=np.int32)
        ###     class_ids = np.array(class_ids, dtype=np.int32)
        ###     masks = np.stack(instance_masks, axis=2)
        ###     return keypoints, masks, class_ids
        ### else:
        ###     # Call super class to return an empty mask
        ###     return super(CocoDataset, self).load_keypoints(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CarsConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=['mrcnn_bbox_fc', 'mrcnn_class_logits', 'mrcnn_mask'])

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CarsDataset()
        dataset_train.load_cars(args.dataset, "train")
        # dataset_train.load_cars(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()
        pdb.set_trace()

        # Validation dataset
        dataset_val = CarsDataset()
        dataset_val.load_cars(args.dataset, "minival")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CarsDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
