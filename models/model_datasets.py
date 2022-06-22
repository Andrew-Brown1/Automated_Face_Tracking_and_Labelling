import torch
import os
from torch.utils.data import Dataset, Sampler
import pdb
import PIL
import numpy as np
import random
from torch.autograd import Variable
import cv2
import scipy.misc

class detection_dataset_mobilenet(Dataset):

    def __init__(self, path_to_ims, resize):
        self.resize = resize

        self.testset_folder = path_to_ims

        self.test_dataset = [f for f in os.listdir(self.testset_folder)]

        self.num_images = len(self.test_dataset)

        if len(self.test_dataset) > 0:

            image_path = os.path.join(self.testset_folder, self.test_dataset[0])

            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

            img = np.float32(img_raw)
            if self.resize != 1:
                img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
            self.im_height, self.im_width, _ = img.shape

            self.scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        else:

            self.scale = torch.Tensor([1])


    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        image_path = os.path.join(self.testset_folder, self.test_dataset[index])
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        # im_height, im_width, _ = img.shape

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        # img = img.to(device)
        # scale = scale.to(device)

        return img

class detection_downloaded_image_dir(Dataset):
    def __init__(self, root_image_dir):

        self.root_image_dir = root_image_dir
        self.test_dataset = []
        self.GetImages()
        self.scale = None

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, index):

        image_path = self.test_dataset[index]
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        self.scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        self.im_height, self.im_width, _ = img.shape
        
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        return img

    def GetImages(self):

        dataset = []
        images = [f for f in os.listdir(self.root_image_dir) if 'clean' in f]
        for ind, image in enumerate(images):

            if image[-3:] == 'jpg' or image[-3:] == 'png':

                dataset.append(os.path.join(self.root_image_dir, image))

        self.test_dataset = dataset
        

class Extract_Dataset(Dataset):

    def __init__(self, detection_info, root_image_directory, face_conf_thresh=0.8,bbx_extension=0.4):
        self.detection_info = detection_info
        self.face_conf_thresh = face_conf_thresh
        self.root_dir = root_image_directory
        self.dataset = self.ReadDataset()

        self.bbx_extension = bbx_extension
        self.image_shape = (224, 224, 3)
        self.mean = (131.0912, 103.8827, 91.4953)
        # compute average image

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        imagepath = self.dataset[index][0]
        ROI = self.dataset[index][1]
        image = PIL.Image.open(imagepath)
        try:
            image = self.preprocess(image, ROI, self.bbx_extension)

            return torch.FloatTensor(image)
        except:
            return torch.FloatTensor(0)

    def ReadDataset(self):
        dataset = []
        paths = []
        ROIs = []
        confs = []

        
        face_info = self.detection_info

        for frame in face_info:
            for detection in range(len(face_info[frame][3])):
                if face_info[frame][3][detection] > self.face_conf_thresh:
                    confs.append(face_info[frame][3][detection])
                    paths.append(frame)
                    ROIs.append([face_info[frame][2][detection][0], face_info[frame][2][detection][1],
                                 face_info[frame][2][detection][2] - face_info[frame][2][detection][0],
                                 face_info[frame][2][detection][3] - face_info[frame][2][detection][1]])
                    dataset.append([os.path.join(self.root_dir, frame),
                                    [face_info[frame][2][detection][0], face_info[frame][2][detection][1],
                                     face_info[frame][2][detection][2] - face_info[frame][2][detection][0],
                                     face_info[frame][2][detection][3] - face_info[frame][2][detection][1]],
                                    face_info[frame][3][detection]])

        self.ROIs = ROIs
        self.confs = confs
        self.paths = paths
        return dataset

    def preprocess(self, img, ROI, extension):
        # process the images before feeding through the network
        # 1) crop and extend the images
        width = ROI[2]
        height = ROI[3]
        Length = (width + height) / 2
        centrepoint = [int(ROI[0]) + (width / 2), int(ROI[1]) + (height / 2)]
        x1 = int(centrepoint[0] - int((1 + extension) * Length / 2))
        y1 = int(centrepoint[1] - int((1 + extension) * Length / 2))
        x2 = int(centrepoint[0] + int((1 + extension) * Length / 2))
        y2 = int(centrepoint[1] + int((1 + extension) * Length / 2))
        x1 = max(1, x1)
        y1 = max(1, y1)
        x2 = min(x2, img.size[0])
        y2 = min(y2, img.size[1])
        img = img.crop((x1,y1,x2,y2))

        # 2) reshape

        short_size = 224.0
        crop_size = self.image_shape
        im_shape = np.array(img.size)  # in the format of (width, height, *)
        img = img.convert('RGB')

        ratio = float(short_size) / np.min(im_shape)
        img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),  # width
                               int(np.ceil(im_shape[1] * ratio))),  # height
                         resample=PIL.Image.BILINEAR)

        x = np.array(img)  # image has been transposed into (height, width)
        newshape = x.shape[:2]

        h_start = (newshape[0] - crop_size[0]) // 2
        w_start = (newshape[1] - crop_size[1]) // 2

        x = x[h_start:h_start + crop_size[0], w_start:w_start + crop_size[1]]
        x = x[:, :, :] - self.mean
        x = np.transpose(x, (2, 0, 1))
        return x

