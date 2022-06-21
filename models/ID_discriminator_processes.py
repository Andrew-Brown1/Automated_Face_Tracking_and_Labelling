
from .model_datasets import Extract_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import numpy as np
import os, pdb


def Extract_Features(args, episode, detection_dict, model):
    
    dataset = Extract_Dataset(detection_dict, os.path.join(args.temp_dir, episode), face_conf_thresh=args.face_conf_thresh)
    dataloader = DataLoader(dataset, batch_size=args.recog_batch_size, shuffle=False,
                            num_workers=int(args.num_workers), drop_last=False,
                            sampler=SequentialSampler(dataset))

    allfeatures = np.zeros((len(dataset.dataset), 256))

    for index, batch in enumerate(dataloader):
        if len(batch.size()) == 4:
            batch_input = batch.float().cuda()
            features = model(batch_input)
            allfeatures[index * args.recog_batch_size:min((index + 1) * args.recog_batch_size, len(dataset.dataset)),
            :] = features.detach().cpu().numpy()
        else:
            print('failed')

    TrackInfo = {}
    TrackInfo['x'] = []
    TrackInfo['y'] = []
    TrackInfo['w'] = []
    TrackInfo['h'] = []
    TrackInfo['Features'] = []
    TrackInfo['conf'] = []
    TrackInfo['ImageNames'] = []
    for ind in range(len(dataset.confs)):
        if not np.array_equal(allfeatures[ind, :], np.zeros((256,))):
            box = dataset.ROIs[ind]
            TrackInfo['x'].append(box[0])
            TrackInfo['y'].append(box[1])
            TrackInfo['w'].append(box[2])
            TrackInfo['h'].append(box[3])
            TrackInfo['Features'].append(allfeatures[ind, :])
            TrackInfo['conf'].append(dataset.confs[ind])
            TrackInfo['ImageNames'].append(dataset.paths[ind])

    return TrackInfo