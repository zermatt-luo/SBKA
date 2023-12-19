import argparse
import pickle
import numpy as np
import datetime
from torch.utils.data import DataLoader

from utils.pq import PQ
from utils.eval_tools import eval_precision, eval_AP_inner
from Sketchy import SketchyDataset
from scipy.spatial.distance import cdist
from models import convnextv2 as convnextv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os


def prepare_features():
    model = convnextv2.__dict__[args.model](num_classes=1000, linear_num=args.nb_classes)
    device_ids = [0, 1]   # we use two GPUs to train our model
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    resume = os.path.join(args.resume_dir, 'model_best.pth.tar')
    checkpoint = torch.load(resume)
    save_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    print(model_dict.keys())
    print(save_dict.keys())

    trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
    print('trashed vars from resume dict:')
    print(trash_vars)
    resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
    model_dict.update(resume_dict)
    model.load_state_dict(model_dict, strict=True)

    cudnn.benchmark = True

    # load data
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])

    sketchy_zero_ext = SketchyDataset(split='zero', version='all_photo', zero_version=args.zero_version, \
                                      transform=transformations, aug=False)

    zero_loader_ext = DataLoader(dataset=sketchy_zero_ext, \
                                 batch_size=args.batch_size, shuffle=False, num_workers=0)

    sketchy_zero = SketchyDataset(split='zero', zero_version=args.zero_version, transform=transformations, aug=False)
    zero_loader = DataLoader(dataset=sketchy_zero, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(str(datetime.datetime.now()) + ' data loaded.')

    predicted_features_gallery, gt_labels_gallery = get_features(zero_loader_ext, model)
    predicted_features_query, gt_labels_query = get_features(zero_loader, model)

    scores = cdist(predicted_features_query, predicted_features_gallery)
    print('euclidean distance calculated')

    with open(os.path.join(args.resume_dir, 'features_zero.pickle'), 'wb') as fh:
        pickle.dump([predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, None],
                    fh)

    return predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores


def get_features(data_loader, model):
    model.eval()
    features_all = []
    targets_all = []

    avgpool = nn.AdaptiveAvgPool2d(1).cuda()
    for i, (input, target) in enumerate(data_loader):
        if i % 10 == 0:
            print(i, end=' ', flush=True)

        input = torch.autograd.Variable(input, requires_grad=False).cuda()

        # compute output
        with torch.no_grad():
            _, _, features = model(input)
        features = F.normalize(features)

        features = features.cpu().detach().numpy()

        features_all.append(features.reshape(input.size()[0], -1))
        targets_all.append(target.detach().numpy())

    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)

    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))

    return features_all, targets_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One-to-Many cluster matching by the Gaussian Mixture Model(GMM)')
    parser.add_argument('--model', default='convnextv2_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--resume_dir', '-s', metavar='DIR', default='../checkpoint/sketchy/zeroshot1')
    parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot1', type=str,
                        help='zeroshot version for training and testing (default: zeroshot1)')
    parser.add_argument('--batch_size', default=80, type=int, metavar='N',
                        help='number of samples per batch')
    parser.add_argument('--nb_classes', default=100, type=int,
                        help='number of the classification types')
    parser.add_argument('--mode', default=2, type=int, metavar='N',
                        help='model 1 is no one_to_many cluster matching, model 2 is one-to-many cluster matching')

    args = parser.parse_args()
    print(args)
    feature_file = os.path.join(args.resume_dir, 'features_zero.pickle')
    if os.path.isfile(feature_file):
        print('load saved SBIR features')
        with open(feature_file, 'rb') as fh:
            predicted_features_gallery, gt_labels_gallery, \
            predicted_features_query, gt_labels_query, \
            scores = pickle.load(fh)

        if scores is None:
            print('euclidean distance is being calculated')
            scores = cdist(predicted_features_query, predicted_features_gallery)

    else:
        print('prepare SBIR features using saved model')
        predicted_features_gallery, gt_labels_gallery, \
        predicted_features_query, gt_labels_query, \
        scores = prepare_features()

    # print(scores)
    # print(scores.shape)

    if args.mode == 2:
        print("M, K, mAP, Prec,")
        Ms = [2, 4]
        Ks = [22, 23, 24, 25, 26, 27, 28]

        for M in Ms:
            for K in Ks:
                pq = PQ(M=M, verbose=False, Ks=K)
                pq.fit(predicted_features_gallery)

                predicted_features_gallery_code = pq.encode(predicted_features_gallery)
                binary_scores = []
                for query in predicted_features_query:
                    dists = pq.dtable(query).adist(predicted_features_gallery_code)
                    binary_scores.append(dists)
                binary_scores = np.array(binary_scores)
                binary_scores = 0.8 * binary_scores + 0.2 * scores

                mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
                for fi in range(predicted_features_query.shape[0]):
                    mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery, top=None)
                    mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)

                mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()

                prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
                for fi in range(predicted_features_query.shape[0]):
                    prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery, top=100)
                    prec_ls_binary[gt_labels_query[fi]].append(prec_binary)
                prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
                print("{}, {}, {}, {},".format(M, K, mAP_binary, prec_binary))

    elif args.mode == 1:
        mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
        for fi in range(predicted_features_query.shape[0]):
            mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=None)
            mAP_ls[gt_labels_query[fi]].append(mapi)

        for mAPi, mAPs in enumerate(mAP_ls):
            print(str(mAPi) + ' ' + str(np.nanmean(mAPs)) + ' ' + str(np.nanstd(mAPs)))

        prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
        for fi in range(predicted_features_query.shape[0]):
            prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery, top=100)
            prec_ls[gt_labels_query[fi]].append(prec)

        for preci, precs in enumerate(prec_ls):
            print(str(preci) + ' ' + str(np.nanmean(precs)) + ' ' + str(np.nanstd(precs)))

        mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()
        prec_binary = np.array([np.nanmean(pre) for pre in prec_ls]).mean()
        print(mAP_binary, prec_binary)
    else:
        print("error! no this mode")
