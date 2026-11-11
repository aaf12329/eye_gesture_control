from __future__ import print_function
import numpy as np
from numpy.linalg import inv
import os
import cv2
import sys
sys.path.append('e:/retinaface_arcface')
from retinaface_arcface.models import *
import torch
import time
import sys



class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = '/data/Datasets/webface/CASIA-maxpy-clean-crop-144/'
    train_list = '/data/Datasets/webface/train_data_13938.txt'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/home/liuyuan/final_design/face/arcface-pytorch/lfw-align-128'
    lfw_test_list = '/home/liuyuan/final_design/face/arcface-pytorch/lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    
def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(face_img):
    image = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) # 128 128 灰度图
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image))) # 128 128 2
    image = image.transpose((2, 0, 1)) # 2 128 128
    image = image[:, np.newaxis, :, :] # 2 1 128 128
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=1):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path) # 2 1 128 128
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1
            with torch.no_grad():
                data = torch.from_numpy(images)
                data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # 60 1 128 128
                output = model(data) # 60 512
                output = output.data.cpu().numpy()

                fe_1 = output[::2] # 30 512
                fe_2 = output[1::2] # 30 512
                feature = np.hstack((fe_1, fe_2)) # 30 1024
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size) # (7701, 1024) 257
    print(features.shape, cnt)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc

def getTransMatrix(src_pts, dst_pts):

    num_pts = src_pts.shape[0]
    x = src_pts[:, 0].reshape((-1, 1))
    y = src_pts[:, 1].reshape((-1, 1))
    temp1 = np.hstack((x, -y, np.ones((num_pts, 1)), np.zeros((num_pts, 1))))
    temp2 = np.hstack((y, x, np.zeros((num_pts, 1)), np.ones((num_pts, 1))))
    X_src = np.vstack((temp1, temp2))
    Y_dst = np.vstack((dst_pts[:, 0].reshape((-1, 1)), dst_pts[:, 1].reshape((-1, 1))))

    X = np.dot(X_src.T, X_src) + np.eye(4, 4) * 1e-6  # (X'X + epsilon * I)
    Y = np.dot(X_src.T, Y_dst)
    r = np.dot(inv(X), Y)

    ref_data = np.dot(X_src, r)[:, 0]
    ref_pts = np.zeros((num_pts, 2))
    for i in range(num_pts):
        ref_pts[i] = [ref_data[i], ref_data[num_pts+i]]

    r = r[:, 0]
    Trans = np.array([[r[0], -r[1], r[2]],[r[1],  r[0], r[3]]])

    return Trans, ref_pts

def process(retinaface, image,  dst_pts, model):
    """进行人识别"""
    if image is None:
        print("Error: Failed to load image")
        return False, None, None
    width = image.shape[1]
    height = image.shape[0]
    ali_imgs = None
    feature = None
    flag = False
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    b1s, b2s = retinaface.detect_image(image)
    if len(b1s) > 0:
        for i in range(len(b1s)):
            b1 = b1s[i]
            b2 = b2s[i]
            b1 = list(map(int, b1))
            b1[1] = 0 if b1[1]<0 else b1[1]
            b1[0] = 0 if b1[0]<0 else b1[0]
            b1[2] = width if b1[2]>width else b1[2]
            b1[3] = height if b1[3]>height else b1[3]
            faces = image[b1[1]:b1[3], b1[0]:b1[2],:]
            b2[:,0] = b2[:, 0] - b1[0]
            b2[:,1] = b2[:, 1] - b1[1]
            Trans, ref_pts = getTransMatrix(b2, dst_pts)
            # faces = cv2.resize(faces, (100, 100))
            ali_img = cv2.warpAffine(faces, Trans, (128, 128), borderMode=3, borderValue=(0,0,0)) # 3 4
            # cv2.imshow('dsd', ali_img)
            ali_img = load_image(ali_img) # 8 1 128 128
            if ali_imgs is None:
                ali_imgs = ali_img
            else:
                ali_imgs = np.concatenate((ali_imgs, ali_img), axis=0) # 16 1 128 128
        with torch.no_grad():
            data = torch.from_numpy(ali_imgs)
            data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # 60 1 128 128
            output = model(data) # 60 512
            output = output.data.cpu().numpy()

            fe_1 = output[::2] # 30 512
            fe_2 = output[1::2] # 30 512
            feature = (fe_1 + fe_2) / 2
            # feature = np.hstack((fe_1, fe_2)) # 8 1024
            flag = True
    return flag, feature, b1s

def img_floder(floder_path, retinaface, dst_pts, model):
    if str(floder_path)[-1] != '/':
        floder_path = floder_path + '/'
    if os.path.isdir(floder_path):
        known_face_names = []
        known_face_encodings = None
        img_formats = ['.jpg', 'jpeg', '.png', '.tif']
        paths = os.listdir(floder_path)
        if '__pycache__' in paths:
            paths.remove('__pycache__')
        # print(paths)
        for path in paths:
            if os.path.isdir(floder_path + path):  
                face_encoding = [np.zeros(512,)]
                pic_num = 1
                if len(os.listdir(floder_path + path)) == 0:
                    continue
                known_face_names.append(str(path))
                for img_path in os.listdir(floder_path + path):
                    if str(img_path[-4:]) in img_formats:
                        pic_num += 1
                        orig_image = cv2.imread(floder_path + path + '/' + img_path)
                        _, image_encoding, _ = process(retinaface, orig_image,  dst_pts, model)
                        assert len(image_encoding)==1, floder_path + path + '/' + img_path+'照片没有检测到人脸，请检测重试'
                        face_encoding += image_encoding[0]
                face_encoding = face_encoding / (pic_num - 1)
                if known_face_encodings is None:
                    known_face_encodings = face_encoding
                else:
                    known_face_encodings = np.vstack((known_face_encodings, face_encoding))
                # known_face_encodings.append(face_encoding[0])
        return known_face_names, known_face_encodings
    else:
        raise Warning('not a floder')
        return -1
