import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
# from unet2 import Model
# from unet_att import Model

import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="hubert")
parser.add_argument('--dataset', type=str, default="")  
parser.add_argument('--audio_feat', type=str, default="")
parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
dataset_dir = args.dataset
audio_feat_path = args.audio_feat
mode = args.asr
device = args.device
bs = args.batch_size

def get_audio_features(features, index):
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds

audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

if mode=="hubert":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
if mode=="wenet":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))
step_stride = 0
img_idx = 0

net = Model(6, mode)
net.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
if device == "hpu":
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.gpu_migration
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    net = wrap_in_hpu_graph(net)
net.to(device)
net.eval()

total_frame_num = audio_feats.shape[0]
total_run = (total_frame_num + bs - 1) // bs

S = time.time()

for i in tqdm(range(total_run)):
    # buffer the images for post-processing
    batch_img = []
    batch_crop_img_ori = []
    batch_xmin = []
    batch_xmax = []
    batch_ymin = []
    batch_ymax = []
    batch_h = []
    batch_w = []

    for ii in range(bs):
        if i*bs + ii >= total_frame_num: # if current batch frame idx exceed audio length, just forward the unpadded batch
            break

        if img_idx>len_img - 1: # circulate when audio length is longer than video imgs length
            step_stride = -1
        if img_idx<1:
            step_stride = 1
        img_idx += step_stride
        img_path = img_dir + str(img_idx)+'.jpg'
        lms_path = lms_dir + str(img_idx)+'.lms'
        img = cv2.imread(img_path)
        batch_img.append(img)
        # TODO preload in memory
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        batch_xmin.append(xmin)
        batch_ymin.append(ymin)
        batch_xmax.append(xmax)
        batch_ymax.append(ymax)

        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        batch_h.append(h)
        batch_w.append(w)
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        crop_img_ori = crop_img.copy()

        batch_crop_img_ori.append(crop_img_ori)

        img_real_ex = crop_img[4:164, 4:164].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
        
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
        
        audio_feat = get_audio_features(audio_feats, i*bs + ii) # index change
        if mode=="hubert":
            audio_feat = audio_feat.reshape(32,32,32)

        if mode=="wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        audio_feat = audio_feat[None]
        audio_feat = audio_feat.to(device)
        img_concat_T = img_concat_T.to(device)

        # batch imgs here! prepare net(img_concat_T, audio_feat)
        if ii == 0:
            batch_img_concat_T = img_concat_T
            batch_audio_feat = audio_feat
        else:
            batch_img_concat_T = torch.cat([batch_img_concat_T, img_concat_T])
            batch_audio_feat = torch.cat([batch_audio_feat, audio_feat])

    with torch.no_grad():
        # Control here to close the mouth?
        # audio_feat.fill_(0)
        # feed the batched input
        # print(batch_audio_feat.size(0))
        # If not full batch, pad to full batch
        # This is useful for HPU static shape
        if device == "hpu" and batch_audio_feat.size(0) < bs:
            concrete_bs = batch_audio_feat.size(0)
            batch_img_concat_T = torch.nn.functional.pad(batch_img_concat_T, (0,0,0,0,0,0,0,bs-concrete_bs), value=0)
            batch_audio_feat = torch.nn.functional.pad(batch_audio_feat, (0,0,0,0,0,0,0,bs-concrete_bs), value=0)
            
            batch_pred = net(batch_img_concat_T, batch_audio_feat)[:concrete_bs]
            # print(batch_img_concat_T.size(0))
            # print(batch_pred.size(0))
        else:
            batch_pred = net(batch_img_concat_T, batch_audio_feat)

    for jj in range(bs):
        if i*bs + jj >= total_frame_num: # if current batch frame idx exceed audio length, just forward the unpadded batch
            break
        pred = batch_pred[jj].cpu().numpy().transpose(1,2,0)*255
        pred = np.array(pred, dtype=np.uint8)
        crop_img_ori = batch_crop_img_ori[jj]
        crop_img_ori[4:164, 4:164] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (batch_w[jj], batch_h[jj]))
        img = batch_img[jj]
        img[batch_ymin[jj]:batch_ymax[jj], batch_xmin[jj]:batch_xmax[jj]] = crop_img_ori
        video_writer.write(img)
video_writer.release()

print(f"Unet generation and video write takes {time.time()-S} sec")

# for i in tqdm(range(audio_feats.shape[0])):
#     if img_idx>len_img - 1:
#         step_stride = -1
#     if img_idx<1:
#         step_stride = 1
#     img_idx += step_stride
#     img_path = img_dir + str(img_idx)+'.jpg'
#     lms_path = lms_dir + str(img_idx)+'.lms'
    
#     img = cv2.imread(img_path)
#     lms_list = []
#     with open(lms_path, "r") as f:
#         lines = f.read().splitlines()
#         for line in lines:
#             arr = line.split(" ")
#             arr = np.array(arr, dtype=np.float32)
#             lms_list.append(arr)
#     lms = np.array(lms_list, dtype=np.int32)
#     xmin = lms[1][0]
#     ymin = lms[52][1]

#     xmax = lms[31][0]
#     width = xmax - xmin
#     ymax = ymin + width
#     crop_img = img[ymin:ymax, xmin:xmax]
#     h, w = crop_img.shape[:2]
#     crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
#     crop_img_ori = crop_img.copy()
#     img_real_ex = crop_img[4:164, 4:164].copy()
#     img_real_ex_ori = img_real_ex.copy()
#     img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
    
#     img_masked = img_masked.transpose(2,0,1).astype(np.float32)
#     img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
#     img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
#     img_masked_T = torch.from_numpy(img_masked / 255.0)
#     img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    
#     audio_feat = get_audio_features(audio_feats, i)
#     if mode=="hubert":
#         audio_feat = audio_feat.reshape(32,32,32)
#     if mode=="wenet":
#         audio_feat = audio_feat.reshape(256,16,32)
#     audio_feat = audio_feat[None]
#     audio_feat = audio_feat.to(device)
#     img_concat_T = img_concat_T.to(device)
    
#     with torch.no_grad():
#         # Control here to close the mouth?
#         # audio_feat.fill_(0)
#         pred = net(img_concat_T, audio_feat)[0]
        
#     pred = pred.cpu().numpy().transpose(1,2,0)*255
#     pred = np.array(pred, dtype=np.uint8)
#     crop_img_ori[4:164, 4:164] = pred
#     crop_img_ori = cv2.resize(crop_img_ori, (w, h))
#     img[ymin:ymax, xmin:xmax] = crop_img_ori
#     video_writer.write(img)
# video_writer.release()

# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
# os.system(f"ffmpeg -i {save_path} -i ./female_gaudi.wav -c:v libx264 -c:a aac result_test.mp4")