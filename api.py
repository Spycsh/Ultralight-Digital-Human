import time
import os
import uuid
import cv2
import tqdm
import shutil
from starlette.middleware.cors import CORSMiddleware
import argparse
import base64
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi import File, UploadFile, HTTPException
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import librosa

from unet import Model

# from unet2 import Model
# from unet_att import Model

import time
from tqdm import tqdm


app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


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

@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model
    if speech.ndim ==2:
        speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it            
        hidden_states = hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

def make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor

def convert_audio_to_hubert_feat(wav_path, return_ori_audio=False):
    speech, sr = sf.read(wav_path)
    speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
    print("SR: {} to {}".format(sr, 16000))
    hubert_hidden = get_hubert_from_16k_speech(speech_16k, device=device)
    hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)
    # np.save(wav_name.replace('.wav', '_hu.npy'), hubert_hidden.detach().numpy())
    # print(hubert_hidden.detach().numpy().shape)
    if return_ori_audio:
        return hubert_hidden.detach().numpy(), speech_16k
    return hubert_hidden.detach().numpy()

def generate_video(output_video_path):
    with open(output_video_path, mode="rb") as file_like:
        yield from file_like

def generate_video_as_text_stream(output_video_path):
    with open(output_video_path, mode="rb") as file_like:
        bytes = file_like.read()
        b64_str = base64.b64encode(bytes).decode()
        yield f"data: {b64_str}\n\n"

async def dh_infer_gen(file_path, bs=50):
    """Input audio file path, Output batched video generator."""
    audio_feats, speech_16k = convert_audio_to_hubert_feat(file_path, return_ori_audio=True)
    uid = str(uuid.uuid4())

    total_frame_num = audio_feats.shape[0]
    total_run = (total_frame_num + bs - 1) // bs

    img_dir = os.path.join(dataset_dir, "full_body_img/")
    lms_dir = os.path.join(dataset_dir, "landmarks/")
    len_img = len(os.listdir(img_dir)) - 1
    exm_img = cv2.imread(img_dir+"0.jpg")
    init_h, init_w = exm_img.shape[:2]

    step_stride = 0
    img_idx = 0

    S = time.time()

    for i in tqdm(range(total_run)):
        save_path = f"{uid}_{i}.mp4"
        if mode=="hubert":
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (init_w, init_h))
        if mode=="wenet":
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (init_w, init_h))


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

        # # TODO bind audio with the save_path video
        # # clip the partial audio
        # result_path = f"result_{save_path}"
        # partial_audio_path = f"result_{save_path.split('.')[0]}.wav"
        # # Duration of each video chunk in seconds
        # chunk_duration = bs / 25 if mode == "hubert" else 20
        # # Convert chunk duration to samples
        # sample_wav_for_each_run = int(chunk_duration * 16000)
        # partial_audio = speech_16k[i*sample_wav_for_each_run: (i+1)*sample_wav_for_each_run]
        # sf.write(partial_audio_path, partial_audio, 16000)
        # TODO This gernerated clips is not length identical to the original audio/video!!
        # os.system(f"ffmpeg -i {save_path} -i {partial_audio_path} -c:v libx264 -c:a aac -ar 16000 -r 25 -shortest {result_path}")
        # os.remove(partial_audio_path)
        # for res_str in generate_video_as_text_stream(result_path):
        for res_str in generate_video_as_text_stream(save_path):
            yield res_str
        if i == 0:
            print(f"First batch video returned in {time.time()-S} sec")
        # os.remove(save_path)

    print(f"Unet generation and video write takes {time.time()-S} sec")
    yield "data: [DONE]\n\n"



def dh_infer(file_name):
    uid = file_name.split(".")[0]
    audio_feats = convert_audio_to_hubert_feat(file_name)

    step_stride = 0
    img_idx = 0

    save_path = uid + ".mp4"

    # audio_feats = np.load(audio_feat_path)
    img_dir = os.path.join(dataset_dir, "full_body_img/")
    lms_dir = os.path.join(dataset_dir, "landmarks/")
    len_img = len(os.listdir(img_dir)) - 1
    exm_img = cv2.imread(img_dir+"0.jpg")
    h, w = exm_img.shape[:2]

    if mode=="hubert":
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
    if mode=="wenet":
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))

    for i in tqdm(range(audio_feats.shape[0])):
        if img_idx>len_img - 1:
            step_stride = -1
        if img_idx<1:
            step_stride = 1
        img_idx += step_stride
        img_path = img_dir + str(img_idx)+'.jpg'
        lms_path = lms_dir + str(img_idx)+'.lms'
        
        img = cv2.imread(img_path)
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
        crop_img = img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:164, 4:164].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
 
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)

        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
        
        audio_feat = get_audio_features(audio_feats, i)
        if mode=="hubert":
            audio_feat = audio_feat.reshape(32,32,32)
        if mode=="wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        audio_feat = audio_feat[None]
        audio_feat = audio_feat.to(device)
        img_concat_T = img_concat_T.to(device)
        
        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)[0]
            
        pred = pred.cpu().numpy().transpose(1,2,0)*255
        pred = np.array(pred, dtype=np.uint8)
        crop_img_ori[4:164, 4:164] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (w, h))
        img[ymin:ymax, xmin:xmax] = crop_img_ori
        video_writer.write(img)

    video_writer.release()

    # combine together the audio and video

    os.system(f"ffmpeg -i {save_path} -i {file_name} -c:v libx264 -c:a aac {uid}_final.mp4")
    # Remove upload audio file
    os.remove(file_name)
    os.remove(save_path)
    print(f"write to {uid}_final.mp4")
    if not os.path.exists(f"{uid}_final.mp4"):
        raise FileNotFoundError(f"Output file not generated: {uid}_final.mp4")

    return f"{uid}_final.mp4"


@app.get("/v1/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/v1/digital_human")
async def digital_human(file: UploadFile = File(...)):
    """Input: audio file, Output: Streaming video response."""
    print("Digital human inference begin.")
    print(file.content_type)
    if file.content_type not in ["audio/wav", "application/octet-stream", "audio/wave", ]:
        raise HTTPException(status_code=400, detail="File must be a WAV format")

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    # Save the uploaded file
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    res_path_gen = dh_infer(file_name,)

    return StreamingResponse(generate_video(res_path_gen), media_type="video/mp4")

@app.post("/v1/digital_human_flow")
async def digital_human(request: Request):
    """Input: dict, Output: Streaming video response."""
    print("Digital human inference begin.")

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    request_dict = await request.json()
    audio_b64_str = request_dict.pop("audio")
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(audio_b64_str))

    res_path = dh_infer(file_name,)

    return StreamingResponse(generate_video(res_path), media_type="video/mp4")

@app.post("/v1/digital_human_fast")
async def digital_human_fast(request: Request):
    """Input: path to the audio, Output: Streaming video response."""
    print("Digital human inference begin.")

    request_dict = await request.json()
    file_path = request_dict.pop("audio_path")

    # FIXME only the first clip is returned!
    # return StreamingResponse(dh_infer_gen(file_path,), media_type="video/mp4")
    return StreamingResponse(dh_infer_gen(file_path,), media_type="text/event_stream")

    # video_files = [f"result_1f4d2584-e9e6-47a5-8f96-3fca0f510755_{i}.mp4" for i in range(5)]
    # async def generate_video_stream(file_paths):
    #     for file_path in file_paths:
    #         print(f"xxx: {file_path}")

    #         with open(file_path, 'rb') as video_file:
    #             # yield from video_file  # Stream the file content in chunks FIXME Why only return the first??
    #             bytes = video_file.read()
    #         b64_str = base64.b64encode(bytes).decode()
    #         # breakpoint()
    #         yield f"data: {b64_str}\n\n"
    #         time.sleep(5)
    #     yield "data: [DONE]\n\n"

    # return StreamingResponse(generate_video_stream(video_files), media_type="text/event_stream")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='API Server', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)

    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--dataset', type=str, default="data_utils")  
    # parser.add_argument('--audio_feat', type=str, default="")
    # parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    checkpoint = args.checkpoint
    # save_path = args.save_path
    dataset_dir = args.dataset
    # audio_feat_path = args.audio_feat
    mode = args.asr
    device = args.device

    net = Model(6, mode)
    net.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
    if device == "hpu":
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch.gpu_migration
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        net = wrap_in_hpu_graph(net)
    net.to(device)
    net.eval()

    print("Loading the Wav2Vec2 Processor...")
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    print("Loading the HuBERT Model...")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model = hubert_model.to(device)

    uvicorn.run(app, host=args.host, port=args.port)



