# Ultralight Digital Human

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10-aff.svg"></a>
    <a href="https://github.com/anliyuan/Ultralight-Digital-Human/stargazers"><img src="https://img.shields.io/github/stars/anliyuan/Ultralight-Digital-Human?color=ccf"></a>
  <br>
    <br>
</p>

A Ultralight Digital Human model can run on mobile devices in real time!!!

一个能在移动设备上实时运行的数字人模型,据我所知，这应该是第一个开源的如此轻量级的数字人模型。

Lets see the demo.⬇️⬇️⬇️

先来看个demo⬇️⬇️⬇️

![DigitalHuman](https://github.com/user-attachments/assets/9d0b37ee-2076-4b4f-93ba-eb939a9fb427)

## 最近issue里提到很多效果不佳的案例。我尝试复现，并没有复现出来。但是我发现如果你视频中声音质量比较差的话，效果大概率不会好。声音质量比较差指的是：1）存在难以忽略的噪声。2）在空旷的房间里录制的视频有回音。3）视频人声不清楚。建议录制视频时候使用外接麦克风，不用拍摄设备自带的麦克风。我自己尝试了声音清晰的情况，不论是wenet还是hubert，效果都非常棒。

## Train

It's so easy to train your own digital human.I will show you step by step.

训练一个你自己的数字人非常简单，我将一步步向你展示。

### install pytorch and other libs

``` bash
conda create -n dh python=3.10
conda activate dh
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install mkl=2024.0
pip install opencv-python
pip install transformers
pip install numpy==1.23.5
pip install soundfile
pip install librosa
pip install onnxruntime
```

I only ran on pytorch==1.13.1, Other versions should also work.

我是在1.13.1版本的pytorch跑的，其他版本的pytorch应该也可以。

Download wenet encoder.onnx from https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link 

and put it in data_utils/

### Data preprocessing

Prepare your video, 3~5min is good. Make sure that every frame of the video has the person's full face exposed and the sound is clear without any noise, put it in a new folder.I will provide a demo video.

准备好你的视频，3到5分钟的就可以，必须保证视频中每一帧都有整张脸露出来的人物，声音清晰没有杂音，把它放到一个新的文件夹里面。我会提供一个demo视频，来自康辉老师的口播，侵删。

First of all, we need to extract audio feature.I'm using 2 different extractor from wenet and hubert, thank them for their great work.

wenet的代码和与训练模型来自:https://github.com/Tzenthin/wenet_mnn

首先我们需要提取音频特征，我用了两个不同的特征提取起，分别是wenet和hubert，感谢他们。

When you using wenet, you neet to ensure that your video frame rate is 20, and for hubert,your video frame rate should be 25.

如果你选择使用wenet的话，你必须保证你视频的帧率是20fps，如果选择hubert，视频帧率必须是25fps。

In my experiments, hubert performs better, but wenet is faster and can run in real time on mobile devices.

在我的实验中，hubert的效果更好，但是wenet速度更快，可以在移动端上实时运行

And other steps are in data_utils/process.py, you just run it like this.

其他步骤都写在data_utils/process.py里面了，没什么特别要注意的。

``` bash
cd data_utils
python process.py YOUR_VIDEO_PATH --asr hubert
```

Then you wait.

然后等它运行完就行了

### train

After the preprocessing step, you can start training the model.

上面步骤结束后，就可以开始训练模型了。

Train a syncnet first for better results.

先训练一个syncnet，效果会更好。

``` bash
cd ..
python syncnet.py --save_dir ./syncnet_ckpt/ --dataset_dir ./data_utils/ --asr hubert
```

Then find a best one（low loss） to train digital human model.

然后找一个loss最低的checkpoint来训练数字人模型。

``` bash
cd ..
python train.py --dataset_dir ./data_utils/ --save_dir ./checkpoint/ --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpt
```

## inference

Before run inference, you need to extract test audio feature(i will merge this step and inference step), run this

在推理之前，需要先提取测试音频的特征（之后会把这步和推理合并到一起去），运行(音频采样率需要是16000)

``` bash
python data_utils/hubert.py --wav your_test_audio.wav  # when using hubert

or

python data_utils/python wenet_infer.py your_test_audio.wav  # when using wenet
```

then you get your_test_audio_hu.npy or your_test_audio_wenet.npy

then run
``` bash
python inference.py --asr hubert --dataset ./data_utils/ --audio_feat male_gaudi_1_hu.npy --save_path out.mp4 --checkpoint checkpoint/5.pth
```

To merge the audio and the video, run

``` bash
ffmpeg -i xxx.mp4 -i your_audio.wav -c:v libx264 -c:a aac result_test.mp4
```

## Enjoy🎉🎉🎉

这个模型是支持流式推理的，但是代码还没有完善，之后我会提上来。

关于在移动端上运行也是没问题的，只需要把现在这个模型通道数改小一点，音频特征用wenet就没问题了。相关代码我也会在之后放上来。

if you have some advice, open an issue or PR.

如果你有改进的建议，可以提个issue或者PR。

If you think this repo is useful to you, please give me a star.

如果你觉的这个repo对你有用的话，记得给我点个star

BUY ME A CUP OF COFFE⬇️⬇️⬇️
<table>
  <tr>
    <td><img src="demo/15bef5a6d08434c0d70f0ba39bb14fc0.JPG" width="180"/></td>
    <td><img src="demo/36d2896f13bee68247de6ccc89b17a94.JPG" width="180"/></td>
  </tr>
</table>
