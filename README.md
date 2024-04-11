# Deep Exemplar-based Video Colorization (Pytorch Remplementation)
This is a reimplement of [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/).  We build this branch with the purpose to provide a training demo as requested by [@doolachen](https://github.com/doolachen) in [issue 8](https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/issues/8) for the excellent work [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/). 

Note that, different from the original code that use pre-computed optical flow, we use [RAFT](https://github.com/princeton-vl/RAFT/) to calculate the optical flow on the fly. In addition, we only consider video dataset while the authors use both video and image datasets. Besides, we only want to provide an example for preparing your training dataset and didn't expect that this project would produce comparable performance to the original method. 

For more details and license, please refer to the original git repository. Thanks again for the great work from the authors of [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/). 

# Environment
```
conda create -n ColorVid_py38 python=3.8
conda activate ColorVid_py38

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt

# support for optical flow warping ops
pip install -U openmim
mim install mmcv-full
git clone -b 0.x https://github.com/open-mmlab/mmediting.git
pip3 install -e ./mmagic/

```
# Training 
```
CUDA_VISIBLE_DEVICES=3 python train.py
```