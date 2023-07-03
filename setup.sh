# 1. Install `Timesformer` and download the pretrained model "TimeSformer_divST_8x32_224_HowTo100M.pyth".
mkdir ext
cd ext
git clone git@github.com:facebookresearch/TimeSformer.git
cd TimeSformer
python setup.py build develop
cd ../..
mkdir models
cd models
wget https://www.dropbox.com/s/9v8hcm88b9tc6ff/TimeSformer_divST_8x32_224_HowTo100M.pyth
cd ..

# 2. Install `BYOL-A` and download the pretrained model "AudioNTT2020-BYOLA-64x96d2048.pth".
cd ext
git clone git@github.com:nttcslab/byol-a.git
cd byol-a
pip install -r requirements.txt
cd ../..
mv ext/byol-a ext/byol_a
mv ext/byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth ./models
