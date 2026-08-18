# Learning How the World Evolves: Extrapolative Video World Models via Latent Dynamics Reasoning

[![Page](https://img.shields.io/badge/Project-Page-pink?logo=googlechrome&logoColor=white)](https://lat-dyn-reason.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.09926)
[![GitHub](https://img.shields.io/github/stars/Lat-Dyn-Reason/Lat-Dyn-Reason?style=default&label=GitHub%20★&logo=github)](https://github.com/Lat-Dyn-Reason/Lat-Dyn-Reason)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Data-yellow)](https://huggingface.co/datasets/haodongli/LDR)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-yellow)](https://huggingface.co/haodongli/LDR)

<p>
  <a href="https://haodong2000.github.io/" target="_blank" rel="noopener">Haodong Li</a><sup>12</sup>&nbsp;
  <a href="https://www.shaotengliu.com/" target="_blank" rel="noopener">Shaoteng Liu</a><sup>2</sup>&nbsp;
  <a href="https://stevewongv.github.io/" target="_blank" rel="noopener">Tianyu Wang</a><sup>2</sup>&nbsp;
  <a href="https://chongjiange.github.io/" target="_blank" rel="noopener">Chongjian Ge</a><sup>2</sup>&nbsp;
  <a href="https://sihuiji.github.io/" target="_blank" rel="noopener">Sihui Ji</a><sup>2</sup>&nbsp;
  <a href="https://me.jiahanzhang.top/" target="_blank" rel="noopener">Jiahan Zhang</a><sup>2</sup>&nbsp;
  <a href="https://linxin0.github.io/" target="_blank" rel="noopener">Xin Lin</a><sup>12</sup>&nbsp;
  <a href="https://suikasibyl.github.io/" target="_blank" rel="noopener">Haolin Lu</a><sup>1</sup>&nbsp;
  <a href="https://sites.google.com/site/zhelin625/home" target="_blank" rel="noopener">Zhe Lin</a><sup>2</sup>&nbsp;
  <a href="https://cseweb.ucsd.edu/~mkchandraker/" target="_blank" rel="noopener">Manmohan Chandraker</a><sup>1</sup>
<br>
  <sup>1</sup>UCSD&nbsp;
  <sup>2</sup>Adobe
</p>

## Setup

> This installation was tested on: Ubuntu 22.04 LTS, Python 3.10, CUDA 12.1, NVIDIA A100-80GB.

1. Clone the repository
```bash
git clone https://github.com/Lat-Dyn-Reason/Lat-Dyn-Reason.git
cd Lat-Dyn-Reason
```
2. Install dependencies
```bash
conda env create -f environment.yml
conda activate ldr
```
3. Download checkpoints and data
```bash
hf auth login
hf download haodongli/LDR --local-dir checkpoints
hf download haodongli/LDR --repo-type dataset --local-dir data
```

## Inference

1. `examples/` holds the first three conditioning frames (`00.png`, `01.png`, `02.png`) of every case shown on the [project page](https://lat-dyn-reason.github.io/), one folder per case. Roll them all out to `logs/examples/` (8 cases in total):
```bash
scripts/infer_examples.sh
```
2. To roll out a specific clip from a `*.hdf5` dataset instead (adding `--side_by_side` writes a three-panel video: ground truth | error map | prediction):
```bash
GROUP=00000
INDEX=0
python infer.py \
  --ckpt checkpoints/256x256/single_task/uniform.pt \
  --eval_data data/eval/uniform.hdf5 --group $GROUP --index $INDEX \
  --img_size 256 --out logs/examples/uniform_${GROUP}_${INDEX}.mp4 --side_by_side
```

## Evaluation

1. Evaluate a single task of a single checkpoint:
```bash
# 256² uniform motion
python eval.py \
  --ckpt checkpoints/256x256/single_task/uniform.pt --task uniform \
  --eval_data data/eval/uniform.hdf5 --split data/splits/uniform.json --img_size 256
```
2. Evaluate all tasks of all checkpoints:
```bash
for t in uniform parabola collision bouncing looming; do
  # 256² single-task
  python eval.py --ckpt checkpoints/256x256/single_task/$t.pt --task $t \
    --eval_data data/eval/$t.hdf5 --split data/splits/$t.json --img_size 256
  # 256² joint 5-task
  python eval.py --ckpt checkpoints/256x256/joint_task/joint_5task.pt --task $t \
    --eval_data data/eval/$t.hdf5 --split data/splits/$t.json --img_size 256
  # 128² single-task
  python eval.py --ckpt checkpoints/128x128/single_task/$t.pt --task $t \
    --eval_data data/eval/$t.hdf5 --split data/splits/$t.json --img_size 128
  # 128² joint 5-task
  python eval.py --ckpt checkpoints/128x128/joint_task/joint_5task.pt --task $t \
    --eval_data data/eval/$t.hdf5 --split data/splits/$t.json --img_size 128
done
```

## Training

1. Set the nodes. By default, we train each LDR model from scratch on 8×8 A100-80GB GPUs (i.e., 8 nodes with passwordless ssh and a shared filesystem). Each node must have the `ldr` conda environment installed.
```bash
export NODES="node-0 node-1 node-2 node-3 node-4 node-5 node-6 node-7"  # your 8 node hostnames
export MASTER="node-0"
export PORT=29500
```
2. Start the training! Checkpoints and per-checkpoint ID/OOD metrics are written to `logs/${DATA}_${RESOLUTION}/`.
```bash
RESOLUTION=128        # {128, 256}
DATA=uniform          # {uniform, parabola, collision, bouncing, looming, joint_5task}
TRAINING_STEPS=10000  # {10000, 20000}
scripts/train.sh $RESOLUTION $DATA $TRAINING_STEPS
```

## BibTeX

```bibtex
@article{li2026learning,
  title={Learning How the World Evolves: Extrapolative Video World Models via Latent Dynamics Reasoning},
  author={Li, Haodong and Liu, Shaoteng and Wang, Tianyu and Ge, Chongjian and Ji, Sihui and Zhang, Jiahan and Lin, Xin and Lu, Haolin and Lin, Zhe and Chandraker, Manmohan},
  journal={arXiv preprint arXiv:2608.09926},
  year={2026}
}
```

## Acknowledgement

- [PhyWorld](https://phyworld.github.io/)
- [PyTorch](https://pytorch.org/)
- [Huggingface](https://huggingface.co/)
