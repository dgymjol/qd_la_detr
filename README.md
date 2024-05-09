# LA-DETR : Length-Aware DETR for Robust Moment Retrieval


## Setup : Environment
<b>step 0. Make a conda environment</b>

```
pip install -r requirements.txt
```

or

```
conda create -n mr python==3.9
conda activate mr
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm ipython easydict tensorboard tabulate scikit-learn pandas ipykernel ftfy
```

## Dataset Preparation
Please refer to [CG-DETR](https://github.com/wjun0830/CGDETR).



## Training (video+text)
### QVHighlight
```
bash la_detr/scripts/train.sh
```
### w/ audio modality
```
bash la_detr/scripts/train_audio.sh
```
### w/ ASR caption
```
bash la_detr/scripts/pretrain.sh
bash la_detr/scripts/train_finetune.sh # set "--resume" to pretrain result path
```

### Charades-STA
```
# SlowFast + CLIP
bash la_detr/scripts/charades_sta/train.sh

# VGG + Glove
bash la_detr/scripts/charades_sta/train_vgg.sh
```
### TACoS
```
bash la_detr/scripts/tacos/train.sh
```


## Inference
```
bash la_detr/scripts/inference.sh {exp_dir}/model_best.ckpt 'val'
bash la_detr/scripts/inference.sh {exp_dir}/model_best.ckpt 'test'
```
To achieve the result of test, please refer to [Moment-DETR evaluation](standalone_eval/README.md).


## Checkpoints

Method | config | log | result | checkpoint
 -- | -- | -- | -- | --
LA-DETR  | [opt.json](results/ours/opt.json)| [train.log.txt](results/ours/train.log.txt)| [metric](results/ours/hl_test_metrics.json)|[mode_best.pth](results/ours/model_best.ckpt)
LA-DETR (+Audio) | [opt.json](results_audio/ours/opt.json)| [train.log.txt](results_audio/ours/train.log.txt)| [metric](results_audio/ours/hl_test_metrics.json)|  [mode_best.pth](results_audio/ours/model_best.ckpt)
LA-DETR (+ASR pretrain) | [opt.json](results_finetune/ours/opt.json)| [train.log.txt](results_finetune/ours/train.log.txt)| [metric](results_finetune/ours/hl_test_metrics.json)| [mode_best.pth](results_finetune/ours/model_best.ckpt)
LA-DETR (on Charades-STA)  |[opt.json](results_charades_sta/ours/opt.json) | [train.log.txt](results_charades_sta/ours/train.log.txt)| [metric](results_charades_sta/ours/hl_test_submission_metrics.json)| [mode_best.pth](results_charades_sta/ours/model_best.ckpt)
LA-DETR (on TACOS)  | [opt.json](results_tacos/ours/opt.json)| [train.log.txt](results_tacos/train.log.txt) | [metric](results_tacos/ours/hl_test_submission_metrics.json)|[mode_best.pth](results_tacos/ours/model_best.ckpt)


## Acknowledgement
The code and annotation files are highly borrowed [Moment-DETR](https://github.com/jayleicn/moment_detr), [QD-DETR](https://github.com/wjun0830/QD-DETR), and [CG-DETR](https://github.com/wjun0830/CGDETR).

Thanks to Jie Lei, and WonJun Moon.