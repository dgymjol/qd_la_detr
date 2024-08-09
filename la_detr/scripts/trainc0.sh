dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_base
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32

gpunum=0

# list="2021 2022 2023 2024 2025"
# results_root=results_base

# for seed in $list
# do
#   echo $seed


# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_${seed} \
# --seed ${seed} \
# ${@:1}

# done


# list="2021 2022 2023 2024 2025"
# results_root=results_10_39_70

# for seed in $list
# do
#   echo $seed

# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_tgt_cc${seed} \
# --seed ${seed} \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --tgt_embed \
# ${@:1}

# done


# list="2021 2022 2023 2024 2025"
# results_root=results_12_18

# for seed in $list
# do
#   echo $seed

# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_tgt_cc${seed} \
# --seed ${seed} \
# --m_classes "[12, 18, 150]" \
# --cc_matching \
# --tgt_embed \
# ${@:1}

# done



# list="2021 2022 2023 2024 2025"
# results_root=results_15_41

# for seed in $list
# do
#   echo $seed

# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_tgt_cc${seed} \
# --seed ${seed} \
# --m_classes "[15, 41, 150]" \
# --cc_matching \
# --tgt_embed \
# ${@:1}

# done


list="2021 2022 2023 2024 2025"
results_root=results_new_crop_test_10_35_65

for seed in $list
do
  echo $seed

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id base_crop_all_${seed} \
--m_classes "[10, 36, 65, 150]" \
--cc_matching \
--tgt_embed \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--crop_all \
${@:1}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id base_crop_merge_${seed} \
--m_classes "[10, 36, 65, 150]" \
--cc_matching \
--tgt_embed \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--merge \
${@:1}

CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id base_crop_all_merge_${seed} \
--m_classes "[10, 36, 65, 150]" \
--cc_matching \
--tgt_embed \
--seed ${seed} \
--crop \
--fore_min 10 \
--back_min 10 \
--mid_min 10 \
--crop_random \
--merge \
--crop_all \
${@:1}

done



# list="2021 2022 2023 2024 2025"
# results_root=results_10_80_135

# for seed in $list
# do
#   echo $seed

# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_tgt_cc${seed} \
# --seed ${seed} \
# --m_classes "[10, 80, 135, 150]" \
# --cc_matching \
# --tgt_embed \
# ${@:1}
  
# done


# list="2021 2022 2023 2024 2025"
# results_root=results_10_90_120

# for seed in $list
# do
#   echo $seed

# CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
# --dset_name ${dset_name} \
# --ctx_mode ${ctx_mode} \
# --train_path ${train_path} \
# --eval_path ${eval_path} \
# --eval_split_name ${eval_split_name} \
# --v_feat_dirs ${v_feat_dirs[@]} \
# --v_feat_dim ${v_feat_dim} \
# --t_feat_dir ${t_feat_dir} \
# --t_feat_dim ${t_feat_dim} \
# --bsz ${bsz} \
# --results_root ${results_root} \
# --exp_id base_tgt_cc${seed} \
# --seed ${seed} \
# --m_classes "[10, 36, 65, 150]" \
# --cc_matching \
# --tgt_embed \
# ${@:1}
  
# done
