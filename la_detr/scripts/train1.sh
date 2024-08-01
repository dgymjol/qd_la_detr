dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
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

gpunum=1


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
--exp_id "base_crop_adv" \
--crop \
${@:1}


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
# --exp_id "cc" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# ${@:1}


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
# --exp_id "cc_class_moe" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --class_moe \
# ${@:1}

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
# --exp_id "cc_span_moe" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --span_moe \
# ${@:1}



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
# --exp_id "cc_ffn_class_moe" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --class_moe \
# --ffn_moe \
# ${@:1}



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
# --exp_id "cc_ffn_span_moe" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --span_moe \
# --ffn_moe \
# ${@:1}




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
# --exp_id "cc_class_span_moe" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --class_moe \
# --span_moe \
# ${@:1}

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
# --exp_id "cc_ffn_class_span_moe" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --class_moe \
# --span_moe \
# --ffn_moe \
# ${@:1}

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
# --exp_id "cc_anc" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --class_anchor \
# ${@:1}

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
# --exp_id "cc_moe_anc" \
# --m_classes "[10, 30, 70, 150]" \
# --cc_matching \
# --class_anchor \
# --moe \
# ${@:1}

# list="2018 2019 2020 2021 2022"
# pqn=10

# for var in $list
# do
#   echo $var

#   CUDA_VISIBLE_DEVICES=${gpunum} PYTHONPATH=$PYTHONPATH:. python la_detr/train.py \
#   --dset_name ${dset_name} \
#   --ctx_mode ${ctx_mode} \
#   --train_path ${train_path} \
#   --eval_path ${eval_path} \
#   --eval_split_name ${eval_split_name} \
#   --v_feat_dirs ${v_feat_dirs[@]} \
#   --v_feat_dim ${v_feat_dim} \
#   --t_feat_dir ${t_feat_dir} \
#   --t_feat_dim ${t_feat_dim} \
#   --bsz ${bsz} \
#   --results_root ${results_root} \
#   --exp_id la_detr_p${pqn}-${var} \
#   --m_classes "[10, 30, 70, 150]" \
#   --tgt_embed \
#   --cc_matching \
#   --num_queries 10 \
#   --pos_query ${pqn} \
#   ${@:1}

# done
