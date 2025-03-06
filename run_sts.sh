#!/bin/bash
model=${MODEL:-princeton-nlp/sup-simcse-roberta-large}  # 
encoding=${ENCODER_TYPE:-bi_encoder}  # cross_encoder, bi_encoder, tri_encoder
lr=${LR:-1e-5}  # learning rate
wd=${WD:-0.1}  # weight decay

transform=${TRANSFORM:-False}  # whether to use an additional linear layer after the encoder
routing_start=${ROUT_START:--4} # where to start using routing
routing_end=${ROUT_END:-24} # where to end using routing
router_type=${ROUTER_TYPE:-3}
mask_type=${MASK_TYPE:-0} # mask type: 0-5
objective=${OBJECTIVE:-mse}  # mse, triplet, triplet_mse

triencoder_head=${TRIENCODER_HEAD:-None}  # hadamard, concat (set for tri_encoder)
num_train_epochs=${NUM_EPOCHS:-3}
seed=${SEED:-45}
output_dir=${OUTPUT_DIR:-output}
basic_config=model_${model//\//__}__enc_${encoding}
if [ "$routing_start" == "$routing_end" ]; then
    config=trans_${transform}__obj_${objective}__mask_${mask_type}
else
    config=trans_${transform}__obj_${objective}__from_${routing_start}_to_${routing_end}__mask_${mask_type}
fi
config=${config}__lr_${lr}__wd_${wd}__s_${seed}

train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}

python run_sts.py \
  --output_dir "${output_dir}/${basic_config}/${config}" \
  --model_name_or_path ${model} \
  --objective ${objective} \
  --encoding_type ${encoding} \
  --pooler_type cls \
  --freeze_encoder False \
  --transform ${transform} \
  --triencoder_head ${triencoder_head} \
  --max_seq_length 512 \
  --train_file ${train_file} \
  --validation_file ${eval_file} \
  --test_file ${test_file} \
  --condition_only False \
  --sentences_only False \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate ${lr} \
  --weight_decay ${wd} \
  --max_grad_norm 0.0 \
  --num_train_epochs ${num_train_epochs} \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --log_level info \
  --disable_tqdm True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --seed ${seed} \
  --data_seed ${seed} \
  --fp16 True \
  --log_time_interval 15 \
  --overwrite_output_dir True \
  --num_show_examples 8 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_spearmanr \
  --greater_is_better True \
  --mask_type ${mask_type} \
  --routing_start ${routing_start} \
  --routing_end ${routing_end} \
  --router_type ${router_type} \

  
