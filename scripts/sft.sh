model_name_or_path=${1:-"/workspace/qingfei/model/chatglm3-6b-32k"}
template=${2:-"chatglm3"}
cutoff_len=${3:-30000}
output_dir="../model/LongRAG_$(basename $model_name_or_path)"

deepspeed --include localhost:0,1,2,3,4,5,6,7 ../LLaMA-Factory/src/train_bash.py\
    --deepspeed ../config/stage3.json \
    --stage sft \
    --do_train \
    --model_name_or_path $model_name_or_path \
    --dataset LRGinstruction \
    --template $template \
    --finetuning_type full \
    --output_dir $output_dir \
    --overwrite_output_dir\
    --overwrite_cache \
    --cutoff_len $cutoff_len \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --logging_steps 13 \
    --save_steps 27 \
    --eval_steps 27 \
    --evaluation_strategy steps \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.005 \
    --plot_loss \
    --bf16 \
    --export_legacy_format True \
    --gradient_checkpointing 1 \
    --report_to wandb \
    --flash_attn \
    --save_only_model
