#! /bin/bash
./run_clip_flax.py \
    --output_dir $HOME/CLIP-RSICD \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --dataset_name $HOME/CLIP-rsicd/rsicd.py \
    --data_dir /home/shared/data \
    --text_column_name="func_code_string" \
    --do_train --do_eval \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16" \
    --preprocessing_num_workers="16" \
    --learning_rate="3e-4" \
    --adafactor \
    --warmup_steps="100" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --logging_steps="200" \
    --eval_steps="200" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="200" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples="10000" \
    # --max_eval_samples="1000"
