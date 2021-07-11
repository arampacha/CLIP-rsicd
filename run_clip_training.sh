#! /bin/bash
./run_clip_flax.py \
    --output_dir $HOME/models/clip-rsicd \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --dataset_name $HOME/CLIP-rsicd/rsicd.py \
    --data_dir /home/shared/data \
    --text_column_name sentences \
    --do_train --do_eval \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --preprocessing_num_workers="16" \
    --learning_rate="3e-4" \
    --adafactor \
    --warmup_steps="100" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --logging_steps="20" \
    --eval_steps="20" \
    --push_to_hub="False" \
    --report_to="all" \
    --dtype="bfloat16" \
    --skip_memory_metrics="False" \
    --save_steps="200" \
    --save_total_limit 2 \
    --gradient_accumulation_steps 1 \
    --max_train_samples 10240 \
    --report_to wandb \
    --run_name="testing-bs-128" \
    # --augment_images \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_eval_samples="1000"
