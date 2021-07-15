#! /bin/bash
./run_clip_flax_tv.py \
    --output_dir /home/shared/models/clip-rsicd/bs128x8-lr1e-4-augs \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --dataset_name $HOME/CLIP-rsicd/rsicd.py \
    --data_dir /home/shared/data \
    --train_file /home/shared/data/train_rsicd.jsonl \
    --validation_file /home/shared/data/valid_rsicd.jsonl \
    --text_column_name captions \
    --do_train --do_eval \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --preprocessing_num_workers="16" \
    --learning_rate="1e-4" \
    --adafactor \
    --warmup_steps="50" \
    --adam_beta1="0.9" \
    --adam_beta2="0.98" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --logging_steps="2" \
    --eval_steps="20" \
    --push_to_hub="False" \
    --dtype="bfloat16" \
    --skip_memory_metrics="True" \
    --save_steps="200" \
    --save_total_limit 10 \
    --gradient_accumulation_steps 1 \
    --report_to all \
    --run_name="bs128x8-lr1e-4-augs" \
    --save_strategy epoch \
    --save_optimizer="False" \
    --captions_per_image 5 \
    --augment_images \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples 10240 \
    # --max_eval_samples="1000"
