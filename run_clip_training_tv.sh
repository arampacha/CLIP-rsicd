#! /bin/bash
./run_clip_flax_tv.py \
    --output_dir /home/shared/models/clip-rsicd/bs256x8-lr5e-6-adam \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --dataset_name $HOME/CLIP-rsicd/rsicd.py \
    --data_dir /home/shared/data \
    --train_file /home/shared/data/train_rsicd.jsonl \
    --validation_file /home/shared/data/valid_rsicd.jsonl \
    --text_column_name captions \
    --do_train --do_eval \
    --per_device_train_batch_size="256" \
    --preprocessing_num_workers="16" \
    --learning_rate="2e-6" \
    --adafactor false \
    --warmup_steps="50" \
    --adam_beta1="0.9" \
    --adam_beta2="0.95" \
    --weight_decay="0.1" \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --logging_steps="2" \
    --eval_steps="20" \
    --push_to_hub="False" \
    --dtype="bfloat16" \
    --skip_memory_metrics="True" \
    --save_steps="200" \
    --save_total_limit 5 \
    --gradient_accumulation_steps 1 \
    --report_to all \
    --save_strategy epoch \
    --save_optimizer="False" \
    --captions_per_image 5 \
    --augment_images true \
    --augment_captions true
    # --run_name="test_run" \
    # --resume_from_checkpoint $HOME/gpt-neo-125M-code-clippy/ckpt_201 \
    # --max_train_samples 10240 \
    # --max_eval_samples="1000"
