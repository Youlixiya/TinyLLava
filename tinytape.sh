deepspeed --include "localhost:0,1,2,3,4,5,6,7" llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower tap \
    --mm_projector_type tap \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

deepspeed --include "localhost:0,1,2,3,4,5,6,7" llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
    --version llava_llama_2 \
    --data_path playground/data/llava_v1_5_mix665k.json+playground/data/Flickr30k_train.json \
    --image_folder playground/data+playground/data/flickr30k-images\
    --vision_tower tap \
    --pretrain_mm_mlp_adapter ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-pretrain/mm_projector.bin \
    --mm_projector_type tap \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

python scripts/merge_lora_weights.py --model-path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-lora \
                                  --model-base ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
                                  --save-model-path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b

deepspeed --include "localhost:0,1,2,3,4,5,6,7" llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b \
    --version llava_llama_2 \
    --data_path playground/data/llava_instruct_150k.json+playground/data/Flickr30k_train.json \
    --image_folder playground/data/coco/train2017+playground/data/flickr30k-images \
    --vision_tower tap \
    --mm_projector_type tap \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-rec-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

python scripts/merge_lora_weights.py --model-path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-rec-lora \
                                  --model-base ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b \
                                  --save-model-path ./checkpoints/tinytape-llava-llama-2-v1.0-1.1b-rec