CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29500 \
    train_ddp.py \
    --domain_name quadruped \
    --task_name walk \
    --encoder_type pixel_rnn --work_dir ./tmp \
    --action_repeat 4 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 108 \
    --agent rad_sac_ddp --frame_stack 3 --data_augs translate  \
    --seed 1208 --critic_lr 2e-4 --actor_lr 2e-4 --eval_freq 10000 \
    --batch_size 128 --num_train_steps 600000 --init_steps 10000 --camera_id 2 \
    --augment_target_same_rnd --num_filters 32 --encoder_feature_dim 64  --replay_buffer_capacity 100000 \
