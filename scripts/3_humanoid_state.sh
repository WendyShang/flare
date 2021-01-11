CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29500 \
    train_ddp.py \
    --domain_name humanoid \
    --task_name stand \
    --encoder_type identity --work_dir ./tmp/ \
    --action_repeat 1 --num_eval_episodes 1 \
    --agent rad_sac_ddp \
    --seed 1208 --critic_lr 1e-4 --actor_lr 1e-4 --eval_freq 20000 \
    --batch_size 1024 --num_train_steps 4000000 --init_steps 5000 \
    --hidden_dim 1024 --actor_log_std_min -5 --alpha_beta 0.9 \
    --critic_tau 0.005 --actor_update_freq 1 --replay_buffer_capacity 4000000 \
