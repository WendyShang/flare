import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy
import cv2

import utils
from logger import Logger
from video import VideoRecorder

from sac_ddp import RadSacAgentDDP
from torchvision import transforms
import data_augs as rad

import torch.distributed

def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--camera_id', default=0, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='rad_sac_ddp', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # data augs
    parser.add_argument('--data_augs', default='crop', type=str)
    parser.add_argument('--augment_target_same_rnd', default=False, action='store_true')
    # misc
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)
    # DistributeDataParallel + PyTorch launcher utility.
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--print_param_check', default=False, action='store_true')

    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            save_name = args.image_dir + '/step_' + str(step) + '_eps_' + str(i) + '.pt'
            state_obs = []
            pixel_obs = []
            if 'pixel' in args.encoder_type:
                obs, qpos = env.reset()
            else:
                obs =  env.reset()    

            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if 'pixel' in args.encoder_type and 'crop' in args.data_augs:
                    obs = utils.center_crop_image(obs,args.image_size)
                if 'pixel' in args.encoder_type and 'translate' in args.data_augs:
                    obs = utils.center_crop_image(obs, args.pre_transform_image_size)
                    obs = np.expand_dims(obs,0)
                    obs = rad.center_translate(obs,args.image_size)
                    obs = np.squeeze(obs,0)
                if 'pixel' in args.encoder_type and 'window' in args.data_augs:
                    obs = np.expand_dims(obs,0)
                    obs = rad.center_window(obs,args.image_size)
                    obs = np.squeeze(obs,0)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        if 'pixel' in args.encoder_type:
                            action = agent.sample_action(obs / 255.)
                        else:
                            action = agent.sample_action(obs)
                    else:
                        if 'pixel' in args.encoder_type:
                            action = agent.select_action(obs / 255.)
                        else:
                            action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
       
            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        filename = args.work_dir + '/' + args.domain_name + '--'+args.task_name + '-' + args.data_augs + '--s' + str(args.seed) + '--eval_scores.npy'
        key = args.domain_name + '-' + args.task_name + '-' + args.data_augs
        try:
            log_data = np.load(filename,allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}
            
        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step 
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward 
        log_data[key][step]['max_ep_reward'] = best_ep_reward 
        log_data[key][step]['std_ep_reward'] = std_ep_reward 
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename,log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, action_range, args, device, image_channel=3):
    if args.agent == 'rad_sac_ddp':
        return RadSacAgentDDP(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            rank=args.local_rank,
            print_param_check=args.print_param_check,
            action_range=action_range,
            image_channel=image_channel,
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    if 'crop' in args.data_augs:
        pre_transform_image_size = args.pre_transform_image_size
    else:
        pre_transform_image_size = args.image_size

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=('pixel' in args.encoder_type),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat,
        camera_id=args.camera_id,
    )

    env_render = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=('pixel' in args.encoder_type),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat,
        camera_id=args.camera_id,
    )
 
 
    env.seed(args.seed)
    action_range = [float(env.action_space.low.min()), float(env.action_space.high.max())]

    # stack several consecutive frames together
    if 'pixel' in args.encoder_type:
        env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-f' + str(args.num_filters) + '-s' + str(args.seed)  + '-' + args.encoder_type
    exp_name = exp_name + '-alr' + str(args.actor_lr)
    exp_name = exp_name + '_' + 'efficieint'
    args.work_dir = args.work_dir + '/'  + exp_name

    rank = args.local_rank
    if rank == 0:
        utils.make_dir(args.work_dir)
        video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
        model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
        image_dir = utils.make_dir(os.path.join(args.work_dir, 'image'))
        args.image_dir = image_dir

        video = VideoRecorder(video_dir if args.save_video else None)

        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer' + str(rank)))

    device = torch.device(f'cuda:{rank}' if
            torch.cuda.is_available() else 'cpu')
    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
    )
    world_size = torch.distributed.get_world_size()
    print(f"IN train_ddp.main()--RANK: {rank};  MY DEVICE: {device};  WORLD_SIZE: {world_size}")
    if rank == 0:
        print(f"USING BACKEND: {torch.distributed.get_backend()}")

    action_shape = env.action_space.shape

    if 'pixel' in args.encoder_type:
        number_channel = 3
        obs_shape_single = (number_channel, args.image_size, args.image_size)
        obs_shape = (number_channel*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (number_channel*args.frame_stack, 
            pre_transform_image_size, 
            pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        obs_shape_single = obs_shape

    capacity = args.replay_buffer_capacity // world_size
    batch_size = args.batch_size // world_size
    if rank == 0:
        print(f"USING PER-PROCESS REPLAY CAPACITY: {capacity}")
        print(f"USING PER-PROCESS BATCH SIZE: {batch_size}")

    replay_buffer = utils.ReplayBufferEfficient(
        obs_shape=obs_shape_single,
        action_shape=action_shape,
        capacity=capacity,
        batch_size=batch_size,
        device=device,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        augment_target_same_rnd=args.augment_target_same_rnd,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
        action_range=action_range,
    )

    if rank == 0:
        L = Logger(args.work_dir, use_tb=args.save_tb)
    else:
        L = None

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    total_loop_time = -time.time()
    # variable 'step' counts the total across all processes.
    obs_save = []
    reward_save = []
    action_save = []
    obs_next_save = []
    for step in range(0, args.num_train_steps, world_size):  # increment by world_size
        # evaluate agent periodically
        if step % args.eval_freq < world_size:
            if rank == 0:
                L.log('eval/episode', episode, step)
                print(f"RANK {rank} performing evaluation.")
                evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
                if args.save_model:
                    agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if rank == 0:
                if step > 0:
                    if step % args.log_interval < world_size:
                        L.log('train/duration', time.time() - start_time, step, n=world_size)  # needs n?
                        L.dump(step)
                    start_time = time.time()
                if step % args.log_interval < world_size:
                    L.log('train/episode_reward', episode_reward, step)

            if 'pixel' in args.encoder_type:
                obs, qpos = env.reset()
            else:
                obs =  env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1 * world_size  # Assume they all finish at same time limit.
            if rank == 0:
                if step % args.log_interval < world_size:
                    L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                if 'pixel' in args.encoder_type:
                    action = agent.sample_action(obs / 255.)
                else:
                    action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 * world_size  # Keep the same replay ratio.
            for step_incr in range(num_updates):
                agent.update(replay_buffer, L, step + step_incr)  # step arg increments by one.
        next_obs, reward, done, info = env.step(action)
        next_qpos = info

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )

        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool, float(done))
        obs = next_obs
        episode_step += 1

    total_loop_time += time.time()
    print(f"total loop time: {total_loop_time}")


if __name__ == '__main__':
    main()
