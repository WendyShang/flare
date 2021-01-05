# Reinforcement Learning with Latent Flow

Official codebase for [Reinforcement Learning with Latent Flow (Flare)](https://drive.google.com/file/d/1p-kEyaCZZ8rum9SBLn8DPZP7Qy-eo4ni/view?usp=sharing). This codebase was originally forked from [Reinforcement Learning with Augmented Data (RAD)](https://mishalaskin.github.io/rad). 

## BibTex

If you use our implementations, please cite: 

```
@unpublished{shang_wang2020flare,
  title={Reinforcement Learning with Latent Flow},
  author={Shang, Wenling and Wang, Xiaofei and Srinivas, Aravind and Rajeswaran, Aravind and Gao, Yang and Abbeel, Pieter and Laskin, Michael},
  booktitle={Neurips Deep Reinforcement Learning Workshop 2020}
}
```

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

## Main differences between RAD repo and Flare repo
- For image-based observations, the new Flare architecture is added. For algorithm details, see [Flare paper](https://drive.google.com/file/d/1p-kEyaCZZ8rum9SBLn8DPZP7Qy-eo4ni/view?usp=sharing). 
- Training from state-based observations is enabled. 
- The replay buffer for image-based observations is improved, allowing a larger buffer capacity. 
- Training can be performed on multiple GPUs, allowing larger batch size and faster training. 


## Examples
- To train a RAD agent on the `quadruped walk` task from image-based observations with translation augmentation on a single GPU, run `bash script/1_quadruped_translate.sh` from the root of this directory.
- To train a Flare-RAD agent on the `quadruped walk` task from image-based observations with translation augmentation on a single GPU, run `bash script/2_quadruped_translate_flare.sh` from the root of this directory.
- To train a SAC agent on the `humanoid stand` task from state-based observations on a single GPU, run `bash script/3_humanoid_state.sh` from the root of this directory.
- To train a RAD agent on the `cheetah run` task from image-based observations on multiple GPUs, run `bash script/4_cheetah_translate_ddp.sh` from the root of this directory.


## Logging 

In your console, you should see printouts that look like this:

```
| train | E: 13 | S: 2000 | D: 9.1 s | R: 48.3056 | BR: 0.8279 | A_LOSS: -3.6559 | CR_LOSS: 2.7563
| train | E: 17 | S: 2500 | D: 9.1 s | R: 146.5945 | BR: 0.9066 | A_LOSS: -5.8576 | CR_LOSS: 6.0176
| train | E: 21 | S: 3000 | D: 7.7 s | R: 138.7537 | BR: 1.0354 | A_LOSS: -7.8795 | CR_LOSS: 7.3928
| train | E: 25 | S: 3500 | D: 9.0 s | R: 181.5103 | BR: 1.0764 | A_LOSS: -10.9712 | CR_LOSS: 8.8753
| train | E: 29 | S: 4000 | D: 8.9 s | R: 240.6485 | BR: 1.2042 | A_LOSS: -13.8537 | CR_LOSS: 9.4001
```
The above output decodes as:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh.

