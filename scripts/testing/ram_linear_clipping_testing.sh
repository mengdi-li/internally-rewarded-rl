python main.py --dynamic_exploration_time False --use_wandb False --device_id 0 --is_train False \
--resume_ckpt "./exps/ckpt_linear_clipping/ckpt/ram_18_4x4_1_ckpt_1400.pth.tar" \
--batch_size 256 --patch_size 4 --num_glimpses 18 \
--dataset cluttered-mnist-60x60-4-8x8 --use_information_maximization_reward True --information_maximization_reward_type linear_clipping \
--n_classifiers 1 --plot_freq 50 --ckp_freq 200 --random_seed 1 \