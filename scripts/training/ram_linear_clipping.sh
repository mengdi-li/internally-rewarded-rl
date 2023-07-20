python main.py --dynamic_exploration_time False --use_wandb True --device_id 0 --batch_size 256 --patch_size 4 --num_glimpses 18 \
--train_patience 1500 --epochs 1500 --lr_patience 1500 --dataset cluttered-mnist-60x60-4-8x8 --use_information_maximization_reward True \
--information_maximization_reward_type linear_clipping --n_classifiers 1 --plot_freq 50 --ckp_freq 200 --random_seed 1
