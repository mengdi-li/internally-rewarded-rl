# generate videos
python plot_glimpses.py --plot_type video_all \
--plot_dir "./exps/ckpt_linear_clipping/plots/ram_18_4x4_1" --train_or_eval test --epoch 1

# generate figures 
python plot_glimpses.py --plot_type figure_all \
--plot_dir "./exps/ckpt_linear_clipping/plots/ram_18_4x4_1" --train_or_eval test --epoch 1