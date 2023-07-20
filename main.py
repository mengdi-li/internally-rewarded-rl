import os
import torch

import utils
import data_loader

from trainer import Trainer
from config import get_config

import datetime
import dateutil
import dateutil.tz

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)

    if config.is_train:
        if config.exp_dir == "./exps": 
            config.exp_dir = os.path.join(config.exp_dir, str(datetime.datetime.now(dateutil.tz.tzlocal()))+"_exp")
        else: 
            config.exp_dir = config.exp_dir
        config.ckpt_dir = os.path.join(config.exp_dir, config.ckpt_dir)
        config.logs_dir = os.path.join(config.exp_dir, config.logs_dir)
        config.data_dir = config.data_dir

        utils.prepare_dirs(config)
    else:
        # "./exps/2022-10-17 12:52:43.942505+02:00_exp/ckpt/ram_18_4x4_1_ckpt_200.pth.tar"
        assert config.resume_ckpt != ""
        ckpt_path = config.resume_ckpt.split("/")
        config.exp_dir = os.path.join(*ckpt_path[:-2])
        config.logs_dir = os.path.join(config.exp_dir, config.logs_dir)
        config.data_dir = config.data_dir

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    
    # parameters checking
    if config.reward_hacking:
        assert config.is_train == True
    if config.noise_visualization:
        assert config.is_train == False

    # instantiate data loaders
    if config.is_train:
        dloader = data_loader.get_train_valid_loader(
            config.dataset, 
            config.data_dir,
            config.batch_size,
            config.random_seed,
            config.valid_size,
            config.shuffle,
            config.show_sample,
            **kwargs,
        )
    else:
        dloader = data_loader.get_test_loader(
            config.dataset, config.data_dir, config.batch_size, **kwargs,
        )

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
