"""ICONIP MRAM — train/test entry point.

Examples
--------
# MRAM on MNIST (Table 1, "MRAM, 7 glimpses")
python main.py --model_choose MRAM --data_choose MNIST --num_glimpses 7 \
               --patch_size 8 --glimpse_scale 1

# MRAM on FashionMNIST (Table 2)
python main.py --model_choose MRAM --data_choose FashionMNIST --num_glimpses 12 \
               --patch_size 8 --glimpse_scale 1

# MRAM on FER2013 (Table 3, single-scale)
python main.py --model_choose MRAM --data_choose FER --num_glimpses 12 \
               --patch_size 8 --glimpse_scale 1
"""
import sys

import torch

import data_loader
import utils
from config import get_config
from trainer import Trainer


def main(config):
    utils.prepare_dirs(config)

    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    if config.is_train:
        dloader = data_loader.get_train_valid_loader(
            config.data_choose,
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
            config.data_choose, config.data_dir, config.batch_size, **kwargs,
        )

    trainer = Trainer(config, dloader)

    if config.is_train:
        utils.save_config(config)
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
