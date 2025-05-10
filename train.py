# -*- coding: utf-8 -*-
import os
from config import global_args

def main():
    trainer_backend = global_args["trainer_backend"]
    if trainer_backend == "pl":
        from training.train_pl import main as main_execute
    else:
        raise ValueError(f"{trainer_backend} NotImplemented ")

    main_execute()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
