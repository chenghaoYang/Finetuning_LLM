# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/25 12:29
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))


import logging
import math
import datasets
import torch
import transformers
from deep_training.trainer.hf.trainer import TrainerHF
from transformers import (
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from data_utils import NN_DataHelper, config_args, get_deepspeed_config, global_args
from transformers import HfArgumentParser, PreTrainedTokenizer,PretrainedConfig
from deep_training.zoo.model_zoo.llm.llm_model import MyTransformer, PetlArguments, LoraConfig, PromptArguments
from deep_training.data_helper import ModelArguments, DataArguments,TrainingArgumentsHF

assert global_args["trainer_backend"] == "hf"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.33.2")

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    setup_model_profile()
    training_args: TrainingArgumentsHF
    parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments, PromptArguments),
                              conflict_handler='resolve')
    model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(config_args,allow_extra_keys=True,)
    lora_args = lora_args.config
    prompt_args = prompt_args.config

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    config_kwargs = {"torch_dtype": torch.float16}
    if global_args['config_merge']:
        config_kwargs.update(global_args['config_merge'])

    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)

    with training_args.main_process_first(desc="make_dataset_all"):
        dataHelper.make_dataset_all()

    is_bf16_supported = torch.cuda.is_bf16_supported()
    precision = global_args["precision"]
    if precision == "auto":
        # 精度 根据实际情况做调整
        if is_bf16_supported:
            precision = 'bf16'
        else:
            precision = '16'

        if global_args["quantization_config"] is not None and global_args["quantization_config"].load_in_8bit:
            precision = "32"

    # 精度 根据实际情况做调整
    if precision == "bf16":
        config.bf16 = True
        config.fp16 = False
        config.fp32 = False
    elif precision == "16":
        config.bf16 = False
        config.fp16 = True
        config.fp32 = False
    elif precision == "32":
        config.bf16 = False
        config.fp16 = False
        config.fp32 = True
    else:
        raise NotImplemented

    if str(precision) == '16':
        training_args.fp16 = True
    elif str(precision) == 'bf16':
        training_args.bf16 = True
    else:
        training_args.fp16 = False
        training_args.bf16 = False

    deepspeed_config = get_deepspeed_config(precision)
    if deepspeed_config:
        training_args.deepspeed = deepspeed_config

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    world_size,local_rank,process_index = training_args.world_size,training_args.local_rank,training_args.process_index

    transformer_args = dict(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args,
                            prompt_args=prompt_args,
                            quantization_config=global_args["quantization_config"],
                            device_map={"": local_rank} if world_size > 1 else "auto",
                            torch_dtype=torch.float16,
                            # new_num_tokens=len(tokenizer),  # 可能扩充词
                            auto_prepare_kbit_training=True,
                            use_gradient_checkpointing=False
                            )

    if transformer_args["quantization_config"] is None:
        transformer_args.pop("device_map")

    pl_model = MyTransformer(**transformer_args)

    config.save_pretrained(training_args.output_dir)

    # 加载sft权重
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    # # Finetune
    # if config.bf16:
    #     pl_model = pl_model.bfloat16()
    # else:
    #     pl_model = pl_model.float()

    train_datasets = None
    if training_args.do_train:
        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.load_dataset_files()["train_files"],
            with_load_memory=data_args.data_backend == 'record',
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=training_args.dataloader_drop_last,  # 多卡建议扔掉
            num_processes=world_size, process_index=process_index,
            num_workers = training_args.dataloader_num_workers,
            pin_memory = training_args.dataloader_pin_memory,
        )



    # Initialize our Trainer
    trainer = TrainerHF(
        model=pl_model,
        args=training_args,
        train_dataset=train_datasets,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_datasets)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()




def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
