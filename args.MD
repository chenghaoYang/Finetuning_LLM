
## Switching Training Mode Configurations
    Please Modify config/main.py
    enable_deepspeed = False
    enable_lora = True
    load_in_bit = 0  # 4 load_in_4bit, 8 load_in_8bit  other  0

## optimizer
    # lamb,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,
    # adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion,lion_8bit,lion_32bit,
    # paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,
    # lamb_fused_dp adagrad_cpu_dp adam_cpu_dp adam_fused_dp

## scheduler
    # linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,
    # constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau

### single computer with mutiple display card
```text
config_args = {
    'devices': 2,
}

config_args = {
    'devices': [0,2],
}
```

## Precision training
    Trainer.precision = '16' # Semi-precision training "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"



### lora finetuning
```text
        global_args = {
           "load_in_8bit": False, # lora need dependency: pip install bitsandbytes
           "num_layers_freeze": -1, # Non-lora, non-p-tuning mode. ， <= config.json num_layers
           "num_layers": -1, # Whether to use all layers of the backbone network Maximum 1-28, -1 means all layers, otherwise only N layers are used.
        }
         lora_info_args = {
               'with_lora': True,  # Whether to enable the lora module
               'r': 8,
               'target_modules': ['query_key_value'],
               'target_dtype': None,
               'lora_alpha': 32,
               'lora_dropout': 0.1,
               'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
               'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        }
```
    

## Full parameter fine tuning

```text
    global_args = {
            "load_in_8bit": False, # lora 
            "num_layers_freeze": -1, 
            "num_layers": -1, 
        }
   lora_info_args = {
       'with_lora': False,  
        ...
   }
   adalora_info_args = {
       'with_lora': False,  
        ...
   }
```


