
## install
  - pip install -U -r requirements.txt
  
```text

# Flash-attention requires a graphics card with an arithmetic power of 7.5 or higher, the following optional installation, if the card does not support it, you can not install it.
git clone -b https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
pip install csrc/layer_norm
pip install csrc/rotary
```


## weight
 - [Qwen2.5-7B&&Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B&&https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)


## dataset
- 20% of alpaca for training, 500 samples for infer(https://huggingface.co/datasets/tatsu-lab/alpaca)


## infer
    # infer_lora_finetuning.py #Inferring lora fine-tuning models
     python infer_lora_finetuning.py.py


## training
```text
 # Production data
 cd scripts
 bash train_full.sh -m dataset 
 or
 bash train_lora.sh -m dataset 

##num_process_worker is for multi-process data production, if the amount of data is large, appropriately adjusted to the number of cpu
 dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
 
 # full parameter training
     bash train_full.sh -m train
     
 # lora adalora 
     bash train_lora.sh -m train
     
```





## training parameter
[training parameter](args.MD)

## Reference
https://github.com/ssbuild/qwen_finetuning
    





