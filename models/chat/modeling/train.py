
from load_chatbot_dataset import DatasetA, DatasetB, formatting_prompts_func
from utils import load_config, set_env, make_today_path

import torch

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer


def get_dataset(config):
    data_config = config["dataset"]
    if data_config["type"] == "A":
        dataset = DatasetA(data_config["pathA"])
    else:
        dataset = DatasetB(data_config["pathB"])
    return dataset

def get_bnb_config(config):
    q_config = config['quantization']
    if q_config['use_4bit']:
        compute_dtype = getattr(torch, q_config['bnb_4bit_compute_dtype'])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q_config['use_4bit'],
            bnb_4bit_quant_type=q_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=q_config['use_nested_quant'],
        )
        return bnb_config
    else:
        return False

def get_lora_config(config):
    l_config = config['lora']
    peft_config = LoraConfig(
        lora_alpha=l_config['lora_alpha'],
        lora_dropout=l_config['lora_dropout'],
        target_modules=l_config['target_modules'],
        r=l_config['lora_r'],
        bias="none",
        task_type="CAUSAL_LM", # generation task
    )
    return peft_config

def get_model(config):
    t_config = config['train_params']

    model = AutoModelForCausalLM.from_pretrained(
        t_config['model_name'],
        quantization_config=get_bnb_config(config),
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=t_config['device_map'],
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    if t_config["load_adapter"]:
        model.load_adapter(t_config["adapter_path"], "loaded")
        model.set_adapter("loaded")
    
    return model

def get_tokenizer(config):
    t_config = config['train_params']
    tokenizer = AutoTokenizer.from_pretrained(t_config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    return tokenizer

def get_train_args(config):
    t_config = config['train_params']
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=t_config['output_dir'],
        num_train_epochs=t_config['num_train_epochs'],
        per_device_train_batch_size=t_config['batch_size'],
        gradient_accumulation_steps=t_config['gradient_accumulation_steps'],
        optim=t_config['optim'],
        save_steps=10,
        logging_steps=10,
        lr_scheduler_type=t_config["lr_scheduler_type"],
        learning_rate=t_config['lr_rate'],
        weight_decay=t_config["weight_decay"],
        fp16=t_config['fp16'],
        bf16=t_config['bf16'],
        max_grad_norm=t_config["max_grad_norm"],
        max_steps=-1,
        warmup_ratio=0.03,
        save_total_limit=1
    )
    return training_arguments

def get_trainer(model, tokenizer, dataset, peft_config, training_arguments):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=670,
        peft_config=peft_config,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        args=training_arguments
    )
    return trainer


if __name__ == "__main__":
    set_env()
    config = load_config('./config.yaml')
    dataset = get_dataset(config)
    
    peft_config = get_lora_config(config)
    model = get_model(config)
    tokenizer = get_tokenizer(config)
    
    train_args = get_train_args(config)
    trainer = get_trainer(model, tokenizer, dataset.dataset, peft_config, train_args)
    
    # trainer.train()
    
    save_path = make_today_path(config["train_params"]["save_dir"])
    trainer.save_model(save_path)
    
    print("DONE")