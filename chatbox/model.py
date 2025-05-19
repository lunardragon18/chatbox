import gc

import pandas as pd
import os
import sys
import transformers
from peft import PeftModel, LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import regex as re
from datasets import Dataset
import huggingface_hub
import torch


def remove_paranthesis(text):
    result = re.sub(r'\(.*?\)', "", text)
    return result


def count_words(text):
    words = [word for word in text.strip().split(" ")]
    return len(words)


class ChatBot:
    def __init__(self, model_path, data_path="/content/chatbox/Data/naruto.csv", huggingface_token=None):
        self.base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.data_path = data_path

        self.huggingface_token = huggingface_token
        self.device = "cuda"
        self.model_path = model_path

        if huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
            if huggingface_hub.repo_exists(self.model_path):
                self.model = self.load_model(self.model_path)

            else:
                train_dataset = self.load_data()
                self.train(self.base_model_path, train_dataset)
                self.model = self.load_model(self.model_path)

    def chatting(self, message, history):
        messages = []
        messages.append({"role": "system",
                         "content": f'You are Naruto from the show "Naruto". You are supposed to respond based on their personality and speech\n'})
        for msg in history:
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})
        messages.append({"role": "user", "content": message})

        terminator = [self.model.tokenizer.eos_token_id,
                      self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        output = self.model(
            messages,
            max_length=256,
            eos_token_id=terminator,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        print("Raw output from model:", output)
        output_message = output[0]['generated_text'][-1]
        return {"content": output_message}

    def load_model(self, model_path):
        bits_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": bits_config
            }
        )
        return pipeline

    def train(self, model_name, dataset, per_device_train_batch=1, gradient_accumulation_steps=1,
              optim="paged_adamw_32bit", save_steps=200, logging_step=10, max_steps=300, lr=2e-4, scheduler="constant",
              max_grad_norm=0.3, warmup_ratio=0.3):
        out_dir = f'/content/results'
        bits_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bits_config,
                                                     trust_remote_code=True)
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        lora_aplha = 16
        lora_droupout = 0.1
        lora_r = 64
        max_seq_len = 512

        peft_config = LoraConfig(lora_alpha=lora_aplha, lora_dropout=lora_droupout, r=lora_r, bias="none",
                                 task_type="CAUSAL_LM")
        training_args = SFTConfig(
            output_dir=out_dir,
            per_device_train_batch_size=per_device_train_batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_step,
            learning_rate=lr,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=scheduler,
            fp16=True,
            group_by_length=True,
            report_to="none",
            dataset_text_field="prompt",
            max_seq_length=max_seq_len
        )

        trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            train_dataset=dataset,
            args=training_args,
        )
        trainer.tokenizer = tokenizer

        trainer.train()
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          return_dict=True,
                                                          quantization_config=bits_config,
                                                          torch_dtype=torch.float16,

                                                          )
        base_model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del model, base_model
        gc.collect()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df["line"] = df['line'].apply(remove_paranthesis)
        df["word_len"] = df['line'].apply(count_words)
        df["index_value"] = 0
        df.loc[(df["name"] == "Naruto") & (df["word_len"] > 5), 'index_value'] = 1
        indexes = list(df.loc[(df["index_value"] == 1) & (df.index > 0)].index)
        sys_prompt = f'You are Naruto from the show "Naruto". You are supposed to respond based on their personality and speech\n'
        prompts = []
        completions = []
        for index in indexes:

            prompt = sys_prompt
            if index >= 2:
                prompt += f'{df.iloc[index - 2]["name"]}: {df.iloc[index - 2]["line"]}\n'
            prompt += f'{df.iloc[index - 1]["name"]}: {df.iloc[index - 1]["line"]}\n'
            prompt += f'{df.iloc[index]["name"]}: {df.iloc[index]["line"]}'
            completion = f' {df.iloc[index]["line"]}'
            prompts.append(prompt)
            completions.append(completion)

        final_df = pd.DataFrame({"prompt": prompts, "completion": completions})
        dataset = Dataset.from_pandas(final_df)
        return dataset


