import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer

"""#Pre-processing"""

df = pd.read_csv("datasets/vulnerabilities_fixes/dataset.csv")
df = df.rename(columns={"vulnerable_code": "input", "fixed_code": "output"})

#convert to Huggingface Datasets format
dataset = Dataset.from_pandas(df)

dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

print(f"Tamanho do conjunto de treino: {len(train_dataset)}")
print(f"Tamanho do conjunto de teste: {len(test_dataset)}")

"""#Apply instruct prompt template"""

mistral_instruct_template = "[INST]{instruction}[/INST]"

system_prompt = """You are a helpful code assistant specialized in fixing vulnerabilities in code.
Analyze the vulnerable code provided and generate a corrected version that resolves the issue:

<code>
{before_code}
</code>
Begin!
Generate only the corrected code with all required imports within <answer> XML tags."""

assistant_prompt = """<answer>
{after_code}
</answer>"""


def format_dataset(sample):
    system_message = system_prompt.format(
        before_code=sample["input"],
    )
    instruction = f"{system_message}\n\n"
    prompt = mistral_instruct_template.format(instruction=instruction)

    completion = assistant_prompt.format(
        after_code = sample["output"]
    )

    sample["prompt"] = prompt
    sample["completion"] = completion
    return sample

dataset = train_dataset.map(format_dataset, batched=False)
print(f"len(dataset): {len(dataset)}")


"""#Fine tune LLM

##Initialize parameters
"""

### model
model_id = "mistral-community/Codestral-22B-v0.1"

### qlora related
r = 64
lora_alpha = 16
lora_dropout = 0.1
target_modules = [ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
task_type = "CAUSAL_LM"

### bitsandbytes related
load_in_4bit=True
bnb_4bit_use_double_quant=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype="bfloat16"


### training related
base_dir = "results"

output_dir = f"{base_dir}/checkpoints"
save_model_dir = f"{base_dir}/model"
offload_folder = f"{base_dir}/offload"
logging_dir = f"{output_dir}/logs"

num_train_epochs = 1
max_steps = 100 # mumber of training steps (overrides num_train_epochs)

per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
gradient_checkpointing = True

bf16 = True
fp16 = False

max_grad_norm = 0.3
weight_decay = 0.001
# optim = "paged_adamw_32bit"
optim = "adamw_torch"

learning_rate = 2e-4
warmup_ratio = 0.03
lr_scheduler_type = "constant"

save_strategy = "no"
logging_steps = 25
logging_strategy = "steps"
group_by_length = True

max_seq_length = 4096
packing = False

"""##Instantiate tokenizer and quantized model"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# define tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# define 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
)
# define model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_cache=False if gradient_checkpointing else True,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1 # num_of_gpus
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

"""##Define lora config"""

import bitsandbytes as bnb
from peft import LoraConfig

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# get lora target modules
modules = find_all_linear_names(model)
print(modules) # NOTE: update target_modules with these values

lora_config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    bias="none",
    task_type=task_type,
)

"""##define training args, collator, trainer"""

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# set training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    weight_decay=weight_decay,
    optim=optim,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    save_strategy=save_strategy,
    logging_steps=logging_steps,
    logging_strategy=logging_strategy,
    group_by_length=group_by_length,
)

# checkout for more info: Train on completions only https://huggingface.co/docs/trl/en/sft_trainer

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]}\n\n ### Answer: {example['completion'][i]}"
        output_texts.append(text)
    return output_texts

collator = DataCollatorForCompletionOnlyLM(
    response_template="### Answer:",
    tokenizer=tokenizer
)

# initialize sft trainer
trainer = SFTTrainer(
    args=training_arguments,
    model=model,
    peft_config=lora_config,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
    packing=packing
)

"""## start training and save finetuned adapter weights"""

trainer.train()

trainer.model.save_pretrained(output_dir, safe_serialization=False)

# clear memory
del model
del trainer
torch.cuda.empty_cache()

"""##merge adapter weights and base model"""

from peft import AutoPeftModelForCausalLM

# load PEFT model in fp16
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,  # ATTENTION: This allows remote code execution
)

print(model)

merged_model = model.merge_and_unload()

print(merged_model)

# save merged model
merged_model.save_pretrained(save_model_dir, safe_serialization=True,  max_shard_size="2GB")

# save tokenizer for easy inference
tokenizer.save_pretrained(save_model_dir)

del model
del merged_model
del tokenizer

torch.cuda.empty_cache()