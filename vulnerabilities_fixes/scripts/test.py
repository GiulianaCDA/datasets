import gc, torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.cuda.empty_cache()
gc.collect()

df = pd.read_csv("datasets/vulnerabilities_fixes/dataset.csv")
df = df.rename(columns={"vulnerable_code": "input", "fixed_code": "output"})
dataset = Dataset.from_pandas(df)
dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

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

dataset = test_dataset.map(format_dataset, batched=False)

model_local_path = "/home/levi/Documents/fInetuneLLM/results/model"
print(f"model_local_path: {model_local_path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_local_path, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

sft_model = AutoModelForCausalLM.from_pretrained(
    model_local_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

sft_model.eval()
results = []

for i, sample in enumerate(test_dataset): 
    eval_prompt, eval_completion = sample["prompt"], sample["completion"]

    print(f"Processing sample {i + 1}/{len(test_dataset)}")
    print(f"Prompt: {eval_prompt}")

    model_inputs = tokenizer([eval_prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = sft_model.generate(
            **model_inputs, max_new_tokens=1000, do_sample=True
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    results.append({
        "prompt": eval_prompt,
        "expected_completion": eval_completion,
        "generated_completion": generated_text
    })

results_df = pd.DataFrame(results)
results_df.to_csv("test_results.csv", index=False)