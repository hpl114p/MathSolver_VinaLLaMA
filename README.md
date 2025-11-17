# MathSolver VinaLLaMA
Implementation of a Vietnamese Math Reasoning LLM using VinaLLaMA and supervised fine-tuning (SFT) on math word problems.

This project fine-tunes the VinaLLaMA family model to solve Vietnamese math word problems.
We load the pretrained model, evaluate baseline performance, then fine-tune using custom math datasets.

## Project Workflow
* Step 1: Import libraries/modules
```python
import json
import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers

from pprint import pprint
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
* Step 2: Load pre-trained model
```python
MODEL_NAME = "vilm/vinallama-7b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
```
* Step 3: Configurate LLMs
```python 
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```
```python
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
```
```python
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
```
* Step 4: Download dataset
```python
data = load_dataset('hllj/vi_grade_school_math_mcq')
```
* Step 5: Create generate prompt function
```python
def generate_prompt(question, choices, explanation):
    return f"""
<|im_start|>system
Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.

<|im_start|>user
### Câu hỏi:
{question}
### Các lựa chọn:
{choices}
### Câu trả lời:

<|im_start|>assistant
{explanation}
""".strip()

def generate_and_tokenize_prompt(question, choices, explanation):
    full_prompt = generate_prompt(question, choices, explanation)
    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )

    return tokenized_full_prompt
```
* Step 6: Create training samples
```python
training_samples = []
for sample in tqdm(data['train']):
    for quest in sample['problems']:
        choices = quest['choices']
        explanation = quest['explanation'].strip()
        question = quest['question']

        if explanation == '' or question == '' or choices == []:
            continue

        try:
            question = question.split('\n \n')[1].strip()
        except:
            continue

        choices = '\n'.join(choices)
        training_sample = generate_and_tokenize_prompt(
            question, choices, explanation
        )

        training_samples.append(training_sample)
```
* Step 8: Training
```python
training_args = transformers.TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=4,
      num_train_epochs=1,
      learning_rate=2e-4,
      fp16=True,
      save_total_limit=3,
      logging_steps=1,
      output_dir="experiments",
      optim="paged_adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=choices_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()
```

## About datasets
In this project, I use one main dataset: 
* [hllj/vi_grade_school_math_mcq](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset): A Vietnamese grade-school math multiple-choice dataset. It contains elementary-level math questions in MCQ format, including problem statements, answer options, and correct labels.

## Fine-tuning results
Before fine-tuning:
```base
<|im_start|> system
Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.

<|im_start|> user
### Câu hỏi:
Số gồm 1 đơn vị và 3 chục đọc là :
### Các lựa chọn:
A. 30
B. 31
C. 20
D. 21
### Câu trả lời:

<|im_start|> assistant
Để giải quyết vấn đề này, chúng ta cần hiểu rằng 1 đơn vị có nghĩa là một nhóm gồm một chục và 3 chục có nghĩa là ba nhóm mười.

Chúng ta có thể viết lại số này là 3 x 10 = 30

Do đó, số đọc là 30.

Lựa chọn đúng là A. 30
CPU times: user 9.13 s, sys: 295 ms, total: 9.43 s
Wall time: 9.46 s
```

After fine-tuning:
```base
<|im_start|> system
Bạn là một chuyên gia về toán. Bạn sẽ nhận câu hỏi trắc nghiệm kèm theo các lựa chọn, hãy giải step by step nếu có và chọn phương án đúng.

<|im_start|> user
### Câu hỏi:
Số gồm 1 đơn vị và 2 chục đọc là :
### Các lựa chọn:
A. 20
B. 21
C. 30
D. 31
### Câu trả lời:

<|im_start|> assistant
Đáp án B Số gồm 1 đơn vị và 2 chục đọc là 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần chọn là: B. 21 . Đáp án cần
CPU times: user 2min 41s, sys: 1min 31s, total: 4min 13s
Wall time: 4min 14s
```

Compare Table:
|            | After Fine-tuning  | Before Fine-tuning |
|----------------|--------------|---------------------|
|   Inference Example Time | ~4m 14s | ~9.46s         |

## Authors
Hoang Phuc Lam
- Github: https://github.com/hpl114p
- Email: hpl114p@gmail.com