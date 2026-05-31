import argparse, json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

def load_jsonl(path):
    rows=[]
    for line in open(path, encoding='utf-8'):
        if line.strip():
            o=json.loads(line)
            rows.append({'text': f"### Instruction:\n{o['prompt']}\n\n### Response:\n{o['completion']}"})
    return Dataset.from_list(rows)

parser=argparse.ArgumentParser()
parser.add_argument('--dataset', default='data/sample_eval/engineering_train.jsonl')
parser.add_argument('--base_model', default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--output_dir', default='fine_tuning/outputs/qwen-engineering-lora')
parser.add_argument('--epochs', type=int, default=1)
args=parser.parse_args()

tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model=AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', trust_remote_code=True)
peft=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
trainer=SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=load_jsonl(args.dataset),
    peft_config=peft,
    dataset_text_field='text',
    max_seq_length=512,
    args=TrainingArguments(output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=1, save_strategy='epoch')
)
trainer.train(); trainer.save_model(args.output_dir)
print('Saved LoRA adapter to', args.output_dir)
