import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

parser=argparse.ArgumentParser()
parser.add_argument('--base_model', default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--adapter', required=True)
parser.add_argument('--prompt', default='Extract the open point, owner and closure action from this RFI: Missing load calculation evidence for bracket design.')
args=parser.parse_args()

def generate(model, tok, prompt):
    ids=tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad(): out=model.generate(**ids, max_new_tokens=120)
    return tok.decode(out[0], skip_special_tokens=True)

tok=AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
base=AutoModelForCausalLM.from_pretrained(args.base_model, device_map='auto', trust_remote_code=True)
ft=PeftModel.from_pretrained(base, args.adapter)
print('\nPROMPT:', args.prompt)
print('\nFINE-TUNED OUTPUT:\n', generate(ft, tok, args.prompt))
