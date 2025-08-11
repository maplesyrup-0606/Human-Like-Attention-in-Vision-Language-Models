import argparse
import json
import os
import torch
import random 

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from judge_prompt import ATTRIBUTE_JUDGE_PROMPT

def generate_judgments(attributes_path, captions_path, save_dir): 
    captions_path = os.path.expanduser(captions_path)
    attributes_path = os.path.expanduser(attributes_path)
    file_name = os.path.basename(captions_path)
    base_name = os.path.splitext(file_name)[0]

    save_file = os.path.join(save_dir, base_name + "_attribute_judgements.json")

    model_name = "/model-weights/Qwen3-32B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    with open(attributes_path, "r") as f:
        gt_attributes = json.load(f)
    
    with open(captions_path, "r") as f:
        gen_captions = json.load(f)
    
    results = {}
    
    common_ids = list(set(gt_attributes) & set(gen_captions))
    sample_ids = random.sample(common_ids, k=min(100, len(common_ids)))
    
    for image_id in tqdm(sample_ids): 
        if image_id not in gen_captions :
            continue
    
        
        attributes = gt_attributes[image_id]
        captions = gen_captions[image_id]
        inputs = []
        for caption in captions :
            prompt = ATTRIBUTE_JUDGE_PROMPT.format(attributes=attributes,caption=caption)
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            inputs.append(text)
        
        model_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
        input_lengths = model_inputs.input_ids.shape[1]

        with torch.no_grad() :
            gen_output_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0
            )
        gen_output_ids = [output[input_lengths:] for output in gen_output_ids]
        decoded_outputs = tokenizer.batch_decode(gen_output_ids, skip_special_tokens=True)
        
        results[image_id] = decoded_outputs

    with open(save_file, "w") as f :
        json.dump(results, f, indent=2)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based attribute judgment on generated captions.")
    parser.add_argument("--attributes", type=str, help="Path to attributes JSON")
    parser.add_argument("--captions", type=str, help="Path to generated captions JSON")
    parser.add_argument("--save-dir", type=str, help="Output path to save judged results")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"LLM-as-a-judge for {args.captions}", flush=True)
    generate_judgments(args.attributes, args.captions, args.save_dir)
