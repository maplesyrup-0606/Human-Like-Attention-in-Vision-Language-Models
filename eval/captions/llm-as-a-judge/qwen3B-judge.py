import argparse
import json
import os
import torch
import random

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from judge_prompt import JUDGE_PROMPT  # assumes this is a string template

def generate_judgments(gt_path, gen_path, save_dir):
    file_name = os.path.basename(gen_path)
    base_name = os.path.splitext(file_name)[0]

    save_file = os.path.join(save_dir, base_name + "_ratings.json")

    model_name = "/model-weights/Qwen3-32B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    with open(gt_path, "r") as f:
        gt_captions = json.load(f)

    with open(gen_path, "r") as f:
        gen_captions = json.load(f)

    results = {}

    common_ids = list(set(gt_captions) & set(gen_captions))
    sample_ids = random.sample(common_ids, k=min(100, len(common_ids)))
    print("Sampled 100 image ids!",flush=True)

    for image_id in tqdm(sample_ids):
        if image_id not in gen_captions:
            continue
        gt_caption = gt_captions[image_id]
        inputs = []
        for gen_caption in gen_captions[image_id]:
            prompt = JUDGE_PROMPT.format(groundtruths=gt_caption, generated=gen_caption)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
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
        
        for output_idx, output in enumerate(decoded_outputs) :
            # output format checking
            if "Evaluation" not in output or "Total rating:" not in output :
                decoded_outputs[output_idx] = "[INVALID]" + decoded_outputs[output_idx]
            
        results[image_id] = decoded_outputs
        print(f"Finished evaluating image : {image_id} ✅", flush=True)

    with open(save_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {save_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based judgment on generated captions.")
    parser.add_argument("groundtruth", type=str, help="Path to ground truth captions JSON")
    parser.add_argument("generated", type=str, help="Path to generated captions JSON")
    parser.add_argument("save_dir", type=str, help="Output path to save judged results")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"LLM-as-a-judge for {args.generated}", flush=True)
    generate_judgments(args.groundtruth, args.generated, args.save_dir)