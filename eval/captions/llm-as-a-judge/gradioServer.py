import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")

path = "OpenGVLab/InternVL3-8B"
# path = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval().cuda()

print("Model loaded. ✅",flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    path, 
    trust_remote_code=True, 
    use_fast=False
)
generation_config = dict(max_new_tokens=1024, do_sample=True)

def evaluate(prompt, temperature=0.7) :
    # InternVL
    with torch.no_grad() :
        response, _ = model.chat(
            tokenizer,
            None,
            prompt,
            {**generation_config, "temperature":temperature},
            history=None,
            return_history=True
        )
    return response


server = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Response"),
    title="InternVL Evaluation",
    allow_flagging="never"
)

server.launch(enable_queue=False)