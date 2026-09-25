import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from find_max_context import build_cache

def benchmark_generation(model_name="Qwen/Qwen2.5-3B-Instruct"):
    print(f"Loading {model_name} in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.set_attn_implementation("eager")
    cfg = model.config

    prompt = "Explain the history of artificial intelligence in exactly 300 words. Be concise and accurate."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    print("\n==========================================")
    print("Testing BASELINE (Standard PyTorch Eager)")
    print("==========================================")
    torch.cuda.empty_cache()
    
    start_time = time.time()
    with torch.no_grad():
        outputs_baseline = model.generate(
            input_ids,
            max_new_tokens=150,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    generated_tokens_baseline = outputs_baseline.shape[1] - input_ids.shape[1]
    time_baseline = end_time - start_time
    tps_baseline = generated_tokens_baseline / time_baseline
    
    print(f"Time: {time_baseline:.2f}s")
    print(f"Tokens Generated: {generated_tokens_baseline}")
    print(f"Speed: {tps_baseline:.2f} tokens/second")
    print(f"Output:\n{tokenizer.decode(outputs_baseline[0][input_ids.shape[1]:], skip_special_tokens=True)}")


    print("\n==========================================")
    print("Testing TURBOQUANT (Custom Triton Kernel)")
    print("==========================================")
    torch.cuda.empty_cache()
    
    cache = build_cache(model, cfg, 3, 4, True)
    
    start_time = time.time()
    with torch.no_grad():
        outputs_tq = model.generate(
            input_ids,
            max_new_tokens=150,
            use_cache=True,
            past_key_values=cache,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    generated_tokens_tq = outputs_tq.shape[1] - input_ids.shape[1]
    time_tq = end_time - start_time
    tps_tq = generated_tokens_tq / time_tq
    
    print(f"Time: {time_tq:.2f}s")
    print(f"Tokens Generated: {generated_tokens_tq}")
    print(f"Speed: {tps_tq:.2f} tokens/second")
    print(f"Output:\n{tokenizer.decode(outputs_tq[0][input_ids.shape[1]:], skip_special_tokens=True)}")


if __name__ == "__main__":
    benchmark_generation()
