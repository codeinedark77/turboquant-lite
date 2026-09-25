import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"

def run_benchmark(context_size, prompt_tokens, generate_tokens=150):
    print(f"\n==========================================")
    print(f"Testing OLLAMA (Model: {MODEL})")
    print(f"Context Window Target: {context_size} tokens")
    print(f"Prompt Length: {prompt_tokens} tokens")
    print(f"==========================================")
    
    # Generate a massive prompt to test the context window
    # "apple " is 1 token in most tokenizers
    dummy_prompt = "apple " * prompt_tokens
    
    payload = {
        "model": MODEL,
        "prompt": f"{dummy_prompt}\n\nExplain the history of artificial intelligence in exactly 300 words. Be concise and accurate.",
        "stream": False,
        "options": {
            "num_predict": generate_tokens,
            "num_ctx": context_size,
            "temperature": 0.0
        }
    }
    
    print("Sending prompt to Ollama (Prefilling and Generating)...")
    start_time = time.time()
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        
        # Ollama provides exact nanosecond timings
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 1)
        
        # Calculate Tokens Per Second
        generation_tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0
        prefill_tps = prompt_eval_count / (prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns > 0 else 0
        
        print(f"\n--- OLLAMA CUDA RESULTS ---")
        print(f"Total Time: {end_time - start_time:.2f}s")
        print(f"Context Processed: {prompt_eval_count} tokens")
        print(f"Prefill Speed: {prefill_tps:.2f} tokens/second")
        print(f"Tokens Generated: {eval_count}")
        print(f"Generation Speed: {generation_tps:.2f} tokens/second")
        print(f"\nOutput Snippet:\n{data['response'][:250]}...")
        
    except requests.exceptions.RequestException as e:
        print(f"Ollama API Error (Potential OOM): {e}")

if __name__ == "__main__":
    # Test 1: Standard context to check raw speed
    run_benchmark(context_size=4096, prompt_tokens=500)
    
    # Test 2: Push the context window to see if it survives (8.5k context)
    run_benchmark(context_size=12000, prompt_tokens=8500)
