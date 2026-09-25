import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

from core.quantizer import TurboQuantProd, TurboQuantMSE
from inference.kv_cache import TurboQuantCacheLayer

def build_cache(model, cfg, k_bits, v_bits, use_prod):
    layers = []
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    for _ in range(cfg.num_hidden_layers):
        layers.append(TurboQuantCacheLayer(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits))
    
    # We use dynamic cache wrapper for full HF compatibility in the generation test
    from transformers.cache_utils import DynamicCache
    class TurboQuantCache(DynamicCache):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return self.layers[layer_idx].update(key_states, value_states)
    
    # Apply Triton patch
    from inference.patch_attention import apply_triton_attention_patch
    apply_triton_attention_patch(model)
    
    return TurboQuantCache(layers)

def find_max_context(model_name="Qwen/Qwen2.5-3B-Instruct"):
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
    
    print("\n--- Finding Max Context for Baseline ---")
    max_baseline = 0
    for ctx in range(2048, 16384, 512):
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx), device="cuda")
        try:
            with torch.no_grad():
                model(input_ids, use_cache=True)
            max_baseline = ctx
            print(f"Baseline passed: {ctx} tokens")
        except torch.OutOfMemoryError:
            print(f"Baseline OOM at {ctx} tokens")
            break
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"Baseline OOM at {ctx} tokens")
                break
            raise
        
        torch.cuda.empty_cache()
        gc.collect()

    print("\n--- Finding Max Context for TurboQuant (K3/V4) ---")
    torch.cuda.empty_cache()
    gc.collect()
    
    max_tq = 0
    for ctx in range(2048, 16384, 512):
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx), device="cuda")
        try:
            cache = build_cache(model, cfg, 3, 4, True)
            with torch.no_grad():
                model(input_ids, use_cache=True, past_key_values=cache)
            max_tq = ctx
            print(f"TurboQuant passed: {ctx} tokens")
        except torch.OutOfMemoryError:
            print(f"TurboQuant OOM at {ctx} tokens")
            break
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"TurboQuant OOM at {ctx} tokens")
                break
            raise
            
        torch.cuda.empty_cache()
        gc.collect()
        
    print("\n=================================")
    print(f"MAX BASELINE CONTEXT: {max_baseline} tokens")
    print(f"MAX TURBOQUANT CONTEXT: {max_tq} tokens")
    print("=================================")

if __name__ == "__main__":
    find_max_context()
