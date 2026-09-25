import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import math

from kernels.fused_kernel_tiled import fused_nf4_matmul_tiled

def apply_triton_attention_patch(model):
    """
    Monkey-patches Qwen2Attention in the loaded model to use the Triton NF4 fused kernel 
    if the KV cache contains packed NF4 values.
    """
    original_forward = Qwen2Attention.forward

    def patched_forward(self, *args, **kwargs):
        # Extract arguments for transformers 5.15.1 signature:
        # (self, hidden_states, position_embeddings, attention_mask, past_key_values, ...)
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        position_embeddings = args[1] if len(args) > 1 else kwargs.get("position_embeddings")
        attention_mask = args[2] if len(args) > 2 else kwargs.get("attention_mask")
        past_key_values = args[3] if len(args) > 3 else kwargs.get("past_key_values")
        cache_position = kwargs.get("cache_position", None)
        
        # We only intercept if it's our custom TurboQuant cache with NF4 packed values
        is_turboquant = past_key_values is not None and hasattr(past_key_values, "layers") and hasattr(past_key_values.layers[0], "_v_packed")
        
        if not is_turboquant:
            return original_forward(self, *args, **kwargs)

        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary pos emb directly using the precomputed cos/sin tuple
        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update cache (returns dummy tensors for K and V)
        past_key_values.update(key_states, value_states, self.layer_idx, {"cache_position": cache_position})
        
        # Extract the NF4 packed values directly from the cache
        cache_layer = past_key_values.layers[self.layer_idx]
        seq_len = cache_layer._seq_len
        
        # For decoding, the cache accumulates chunks. We concatenate them for the kernel.
        # In a fully optimized C++ engine, the kernel would accept a list of pointers (PagedAttention).
        k_packed = torch.cat(cache_layer._k_packed, dim=0) if len(cache_layer._k_packed) > 1 else cache_layer._k_packed[0]
        k_absmax = torch.cat(cache_layer._k_absmax, dim=0) if len(cache_layer._k_absmax) > 1 else cache_layer._k_absmax[0]
        v_packed = torch.cat(cache_layer._v_packed, dim=0) if len(cache_layer._v_packed) > 1 else cache_layer._v_packed[0]
        v_absmax = torch.cat(cache_layer._v_absmax, dim=0) if len(cache_layer._v_absmax) > 1 else cache_layer._v_absmax[0]
        
        # Reshape the packed tensors to include batch and head dims for the Triton kernel
        # Elements per KV head
        elements_per_head = seq_len * self.head_dim
        bytes_per_head = elements_per_head // 2
        blocks_per_head = elements_per_head // cache_layer.blocksize
        
        # The packed tensors are flat: [bsz * num_kv_heads * seq_len * head_dim / 2]
        # We need to reshape them to [bsz, num_kv_heads, bytes_per_head]
        k_packed_view = k_packed.view(bsz, self.config.num_key_value_heads, bytes_per_head)
        v_packed_view = v_packed.view(bsz, self.config.num_key_value_heads, bytes_per_head)
        
        k_absmax_view = k_absmax.view(bsz, self.config.num_key_value_heads, blocks_per_head)
        v_absmax_view = v_absmax.view(bsz, self.config.num_key_value_heads, blocks_per_head)
        
        # GQA: Repeat the KV heads to match num_heads
        k_packed_rep = k_packed_view.repeat_interleave(self.num_key_value_groups, dim=1)
        v_packed_rep = v_packed_view.repeat_interleave(self.num_key_value_groups, dim=1)
        k_absmax_rep = k_absmax_view.repeat_interleave(self.num_key_value_groups, dim=1)
        v_absmax_rep = v_absmax_view.repeat_interleave(self.num_key_value_groups, dim=1)

        # Branch logic: Prefill vs Decoding
        if q_len > 1:
            # Prefill: Use Triton NF4 kernel to avoid massive O(N^2) VRAM allocation
            from kernels.flash_attention_nf4 import flash_attention_nf4
            attn_output = flash_attention_nf4(query_states, k_packed_rep, k_absmax_rep, v_packed_rep, v_absmax_rep)
        else:
            # Decoding: q_len == 1. The O(N^2) wall doesn't exist here (matrix is just 1xN).
            # We can dequantize using bitsandbytes and use PyTorch SDPA for perfect accuracy.
            import bitsandbytes.functional as bnbF
            
            k_dequant_chunks = []
            v_dequant_chunks = []
            
            # Dequantize each chunk individually to preserve the [batch, head, seq, dim] memory layout
            for i in range(len(cache_layer._k_packed)):
                k_chunk_packed = cache_layer._k_packed[i]
                k_chunk_absmax = cache_layer._k_absmax[i]
                v_chunk_packed = cache_layer._v_packed[i]
                v_chunk_absmax = cache_layer._v_absmax[i]
                
                # The length of this chunk can be inferred from the packed tensor size
                chunk_elements = k_chunk_packed.numel() * 2
                chunk_seq_len = chunk_elements // (bsz * self.config.num_key_value_heads * self.head_dim)
                
                k_quant_state = bnbF.QuantState(
                    absmax=k_chunk_absmax,
                    shape=torch.Size([bsz, self.config.num_key_value_heads, chunk_seq_len, self.head_dim]),
                    blocksize=cache_layer.blocksize,
                    quant_type="nf4",
                    dtype=query_states.dtype
                )
                k_dequant_chunk = bnbF.dequantize_4bit(k_chunk_packed, k_quant_state)
                k_dequant_chunks.append(k_dequant_chunk)
                
                v_quant_state = bnbF.QuantState(
                    absmax=v_chunk_absmax,
                    shape=torch.Size([bsz, self.config.num_key_value_heads, chunk_seq_len, self.head_dim]),
                    blocksize=cache_layer.blocksize,
                    quant_type="nf4",
                    dtype=query_states.dtype
                )
                v_dequant_chunk = bnbF.dequantize_4bit(v_chunk_packed, v_quant_state)
                v_dequant_chunks.append(v_dequant_chunk)
                
            k_dequant = torch.cat(k_dequant_chunks, dim=2)
            v_dequant = torch.cat(v_dequant_chunks, dim=2)
            
            # GQA: Repeat the KV heads to match num_heads
            k_dequant = k_dequant.repeat_interleave(self.num_key_value_groups, dim=1)
            v_dequant = v_dequant.repeat_interleave(self.num_key_value_groups, dim=1)
            
            # Use PyTorch SDPA
            import torch.nn.functional as F
            attn_output = F.scaled_dot_product_attention(
                query_states, 
                k_dequant, 
                v_dequant, 
                is_causal=False
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None

    Qwen2Attention.forward = patched_forward
