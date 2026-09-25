import torch
import triton
import triton.language as tl
import math

from ._lut import codebook_kwargs, nf4_lut_lookup

@triton.jit
def _flash_attn_nf4_fwd_kernel(
    Q, K_packed, K_absmax, V_packed, V_absmax, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    C8: tl.constexpr, C9: tl.constexpr, C10: tl.constexpr, C11: tl.constexpr,
    C12: tl.constexpr, C13: tl.constexpr, C14: tl.constexpr, C15: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k_packed = off_hz * stride_kh + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]) // 2
    off_v_packed = off_hz * stride_vh + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]) // 2
    
    off_k_absmax = off_hz * (N_CTX * BLOCK_DMODEL // 64) + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]) // 64
    off_v_absmax = off_hz * (N_CTX * BLOCK_DMODEL // 64) + (offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]) // 64
    
    q_ptrs = Q + off_q
    k_packed_ptrs = K_packed + off_k_packed
    v_packed_ptrs = V_packed + off_v_packed
    k_absmax_ptrs = K_absmax + off_k_absmax
    v_absmax_ptrs = V_absmax + off_v_absmax
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q (it's float16)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = (q * sm_scale).to(tl.float32)
    
    # To determine high/low nibble for packing
    flat_offs = offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
    is_high = (flat_offs % 2) == 0

    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K (NF4)
        k_p = tl.load(k_packed_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0).to(tl.int32)
        k_s = tl.load(k_absmax_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=1.0)
        
        k_nib = tl.where(is_high, (k_p >> 4) & 0x0F, k_p & 0x0F)
        k_val = nf4_lut_lookup(k_nib, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15)
        k = k_val * k_s
        
        # qk = q @ k.T
        qk = tl.dot(q, k.trans(1, 0))
        qk = tl.where((start_n + offs_n)[None, :] <= offs_m[:, None], qk, float("-inf"))
        
        # Compute new m
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Load V (NF4)
        v_p = tl.load(v_packed_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0).to(tl.int32)
        v_s = tl.load(v_absmax_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=1.0)
        
        v_nib = tl.where(is_high, (v_p >> 4) & 0x0F, v_p & 0x0F)
        v_val = nf4_lut_lookup(v_nib, C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15)
        v = v_val * v_s
        
        # Scale acc by alpha and add p @ v
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        
        # Update l_i and m_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Advance pointers
        k_packed_ptrs += BLOCK_N * BLOCK_DMODEL // 2
        v_packed_ptrs += BLOCK_N * BLOCK_DMODEL // 2
        k_absmax_ptrs += BLOCK_N * BLOCK_DMODEL // 64
        v_absmax_ptrs += BLOCK_N * BLOCK_DMODEL // 64

    acc = acc / l_i[:, None]
    
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def flash_attention_nf4(q, k_packed, k_absmax, v_packed, v_absmax):
    # q: (batch, num_heads, seq_len, head_dim)
    # k_packed, v_packed: (batch, num_heads, seq_len * head_dim // 2)
    # k_absmax, v_absmax: (batch, num_heads, seq_len * head_dim // 64)
    
    Lq, Lk, Lv = q.shape[-1], q.shape[-1], q.shape[-1]
    assert Lq in {16, 32, 64, 128}
    assert q.dim() == 4
    
    batch, num_heads, seq_len, head_dim = q.shape
    
    out = torch.empty_like(q)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)
    
    _flash_attn_nf4_fwd_kernel[grid](
        q, k_packed, k_absmax, v_packed, v_absmax,
        1.0 / math.sqrt(head_dim),
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        seq_len * head_dim // 2, seq_len * head_dim // 2, 1, # k strides
        seq_len * head_dim // 2, seq_len * head_dim // 2, 1, # v strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, num_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
        num_stages=1, num_warps=2,
        **codebook_kwargs()
    )
    return out
