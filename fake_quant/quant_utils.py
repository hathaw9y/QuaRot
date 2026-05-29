import math
import transformers
import torch
import inspect
import utils
import hadamard_utils
import fast_hadamard_transform

BFP_DEFAULT_BLOCK_SIZE = 32

def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def bfp_quant(x, scale, minq, maxq):
    scale = scale.to(x.device)
    minq = minq.to(x.device)
    maxq = maxq.to(x.device)
    q = torch.clamp(torch.round(x / scale), minq, maxq)
    return q, scale

def bfp_quant_dequant(x, scale, minq, maxq):
    q, scale = bfp_quant(x, scale, minq, maxq)
    return scale * q

def bfp_quant_dequant_dynamic(x, bits, groupsize=-1, clip_ratio=1.0):
    if bits == 16:
        return x
    if groupsize == -1:
        groupsize = BFP_DEFAULT_BLOCK_SIZE

    x_dtype = x.dtype
    minq, maxq = get_minq_maxq(bits, True)
    minq = minq.to(x.device)
    maxq = maxq.to(x.device)

    init_shape = x.shape
    pad = (groupsize - x.shape[-1] % groupsize) % groupsize
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))
    reshaped_x = x.reshape(-1, x.shape[-1] // groupsize, groupsize)
    xmax = torch.amax(torch.abs(reshaped_x), dim=-1, keepdim=True) * clip_ratio
    tmp = xmax == 0
    scale = torch.pow(2.0, torch.ceil(torch.log2(xmax / maxq)))
    scale[tmp] = 1
    q = torch.clamp(torch.round(reshaped_x / scale), minq, maxq)
    out = (q * scale).reshape(x.shape)
    if pad > 0:
        out = out[..., :init_shape[-1]]
    return out.reshape(init_shape).to(x_dtype)

def hadamard_transform_blockwise(x, block_size=None):
    n = x.shape[-1]
    if block_size is None:
        block_size = n
    assert (block_size & (block_size - 1) == 0) and block_size > 0, 'Hadamard block_size must be a power of 2'
    assert n % block_size == 0, 'The last dimension should be divisible by Hadamard block_size'

    if block_size == n:
        return fast_hadamard_transform.hadamard_transform(
            x.contiguous(), scale=1 / math.sqrt(n)
        )

    init_shape = x.shape
    x = x.reshape(-1, n // block_size, block_size)
    x = fast_hadamard_transform.hadamard_transform(
        x.contiguous(), scale=1 / math.sqrt(block_size)
    )
    return x.reshape(init_shape)


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0>=8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1>=8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)

class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('minq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        self.quant_method = 'int'

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.quant_method == 'bfp':
            return bfp_quant_dequant(x, self.scale, self.minq, self.maxq).to(x_dtype)
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.quant_method == 'bfp':
            return bfp_quant(x, self.scale, self.minq, self.maxq)
        elif self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0, quant_method='int'):
        assert quant_method in ['int', 'bfp'], "quant_method should be one of ['int', 'bfp']"
        effective_sym = True if quant_method == 'bfp' else sym
        if quant_method == 'bfp' and groupsize == -1:
            groupsize = BFP_DEFAULT_BLOCK_SIZE
        self.minq, self.maxq = get_minq_maxq(bits, effective_sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.quant_method = quant_method
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def _bfp_scale(self, xmax):
        tmp = xmax == 0
        scale = torch.pow(2.0, torch.ceil(torch.log2(xmax / self.maxq)))
        scale[tmp] = 1
        return scale

    def find_bfp_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(torch.abs(reshaped_x), dim=3, keepdim=True) * self.clip_ratio
        self.scale = self._bfp_scale(xmax)
        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = torch.zeros_like(self.scale)

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.minq = self.minq.to(dev)
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            if self.quant_method == 'bfp':
                self.find_bfp_params_per_token_groupwise(x)
            else:
                self.find_params_per_token_groupwise(x)
            utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        if self.quant_method == 'bfp':
            xmax = torch.amax(torch.abs(reshaped_x), dim=1) * self.clip_ratio
            self.scale = self._bfp_scale(xmax).unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
            return

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.online_had_block_size = None
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            if self.quantizer.quant_method == 'bfp':
                str_ += ' (BFP Per-Token)'
            else:
                str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            if self.out_quantizer.quant_method == 'bfp':
                str_ += ' (BFP Per-Token)'
            else:
                str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.online_had_block_size is not None:
                had_x = x.float() if self.fp32_had else x
                x = hadamard_transform_blockwise(had_x, self.online_had_block_size).to(x_dtype)
            elif self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
            
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            
            if self.fp32_had:
                x = x.float()
                
            init_shape = x.shape
            if self.online_had_block_size is not None:
                x = hadamard_transform_blockwise(x, self.online_had_block_size)
            elif self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
                
            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16: #Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x



class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class BFPAttentionOpsWrapper(torch.nn.Module):
    def __init__(
        self,
        module,
        qk_bits,
        qk_groupsize,
        qk_clip_ratio,
        qk_quant_method,
        av_bits,
        av_groupsize,
        av_clip_ratio,
        av_quant_method,
        rotation_block_size,
    ):
        super().__init__()
        self.module = module
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.num_heads = module.num_heads
        self.num_key_value_heads = module.num_key_value_heads
        self.num_key_value_groups = module.num_key_value_groups
        self.head_dim = module.head_dim
        self.hidden_size = module.hidden_size
        self.scaling = getattr(module, 'scaling', self.head_dim ** -0.5)
        self.attention_dropout = getattr(module, 'attention_dropout', 0.0)
        self.qk_bits = qk_bits
        self.qk_groupsize = qk_groupsize
        self.qk_clip_ratio = qk_clip_ratio
        self.qk_quant_method = qk_quant_method
        self.av_bits = av_bits
        self.av_groupsize = av_groupsize
        self.av_clip_ratio = av_clip_ratio
        self.av_quant_method = av_quant_method
        self.rotation_block_size = rotation_block_size
        self.legacy_return = 'past_key_value' in inspect.signature(module.forward).parameters

    def _apply_rotary_pos_emb(self, query_states, key_states, value_states, position_ids, position_embeddings):
        from transformers.models.llama import modeling_llama

        if position_embeddings is None:
            if not hasattr(self.module, 'rotary_emb'):
                raise ValueError('position_embeddings is required for this transformers LLaMA attention version')
            try:
                cos, sin = self.module.rotary_emb(value_states, position_ids)
            except TypeError:
                cos, sin = self.module.rotary_emb(value_states, seq_len=key_states.shape[-2])
        else:
            cos, sin = position_embeddings

        try:
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        except TypeError:
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, cos, sin

    def _repeat_kv(self, key_states, value_states):
        from transformers.models.llama import modeling_llama

        return (
            modeling_llama.repeat_kv(key_states, self.num_key_value_groups),
            modeling_llama.repeat_kv(value_states, self.num_key_value_groups),
        )

    def _maybe_bfp(self, x, bits, groupsize, clip_ratio, quant_method):
        if bits >= 16 or quant_method != 'bfp':
            return x
        return bfp_quant_dequant_dynamic(x, bits, groupsize=groupsize, clip_ratio=clip_ratio)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        query_states = self.module.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        key_states = self.module.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.module.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        query_states, key_states, cos, sin = self._apply_rotary_pos_emb(
            query_states, key_states, value_states, position_ids, position_embeddings
        )

        past_key_value = past_key_values if past_key_values is not None else past_key_value
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            try:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            except TypeError:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        if self.qk_bits < 16:
            dtype = query_states.dtype
            query_states = hadamard_transform_blockwise(
                query_states.float(), self.rotation_block_size
            ).to(dtype)
            key_states = hadamard_transform_blockwise(
                key_states.float(), self.rotation_block_size
            ).to(dtype)

        key_states, value_states = self._repeat_kv(key_states, value_states)

        query_states = self._maybe_bfp(
            query_states, self.qk_bits, self.qk_groupsize, self.qk_clip_ratio, self.qk_quant_method
        )
        key_states = self._maybe_bfp(
            key_states, self.qk_bits, self.qk_groupsize, self.qk_clip_ratio, self.qk_quant_method
        )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                causal_mask = attention_mask[:, None, None, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask[:, :, : query_states.shape[-2], : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self._maybe_bfp(
            attn_weights, self.av_bits, self.av_groupsize, self.av_clip_ratio, self.av_quant_method
        )
        value_states = self._maybe_bfp(
            value_states, self.av_bits, self.av_groupsize, self.av_clip_ratio, self.av_quant_method
        )
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, self.hidden_size)
        attn_output = self.module.o_proj(attn_output)
        returned_attn_weights = attn_weights if output_attentions else None

        if self.legacy_return:
            return attn_output, returned_attn_weights, past_key_value
        return attn_output, returned_attn_weights


def add_bfp_attention_ops(model, args):
    from transformers.models.llama import modeling_llama

    for layer in model.model.layers:
        if isinstance(layer.self_attn, BFPAttentionOpsWrapper):
            continue
        if not isinstance(layer.self_attn, modeling_llama.LlamaAttention):
            raise NotImplementedError('BFP attention ops are only implemented for LLaMA attention')
        layer.self_attn = BFPAttentionOpsWrapper(
            layer.self_attn,
            qk_bits=args.k_bits,
            qk_groupsize=args.k_groupsize,
            qk_clip_ratio=args.k_clip_ratio,
            qk_quant_method=args.k_quant_method,
            av_bits=args.v_bits,
            av_groupsize=args.v_groupsize,
            av_clip_ratio=args.v_clip_ratio,
            av_quant_method=args.v_quant_method,
            rotation_block_size=args.rotation_block_size
            if args.rotation_block_size > 0
            else (
                args.a_groupsize
                if args.rotation_block_size == -1 and args.a_quant_method == 'bfp' and args.a_groupsize > 0
                else BFP_DEFAULT_BLOCK_SIZE
                if args.rotation_block_size == -1 and args.a_quant_method == 'bfp'
                else None
            ),
        )


def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        try:
            tmp = getattr(module, attr)
        except AttributeError:
            continue
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
