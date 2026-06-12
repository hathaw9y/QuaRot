import math
import transformers
import torch
import utils
import hadamard_utils
import fast_hadamard_transform

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

def _mxfp4_block_scales(x, block_size=32, clip_ratio=1.0):
    max_abs = torch.amax(torch.abs(x), dim=-1, keepdim=True) * clip_ratio
    safe_max = torch.clamp(max_abs, min=1e-30)
    scale = torch.pow(2.0, torch.ceil(torch.log2(safe_max / 6.0)))
    return torch.where(max_abs == 0, torch.ones_like(scale), scale)

def _mxfp4_quant_dequant_with_scale(x, scale):
    x_dtype = x.dtype
    codebook = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            device=x.device, dtype=torch.float32)
    scaled = x.float() / scale.float()
    sign = torch.sign(scaled)
    abs_scaled = torch.abs(scaled)
    indexes = torch.argmin(torch.abs(abs_scaled.unsqueeze(-1) - codebook), dim=-1)
    quantized = codebook[indexes] * sign
    return (quantized * scale.float()).to(x_dtype)

def mxfp4_quant_dequant(x, block_size=32, clip_ratio=1.0):
    assert block_size > 0, 'MX block size should be positive'
    assert clip_ratio <= 1 and clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    init_shape = x.shape
    last_dim = init_shape[-1]
    pad = (block_size - last_dim % block_size) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))

    blocked = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scale = _mxfp4_block_scales(blocked, block_size=block_size, clip_ratio=clip_ratio)
    quantized = _mxfp4_quant_dequant_with_scale(blocked, scale).reshape(*x.shape)
    if pad:
        quantized = quantized[..., :last_dim]
    return quantized.reshape(init_shape)


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
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16
        self.quant_format = "int"
        self.mx_block_size = 32

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        if self.quant_format == "mxfp4":
            return mxfp4_quant_dequant(x, self.mx_block_size, self.clip_ratio).to(x_dtype)
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.quant_format == "mxfp4":
            return mxfp4_quant_dequant(x, self.mx_block_size, self.clip_ratio)
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0,
                  quant_format="int", mx_block_size=32):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.quant_format = quant_format
        self.mx_block_size = mx_block_size
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

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
        if self.quant_format == "mxfp4":
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

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
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (MXFP4, block={self.quantizer.mx_block_size})' if self.quantizer.quant_format == "mxfp4" else (
                f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)')

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (MXFP4, block={self.out_quantizer.mx_block_size})' if self.out_quantizer.quant_format == "mxfp4" else (
                f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)')

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            
            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
            
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            
            if self.fp32_had:
                x = x.float()
                
            init_shape = x.shape
            if self.K == 1:
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
        self.quant_format = "int"
        self.mx_block_size = 32
        self.clip_ratio = 1.0
        self._mx_col_offset = 0

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        quant_format="int", mx_block_size=32, clip_ratio=1.0,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.quant_format = quant_format
        self.mx_block_size = mx_block_size
        self.clip_ratio = clip_ratio
        self._mx_col_offset = 0
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

        if self.quant_format == "mxfp4":
            pad = (self.mx_block_size - x.shape[-1] % self.mx_block_size) % self.mx_block_size
            if pad:
                x = torch.nn.functional.pad(x, (0, pad))
            blocked = x.reshape(x.shape[0], x.shape[1] // self.mx_block_size, self.mx_block_size)
            if self.mse:
                best = torch.full(blocked.shape[:-1], float('inf'), device=dev)
                best_scale = _mxfp4_block_scales(blocked, self.mx_block_size, self.clip_ratio)
                for i in range(int(self.maxshrink * self.grid)):
                    p = 1 - i / self.grid
                    scale1 = _mxfp4_block_scales(blocked, self.mx_block_size, self.clip_ratio * p)
                    q = _mxfp4_quant_dequant_with_scale(blocked, scale1)
                    err = torch.sum(torch.abs(q - blocked).pow(self.norm), dim=-1)
                    tmp = err < best
                    if torch.any(tmp):
                        best[tmp] = err[tmp]
                        best_scale[tmp] = scale1[tmp]
                self.scale = best_scale.squeeze(-1)
            else:
                self.scale = _mxfp4_block_scales(blocked, self.mx_block_size, self.clip_ratio).squeeze(-1)
            self.zero = torch.zeros_like(self.scale)
            self._mx_col_offset = 0
            return

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
            if self.quant_format == "mxfp4":
                if x.shape[-1] == 1 and self.scale.dim() == 2:
                    block_idx = min(self._mx_col_offset // self.mx_block_size, self.scale.shape[-1] - 1)
                    scale = self.scale[:, block_idx].reshape(-1, 1)
                    self._mx_col_offset += 1
                    return _mxfp4_quant_dequant_with_scale(x, scale).to(x_dtype)
                if x.dim() == 2 and self.scale.dim() == 2 and x.shape[0] == self.scale.shape[0]:
                    last_dim = x.shape[-1]
                    pad = (self.mx_block_size - last_dim % self.mx_block_size) % self.mx_block_size
                    x_pad = torch.nn.functional.pad(x, (0, pad)) if pad else x
                    blocked = x_pad.reshape(x_pad.shape[0], x_pad.shape[1] // self.mx_block_size, self.mx_block_size)
                    scale = self.scale[:, :blocked.shape[1]].unsqueeze(-1)
                    q = _mxfp4_quant_dequant_with_scale(blocked, scale).reshape_as(x_pad)
                    if pad:
                        q = q[:, :last_dim]
                    return q.to(x_dtype)
                return mxfp4_quant_dequant(x, self.mx_block_size, self.clip_ratio).to(x_dtype)
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)



def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr, tmp in list(module._modules.items()):
        if tmp is None:
            continue
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
            continue
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
            continue
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
            continue
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
