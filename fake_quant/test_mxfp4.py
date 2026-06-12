import os
import sys
import types

import torch


sys.path.insert(0, os.path.dirname(__file__))
fast_hadamard_transform = types.ModuleType("fast_hadamard_transform")
fast_hadamard_transform.hadamard_transform = lambda x, *args, **kwargs: x
sys.modules.setdefault("fast_hadamard_transform", fast_hadamard_transform)
import quant_utils


def test_exact_codebook_values():
    values = torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0,
                            -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0]])
    out = quant_utils.mxfp4_quant_dequant(values, block_size=32)
    assert torch.equal(out, values)


def test_non_multiple_block_size_shape():
    x = torch.linspace(-7, 7, steps=37).reshape(1, 37)
    out = quant_utils.mxfp4_quant_dequant(x, block_size=32)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_zero_blocks_are_finite_zero():
    x = torch.zeros(2, 35)
    out = quant_utils.mxfp4_quant_dequant(x, block_size=32)
    assert torch.equal(out, x)
    assert torch.isfinite(out).all()


def test_quantizers_dispatch_to_mxfp4():
    x = torch.tensor([[0.1, 0.7, -2.4, 5.2]])

    act = quant_utils.ActQuantizer()
    act.configure(bits=4, quant_format="mxfp4", mx_block_size=32)
    assert torch.isfinite(act(x)).all()

    v_cache = quant_utils.ActQuantizer()
    v_cache.configure(bits=4, quant_format="mxfp4", mx_block_size=32)
    assert torch.isfinite(v_cache(x)).all()

    k_cache = quant_utils.ActQuantizer()
    k_cache.configure(bits=4, quant_format="mxfp4", mx_block_size=32)
    assert torch.isfinite(k_cache(x)).all()

    weight = quant_utils.WeightQuantizer()
    weight.configure(bits=4, perchannel=True, quant_format="mxfp4", mx_block_size=32)
    weight.find_params(x)
    assert torch.isfinite(weight.quantize(x)).all()


if __name__ == "__main__":
    test_exact_codebook_values()
    test_non_multiple_block_size_shape()
    test_zero_blocks_are_finite_zero()
    test_quantizers_dispatch_to_mxfp4()
    print("MXFP4 fake quant tests passed.")
