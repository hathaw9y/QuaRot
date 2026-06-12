# Fake Quantization in QuaRot


In this directory, we provide the torch scripts for the experiments in QuaRot. 


## Language Generation and Zero-Shot Evaluations

Currently, we only support **LLaMa-2** models. You can simply run the `main.py` to reproduce the results in the paper. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--bsz`: the batch size for PPL evaluation
- `--rotate`: whether we want to rotate the model
- `--lm_eval`: whether we want to run LM-Eval for Zero-Shot tasks
- `--tasks`: the tasks for LM-Eval
- `--cal_dataset`: the calibration dataset for GPTQ quantization
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization
- `--k_bits`: the number of bits for key quantization
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--k_clip_ratio`: The ratio of clipping for key
- `--v_clip_ratio`: The ratio of clipping for value
- `--w_asym`: Whether we want to use asymmetric quantization for weights
- `--a_asym`: Whether we want to use asymmetric quantization for activation
- `--v_asym`: Whether we want to use asymmetric quantization for value
- `--k_asym`: Whether we want to use asymmetric quantization for key
- `--a_groupsize`: The group size for activation quantization
- `--w_groupsize`: The group size for weight quantization
- `--v_groupsize`: The group size for value quantization
- `--k_groupsize`: The group size for key quantization
- `--quant_format`: The fake quantization format (`int` or `mxfp4`)
- `--mx_block_size`: The MXFP4 block size. The default is 32.
  
For example, to run the perplexity of `LLaMA2-7B` model with quantizing all weights and activations, you can run the following command:

```bash
/bin/python main.py --model meta-llama/Llama-2-7b-hf  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip
```

To simulate MXFP4 for weights, linear activations, and K/V cache, run:

```bash
python main.py --model meta-llama/Llama-2-7b-hf --rotate --quant_format mxfp4 --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4
```

MXFP4 simulation uses E2M1 FP4 values with one power-of-two scale per contiguous block of 32 values by default. This path is fake quantization only; the packed int4 CUDA runtime and checkpoint export are unchanged.
