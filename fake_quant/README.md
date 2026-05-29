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
- `--a_quant_method`: the activation quantization method (`int` or `bfp`)
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization
- `--v_quant_method`: the value quantization method (`int` or `bfp`)
- `--k_bits`: the number of bits for key quantization
- `--k_quant_method`: the key quantization method (`int` or `bfp`)
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
  
For example, to run the perplexity of `LLaMA2-7B` model with quantizing all weights and activations, you can run the following command:

```bash
/bin/python main.py --model meta-llama/Llama-2-7b-hf  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip
```

To use block floating point for activation quantization:

```bash
/bin/python main.py --model meta-llama/Llama-2-7b-hf --rotate --a_bits 4 --a_quant_method bfp --w_bits 4 --w_clip
```

When `--a_quant_method bfp` is used with the default `--a_groupsize -1`, the BFP block size defaults to 32.
The random Hadamard rotation and online Hadamard rotations also use the same BFP block size.
Use `--rotation_block_size 0` to compare against full hidden-size Hadamard rotation.
Use `--no-online_o_proj_had` to disable the online partial Hadamard before `o_proj`.

To keep weights in the original precision and use BFP only for activations and KV:

```bash
/bin/python main.py --model meta-llama/Llama-2-7b-hf --rotate --a_bits 4 --a_quant_method bfp --v_bits 4 --v_quant_method bfp --k_bits 4 --k_quant_method bfp --w_bits 16
```

Add `--bfp_attn_ops` to also BFP-quantize Q/K before the QK matmul and attention/V before the AV matmul.
