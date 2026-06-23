# Supported Models

TensorFold Runtime targets MoE checkpoints where only a small active expert set
is needed for each token.

## Supported Families

- Qwen-style MoE MLX checkpoints used by the current low-memory runtime.
- GPT-OSS MLX checkpoints used by the exact-first streaming path.

## Experimental Families

- Large sparse MoE models with compatible safetensors/MLX tensor naming.
- Models that need custom route-union or page-backed expert kernels.

## Not Yet General

TensorFold does not currently promise universal support for every transformer
architecture. Dense models can be inspected and planned, but the strongest
memory wins come from MoE sparsity and expert packing.

## Compatibility Checklist

A model is a good candidate when:

- weights are available in local safetensors or MLX-compatible format
- the tokenizer can be loaded by the existing MLX stack
- MoE expert tensors have stable names and shapes
- selected experts can be packed or mapped into a bounded resident arena
- exactness can be checked against a native or direct-QMM reference
