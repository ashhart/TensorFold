# Observed Qwen35 GGUF schema

The key/type/scalar list and every tensor name/type/shape come from ggml-org/Qwen3.8-27B-GGUF revision a316ea527fde32c12487be7574e2db5ac2ad2a7e, Qwen3.8-27B-Q4_K_M.gguf. Core's GGUF reader parsed its header and tensor index from a readonly, non-prefaulting map. The header/data offset is 10994112 bytes and that header's SHA256 is aa35b1a4418dbdad0d6ea93d238291ec56e55d379ab524b60a292f0386f8402f. The Q8_0 file has the same name/shape list.

The data lists 40 metadata keys and 851 tensors. Large metadata arrays record their type/count; the template records its length/hash. Token, merge and template contents are not copied. The JSON is the corresponding native Config input derived from these metadata values, not an MLX checkpoint config. The fixture contains no projection payload and needs no checkpoint file during tests.

This schema fixture establishes host identities and dimensions. The paired norm sample data settle effective gains for these supplied checkpoints; other source conventions and inference qualification need their own evidence. The source repository declares Apache-2.0.

The norm TSV columns are GGUF name, corresponding native name, element index, GGUF F32 bits in hexadecimal and MLX BF16 bits in hexadecimal. There are 16 samples from each of six classes: final, input, post-attention, gated-linear, Q and K norms. Both GGUF variants have identical sample bits. Every GGUF value rounded to BF16 matches the MLX value; neither adding nor subtracting one matches. These weights are effective multiplicative RMS gains and remain unchanged in host preparation. This does not establish the parameter convention of any original pre-conversion checkpoint.

The supplied Qwen3.8-27B MLX 4-bit checkpoint declares Qwen/Qwen3.8-27B as its base model and Apache-2.0 as its license. Its remote revision was not supplied. Config SHA256 is 14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85 and safetensors index SHA256 is 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8. Only shard headers and 192 norm payload bytes were read; no projection data were read. The three serialized shard header hashes, including their eight-byte size prefixes, are:

- model-00001-of-00003.safetensors: 5455768325c60469d9132adda61063cb15aba6f59d1303228d43e3d26597f990
- model-00002-of-00003.safetensors: 56d696f2640c6610f89cf17fd989fc275b78b5284120e1a2194066bf11b1b847
- model-00003-of-00003.safetensors: a82303e28829cd5442336a07af892a889813bf8e06be6f7484fd15dc3d371efb
