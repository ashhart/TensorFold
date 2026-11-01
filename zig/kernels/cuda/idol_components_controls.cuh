// Idol's controls past the denoiser's, one header a component, in contract_controls.cu's image; none ship.
#pragma once
// Gates 3 and 4: Idol's DINOv3 glue (idol_dino.cu), one bf16 rounding dropped or added, or its sums in fp16.
#include "idol_dino_controls.cuh"
// Idol's image encoder (idol_vae.cu), each control one planted flaw of idol_vae.cuh's bodies.
#include "idol_vae_controls.cuh"
// Idol decoders: their templates at the BREAK values that drop or add one rounding point, sum or rule.
#include "contract_controls_idol_decoder.cuh"
// Idol's background remover's controls (fp16 rounding points or sums, a dropped NaN); none ship.
#include "idol_birefnet_controls.cuh"
