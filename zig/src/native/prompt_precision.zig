//! Backend-specific prompt-precision admission messages.
const api = @import("engine_api");

pub const metal_int8 = "--prompt-precision int8 is the CUDA Qwen3.5 engine's; the Metal engines prompt in bf16: serve without it";
pub const metal_fp8 = "--prompt-precision fp8 is the guarded CUDA Qwen3.5 engine's; the Metal engines prompt in bf16: serve without it";

pub fn metal(o: api.Open) ?[]const u8 {
    return if (o.prompt_int8) metal_int8 else if (o.prompt_fp8) metal_fp8 else null;
}
