//! Force the real profile, contract and Qwen startup bodies without device discovery or CUDA compilation.
export fn promptStages2Consumers() usize {
    return @intFromPtr(&@import("qwen_prompt_gemm_profile.zig").run) ^
        @intFromPtr(&@import("contract_qmm_prefill_stages2.zig").run) ^
        @intFromPtr(&@import("qwen35").engine.Engine.init);
}
