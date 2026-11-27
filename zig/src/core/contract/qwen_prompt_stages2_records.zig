//! The opt-in two-stage prompt launch; missing compiler receipts remain unqualified.
const registry = @import("registry.zig");

pub const cuda = [_]registry.CudaRecord{.{
    .launch = .{
        .name = "qmm_prefill 2x4 two-stage Qwen prompt",
        .symbol = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi2ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii",
        .image = "qmm_prefill",
        .threads = 256,
        .dynamic = 41984,
        .min_sm = 121,
    },
    .usage = registry.none,
}};
