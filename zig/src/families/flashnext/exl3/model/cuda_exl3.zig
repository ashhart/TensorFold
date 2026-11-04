//! EXL3 checkpoints on the Flash Next CUDA family: what the upstream files' one-line hooks call.
pub const config = @import("../pack/cuda_exl3_config.zig");
pub const arch = @import("cuda_exl3_arch.zig");
pub const store = @import("../pack/cuda_exl3_store.zig");
pub const memory = @import("../pack/cuda_exl3_memory.zig");
pub const load = @import("../pack/cuda_exl3_load.zig");
pub const pass = @import("cuda_exl3_pass.zig");
pub const tp = @import("../tp/cuda_exl3_tp.zig");
pub const kv = @import("../kv/cuda_exl3_kv.zig");
pub const fork = @import("../tp/cuda_exl3_fork.zig");
pub const telemetry = @import("cuda_exl3_telemetry.zig");
pub const frequency = @import("../offload/cuda_exl3_frequency.zig");
/// The upstream modules an EXL3 check drives (tf-cuda-test flashnext-exl3): the store, loaders and the batched pass.
pub const prompt_wt = @import("../../cuda_prompt_wt.zig");
pub const prompt_load = @import("../../cuda_prompt_load.zig");
pub const projection_load = @import("../../cuda_projection_load.zig");
pub const batch = @import("../../cuda_batch.zig");

test {
    _ = config;
    _ = arch;
    _ = store;
    _ = memory;
    _ = load;
    _ = pass;
    _ = tp;
    _ = kv;
    _ = @import("../kv/cuda_exl3_kv_test.zig");
    _ = @import("../../cuda_mtp_ops_test.zig");
    _ = fork;
    _ = telemetry;
    _ = frequency;
    _ = @import("../offload/cuda_exl3_frequency_state.zig");
}
