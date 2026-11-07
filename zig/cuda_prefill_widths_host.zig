//! Three served CUDA prompt adapters over recording forwards, without a GPU or shader toolchain.
test {
    _ = @import("prefill_widths_qwen_host.zig");
    _ = @import("prefill_widths_glm_host.zig");
    _ = @import("prefill_widths_nemo_host.zig");
}
