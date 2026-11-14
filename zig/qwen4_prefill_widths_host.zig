//! Select the Metal tag for the existing memory-backed Qwen4 adapter fixture; no device calls.
pub const metal_prefill_widths = true;
test {
    _ = @import("qwen4");
}
