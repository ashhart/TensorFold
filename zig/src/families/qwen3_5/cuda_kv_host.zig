//! TF_QWEN_KV_HOST=1: the pool's KV pages (and runs) in pinned host memory, mapped into the
//! device's address space (cuMemHostAlloc PORTABLE|DEVICEMAP, as GLM's host pages). Qwen3.8's
//! full attention reads every position every token, so a host tier is a capacity mode, not a
//! speed mode: decode pays one PCIe read of the window's KV a token, while prefill streams
//! writes and stays compute-bound. The kernels are untouched: one page base, host or device.
const std = @import("std");

/// An exact-one switch through the profile registry, which is the one reader of a registered name.
pub fn onHost() bool {
    const v = @import("profile").get(.TF_QWEN_KV_HOST) orelse return false;
    return std.mem.eql(u8, v, "1");
}

/// TF_QWEN_KV_HOST_GIB=n caps the host pages; unset: a full window's worth plus its run.
pub fn hostGib() ?u64 {
    const v = std.c.getenv("TF_QWEN_KV_HOST_GIB") orelse return null;
    const n = std.fmt.parseInt(u64, std.mem.span(v), 10) catch return null;
    return if (n == 0) null else n << 30;
}

/// Host bytes the pool's pages need: a full window's pages plus its run, or the _GIB cap.
pub fn hostBytes(capacity: usize, tokens: usize, page_bytes: u64, run_bytes: u64) u64 {
    const t: usize = @intCast(tokens);
    const window_pages = (capacity + t - 1) / t;
    const run_pages = (run_bytes + page_bytes - 1) / page_bytes;
    const all = (window_pages + run_pages) * page_bytes;
    return @min(all, hostGib() orelse all);
}
