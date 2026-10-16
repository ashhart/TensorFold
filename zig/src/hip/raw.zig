//! Device copies and fills on a forward's raw stream handle; the counting stream runs none of them.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const counting = @import("launches/util.zig").counting;

/// `n` bytes from device address `src` into `dst` at `offset`, queued on `stream`.
pub fn copy(dst: DeviceBuffer, offset: usize, src: u64, n: usize, stream: abi.Stream) runtime.Error!void {
    if (offset > dst.len or n > dst.len - offset) return error.Invalid;
    if (n == 0 or stream == counting) return;
    try runtime.check(dst.r.api.hipMemcpyDtoDAsync(@ptrFromInt(dst.base() + offset), @ptrFromInt(src), n, stream));
}

/// Every byte of `b` set to `value`, queued on `stream`.
pub fn fill8(b: DeviceBuffer, value: u8, stream: abi.Stream) runtime.Error!void {
    if (b.len == 0 or stream == counting) return;
    try runtime.check(b.r.api.hipMemsetAsync(b.ptr, value, b.len, stream));
}

/// Every 32-bit word of `b` set to `value`, queued on `stream`.
pub fn fill32(b: DeviceBuffer, value: u32, stream: abi.Stream) runtime.Error!void {
    if (b.len % 4 != 0) return error.Invalid;
    if (b.len == 0 or stream == counting) return;
    try runtime.check(b.r.api.hipMemsetD32Async(b.ptr, @bitCast(value), b.len / 4, stream));
}

/// Pinned host `bytes` into `dst` at `offset`, queued on `stream`; they stay unchanged until it completes.
pub fn upload(dst: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) runtime.Error!void {
    if (offset > dst.len or bytes.len > dst.len - offset) return error.Invalid;
    if (bytes.len == 0 or stream == counting) return;
    try runtime.check(dst.r.api.hipMemcpyHtoDAsync(@ptrFromInt(dst.base() + offset), bytes.ptr, bytes.len, stream));
}

/// `out.len` bytes of `src` from `offset` into pinned host `out`, queued on `stream`.
pub fn download(src: DeviceBuffer, offset: usize, out: []u8, stream: abi.Stream) runtime.Error!void {
    if (offset > src.len or out.len > src.len - offset) return error.Invalid;
    if (out.len == 0 or stream == counting) return;
    try runtime.check(src.r.api.hipMemcpyDtoHAsync(out.ptr, @ptrFromInt(src.base() + offset), out.len, stream));
}
