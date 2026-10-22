//! The batched pass's small launches: the K-slice sum, the conv window shift and column copies.
const cuda = @import("cuda");
const Kernels = @import("cuda_batch_state.zig").Kernels;

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub fn sumSlices(k: *const Kernels, stream: cuda.Stream, part: u64, out: u64, bf16: bool, n: usize, sk: usize) !void {
    var a: cuda.Args = .{};
    a.add(part);
    a.add(out);
    a.add(@as(c_int, @intFromBool(bf16)));
    a.add(@as(c_longlong, @intCast(n)));
    a.add(cint(sk));
    try cuda.launch.launch(k.sum, .{ .grid = .{ .x = @intCast(cdiv(n, 256)) }, .block = .{ .x = 256 } }, stream, &a);
}

pub fn shiftWindow(k: *const Kernels, stream: cuda.Stream, win: u64, rows_ptr: u64, stride: usize, n: usize, taps: usize, c: usize) !void {
    var a: cuda.Args = .{};
    a.add(win);
    a.add(rows_ptr);
    a.add(@as(c_longlong, @intCast(stride)));
    a.add(cint(n));
    a.add(cint(taps));
    a.add(cint(c));
    try cuda.launch.launch(k.shift, .{ .grid = .{ .x = @intCast(cdiv(c, 256)) }, .block = .{ .x = 256 } }, stream, &a);
}

/// rows x width 16-bit columns from src (src_pitch values a row) to dst (dst_pitch); dst 0: none.
pub fn copyCols(k: *const Kernels, stream: cuda.Stream, src: u64, src_pitch: usize, dst: u64, dst_pitch: usize, width: usize, rows: usize) !void {
    if (dst == 0 or rows == 0) return;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(c_longlong, @intCast(src_pitch)));
    a.add(dst);
    a.add(@as(c_longlong, @intCast(dst_pitch)));
    a.add(cint(width));
    try cuda.launch.launch(k.copy_cols, .{ .grid = .{ .x = @intCast(cdiv(width, 256)), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}
