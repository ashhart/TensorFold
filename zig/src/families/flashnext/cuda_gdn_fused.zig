//! The prompt GDN block's launches: fn_gdn_fused (front, chain, gated norm in one kernel a value head), the gated norm.
const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const parts = @import("cuda_batch_state.zig");
const eps = @import("cuda_prompt_wt.zig").eps;

/// One kernel a value head for a lone stream's pass, unless TF_FLASHNEXT_GDN_FUSED=0 (three launches, the same bits).
pub fn on() bool {
    const v = std.c.getenv("TF_FLASHNEXT_GDN_FUSED") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// fn_gdn_fused's inputs: projection rows, conv window and taps, gate parameters, the state and where it goes.
pub const In = struct { proj: u64, conv: u64, cw: u64, a_log: u64, dt: u64, state: u64, state_out: u64, norm: u64 };

/// fn_gdn_front, fn_gdn_chain_tpr and fn_gdn_norm over `rows` rows in one launch (their bits): out and xs.
pub fn launch(f: cuda.Function, stream: cuda.Stream, in: In, rows: usize, out: u64, xs: u64) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ in.proj, in.conv, in.cw, in.a_log, in.dt, in.state, in.state_out }) |v| a.add(v);
    a.add(@as(c_int, @intCast(rows)));
    a.add(in.norm);
    a.add(eps);
    for ([_]u64{ out, xs, 0 }) |v| a.add(v); // no fp32 readouts kept
    try cuda.launch.launch(f, .{ .grid = .{ .x = gdn.nv }, .block = .{ .x = parts.gdn_fused_threads }, .shared = parts.gdn_fused_smem }, stream, &a);
}

/// fn_gdn_norm over `rows` readouts ys: the gated norm into out, its sums into xs.
pub fn norm(f: cuda.Function, stream: cuda.Stream, ys: u64, proj: u64, norm_w: u64, out: u64, xs: u64, rows: usize) !void {
    var b: cuda.Args = .{};
    for ([_]u64{ ys, proj, norm_w }) |v| b.add(v);
    b.add(eps);
    for ([_]u64{ out, xs }) |v| b.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = gdn.nv }, .block = .{ .x = gdn.dv } }, stream, &b);
}
