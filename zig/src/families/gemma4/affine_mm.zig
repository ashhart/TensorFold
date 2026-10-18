//! Gemma-only dense projections: weights rounded to bf16 in the tile, preserving the contributed arithmetic.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

/// Source specialization retained for the contributed dense shader.
pub const Glu = enum { clipped, plain };

/// The weights' bits (2-6 or 8) and group (32 or 64), the output's type and the GLU epilogue.
pub const Format = struct { bits: u8 = 4, group: u16 = 64, out_f32: bool = false, glu: Glu = .clipped };

/// The kernel source for `f` (nax.h inlined for this GPU).
pub fn source(device: mtl.Device, a: std.mem.Allocator, f: Format) ![]u8 {
    if (f.bits < 2 or f.bits > 8 or f.bits == 7 or (f.group != 32 and f.group != 64)) return error.UnsupportedFormat;
    const body = try frags.source(device, a, ks.gemma4_affine_mm);
    defer a.free(body);
    const glu = if (f.glu == .plain) "#define TF_GLU_PLAIN 1\n" else "";
    const out = if (f.out_f32) "float" else "bfloat";
    const head = "#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_OUT_T {s}\n{s}{s}";
    return std.fmt.allocPrint(a, head, .{ f.bits, f.group, out, glu, body });
}

/// Gemma's rounded-weight dense entry point is compiled in its own library.
pub const names = [1][:0]const u8{"tf_affine_mm"};
pub const Pipes = [1]mtl.Pipeline;

pub const Args = extern struct { rows: i32, n: i32, k: i32, experts: i32 };

/// Dense pitches and batch offsets in elements, with weight offsets in rows.
pub const Strides = extern struct { x_row: i32 = 0, y_row: i32 = 0, x_batch: i32 = 0, y_batch: i32 = 0, w_batch: i32 = 0, pad: i32 = 0 };

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// y [rows, n] = x [rows, q.k] W^T; n is q.n rounded up to 64 (the padded weight rows, y's pitch).
pub fn dense(e: mtl.ComputeEncoder, p: Pipes, x: anytype, q: anytype, y: anytype, rows: u32) void {
    const n = std.mem.alignForward(u32, q.n, 64);
    denseBatch(e, p, x, q, y, rows, 1, .{ .x_row = @intCast(q.k), .y_row = @intCast(n) });
}

/// Batched dense products over separate weight rows; q.n is a multiple of 64.
pub fn denseBatch(e: mtl.ComputeEncoder, p: Pipes, x: anytype, q: anytype, y: anytype, rows: u32, batch: u32, s: Strides) void {
    const n = std.mem.alignForward(u32, q.n, 64);
    e.setPipeline(p[0]);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(q.k), .experts = 0 }, 5);
    bind(e, 6, y);
    e.setValue(s, 7);
    e.dispatchGroups(mtl.Size.of(n / 64, (rows + 63) / 64, batch), mtl.Size.of(32, 2, 2));
}
