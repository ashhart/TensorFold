//! An EXL3 pack's linears on the Metal lane: #5050's projection, #5054's gate/up and #5063's experts, 16 rows a launch.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const x3 = @import("core").exl3_projection;
const xe = @import("experts_exl3.zig");
const glue = @import("glue.zig");

const Ref = glue.Ref;
pub const Matrix = x3.Matrix;

/// Rows a launch takes: the projections' row kernels are bit-equal per row from 1 to 16, and the experts take 16.
pub const max_rows = 16;

pub const Pipes = struct {
    proj: x3.Pipelines,
    gate_up: x3.gate_up.Pipelines,
    experts: xe.Pipelines,

    pub fn load(device: mtl.Device) !Pipes {
        var rows = try x3.Pipelines.load(device);
        errdefer rows.deinit();
        var pair = try x3.gate_up.Pipelines.load(device);
        errdefer pair.deinit();
        return .{ .proj = rows, .gate_up = pair, .experts = try xe.Pipelines.load(device) };
    }

    pub fn deinit(p: *Pipes) void {
        p.proj.deinit();
        p.gate_up.deinit();
        p.experts.deinit();
    }
};

/// Each linear's K x N of an EXL3 pack: the kernels decode whole 128-value Hadamard blocks on both sides.
fn shapes(c: *const Config, out: *[11][2]usize) []const [2]usize {
    const kg = @as(usize, c.heads) / c.o_groups * c.head_dim;
    out[0..10].* = .{
        .{ c.hidden, c.q_lora },                          .{ c.hidden, c.head_dim },
        .{ c.q_lora, @as(usize, c.heads) * c.head_dim },  .{ kg, c.o_lora },
        .{ @as(usize, c.o_groups) * c.o_lora, c.hidden }, .{ c.q_lora, @as(usize, c.index_heads) * c.index_dim },
        .{ c.head_dim, c.index_dim },                     .{ c.hidden, c.expert_width },
        .{ c.expert_width, c.hidden },                    .{ c.hidden, c.vocab },
    };
    if (c.engram_layers.len == 0) return out[0..10];
    out[10] = .{ @as(usize, c.hashCols()) * c.engram_head_dim, (@as(usize, c.hc_mult) + 1) * c.hidden };
    return out[0..11];
}

/// A config whose linears an EXL3 pack cannot hold on this lane is refused before any kernel is built.
pub fn check(c: *const Config) !void {
    if (c.format != .exl3) return;
    if (c.o_groups > max_groups) return error.UnsupportedShape;
    var buf: [11][2]usize = undefined;
    for (shapes(c, &buf)) |s| _ = try x3.scratchBytes(s[0], s[1], max_rows);
}

/// wo_a's groups an EXL3 pack may hold (the release has 8).
pub const max_groups = 8;

/// Scratch of one launch at the widest linear (gate and up take two), or 0 for the release's format.
pub fn scratchBytes(c: *const Config) usize {
    if (c.format != .exl3) return 0;
    var most: usize = 0;
    var buf: [11][2]usize = undefined;
    for (shapes(c, &buf)) |s| most = @max(most, x3.scratchBytes(s[0], s[1], max_rows) catch 0);
    const gu = x3.gate_up.scratchBytes(c.hidden, c.expert_width, max_rows) catch 0;
    const call: xe.Call = .{ .rows = max_rows, .k = c.hidden, .n = c.expert_width, .experts = c.experts, .topk = c.topk };
    return @max(most, gu, xe.scratchBytes(call));
}

pub fn view(r: Ref) x3.View {
    return .{ .buffer = r.buf, .offset = r.off };
}

fn at(r: Ref, bytes: usize) Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

/// x [rows][k] bf16 through `m` into y [rows][n] (bf16 or fp32), 16 rows a launch; `scratch` holds one launch.
pub fn proj(p: *const Pipes, e: mtl.ComputeEncoder, m: Matrix, in: Ref, out: Ref, rows: u32, f32_out: bool, scratch: Ref) !void {
    const ob: usize = if (f32_out) 4 else 2;
    var r: u32 = 0;
    while (r < rows) : (r += max_rows) {
        const n = @min(max_rows, rows - r);
        const rs: x3.Rows = .{ .count = n, .x_stride = m.k, .y_stride = m.n, .output = if (f32_out) .f32 else .bf16 };
        try x3.encode(&p.proj, e, m, view(at(in, r * m.k * 2)), view(at(out, r * m.n * ob)), view(scratch), rs);
    }
}

/// The shared expert's w1 and w3 on x [rows][k], its clamped SwiGLU into y and that SwiGLU's act_quant into yq.
pub fn gateUp(p: *const Pipes, e: mtl.ComputeEncoder, w1: Matrix, w3: Matrix, in: Ref, y: Ref, yq: Ref, rows: u32, limit: f32, scratch: Ref) !void {
    var r: u32 = 0;
    while (r < rows) : (r += max_rows) {
        const n = @min(max_rows, rows - r);
        const rs: x3.Rows = .{ .count = n, .x_stride = w1.k, .y_stride = w1.n };
        const yo = r * w1.n * 2;
        try x3.gate_up.encode(&p.proj, &p.gate_up, e, w1, w3, view(at(in, r * w1.k * 2)), view(at(y, yo)), view(at(yq, yo)), view(scratch), rs, limit);
    }
}

/// The routed experts of up to 16 rows: gate and up with the route weight inside, then the down into `out`.
pub fn experts(p: *const Pipes, e: mtl.ComputeEncoder, in: Ref, t: xe.Tables, r: xe.Route, act: Ref, out: Ref, scratch: Ref, c: xe.Call, limit: f32) !void {
    try xe.check(c, t);
    xe.up(&p.experts, e, cast(in), t, r, cast(act), cast(scratch), c, limit);
    xe.down(&p.experts, e, cast(act), t, r, cast(out), cast(scratch), c);
}

fn cast(r: Ref) xe.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

test "an EXL3 config's scratch holds its widest launch and the release's takes none" {
    var c: Config = .{};
    try std.testing.expectEqual(@as(usize, 0), scratchBytes(&.{ .format = .release_fp8 }));
    try check(&c);
    // the head at 16 rows: rotated rows of 5120 then fp32 sums of the vocabulary
    try std.testing.expect(scratchBytes(&c) >= 16 * 5120 * 2 + 16 * @as(usize, c.vocab) * 4);
    c.q_lora = 1200;
    try std.testing.expectError(error.UnsupportedExl3Shape, check(&c));
}
