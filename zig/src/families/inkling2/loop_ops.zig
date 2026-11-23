//! Mini's loop on the device (inkling2/loop.cu): a message visit's raw mixes, messages, rank-64 gates and norms.
const std = @import("std");
const cuda = @import("cuda");

pub const width = 2304; // the kernels' hidden size (loop.cu kWidth)
const rank = 64;
const split = 9;
const most_gates = 3;

/// loop.cu's Gates: a core layer's four gates (q, k, v, MLP) or one gate repeated, bf16 down, forget and gain.
pub const Gates = extern struct { down: [4]u64, forget: [4]u64, gain: [4]u64 };

/// loop.cu's RawMix, field for field.
const RawMix = extern struct {
    blocks: [8]u64 = @splat(0),
    partial: u64,
    query: u64,
    out: u64,
    n_blocks: i32,
    partial_bf16: i32,
    nq: i32,
    rows: i32,
    c: i32,
};

const Norms = extern struct { w: [3]u64 };

/// A core layer's loop weights: its four gates and their queries as one fp32 [4][hidden] table (q, k, v, MLP).
pub const Layer = struct {
    gates: Gates,
    queries: u64,

    /// The gate of stream `j` (q 0, k 1, v 2, MLP 3) in every place, for a launch of that one gate.
    fn one(l: Layer, j: usize) Gates {
        return .{ .down = @splat(l.gates.down[j]), .forget = @splat(l.gates.forget[j]), .gain = @splat(l.gates.gain[j]) };
    }
};

pub const Kernels = struct {
    mod: cuda.Module,
    msg: cuda.Function,
    down8: cuda.Function,
    down16: cuda.Function,
    down64: cuda.Function,
    blend: cuda.Function,
    mix: cuda.Function,
    norm: cuda.Function,
    bank: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_loop);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .msg = try mod.function("tf_ink2_loop_msg"),
            .down8 = try mod.function("tf_ink2_loop_down8"),
            .down16 = try mod.function("tf_ink2_loop_down16"),
            .down64 = try mod.function("tf_ink2_loop_down64"),
            .blend = try mod.function("tf_ink2_loop_blend"),
            .mix = try mod.function("tf_ink2_loop_mix"),
            .norm = try mod.function("tf_ink2_loop_norm"),
            .bank = try mod.function("tf_ink2_loop_bank"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.* = undefined;
    }
};

/// A call's loop scratch for `rows` rows (all zeroed at allocation; the down kernels' counts stay zeroed after each).
pub const Scratch = struct {
    rows: usize = 0,
    bank: [2]u64 = .{ 0, 0 }, // fp32 [rows][sources][hidden]: the bank a pass writes, the next pass reads
    h: u64 = 0, // fp32 [3][rows][hidden]: a visit's raw mixes
    pair: u64 = 0, // bf16 [rows][3][2 hidden]: the gates' down inputs
    msg: u64 = 0, // fp32 [rows][3][hidden]
    part: u64 = 0, // fp32 [rows][3][9][64]: the down slices
    count: u64 = 0, // u32 [row tiles][3][8]
    g: u64 = 0, // bf16 [rows][3][64]
    out: u64 = 0, // fp32 [rows][3][hidden]: the gates' outputs

    /// Scratch for `rows` rows with `sources` bank states, each buffer from A(ctx, bytes) (zeroed).
    pub fn init(rows: usize, sources: usize, ctx: anytype, comptime A: anytype) !Scratch {
        const w: usize = width;
        const g = most_gates;
        return .{
            .rows = rows,
            .bank = .{ try A(ctx, rows * sources * w * 4), try A(ctx, rows * sources * w * 4) },
            .h = try A(ctx, g * rows * w * 4),
            .pair = try A(ctx, rows * g * 2 * w * 2),
            .msg = try A(ctx, rows * g * w * 4),
            .part = try A(ctx, rows * g * split * rank * 4),
            .count = try A(ctx, (rows + 15) / 16 * g * 8 * 4),
            .g = try A(ctx, rows * g * rank * 2),
            .out = try A(ctx, rows * g * w * 4),
        };
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// The messages, gates and blends of `gates` streams of h (fp32, `h_row` / `h_gate` elements apart) into x.out.
fn gate(k: *const Kernels, s: cuda.Stream, x: *const Scratch, bank: u64, sources: usize, queries: u64, h: u64, h_row: usize, h_gate: usize, gates: usize, w: Gates, rows: usize, eps: f32) !void {
    const scale: f32 = @floatCast(std.math.pow(f64, width, -0.5));
    var a: cuda.Args = .{};
    a.add(bank);
    a.add(@as(i32, @intCast(sources)));
    a.add(queries);
    a.add(h);
    a.add(@as(i64, @intCast(h_row)));
    a.add(@as(i64, @intCast(h_gate)));
    a.add(@as(i32, @intCast(gates)));
    a.add(scale);
    a.add(eps);
    a.add(w);
    a.add(x.msg);
    a.add(x.pair);
    try go(k.msg, s, .{ .x = @intCast(rows * gates) }, 256, &a);
    // launch shapes by rows: none of them moves a bit (each output's sums keep one order)
    const out: usize = if (rows <= 16) 8 else if (rows <= 128) 16 else 64;
    var b: cuda.Args = .{};
    b.add(x.pair);
    b.add(w);
    b.add(@as(i32, @intCast(rows)));
    b.add(@as(i32, @intCast(gates)));
    b.add(x.part);
    b.add(x.count);
    b.add(x.g);
    const f = if (out == 8) k.down8 else if (out == 16) k.down16 else k.down64;
    try go(f, s, .{ .x = @intCast((rows + 15) / 16), .y = split, .z = @intCast(gates * (rank / out)) }, 128, &b);
    const per: usize = if (rows <= 16) 16 else 32;
    var c: cuda.Args = .{};
    c.add(h);
    c.add(@as(i64, @intCast(h_row)));
    c.add(@as(i64, @intCast(h_gate)));
    c.add(x.msg);
    c.add(x.g);
    c.add(w);
    c.add(@as(i32, @intCast(rows)));
    c.add(@as(i32, @intCast(gates)));
    c.add(@as(i32, @intCast(per)));
    c.add(x.out);
    try go(k.blend, s, .{ .x = width / 32, .y = @intCast(gates), .z = @intCast((rows + per - 1) / per) }, 256, &c);
}

/// What a message visit's mix reads: the completed blocks and the partial, and the layer's mix queries.
pub const Mix = struct { blocks: []const u64, partial: u64, partial_bf16: bool, query: u64, nq: usize };

fn mix(k: *const Kernels, s: cuda.Stream, m: Mix, out: u64, rows: usize) !void {
    var r: RawMix = .{ .partial = m.partial, .query = m.query, .out = out, .n_blocks = @intCast(m.blocks.len), .partial_bf16 = @intFromBool(m.partial_bf16), .nq = @intCast(m.nq), .rows = @intCast(rows), .c = width };
    for (m.blocks, 0..) |b, j| r.blocks[j] = b;
    var a: cuda.Args = .{};
    a.add(r);
    try go(k.mix, s, .{ .x = @intCast(rows) }, 512, &a);
}

fn norm(k: *const Kernels, s: cuda.Stream, x: u64, w: [3]u64, gates: usize, out: u64, rows: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(Norms{ .w = w });
    a.add(out);
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i32, width));
    a.add(eps);
    try go(k.norm, s, .{ .x = @intCast(rows), .y = @intCast(gates) }, 512, &a);
}

/// A message visit's q, k and v inputs: the mix, the three gates on the previous pass's bank, each input norm (bf16).
pub fn attention(k: *const Kernels, s: cuda.Stream, x: *const Scratch, l: Layer, m: Mix, bank: u64, sources: usize, norms: [3]u64, rows: usize, eps: f32, out: u64) !void {
    try mix(k, s, m, x.h, rows);
    const h_gate: usize = if (m.nq == 3) rows * width else 0;
    try gate(k, s, x, bank, sources, l.queries, x.h, width, h_gate, 3, l.gates, rows, eps);
    const nw: [3]u64 = if (norms[1] == 0) @splat(norms[0]) else norms;
    try norm(k, s, x.out, nw, 3, out, rows, eps);
}

/// A message visit's MLP input: the mix of the fp32 residual, the MLP gate, mlp_norm into bf16 [rows][hidden].
pub fn mlp(k: *const Kernels, s: cuda.Stream, x: *const Scratch, l: Layer, m: Mix, bank: u64, sources: usize, mlp_norm: u64, rows: usize, eps: f32, out: u64) !void {
    try mix(k, s, m, x.h, rows);
    try gate(k, s, x, bank, sources, l.queries + 3 * width * 4, x.h, width, 0, 1, l.one(3), rows, eps);
    try norm(k, s, x.out, @splat(mlp_norm), 1, out, rows, eps);
}

/// bf16 rows `src` into bank source `at` of `sources`.
pub fn save(k: *const Kernels, s: cuda.Stream, src: u64, bank: u64, sources: usize, at: usize, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(src);
    a.add(bank);
    a.add(@as(i32, @intCast(sources)));
    a.add(@as(i32, @intCast(at)));
    a.add(@as(i32, width));
    try go(k.bank, s, .{ .x = @intCast(rows) }, 256, &a);
}

test "loop: the argument structs keep loop.cu's layouts" {
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Gates));
    try std.testing.expectEqual(@as(usize, 8 * 8 + 3 * 8 + 5 * 4 + 4), @sizeOf(RawMix));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Norms));
}
