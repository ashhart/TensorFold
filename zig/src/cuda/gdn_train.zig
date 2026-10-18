//! Flash Next's Gated DeltaNet backward for Living Weights (gdn_train.cu).
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Args = launch_.Args;

pub const dk = 128;
pub const dv = 128;
pub const taps = 4;
/// Chain rows between state checkpoints (gdn_train.cu's CK): 4 keeps the replay area.
pub const ck = 4;
/// Blocks of 256 threads a value head's chain runs in, each 32 of its state rows (gdn_train.cu's SPLIT).
pub const split = 4;

/// The layer's shape as the kernels read it (gdn_train.cu's GdnShape).
pub const Shape = extern struct {
    rows: c_int,
    nk: c_int,
    nv: c_int,
    conv: c_int,
    proj: c_int,

    pub fn init(rows: usize, nk: usize, nv: usize) Shape {
        const conv = 2 * nk * dk + nv * dv;
        return .{ .rows = @intCast(rows), .nk = @intCast(nk), .nv = @intCast(nv), .conv = @intCast(conv), .proj = @intCast(conv + nv * dv + 2 * nv) };
    }

    pub fn chunks(s: Shape) usize {
        return (@as(usize, @intCast(s.rows)) + ck - 1) / ck;
    }
};

/// An MLX affine 4-bit matrix [n, k]: packed words, bf16 scales and biases every `group` inputs.
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, group: usize = 32 };

/// One DeltaNet layer's weights: conv [conv, 4] bf16, A_log and dt_bias [nv] fp32, the norm's [dv] bf16, projections.
pub const Layer = struct {
    nk: usize,
    nv: usize,
    hidden: usize,
    eps: f32,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    in_proj: Q4, // [qkv | z | b | a] x hidden
    out_proj: Q4, // hidden x nv dv
};

/// Both family adapters use the same fixed head width and affine layouts; reject before shape casts or launches.
pub fn validate(l: Layer, rows: usize) !void {
    if (rows == 0 or rows > 1024 or l.nk == 0 or l.nk > 32 or l.nv == 0 or l.nv > 128 or l.nv % l.nk != 0 or
        l.hidden == 0 or l.hidden > 8192 or l.hidden % 64 != 0 or !std.math.isFinite(l.eps) or l.eps <= 0)
        return error.BadGdnShape;
    const proj = (2 * l.nk + 2 * l.nv) * dk + 2 * l.nv;
    if (l.in_proj.n != proj or l.in_proj.k != l.hidden or l.out_proj.n != l.hidden or l.out_proj.k != l.nv * dv)
        return error.BadGdnShape;
    for ([_]Q4{ l.in_proj, l.out_proj }) |q| if ((q.group != 32 and q.group != 64) or q.k % q.group != 0)
        return error.BadGdnShape;
}

const names = [_][:0]const u8{ "tf_gdn_front", "tf_gdn_scan_fwd", "tf_gdn_norm_back", "tf_gdn_scan_back", "tf_gdn_qk_back", "tf_gdn_gates_back", "tf_gdn_conv_back", "tf_gdn_dequant" };
const train_names = [_][:0]const u8{ "tf_train_narrow", "tf_train_gemm" };

fn index(comptime list: []const [:0]const u8, comptime name: []const u8) usize {
    inline for (list, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

pub const Kernels = struct {
    module: Module,
    train: Module,
    fns: [names.len]Function,
    train_fns: [train_names.len]Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .module = try Module.load(d, kernels.gdn_train), .train = undefined, .fns = undefined, .train_fns = undefined };
        errdefer k.module.unload();
        k.train = try Module.load(d, kernels.train);
        errdefer k.train.unload();
        for (names, &k.fns) |n, *f| f.* = try k.module.function(n);
        for (train_names, &k.train_fns) |n, *f| f.* = try k.train.function(n);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.train.unload();
        k.module.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn up(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// The device buffers a pass of up to `rows` rows uses, carved from one allocation of `bytes(...)`.
pub const Scratch = struct {
    act: u64, // [rows, conv] bf16: the conv outputs
    q: u64, // [rows, nk, dk] fp32, normalized and scaled
    k: u64, // [rows, nk, dk] fp32, normalized
    g: u64, // [rows, nv] fp32
    beta: u64, // [rows, nv] fp32
    y: u64, // [rows, nv, dv] bf16: the chain's read-out
    ckpt: u64, // [nv, chunks, dv, dk] fp32
    work: u64, // [nv, ck, dv, dk] fp32
    dout: u64, // [rows, nv dv] fp32: the out projection's input gradient
    dy: u64, // [rows, nv, dv] fp32
    dqp: u64, // [rows, nv, split, dk] fp32: each chain block's share of dq
    dkp: u64, // [rows, nv, split, dk] fp32
    dact: u64, // [rows, conv] fp32
    dg: u64, // [rows, nv, split] fp32: each chain block's share
    dbeta: u64, // [rows, nv, split] fp32
    dproj: u64, // [rows, proj] fp32: the in projection's output gradient
    xb: u64, // [rows, max(proj, hidden)] bf16: a product's narrowed left side
    wt: u64, // [proj, hidden] bf16: a projection dequantized

    const Part = struct { name: []const u8, bytes: usize };

    fn parts(rows: usize, nk: usize, nv: usize, hidden: usize) [18]Part {
        const s = Shape.init(rows, nk, nv);
        const conv: usize = @intCast(s.conv);
        const proj: usize = @intCast(s.proj);
        const state = nv * dv * dk * 4;
        return .{
            .{ .name = "act", .bytes = rows * conv * 2 },              .{ .name = "q", .bytes = rows * nk * dk * 4 },
            .{ .name = "k", .bytes = rows * nk * dk * 4 },             .{ .name = "g", .bytes = rows * nv * 4 },
            .{ .name = "beta", .bytes = rows * nv * 4 },               .{ .name = "y", .bytes = rows * nv * dv * 2 },
            .{ .name = "ckpt", .bytes = s.chunks() * state },          .{ .name = "work", .bytes = ck * state },
            .{ .name = "dout", .bytes = rows * nv * dv * 4 },          .{ .name = "dy", .bytes = rows * nv * dv * 4 },
            .{ .name = "dqp", .bytes = rows * nv * split * dk * 4 },   .{ .name = "dkp", .bytes = rows * nv * split * dk * 4 },
            .{ .name = "dact", .bytes = rows * conv * 4 },             .{ .name = "dg", .bytes = rows * nv * split * 4 },
            .{ .name = "dbeta", .bytes = rows * nv * split * 4 },      .{ .name = "dproj", .bytes = rows * proj * 4 },
            .{ .name = "xb", .bytes = rows * @max(proj, hidden) * 2 }, .{ .name = "wt", .bytes = proj * hidden * 2 },
        };
    }

    pub fn padded(n: usize) usize {
        return (n + 255) / 256 * 256;
    }

    /// Bytes of one allocation holding every buffer, each 256-byte aligned.
    pub fn bytes(rows: usize, nk: usize, nv: usize, hidden: usize) usize {
        var n: usize = 0;
        for (parts(rows, nk, nv, hidden)) |p| n += padded(p.bytes);
        return n;
    }

    /// The buffers at `base`, an allocation of at least bytes(rows, nk, nv, hidden).
    pub fn carve(base: u64, rows: usize, nk: usize, nv: usize, hidden: usize) Scratch {
        var out: Scratch = undefined;
        var at: u64 = base;
        inline for (@typeInfo(Scratch).@"struct".field_names, 0..) |f, i| {
            @field(out, f) = at;
            at += padded(parts(rows, nk, nv, hidden)[i].bytes);
        }
        return out;
    }
};

/// A family's own 4-bit product for backwardAll's two projections, taken instead of the narrow.
pub const Product = struct {
    ctx: *const anyopaque,
    f: *const fn (ctx: *const anyopaque, x: u64, m: Q4, y: u64, rows: usize, add: bool) anyerror!void,
};

/// Launches on one stream.
pub const Ops = struct {
    k: *const Kernels,
    s: Stream,
    q4: ?Product = null, // backwardAll's projections through the family's own product (null: productWith)

    fn go(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *Args) !void {
        try launch_.launch(o.k.fns[index(&names, name)], .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    fn goTrain(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *Args) !void {
        try launch_.launch(o.k.train_fns[index(&train_names, name)], .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    /// Conv, SiLU, q and k norms, g and beta for every row of P [rows, proj] bf16 after hist [3, conv] (null: zeros).
    pub fn front(o: Ops, p: u64, hist: ?u64, conv_w: u64, a_log: u64, dt_bias: u64, sc: Scratch, s: Shape) !void {
        var a: Args = .{};
        for ([_]u64{ p, hist orelse 0, conv_w, a_log, dt_bias, sc.act, sc.q, sc.k, sc.g, sc.beta }) |v| a.add(v);
        a.add(s);
        try o.go("tf_gdn_front", .{ u(@intCast(s.rows)), u(@intCast(s.nk + s.nv)), 1 }, 128, &a);
    }

    /// The chain from init [nv, dv, dk] fp32 (null: zeros): y and the checkpoints.
    pub fn scanFwd(o: Ops, init: ?u64, sc: Scratch, s: Shape) !void {
        var a: Args = .{};
        for ([_]u64{ sc.act, sc.q, sc.k, sc.g, sc.beta, init orelse 0, sc.y, sc.ckpt }) |v| a.add(v);
        a.add(s);
        try o.go("tf_gdn_scan_fwd", .{ u(@intCast(s.nv * split)), 1, 1 }, 256, &a);
    }

    /// dout back through the gated RMS norm: dy, and dz into dproj.
    pub fn normBack(o: Ops, p: u64, norm_w: u64, sc: Scratch, s: Shape, eps: f32) !void {
        var a: Args = .{};
        for ([_]u64{ sc.y, p, norm_w, sc.dout, sc.dy, sc.dproj }) |v| a.add(v);
        a.add(s);
        a.add(eps);
        try o.go("tf_gdn_norm_back", .{ u(@intCast(s.rows)), u(@intCast(s.nv)), 1 }, 128, &a);
    }

    /// dy back through the chain: dq and dk shares, dv into dact, dg and dbeta.
    pub fn scanBack(o: Ops, sc: Scratch, s: Shape) !void {
        return o.scanBackFrom(sc, s, 0);
    }

    /// scanBack walked back only to the chunk holding row `from`: the rows from it on, the same bits.
    pub fn scanBackFrom(o: Ops, sc: Scratch, s: Shape, from: usize) !void {
        var a: Args = .{};
        for ([_]u64{ sc.act, sc.q, sc.k, sc.g, sc.beta, sc.ckpt, sc.dy, sc.work, sc.dqp, sc.dkp, sc.dact, sc.dg, sc.dbeta }) |v| a.add(v);
        a.add(s);
        a.add(@as(c_int, @intCast(from)));
        try o.go("tf_gdn_scan_back", .{ u(@intCast(s.nv * split)), 1, 1 }, 256, &a);
    }

    /// The dq and dk shares summed per key head and their L2 norms undone into dact.
    pub fn qkBack(o: Ops, sc: Scratch, s: Shape) !void {
        var a: Args = .{};
        for ([_]u64{ sc.act, sc.dqp, sc.dkp, sc.dact }) |v| a.add(v);
        a.add(s);
        try o.go("tf_gdn_qk_back", .{ u(@intCast(s.rows)), u(@intCast(s.nk)), 1 }, 128, &a);
    }

    /// dg and dbeta into dproj's a and b columns.
    pub fn gatesBack(o: Ops, p: u64, a_log: u64, dt_bias: u64, sc: Scratch, s: Shape) !void {
        var a: Args = .{};
        for ([_]u64{ p, a_log, dt_bias, sc.g, sc.dg, sc.dbeta, sc.dproj }) |v| a.add(v);
        a.add(s);
        try o.go("tf_gdn_gates_back", .{ u(@intCast(s.rows)), 1, 1 }, up(@intCast(s.nv), 32) * 32, &a);
    }

    /// dact back through the SiLU and the conv into dproj's qkv columns.
    pub fn convBack(o: Ops, p: u64, hist: ?u64, conv_w: u64, sc: Scratch, s: Shape) !void {
        var a: Args = .{};
        for ([_]u64{ p, hist orelse 0, conv_w, sc.dact, sc.dproj }) |v| a.add(v);
        a.add(s);
        try o.go("tf_gdn_conv_back", .{ up(@intCast(s.conv), 256), u(@intCast(s.rows)), 1 }, 256, &a);
    }

    /// A 4-bit matrix as bf16 [n, k] into out.
    pub fn dequant(o: Ops, m: Q4, out: u64) !void {
        var a: Args = .{};
        for ([_]u64{ m.w, m.s, m.b, out }) |v| a.add(v);
        a.add(@as(c_longlong, @intCast(m.n * m.k)));
        a.add(int(m.group));
        try o.go("tf_gdn_dequant", .{ up(m.n * m.k, 256), 1, 1 }, 256, &a);
    }

    /// y [rows, m.k] (+)= bf16(x) [rows, m.n] times the dequantized matrix: its input's gradient from its output's.
    pub fn product(o: Ops, x: u64, m: Q4, y: u64, rows: usize, sc: Scratch, add: bool) !void {
        return o.productWith(x, m, y, rows, sc.xb, sc.wt, add);
    }

    /// product with the narrowed left side in xb [rows, m.n] bf16 and the dequantized matrix in wt.
    fn productWith(o: Ops, x: u64, m: Q4, y: u64, rows: usize, xb: u64, wt: u64, add: bool) !void {
        var a: Args = .{};
        a.add(x);
        a.add(xb);
        a.add(@as(c_longlong, @intCast(rows * m.n)));
        try o.goTrain("tf_train_narrow", .{ up(rows * m.n, 256), 1, 1 }, 256, &a);
        try o.dequant(m, wt);
        var g: Args = .{};
        for ([_]u64{ xb, wt, y }) |v| g.add(v);
        for ([_]usize{ rows, m.k, m.n, m.n, m.k, m.k }) |v| g.add(int(v));
        g.add(@as(c_int, @intFromBool(add)));
        try o.goTrain("tf_train_gemm", .{ up(m.k, 64), up(rows, 64), 1 }, 128, &g);
    }

    /// The forward for training: P [rows, proj] bf16 (the layer's kept in-projection rows) after the prefix's conv.
    pub fn forward(o: Ops, l: Layer, sc: Scratch, p: u64, hist: ?u64, init: ?u64, rows: usize) !void {
        try validate(l, rows);
        if (rows > 512) return error.BadGdnShape;
        const s = Shape.init(rows, l.nk, l.nv);
        try o.front(p, hist, l.conv_w, l.a_log, l.dt_bias, sc, s);
        try o.scanFwd(init, sc, s);
    }

    /// dx [rows, hidden] (+)= the layer's input gradient from gy [rows, hidden] fp32, its output's.
    pub fn backward(o: Ops, l: Layer, sc: Scratch, p: u64, hist: ?u64, init: ?u64, gy: u64, dx: u64, rows: usize, add: bool) !void {
        try validate(l, rows);
        const s = Shape.init(rows, l.nk, l.nv);
        try o.forward(l, sc, p, hist, init, rows);
        try o.product(gy, l.out_proj, sc.dout, rows, sc, false);
        try o.normBack(p, l.norm_w, sc, s, l.eps);
        try o.scanBack(sc, s);
        try o.qkBack(sc, s);
        try o.gatesBack(p, l.a_log, l.dt_bias, sc, s);
        try o.convBack(p, hist, l.conv_w, sc, s);
        try o.product(sc.dproj, l.in_proj, dx, rows, sc, add);
    }

    /// backward for several sequences of one pass side by side (their rows contiguous in gy and dx).
    pub fn backwardAll(o: Ops, l: Layer, sc: Scratch, w: Wide, runs: []const Run, gy: u64, dx: u64, rows: usize, add: bool) !void {
        try validate(l, rows);
        var end: usize = 0;
        for (runs) |r| {
            if (r.rows == 0 or r.rows > 512 or r.at != end or r.rows > rows - end) return error.BadGdnShape;
            end += r.rows;
        }
        if (end != rows) return error.BadGdnShape;
        const proj: usize = @intCast(Shape.init(1, l.nk, l.nv).proj);
        if (o.q4) |q| try q.f(q.ctx, gy, l.out_proj, w.dout, rows, false) else try o.productWith(gy, l.out_proj, w.dout, rows, w.xb, sc.wt, false);
        for (runs) |r| {
            const s = Shape.init(r.rows, l.nk, l.nv);
            var one = sc;
            one.dout = w.dout + r.at * l.nv * dv * 4;
            one.dproj = w.dproj + r.at * proj * 4;
            try o.forward(l, one, r.p, r.hist, r.init, r.rows);
            try o.normBack(r.p, l.norm_w, one, s, l.eps);
            try o.scanBack(one, s); // replay the whole causal prefix; all following reads are initialized
            try o.qkBack(one, s);
            try o.gatesBack(r.p, l.a_log, l.dt_bias, one, s);
            try o.convBack(r.p, r.hist, l.conv_w, one, s);
        }
        if (o.q4) |q| try q.f(q.ctx, w.dproj, l.in_proj, dx, rows, add) else try o.productWith(w.dproj, l.in_proj, dx, rows, w.xb, sc.wt, add);
    }
};

/// One sequence of a pass the layer's backward reads: its kept in-projection rows.
pub const Run = struct { p: u64, hist: ?u64 = null, init: ?u64 = null, rows: usize, at: usize, from: usize = 0 }; // from: the first row whose gradient is needed

/// Buffers over every row of a pass: the out projection's input gradient [rows, nv dv] fp32.
pub const Wide = struct {
    dout: u64,
    dproj: u64,
    xb: u64,

    pub fn bytes(rows: usize, nk: usize, nv: usize, hidden: usize) usize {
        const proj: usize = @intCast(Shape.init(1, nk, nv).proj);
        return Scratch.padded(rows * nv * dv * 4) + Scratch.padded(rows * proj * 4) + Scratch.padded(rows * @max(proj, hidden) * 2);
    }

    pub fn carve(base: u64, rows: usize, nk: usize, nv: usize) Wide {
        const proj: usize = @intCast(Shape.init(1, nk, nv).proj);
        const dproj = base + Scratch.padded(rows * nv * dv * 4);
        return .{ .dout = base, .dproj = dproj, .xb = dproj + Scratch.padded(rows * proj * 4) };
    }
};

test "the scratch buffers tile their allocation" {
    const n = Scratch.bytes(256, 16, 48, 2560);
    const sc = Scratch.carve(0, 256, 16, 48, 2560);
    try std.testing.expect(sc.wt + 16480 * 2560 * 2 <= n);
    try std.testing.expectEqual(@as(c_int, 16480), Shape.init(1, 16, 48).proj);
    try std.testing.expectEqual(@as(usize, 256 / ck), Shape.init(256, 16, 48).chunks());
    try std.testing.expectEqual(@as(usize, 37 / ck + 1), Shape.init(37, 16, 48).chunks());
}

test "Trainer admission: one shared GDN descriptor accepts both group sizes and refuses unsafe shapes before launch" {
    var layer: Layer = .{ .nk = 16, .nv = 48, .hidden = 5120, .eps = 1e-6, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .in_proj = .{ .w = 0, .s = 0, .b = 0, .n = 16480, .k = 5120 }, .out_proj = .{ .w = 0, .s = 0, .b = 0, .n = 5120, .k = 6144 } };
    for ([_]usize{ 32, 64 }) |group| {
        layer.in_proj.group = group;
        layer.out_proj.group = group;
        try validate(layer, 512);
    }
    try std.testing.expectError(error.BadGdnShape, validate(layer, 0));
    try std.testing.expectError(error.BadGdnShape, validate(layer, 1025));
    layer.nk = 0;
    try std.testing.expectError(error.BadGdnShape, validate(layer, 512));
    layer.nk = 16;
    layer.out_proj.group = 63;
    try std.testing.expectError(error.BadGdnShape, validate(layer, 512));
}
