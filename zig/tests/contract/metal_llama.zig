//! Llama's glue, row projections and GEMM under the contract, compiled as the loader compiles them, launched as served.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_llama_controls.zig");
const c = contract.spec;
const g = contract.gates;
const rows_in = contract.rows;
pub const family = tf.llama.contracts;

pub const DH = 128;
const G33 = family.nanbeige43;
const G3 = family.nanbeige41;

/// glue.metal's argument structs, as forward.zig fills them.
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, eps: f32 = 0 };
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos: u32, stride: u32 };

/// A value through fp32 and then bf16, as the kernels round.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn none(_: c.Shape) usize {
    return 0;
}

/// One thread's fma chain over the row (256 threads), the simd tree and the eight partials.
fn normDepth(s: c.Shape) usize {
    return s.k / 256 + 16;
}

const no_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums: elementwise, a gather or a draw" };
// the norm's inner bf16(x * inv) rounds either way under the fp32 sum's error (mids 3): fp16's overflow misses 64x
const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const rms_sums: c.Sums = .{ .f16 = 50, .bf16 = 0, .why = rms_why };
const argmax_rule = "a NaN logit wins (the lowest NaN id); else the largest logit, ties to the lower id";

pub fn pipelineOf(lib: mtl.Library, device: mtl.Device, name: []const u8) !mtl.Pipeline {
    var buf: [48]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&buf, "{s}", .{name}, 0);
    return mtl.Pipeline.init(device, lib, z, false);
}

pub const Which = enum { embed, norm, add, add_norm, silu, prep };

pub fn function(w: Which) []const u8 {
    return switch (w) {
        .embed => "ll_embed",
        .norm => "ll_norm",
        .add => "ll_add",
        .add_norm => "ll_add_norm",
        .silu => "ll_silu_gu",
        .prep => "ll_prep",
    };
}

pub fn spec(w: Which) c.Spec {
    const widths = &[_]c.Shape{ .{ .rows = 32, .n = G33.hidden, .k = G33.hidden }, .{ .rows = 2048, .n = G3.hidden, .k = G3.hidden } };
    const anyk: [3]c.Tile = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } };
    return switch (w) {
        .embed => .{ .name = "llama glue ll_embed", .accum = .none, .depth = none, .out = .bf16, .served = widths, .sums = no_sums },
        .norm => .{ .name = "llama glue ll_norm", .accum = .f32, .depth = normDepth, .ops = 3, .mids = 3, .out = .bf16, .tiles = anyk, .served = widths, .sums = rms_sums },
        .add => .{ .name = "llama glue ll_add", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .served = widths, .sums = no_sums },
        .add_norm => .{ .name = "llama glue ll_add_norm", .accum = .f32, .depth = normDepth, .ops = 3, .mids = 3, .out = .bf16, .in_place = true, .tiles = anyk, .served = widths, .sums = rms_sums },
        .silu => .{ .name = "llama glue ll_silu_gu", .accum = .none, .depth = none, .ops = 3, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 32, .n = G33.intermediate, .k = G33.intermediate }, .{ .rows = 2048, .n = G3.intermediate, .k = G3.intermediate } }, .sums = no_sums },
        .prep => .{ .name = "llama glue ll_prep", .accum = .none, .depth = none, .ops = 5, .mids = 2, .out = .bf16, .served = &.{ .{ .rows = 32, .n = G33.heads + G33.kv_heads, .k = (G33.heads + 2 * G33.kv_heads) * DH }, .{ .rows = 2048, .n = G3.heads + G3.kv_heads, .k = (G3.heads + 2 * G3.kv_heads) * DH } }, .sums = no_sums },
    };
}

/// The glue kernels: the width is the shape's k (n says the same), the rows its rows.
pub const Glue = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    geo: family.Geometry = G33,
    pos: u32 = 0,
    r: ?rows_in.Rows = null,
    x: []u16 = &.{}, // the residual's first contents (add, add_norm), gu (silu), q|k|v (prep), the logits (argmax)
    y: []u16 = &.{}, // the added rows (add, add_norm)
    table: []u16 = &.{},
    ids: []u32 = &.{},
    inv: []f32 = &.{},
    in: [3]rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, mutation: controls.Mutation) !Glue {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .mutation = mutation, .lib = lib, .pipe = try pipelineOf(lib, device, function(which)) };
    }

    fn drop(p: *Glue) void {
        if (p.r) |r| r.deinit();
        p.r = null;
        for ([_]*[]u16{ &p.x, &p.y, &p.table }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.a.free(p.ids);
        p.ids = &.{};
        p.a.free(p.inv);
        p.inv = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Glue) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Glue);

    pub fn kernel(p: *Glue) g.Kernel {
        return controls.mark(.{ .spec = spec(p.which), .ctx = p, .vt = &vt }, p.mutation);
    }

    fn width(p: *const Glue) usize {
        return p.shape.k;
    }

    fn random16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
        for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
    }

    /// NaN, +Inf and -Inf into rows 0, 1 and 2 of `v` (rows `w` wide), as the input families plant them.
    fn plant(v: []u16, w: usize, rows: usize) void {
        const at = [_][2]usize{ .{ 0, w / 3 }, .{ 1, w / 2 }, .{ 2, w - 1 } };
        const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
        for (at, put) |x, f| if (x[0] < rows) {
            v[x[0] * w + x[1]] = c.toBf16(f);
        };
    }

    pub fn prepare(p: *Glue, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const w = s.k;
        const n = s.rows * w;
        switch (p.which) {
            .embed => {
                p.table = try p.a.alloc(u16, 8 * w);
                random16(rng, p.table, -2, 2);
                if (fill == .nonfinite) plant(p.table, w, 8);
                p.ids = try p.a.alloc(u32, s.rows);
                for (p.ids, 0..) |*id, i| id.* = @intCast(i % 8);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.table));
                p.out[0] = try p.rig.output(n * 2);
            },
            .norm, .add_norm => {
                const r = try rows_in.Rows.init(p.a, .rms, .{ .rows = s.rows, .n = w, .k = w });
                p.r = r;
                r.fill(if (p.which == .add_norm and fill == .nonfinite) .random else fill, seed);
                if (p.which == .norm) {
                    p.in[0] = try p.rig.input(std.mem.sliceAsBytes(r.x));
                    p.in[1] = try p.rig.input(std.mem.sliceAsBytes(r.w));
                    p.out[0] = try p.rig.output(n * 2);
                } else {
                    // the added rows: zero for the sums gate (the residual's stress rows stay whole), else small
                    p.y = try p.a.alloc(u16, n);
                    if (fill == .stress) @memset(p.y, 0) else random16(rng, p.y, -0.5, 0.5);
                    if (fill == .nonfinite) plant(p.y, w, s.rows);
                    p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.y));
                    p.in[1] = try p.rig.input(std.mem.sliceAsBytes(r.w));
                    p.out[0] = try p.rig.output(n * 2);
                    p.out[1] = try p.rig.output(n * 2);
                }
            },
            .add => {
                p.x = try p.a.alloc(u16, n);
                p.y = try p.a.alloc(u16, n);
                random16(rng, p.x, -4, 4);
                random16(rng, p.y, -4, 4);
                if (fill == .nonfinite) {
                    plant(p.y, w, s.rows);
                    if (s.rows > 3) { // +Inf added to -Inf: a NaN out of two infinities
                        p.x[3 * w] = c.toBf16(std.math.inf(f32));
                        p.y[3 * w] = c.toBf16(-std.math.inf(f32));
                    }
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.y));
                p.out[0] = try p.rig.output(n * 2);
            },
            .silu => {
                p.x = try p.a.alloc(u16, 2 * n);
                random16(rng, p.x, -6, 6);
                if (fill == .nonfinite) plant(p.x, 2 * w, s.rows);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.out[0] = try p.rig.output(n * 2);
            },
            .prep => {
                p.geo = if (s.n == G3.heads + G3.kv_heads) G3 else G33;
                const stride = (p.geo.heads + 2 * p.geo.kv_heads) * DH;
                if (s.k != stride) return false;
                p.pos = if (s.rows <= 64) 8192 else 0; // a window past a long prompt, or a prompt's first chunk
                p.x = try p.a.alloc(u16, s.rows * stride);
                random16(rng, p.x, -3, 3);
                if (fill == .nonfinite) plant(p.x, stride, s.rows);
                p.inv = try p.a.alloc(f32, DH / 2);
                for (p.inv, 0..) |*f, i| f.* = @floatCast(1.0 / std.math.pow(f64, 1.2e8, @as(f64, @floatFromInt(2 * i)) / DH));
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.inv));
                p.out[0] = try p.rig.output(s.rows * p.geo.heads * DH * 2);
                p.out[1] = try p.rig.output((p.pos + s.rows) * p.geo.kv_heads * DH * 2);
                p.out[2] = try p.rig.output((p.pos + s.rows) * p.geo.kv_heads * DH * 2);
            },
        }
        return true;
    }

    pub fn count(p: *Glue) usize {
        const n = p.shape.rows * p.width();
        return switch (p.which) {
            .add_norm => 2 * n,
            .prep => p.shape.rows * p.geo.heads * DH + 2 * (p.pos + p.shape.rows) * p.geo.kv_heads * DH,
            else => n,
        };
    }

    pub fn run(p: *Glue) !void {
        const s = p.shape;
        const w: u32 = @intCast(p.width());
        const rows: u32 = @intCast(s.rows);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        const grid = mtl.Size.of(p.width(), s.rows, 1);
        const threads = mtl.Size.of(256, 1, 1);
        switch (p.which) {
            .embed => {
                e.setBuffer(p.in[0].buf, p.in[0].off, 0);
                e.setBuffer(p.in[1].buf, p.in[1].off, 1);
                e.setValue(Args{ .rows = rows, .width = w }, 2);
                e.setBuffer(p.out[0].buf, p.out[0].off, 3);
                e.dispatchThreads(grid, threads);
            },
            .norm => {
                e.setBuffer(p.in[0].buf, p.in[0].off, 0);
                e.setBuffer(p.in[1].buf, p.in[1].off, 1);
                e.setValue(Args{ .rows = rows, .width = w, .eps = p.r.?.eps }, 2);
                e.setBuffer(p.out[0].buf, p.out[0].off, 3);
                e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), threads);
            },
            .add => {
                @memcpy(p.rig.bytes(p.out[0], p.x.len * 2), std.mem.sliceAsBytes(p.x)); // the residual before the add
                e.setBuffer(p.out[0].buf, p.out[0].off, 0);
                e.setBuffer(p.in[0].buf, p.in[0].off, 1);
                e.setValue(Args{ .rows = rows, .width = w }, 2);
                e.dispatchThreads(grid, threads);
            },
            .add_norm => {
                const r = p.r.?;
                @memcpy(p.rig.bytes(p.out[0], r.x.len * 2), std.mem.sliceAsBytes(r.x));
                e.setBuffer(p.out[0].buf, p.out[0].off, 0);
                e.setBuffer(p.in[0].buf, p.in[0].off, 1);
                e.setBuffer(p.in[1].buf, p.in[1].off, 2);
                e.setValue(Args{ .rows = rows, .width = w, .eps = r.eps }, 3);
                e.setBuffer(p.out[1].buf, p.out[1].off, 4);
                e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), threads);
            },
            .silu => {
                e.setBuffer(p.in[0].buf, p.in[0].off, 0);
                e.setValue(Args{ .rows = rows, .width = w }, 1);
                e.setBuffer(p.out[0].buf, p.out[0].off, 2);
                e.dispatchThreads(grid, threads);
            },
            .prep => {
                // the cache's slots before the window: unwritten by the kernel, zero in the reference
                for (p.out[1..]) |o| @memset(p.rig.bytes(o, p.pos * p.geo.kv_heads * DH * 2), 0);
                e.setBuffer(p.in[0].buf, p.in[0].off, 0);
                e.setBuffer(p.in[1].buf, p.in[1].off, 1);
                e.setValue(PArgs{ .rows = rows, .heads = p.geo.heads, .kv_heads = p.geo.kv_heads, .pos = p.pos, .stride = w }, 2);
                for (p.out, 3..) |o, slot| e.setBuffer(o.buf, o.off, slot);
                e.dispatchGroups(mtl.Size.of(p.geo.heads + p.geo.kv_heads, s.rows, 1), mtl.Size.of(32, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }

    fn xv(v: []const u16, i: usize) f64 {
        return c.fromBf16(v[i]);
    }

    /// The norm of a row held in `x` (bf16 values), HF's rounding points, into want and mag.
    fn normRow(p: *Glue, x: []const f64, ws: []const u16, want: []f64, mag: []f64) void {
        var ss: f64 = 0;
        for (x) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + @as(f64, p.r.?.eps));
        for (x, 0..) |v, i| {
            want[i] = bf(c.fromBf16(ws[i]) * bf(v * inv));
            mag[i] = @abs(want[i]) / 2;
        }
    }

    pub fn reference(p: *Glue, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const w = p.width();
        switch (p.which) {
            .embed => for (0..s.rows) |r| for (0..w) |i| {
                want[r * w + i] = xv(p.table, p.ids[r] * w + i);
                mag[r * w + i] = @abs(want[r * w + i]);
            },
            .norm => {
                const x = try p.a.alloc(f64, w);
                defer p.a.free(x);
                for (0..s.rows) |r| {
                    for (x, 0..) |*v, i| v.* = xv(p.r.?.x, r * w + i);
                    p.normRow(x, p.r.?.w, want[r * w ..][0..w], mag[r * w ..][0..w]);
                }
            },
            .add => for (0..s.rows * w) |i| {
                want[i] = bf(xv(p.x, i) + xv(p.y, i));
                mag[i] = @abs(want[i]);
            },
            .add_norm => {
                const x = try p.a.alloc(f64, w);
                defer p.a.free(x);
                const n = s.rows * w;
                for (0..s.rows) |r| {
                    for (x, 0..) |*v, i| {
                        v.* = bf(xv(p.r.?.x, r * w + i) + xv(p.y, r * w + i));
                        want[r * w + i] = v.*;
                        mag[r * w + i] = @abs(v.*);
                    }
                    p.normRow(x, p.r.?.w, want[n + r * w ..][0..w], mag[n + r * w ..][0..w]);
                }
            },
            .silu => for (0..s.rows) |r| for (0..w) |i| {
                const gate = xv(p.x, r * 2 * w + i);
                const up = xv(p.x, r * 2 * w + w + i);
                const silu = bf(gate / (1.0 + @exp(-gate)));
                want[r * w + i] = bf(silu * up);
                mag[r * w + i] = @abs(want[r * w + i]);
            },
            .prep => try p.prepReference(want, mag),
        }
    }

    /// HF's rotation of q and k (fp32 angle, bf16 cos and sin), v copied: QN, then caches of pos + rows slots.
    fn prepReference(p: *Glue, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const geo = p.geo;
        const stride = (geo.heads + 2 * geo.kv_heads) * DH;
        const kvw = geo.kv_heads * DH;
        const qn = s.rows * geo.heads * DH;
        const kc = (p.pos + s.rows) * kvw;
        @memset(want[qn .. qn + 2 * kc], 0);
        @memset(mag[qn .. qn + 2 * kc], 0);
        for (0..s.rows) |r| {
            const position: f32 = @floatFromInt(p.pos + r);
            for (0..geo.heads + geo.kv_heads) |unit| {
                const q = unit < geo.heads;
                const head = if (q) unit else unit - geo.heads;
                const src = r * stride + (if (q) 0 else geo.heads * DH) + head * DH;
                const dst = if (q) (r * geo.heads + head) * DH else qn + (p.pos + r) * kvw + head * DH;
                for (0..DH) |d| {
                    const i = d % (DH / 2);
                    const angle: f32 = position * p.inv[i]; // the kernel's fp32 angle
                    const cs = bf(@cos(@as(f64, angle)));
                    const sn = bf(@sin(@as(f64, angle)));
                    const x = xv(p.x, src + d);
                    const other = if (d < DH / 2) -xv(p.x, src + d + DH / 2) else xv(p.x, src + d - DH / 2);
                    want[dst + d] = bf(bf(x * cs) + bf(other * sn));
                    mag[dst + d] = @abs(x * cs) + @abs(other * sn);
                }
            }
            for (0..geo.kv_heads) |head| for (0..DH) |d| {
                const at = qn + kc + (p.pos + r) * kvw + head * DH + d;
                want[at] = xv(p.x, r * stride + (geo.heads + geo.kv_heads) * DH + head * DH + d);
                mag[at] = 0;
            };
        }
    }

    /// The rotation's allowance: a bf16 ulp of each term (its rounding, its cos or sin), then the output's rounding.
    pub fn bounds(p: *Glue, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (p.which != .prep) return;
        for (allow, want, mag) |*l, w, m| l.* = 0x1p-7 * m + 0x1p-8 * @abs(w) + 0x1p-133;
    }

    pub fn simulate(p: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const w = p.width();
        const t = try p.a.alloc(f64, w);
        defer p.a.free(t);
        const r = p.r.?;
        const base: usize = if (p.which == .add_norm) s.rows * w else 0;
        for (0..s.rows) |row| {
            for (t, 0..) |*v, i| {
                const x = if (p.which == .add_norm) bf(xv(r.x, row * w + i) + xv(p.y, row * w + i)) else xv(r.x, row * w + i);
                if (p.which == .add_norm) out[row * w + i] = x;
                v.* = x * x;
            }
            const ss = rows_in.sumIn(acc, order, t);
            const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(w)) + @as(f64, r.eps));
            for (0..w) |i| {
                const x = if (p.which == .add_norm) out[row * w + i] else xv(r.x, row * w + i);
                out[base + row * w + i] = bf(c.fromBf16(r.w[i]) * bf(x * inv));
            }
        }
    }
};

/// ll_argmax, a draw a logits row: the stated rule checks the ids (gates 1 and 3), the bytes repeat (gate 2).
pub const Argmax = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: controls.Mutation) !Argmax {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mutation = mutation, .lib = lib, .pipe = try pipelineOf(lib, device, "ll_argmax") };
    }

    pub fn deinit(p: *Argmax) void {
        p.a.free(p.x);
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Argmax);

    pub fn kernel(p: *Argmax) g.Kernel {
        return controls.mark(.{ .spec = .{ .name = "llama glue ll_argmax", .accum = .none, .depth = none, .out = .raw, .nan = .{ .stated = argmax_rule }, .tiles = .{ .{}, .{}, .{ .size = 1024, .step = 1, .min = 1 } }, .served = &.{ .{ .rows = 1, .n = 1, .k = G33.vocab }, .{ .rows = 1, .n = 1, .k = G3.vocab } }, .sums = no_sums }, .ctx = p, .vt = &vt }, p.mutation);
    }

    /// rows logits rows of k: NaNs and infinities as planted, two NaNs in row 3 and a tie in row 4.
    pub fn prepare(p: *Argmax, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0) return false;
        p.a.free(p.x);
        p.x = &.{};
        p.rig.reset();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const w = s.k;
        p.x = try p.a.alloc(u16, s.rows * w);
        for (p.x) |*v| v.* = c.toBf16(rng.float(f32) * 16 - 8);
        if (fill == .nonfinite) {
            Glue.plant(p.x, w, s.rows);
            if (s.rows > 3 and w > 7) { // two NaNs: the lower id wins
                p.x[3 * w + 7] = c.toBf16(std.math.nan(f32));
                p.x[3 * w + 5] = c.toBf16(std.math.nan(f32));
            }
        }
        if (s.rows > 4 and w > 2) { // a tie: the lower id wins
            p.x[4 * w + 2] = c.toBf16(9);
            p.x[4 * w + w - 1] = c.toBf16(9);
        }
        p.in = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.out = try p.rig.output(s.rows * 4);
        return true;
    }

    pub fn count(p: *Argmax) usize {
        return p.shape.rows * 4;
    }

    pub fn run(p: *Argmax) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (0..s.rows) |r| { // a draw a logits row, as the forward draws its last row
            e.setBuffer(p.in.buf, p.in.off + r * s.k * 2, 0);
            e.setBuffer(p.out.buf, p.out.off + r * 4, 1);
            e.setValue(@as(u32, @intCast(s.k)), 2);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Argmax, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The argmax rule on the host against the drawn ids (gates 1 and 3).
    pub fn check(p: *Argmax, _: g.Gate) !?[]const u8 {
        const w = p.shape.k;
        const ids = std.mem.bytesAsSlice(u32, p.rig.bytes(p.out, p.shape.rows * 4));
        for (0..p.shape.rows) |r| {
            var best: f64 = -std.math.inf(f64);
            var at: usize = 0;
            var nan: ?usize = null;
            for (0..w) |i| {
                const v = c.fromBf16(p.x[r * w + i]);
                if (std.math.isNan(v)) {
                    if (nan == null) nan = i;
                } else if (v > best) {
                    best = v;
                    at = i;
                }
            }
            const want = nan orelse at;
            if (ids[r] != want) return try std.fmt.bufPrint(&p.msg, "row {d}: drew {d} where the rule gives {d}", .{ r, ids[r], want });
        }
        return null;
    }
};
