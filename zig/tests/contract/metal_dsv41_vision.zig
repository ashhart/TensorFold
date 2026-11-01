//! DeepSeek V4.1's vision kernels under the contract on Metal: epilogues, RoPE, attention, SwiGLU, layout.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_vision_ref.zig");

const c = contract.spec;
const g = contract.gates;
const vis = tf.dsv41_metal.vision;
const base = "dsv41 vision ";
const u: f64 = 0x1p-8; // half a bf16 ulp, relative

pub const Fam = enum { epi_bias, epi_gelu, epi_res, epi_res0, qkv, swiglu, unshuffle, merge, attn };

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(f: Fam, name: []const u8) c.Spec {
    const t32: c.Tile = .{ .size = 32 };
    return switch (f) {
        .epi_bias, .epi_gelu, .epi_res, .epi_res0 => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ .{}, t32, .{} }, .served = &.{
            .{ .rows = 300, .n = 1024, .k = 1 }, .{ .rows = 64, .n = 5120, .k = 1 }, .{ .rows = 33, .n = 3072, .k = 1 }, .{ .rows = 5, .n = 2816, .k = 1 }, .{ .rows = 1, .n = 1024, .k = 1 },
        } },
        .qkv => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 64, .step = 2, .min = 2 } }, .served = &.{
            .{ .rows = 96, .n = 16, .k = 64 }, .{ .rows = 7, .n = 2, .k = 64 }, .{ .rows = 1, .n = 16, .k = 64 }, .{ .rows = 33, .n = 3, .k = 64 },
        } },
        .swiglu => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ .{}, t32, .{} }, .served = &.{
            .{ .rows = 128, .n = 2816, .k = 1 }, .{ .rows = 5, .n = 192, .k = 1 }, .{ .rows = 1, .n = 2816, .k = 1 }, .{ .rows = 40, .n = 1000, .k = 1 },
        } },
        .unshuffle => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ .{ .size = 3 }, .{ .size = 3 }, .{ .size = 32 } }, .served = &.{
            .{ .rows = 36, .n = 43, .k = 1024 }, .{ .rows = 5, .n = 7, .k = 128 }, .{ .rows = 1, .n = 1, .k = 64 }, .{ .rows = 9, .n = 4, .k = 100 },
        } },
        .merge => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ .{}, t32, .{} }, .served = &.{
            .{ .rows = 200, .n = 5120, .k = 40 }, .{ .rows = 16, .n = 256, .k = 9 }, .{ .rows = 1, .n = 5120, .k = 1 }, .{ .rows = 70, .n = 100, .k = 20 },
        } },
        .attn => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ .{ .size = 32 }, .{}, .{} }, .served = &.{
            .{ .rows = 96, .n = 16, .k = 64 }, .{ .rows = 37, .n = 4, .k = 64 }, .{ .rows = 1, .n = 16, .k = 64 }, .{ .rows = 130, .n = 2, .k = 64 },
        } },
    };
}

fn pow2(e: i32) f32 {
    return std.math.pow(f32, 2, @floatFromInt(e));
}

fn spread(rng: std.Random, lo: i32, hi: i32) f32 {
    return (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, lo, hi));
}

fn at(r: rig_.Ref, extra: usize) vis.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

/// The five planted values of the rows family (a NaN, both infinities, two at one row's ends) into fp32 rows.
fn plant32(x: []f32, rows: usize, width: usize) usize {
    const place = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, width - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    var n: usize = 0;
    for (place, put) |p, v| if (p[0] < rows) {
        x[p[0] * width + p[1]] = v;
        n += 1;
    };
    return n;
}

fn bf16s(a: std.mem.Allocator, n: usize, rng: std.Random, lo: i32, hi: i32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*o| o.* = c.toBf16(spread(rng, lo, hi));
    return v;
}

fn f32s(a: std.mem.Allocator, n: usize, rng: std.Random, lo: i32, hi: i32) ![]f32 {
    const v = try a.alloc(f32, n);
    for (v) |*o| o.* = spread(rng, lo, hi);
    return v;
}

const vt = g.hooks(Proj);

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    fam: Fam,
    name: []const u8,
    pipes: vis.Pipelines,
    shape: c.Shape = undefined,
    acc: []f32 = &.{},
    x: []u16 = &.{},
    bias: []u16 = &.{},
    res: []u16 = &.{},
    cs: [][2]f32 = &.{},
    src: []i32 = &.{},
    in: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, name: []const u8) !Proj {
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .fam = f, .name = name, .pipes = try vis.Pipelines.load(device) };
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.fam, p.name), .ctx = p, .vt = &vt };
    }

    fn isEpi(p: *const Proj) bool {
        return p.fam == .epi_bias or p.fam == .epi_gelu or p.fam == .epi_res or p.fam == .epi_res0;
    }

    /// qkv: the three parts of every head; the others: a row's outputs.
    pub fn count(p: *Proj) usize {
        const s = p.shape;
        return switch (p.fam) {
            .qkv => s.rows * 3 * s.n * s.k,
            .swiglu => s.rows * s.n,
            .unshuffle => ((s.rows + 2) / 3) * ((s.n + 2) / 3) * s.k * 9,
            .merge => s.rows * 4 * s.n,
            .attn => s.rows * s.n * 64,
            else => s.rows * s.n,
        };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (p.fam == .qkv and shape.k % 2 != 0) return false;
        if (p.fam == .attn and (shape.k != 64 or shape.rows > 500)) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const al = p.arena.allocator();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const s = shape;
        switch (p.fam) {
            .epi_bias, .epi_gelu, .epi_res, .epi_res0 => {
                p.acc = try f32s(al, s.rows * s.n, rng, -5, 3);
                p.bias = try bf16s(al, s.n, rng, -4, 1);
                p.res = try bf16s(al, s.rows * s.n, rng, -4, 3);
                if (nf) _ = plant32(p.acc, s.rows, s.n);
                if (nf and p.fam == .epi_res) p.res[s.n + 1] = c.toBf16(-std.math.inf(f32));
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.acc));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
                p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.res));
            },
            .qkv => {
                const w = 3 * s.n * s.k;
                p.acc = try f32s(al, s.rows * w, rng, -4, 2);
                p.bias = try bf16s(al, w, rng, -4, 1);
                p.cs = try al.alloc([2]f32, s.rows * s.k / 2);
                for (p.cs) |*o| {
                    const angle = (rng.float(f64) * 2 - 1) * std.math.pi;
                    o.* = .{ @floatCast(@cos(angle)), @floatCast(@sin(angle)) };
                }
                if (nf) _ = plant32(p.acc, s.rows, w);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.acc));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
                p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.cs));
            },
            .swiglu => {
                p.acc = try f32s(al, s.rows * 2 * s.n, rng, -5, 3);
                if (nf) _ = plant32(p.acc, s.rows, 2 * s.n);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.acc));
            },
            .unshuffle => {
                p.x = try bf16s(al, s.rows * s.n * s.k, rng, -3, 3);
                if (nf) p.x[0] = c.toBf16(std.math.nan(f32));
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            },
            .merge => {
                p.x = try bf16s(al, s.k * s.n, rng, -3, 3);
                p.bias = try bf16s(al, 3 * s.n, rng, -3, 3);
                p.src = try al.alloc(i32, s.rows);
                for (p.src, 0..) |*o, i| o.* = if (i % 5 == 4) -1 - @as(i32, @intCast(i % 3)) else @intCast(rng.uintLessThan(usize, s.k));
                if (nf) {
                    // a NaN feature row and an infinite delimiter, each read by a row
                    p.x[0] = c.toBf16(std.math.nan(f32));
                    p.bias[s.n + 1] = c.toBf16(std.math.inf(f32));
                    p.src[0] = 0;
                    p.src[4] = -2;
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.bias));
                p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.src));
            },
            .attn => {
                const D = s.n * 64;
                p.x = try bf16s(al, s.rows * 3 * D, rng, -2, 1);
                if (nf) {
                    // NaN and infinities in four queries of the first head only: those rows' head goes non-finite
                    const qs = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) };
                    for (qs, 0..) |v, r| if (r < s.rows) {
                        p.x[r * 3 * D + 3 + r] = c.toBf16(v);
                    };
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            },
        }
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Rows r0 to r0 + n of the prepared call as one launch (the attention and unshuffle take whole calls).
    pub fn encode(p: *Proj, e: anytype, r0: usize, n: usize) !void {
        const s = p.shape;
        const pl = &p.pipes;
        const n32: u32 = @intCast(n);
        switch (p.fam) {
            .epi_bias, .epi_gelu, .epi_res, .epi_res0 => {
                const w = s.n;
                const bias: ?vis.Ref = if (p.fam == .epi_res0) null else at(p.in[1], 0);
                const res: ?vis.Ref = if (p.fam == .epi_res or p.fam == .epi_res0) at(p.in[2], r0 * w * 2) else null;
                vis.epilogue(pl, e, at(p.in[0], r0 * w * 4), bias, res, at(p.out, r0 * w * 2), n32, @intCast(w), p.fam == .epi_gelu);
            },
            .qkv => vis.qkv(pl, e, at(p.in[0], r0 * 3 * s.n * s.k * 4), at(p.in[1], 0), at(p.in[2], r0 * s.k / 2 * 8), at(p.out, r0 * 3 * s.n * s.k * 2), n32, @intCast(s.n), @intCast(s.k)),
            .swiglu => vis.swiglu(pl, e, at(p.in[0], r0 * 2 * s.n * 4), at(p.out, r0 * s.n * 2), n32, @intCast(s.n)),
            .unshuffle => vis.unshuffle(pl, e, at(p.in[0], 0), at(p.out, 0), @intCast(s.rows), @intCast(s.n), @intCast(s.k), 3),
            .merge => vis.merge(pl, e, at(p.in[0], 0), at(p.in[1], 0), at(p.in[2], r0 * 4), at(p.out, r0 * 4 * s.n * 2), n32, @intCast(s.n)),
            .attn => vis.attention(pl, e, at(p.in[0], 0), at(p.out, 0), @intCast(s.rows), @intCast(s.n)),
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const s = p.shape;
        @memset(mag, 0);
        switch (p.fam) {
            .epi_bias, .epi_gelu, .epi_res, .epi_res0 => for (want, mag, 0..) |*w, *m, i| {
                const z: f64 = @as(f64, p.acc[i]) + (if (p.fam == .epi_res0) 0 else c.fromBf16(p.bias[i % s.n]));
                const y = if (p.fam == .epi_gelu) ref.gelu(z) else z;
                const r: f64 = if (p.fam == .epi_res or p.fam == .epi_res0) c.fromBf16(p.res[i]) else 0;
                w.* = y + r;
                m.* = @abs(z) + @abs(r);
            },
            .qkv => for (0..s.rows) |r| for (0..3 * s.n) |ph| for (0..s.k / 2) |j| {
                const part = ph / s.n;
                const o = (r * 3 * s.n + ph) * s.k + j;
                const col = ph * s.k + j;
                const x1: f64 = @as(f64, p.acc[r * 3 * s.n * s.k + col]) + c.fromBf16(p.bias[col]);
                const x2: f64 = @as(f64, p.acc[r * 3 * s.n * s.k + col + s.k / 2]) + c.fromBf16(p.bias[col + s.k / 2]);
                if (part == 2) {
                    want[o] = x1;
                    want[o + s.k / 2] = x2;
                    mag[o] = @abs(x1);
                    mag[o + s.k / 2] = @abs(x2);
                    continue;
                }
                const cs = p.cs[r * s.k / 2 + j];
                const co: f64 = cs[0];
                const si: f64 = cs[1];
                want[o] = x1 * co - x2 * si;
                want[o + s.k / 2] = x2 * co + x1 * si;
                mag[o] = @abs(x1 * co) + @abs(x2 * si);
                mag[o + s.k / 2] = @abs(x2 * co) + @abs(x1 * si);
            },
            .swiglu => for (0..s.rows) |r| for (0..s.n) |j| {
                const gt: f64 = p.acc[r * 2 * s.n + j];
                const up: f64 = p.acc[r * 2 * s.n + s.n + j];
                want[r * s.n + j] = ref.silu(gt) * up;
                mag[r * s.n + j] = @abs(gt); // the gate's own size sets how far its rounding moves the output
            },
            .unshuffle => {
                const lw = (s.n + 2) / 3;
                for (0..((s.rows + 2) / 3) * lw) |l| for (0..s.k) |ch| for (0..9) |kk| {
                    const py = (l / lw) * 3 + kk / 3;
                    const px = (l % lw) * 3 + kk % 3;
                    want[(l * s.k + ch) * 9 + kk] = if (py < s.rows and px < s.n) c.fromBf16(p.x[(py * s.n + px) * s.k + ch]) else 0;
                };
            },
            .merge => for (0..s.rows) |r| for (0..4) |st| for (0..s.n) |d| {
                const sr = p.src[r];
                want[(r * 4 + st) * s.n + d] = if (sr >= 0) c.fromBf16(p.x[@as(usize, @intCast(sr)) * s.n + d]) else c.fromBf16(p.bias[@as(usize, @intCast(-1 - sr)) * s.n + d]);
            },
            .attn => {
                const D = s.n * 64;
                const q = try p.a.alloc(f64, 64);
                defer p.a.free(q);
                const kk = try p.a.alloc(f64, s.rows * 3 * D);
                defer p.a.free(kk);
                for (kk, p.x) |*o, v| o.* = c.fromBf16(v);
                for (0..s.rows) |r| for (0..s.n) |h| {
                    for (q, 0..) |*o, d| o.* = kk[r * 3 * D + h * 64 + d];
                    ref.attend(want[(r * s.n + h) * 64 ..][0..64], mag[(r * s.n + h) * 64 ..][0..64], q, kk[D + h * 64 ..], kk[2 * D + h * 64 ..], s.rows, 64, 3 * D);
                };
            },
        }
    }

    /// The allowed |got - want|: the intermediate bf16 roundings of each formula, fp32's own, then the output's.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        const e32: f64 = 0x1p-24;
        for (allow, want, mag) |*l, w, m| {
            const work: f64 = switch (p.fam) {
                .epi_bias => e32 * m,
                .epi_gelu => 1.2 * u * m + 0x1p-20 * m,
                .epi_res, .epi_res0 => u * m + e32 * m,
                .qkv => (u + 4 * e32) * m,
                .swiglu => u * (3 + m) * @abs(w) + 0x1p-20 * @abs(w),
                .attn => 0x1p-14 * m,
                else => 0,
            };
            l.* = work + (if (p.fam == .unshuffle or p.fam == .merge) 0 else u * (@abs(w) + work)) + 0x1p-133;
        }
    }
};

/// Every registered variant and the launch-split probes after the gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, name: []const u8 };
    const list = [_]E{
        .{ .f = .epi_bias, .name = base ++ "epilogue bias tf_dsv41_vis_epi" },
        .{ .f = .epi_gelu, .name = base ++ "epilogue bias gelu tf_dsv41_vis_epi" },
        .{ .f = .epi_res, .name = base ++ "epilogue bias residual tf_dsv41_vis_epi" },
        .{ .f = .epi_res0, .name = base ++ "epilogue residual tf_dsv41_vis_epi" },
        .{ .f = .qkv, .name = base ++ "qkv bias rope tf_dsv41_vis_qkv" },
        .{ .f = .swiglu, .name = base ++ "swiglu tf_dsv41_vis_swiglu" },
        .{ .f = .unshuffle, .name = base ++ "unshuffle tf_dsv41_vis_unshuffle" },
        .{ .f = .merge, .name = base ++ "merge tf_dsv41_vis_merge" },
        .{ .f = .attn, .name = base ++ "attention tf_dsv41_vis_attn" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.f, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    fn splits(f: Fam) bool {
        return f != .attn and f != .unshuffle;
    }

    /// Launches of 1, 7 and rows/3 rows give the whole launch's bytes.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (!splits(p.fam)) continue;
            kernels += 1;
            const shape = spec(p.fam, p.name).served[1];
            if (!try p.prepare(.{ .rows = shape.rows + 20, .n = shape.n, .k = shape.k }, .random, seed)) return error.Refused;
            try p.run();
            const len = p.count() * 2;
            const whole = try s.a.dupe(u8, p.rig.bytes(p.out, len));
            defer s.a.free(whole);
            const rows = p.shape.rows;
            for ([_]usize{ 1, 7, @max(1, rows / 3) }) |step| {
                @memset(p.rig.bytes(p.out, len), 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < rows) : (r0 += step) {
                    try p.encode(pass.e, r0, @min(step, rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, p.rig.bytes(p.out, len))) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {s}: {d} rows a launch differs from one launch\n", .{ p.name, step });
                }
            }
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and rows/3 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};
