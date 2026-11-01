//! DeepSeek V4.1's phase-1 glue on Metal: rmsnorm, rope, scatter, embed, hcmean against float64; controls 1-4.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const gl = tf.dsv41_metal.glue_tmp;
const base = "dsv41 glue_tmp ";
const eps: f32 = 1e-20; // rms_norm_eps
const vocab: u32 = 300; // the embed probe's table rows
const table_positions: u32 = 8192; // the rope probe's table rows, 32 cos then 32 sin each

pub const Fam = enum { rms, rope, scatter, embed, mean };

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Per thread a chain over every 256th square, then the 5-step butterfly and the eight adds over simdgroups.
fn rmsDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 12;
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const scatter_rule = "a word copy holds no non-finite values; every word matches the source's bytes";

pub fn spec(f: Fam, name: []const u8) c.Spec {
    const t16: c.Tile = .{ .size = 16 };
    return switch (f) {
        .rms => .{ .name = name, .accum = .f32, .depth = rmsDepth, .ops = 8, .out = .bf16, .tiles = .{ t16, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 5120 }, .{ .rows = 1, .n = 5120, .k = 5120 }, .{ .rows = 4, .n = 1280, .k = 1280 },
            .{ .rows = 8, .n = 512, .k = 512 },    .{ .rows = 16, .n = 128, .k = 128 },  .{ .rows = 3, .n = 2048, .k = 2048 },
        }, .sums = .{ .bf16 = 0, .why = rms_why } },
        .rope => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ t16, .{ .size = 8 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 64, .k = 512 }, .{ .rows = 1, .n = 1, .k = 512 }, .{ .rows = 5, .n = 1, .k = 512 },
            .{ .rows = 8, .n = 32, .k = 128 },  .{ .rows = 6, .n = 1, .k = 128 },
        } },
        .scatter => .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = scatter_rule }, .tiles = .{ t16, .{}, .{ .size = 256, .step = 64, .min = 64 } }, .served = &.{
            .{ .rows = 16, .n = 1, .k = 1280 }, .{ .rows = 1, .n = 1, .k = 1280 }, .{ .rows = 4, .n = 1, .k = 512 }, .{ .rows = 3, .n = 1, .k = 2048 },
        } },
        .embed => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ t16, .{ .size = 64, .step = 2, .min = 2 }, .{} }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 4 }, .{ .rows = 1, .n = 5120, .k = 4 }, .{ .rows = 5, .n = 5120, .k = 4 }, .{ .rows = 9, .n = 128, .k = 4 },
        } },
        .mean => .{ .name = name, .accum = .none, .depth = noSums, .out = .bf16, .tiles = .{ t16, .{}, .{} }, .served = &.{
            .{ .rows = 16, .n = 5120, .k = 4 }, .{ .rows = 1, .n = 5120, .k = 4 }, .{ .rows = 6, .n = 512, .k = 4 },
        } },
    };
}

fn pow2(e: i32) f32 {
    return std.math.pow(f32, 2, @floatFromInt(e));
}

fn spread(rng: std.Random, lo: i32, hi: i32) f32 {
    return (rng.float(f32) * 2 - 1) * pow2(rng.intRangeAtMost(i32, lo, hi));
}

fn bf16s(a: std.mem.Allocator, n: usize, rng: std.Random, lo: i32, hi: i32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*o| o.* = c.toBf16(spread(rng, lo, hi));
    return v;
}

fn at(r: rig_.Ref, extra: usize) gl.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

var tab_cache: ?[]f32 = null;

/// The probe's own cos/sin table: any consistent table does, since the reference reads the same floats.
fn ropeTable() ![]f32 {
    if (tab_cache) |t| return t;
    const t = try std.heap.page_allocator.alloc(f32, table_positions * 64);
    for (0..table_positions) |pos| {
        const p: f64 = @floatFromInt(pos);
        for (0..32) |i| {
            const freq = std.math.pow(f64, 10000, -@as(f64, @floatFromInt(2 * i)) / 64.0);
            const ang = p * freq;
            t[pos * 64 + i] = @floatCast(@cos(ang));
            t[pos * 64 + 32 + i] = @floatCast(@sin(ang));
        }
    }
    tab_cache = t;
    return t;
}

const vt_scatter = g.hooks(Proj);
const vt_plain = blk: {
    var v = g.hooks(Proj);
    v.check = null; // the other families are checked against the float64 bound, not a stated rule
    break :blk v;
};

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    fam: Fam,
    ctl: u8,
    name: []const u8,
    pipes: gl.Pipelines,
    shape: c.Shape = undefined,
    rows_f: ?rows_.Rows = null,
    x: []u16 = &.{},
    w32: []f32 = &.{},
    tab: []f32 = &.{},
    pos: []i32 = &.{},
    words: []u32 = &.{},
    dest: []i32 = &.{},
    table: []u16 = &.{},
    ids: []u32 = &.{},
    plants: usize = 0,
    in: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    pre: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .fam = f, .ctl = ctl, .name = name, .pipes = undefined };
        if (ctl == 0) {
            p.pipes = try gl.Pipelines.load(device, sources.dsv41.glue_tmp);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.glue_tmp });
            defer a.free(text);
            p.pipes = try gl.Pipelines.load(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.ctl) {
            0 => null,
            1 => .nonfinite,
            2 => .shapes,
            3 => if (p.fam == .rms) .sums else .shapes, // the mean's fp16 partial shows against the tight sum bound
            else => if (p.fam == .rope) .shapes else .repeat, // rope's extra write lands inside its last row
        };
        return .{ .spec = spec(p.fam, p.name), .ctx = p, .vt = if (p.fam == .scatter) &vt_scatter else &vt_plain, .control = control };
    }

    pub fn count(p: *Proj) usize {
        const s = p.shape;
        return switch (p.fam) {
            .rms => s.rows * s.k,
            .rope => s.rows * s.n * s.k,
            .scatter => s.rows * s.k * 4, // raw bytes
            .embed => s.rows * s.k * s.n,
            .mean => s.rows * s.n,
        };
    }

    fn outLen(p: *Proj) usize {
        return if (p.fam == .scatter) p.count() else p.count() * 2;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        p.plants = 0;
        p.rows_f = null;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        switch (p.fam) {
            .rms => {
                var r = try rows_.Rows.init(a, .rms, .{ .rows = shape.rows, .n = shape.k, .k = shape.k });
                r.eps = eps;
                r.fill(fill, seed);
                p.rows_f = r;
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(r.x));
                p.w32 = try a.alloc(f32, r.w.len);
                for (p.w32, r.w) |*o, w| o.* = c.fromBf16(w);
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.w32));
            },
            .rope => {
                if (shape.k < 64 or shape.k % 2 != 0) return false;
                p.tab = try ropeTable();
                p.x = try bf16s(a, shape.rows * shape.n * shape.k, rng, -3, 3);
                p.pos = try a.alloc(i32, shape.rows);
                for (p.pos) |*o| o.* = @intCast(rng.uintLessThan(u32, table_positions));
                if (nf) p.plants = ref.plant(p.x, shape.rows, shape.n * shape.k, shape.k - 64, 64);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.pos));
                p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.tab));
            },
            .scatter => {
                if (shape.k == 0) return false;
                p.words = try a.alloc(u32, shape.rows * shape.k);
                for (p.words) |*o| o.* = rng.int(u32);
                p.dest = try a.alloc(i32, shape.rows);
                for (p.dest, 0..) |*o, i| o.* = @intCast(i); // identity: every row a destination, the copy the whole output
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.words));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.dest));
            },
            .embed => {
                if (shape.n == 0) return false;
                p.table = try bf16s(a, vocab * shape.n, rng, -2, 2);
                p.ids = try a.alloc(u32, shape.rows);
                for (p.ids) |*id| id.* = rng.uintLessThan(u32, vocab);
                if (nf) {
                    for (0..@min(4, shape.rows)) |i| p.ids[i] = @intCast(i);
                    p.plants = ref.plant(p.table, 4, shape.n, 0, shape.n);
                }
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
                p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.table));
                p.pre = try p.rig.scratch(shape.rows * 16); // the one-hot pre mix, written but not the output here
            },
            .mean => {
                if (shape.n == 0) return false;
                p.x = try a.alloc(u16, shape.rows * 4 * shape.n);
                for (0..shape.rows) |t| for (0..shape.n) |d| {
                    const v0 = spread(rng, -6, 6);
                    const v1 = spread(rng, -6, 6);
                    const v2 = @as(f32, 0.01) * spread(rng, -6, 6);
                    const b = (t * 4 * shape.n) + d;
                    p.x[b] = c.toBf16(v0);
                    p.x[b + shape.n] = c.toBf16(v1);
                    p.x[b + 2 * shape.n] = c.toBf16(v2);
                    p.x[b + 3 * shape.n] = c.toBf16(-(v0 + v1)); // the fourth stream cancels the first two
                };
                if (nf) p.plants = ref.plant(p.x, shape.rows, 4 * shape.n, 0, 4 * shape.n);
                p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            },
        }
        p.out = try p.rig.output(p.outLen());
        return true;
    }

    /// Rows r0 to r0 + n of the prepared call as one launch; p.in[] are whole.
    pub fn encode(p: *Proj, e: anytype, r0: usize, n: usize) !void {
        const s = p.shape;
        const pl = &p.pipes;
        switch (p.fam) {
            .rms => gl.rmsnorm(pl, e, at(p.in[0], r0 * s.k * 2), at(p.in[1], 0), at(p.out, r0 * s.k * 2), @intCast(n), @intCast(s.k), eps),
            .rope => gl.rope(pl, e, at(p.in[0], r0 * s.n * s.k * 2), at(p.out, r0 * s.n * s.k * 2), at(p.in[1], r0 * 4), at(p.in[2], 0), @intCast(n), @intCast(s.n), @intCast(s.k)),
            .scatter => gl.scatter(pl, e, at(p.in[0], r0 * s.k * 4), at(p.out, 0), at(p.in[1], r0 * 4), @intCast(n), @intCast(s.k * 4), 1),
            .embed => gl.embed(pl, e, at(p.in[1], 0), at(p.in[0], r0 * 4), at(p.out, r0 * 4 * s.n * 2), at(p.pre, r0 * 16), @intCast(n), @intCast(s.n)),
            .mean => gl.hcmean(pl, e, at(p.in[0], r0 * 4 * s.n * 2), at(p.out, r0 * s.n * 2), @intCast(n), @intCast(s.n), @intCast(s.n)),
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
            .rms => try p.rows_f.?.reference(want, mag),
            .rope => for (0..s.rows) |t| for (0..s.n) |h| {
                const row = (t * s.n + h) * s.k;
                const pos: usize = @intCast(p.pos[t]);
                for (0..s.k / 2) |i| {
                    const xr: f64 = c.fromBf16(p.x[row + 2 * i]);
                    const xi: f64 = c.fromBf16(p.x[row + 2 * i + 1]);
                    if (2 * i < s.k - 64) {
                        want[row + 2 * i] = xr;
                        want[row + 2 * i + 1] = xi;
                    } else {
                        const pr = i - (s.k - 64) / 2;
                        const co: f64 = p.tab[pos * 64 + pr];
                        const si: f64 = p.tab[pos * 64 + 32 + pr];
                        want[row + 2 * i] = xr * co - xi * si;
                        want[row + 2 * i + 1] = xr * si + xi * co;
                        mag[row + 2 * i] = @abs(xr * co) + @abs(xi * si);
                        mag[row + 2 * i + 1] = @abs(xr * si) + @abs(xi * co);
                    }
                }
            },
            .scatter => for (want, std.mem.sliceAsBytes(p.words)) |*w, b| {
                w.* = @floatFromInt(b);
            },
            .embed => for (0..s.rows) |t| for (0..s.k) |j| for (0..s.n) |d| {
                want[(t * s.k + j) * s.n + d] = c.fromBf16(p.table[@as(usize, p.ids[t]) * s.n + d]);
            },
            .mean => for (0..s.rows) |t| for (0..s.n) |d| {
                var sum: f64 = 0;
                var m: f64 = 0;
                for (0..4) |j| {
                    const v: f64 = c.fromBf16(p.x[(t * 4 + j) * s.n + d]);
                    sum += v;
                    m += @abs(v);
                }
                want[t * s.n + d] = sum / 4; // the kernel writes the mean, s * 0.25
                mag[t * s.n + d] = m;
            },
        }
    }

    /// Copies are exact; the fp32 steps of the mean and RoPE add 2^-24 of their term magnitudes to the bound.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        switch (p.fam) {
            .scatter, .embed => @memset(allow, 0),
            .rope, .mean => for (allow, want, mag) |*l, w, m| {
                const steps: f64 = if (p.fam == .mean) 8 else 4;
                const e32 = steps * 0x1p-24 * m;
                l.* = e32 + 0x1p-8 * (@abs(w) + e32) + 0x1p-133;
            },
            else => {},
        }
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return (p.rows_f orelse return error.NoSums).simulate(acc, order, out);
    }

    /// The scatter's whole output against the source's bytes: exact at every gate, so gate 1 needs no plants.
    pub fn check(p: *Proj, gate: g.Gate) !?[]const u8 {
        if (p.fam != .scatter) return null;
        const got = p.rig.bytes(p.out, p.outLen());
        const xb = std.mem.sliceAsBytes(p.words);
        if (std.mem.indexOfDiff(u8, got, xb)) |i| {
            return try std.fmt.bufPrint(&p.msg, "byte {d} of the copy is 0x{x:0>2} where the source holds 0x{x:0>2}", .{ i, got[i], xb[i] });
        }
        if (gate == .nonfinite and p.plants != 0) return try std.fmt.bufPrint(&p.msg, "{d} plants: no proof", .{p.plants});
        return null;
    }
};

/// Every served kernel and its planted controls, plus the launch-split probe after the gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .f = .rms, .ctl = 0, .name = base ++ "rmsnorm tf_dsv41_g_rmsnorm" },
        .{ .f = .rms, .ctl = 1, .name = base ++ "control: rms non-finite inputs flushed" },
        .{ .f = .rms, .ctl = 2, .name = base ++ "control: rms the first square of every row lost" },
        .{ .f = .rms, .ctl = 3, .name = base ++ "control: rms partial sums rounded to fp16" },
        .{ .f = .rms, .ctl = 4, .name = base ++ "control: rms writes past its output" },
        .{ .f = .rope, .ctl = 0, .name = base ++ "rope tf_dsv41_g_rope" },
        .{ .f = .rope, .ctl = 1, .name = base ++ "control: rope non-finite inputs flushed" },
        .{ .f = .rope, .ctl = 2, .name = base ++ "control: rope the rotation's sign lost" },
        .{ .f = .rope, .ctl = 4, .name = base ++ "control: rope overwrites its last output" },
        .{ .f = .scatter, .ctl = 0, .name = base ++ "scatter tf_dsv41_g_scatter" },
        .{ .f = .scatter, .ctl = 2, .name = base ++ "control: scatter every word one place on" },
        .{ .f = .scatter, .ctl = 4, .name = base ++ "control: scatter writes past its output" },
        .{ .f = .embed, .ctl = 0, .name = base ++ "embed tf_dsv41_g_embed" },
        .{ .f = .embed, .ctl = 2, .name = base ++ "control: embed the last stream never copied" },
        .{ .f = .embed, .ctl = 4, .name = base ++ "control: embed writes past its output" },
        .{ .f = .mean, .ctl = 0, .name = base ++ "hc mean tf_dsv41_g_hcmean" },
        .{ .f = .mean, .ctl = 1, .name = base ++ "control: hc mean non-finite inputs flushed" },
        .{ .f = .mean, .ctl = 2, .name = base ++ "control: hc mean drops the last stream" },
        .{ .f = .mean, .ctl = 3, .name = base ++ "control: hc mean the three-stream partial rounded to fp16" },
        .{ .f = .mean, .ctl = 4, .name = base ++ "control: hc mean writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.f, entry.ctl, entry.name);
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

    fn invShapes(f: Fam) []const c.Shape {
        return switch (f) {
            .rms => &.{ .{ .rows = 33, .n = 5120, .k = 5120 }, .{ .rows = 16, .n = 1280, .k = 1280 } },
            .rope => &.{ .{ .rows = 17, .n = 64, .k = 512 }, .{ .rows = 9, .n = 32, .k = 128 } },
            .scatter => &.{.{ .rows = 20, .n = 1, .k = 1280 }},
            .embed => &.{.{ .rows = 20, .n = 5120, .k = 4 }},
            .mean => &.{.{ .rows = 20, .n = 5120, .k = 4 }},
        };
    }

    /// Launches of 1, 7 and rows/3 rows equal the whole launch's bytes: no kernel carries state across rows.
    fn probe(s: *Suite, p: *Proj, shape: c.Shape, seed: u64, bad: *usize, launches: *usize) !void {
        if (!try p.prepare(shape, .random, seed)) return error.Refused;
        try p.run();
        const len = p.outLen();
        const whole = try s.a.dupe(u8, p.rig.bytes(p.out, len));
        defer s.a.free(whole);
        for ([_]usize{ 1, 7, @max(1, shape.rows / 3) }) |step| {
            @memset(p.rig.bytes(p.out, len), 0x55);
            const pass = p.rig.begin();
            var r0: usize = 0;
            while (r0 < shape.rows) : (r0 += step) {
                try p.encode(pass.e, r0, @min(step, shape.rows - r0));
                launches.* += 1;
            }
            try p.rig.finish(pass);
            if (!std.mem.eql(u8, whole, p.rig.bytes(p.out, len))) {
                bad.* += 1;
                std.debug.print("FAIL row invariance {s} {d}x{d}x{d}: {d} rows a launch differs from one launch\n", .{ p.name, shape.rows, shape.n, shape.k, step });
            }
        }
    }

    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            kernels += 1;
            for (invShapes(p.fam), 0..) |shape, i| try s.probe(p, shape, seed + i, &bad, &launches);
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and rows/3 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};
