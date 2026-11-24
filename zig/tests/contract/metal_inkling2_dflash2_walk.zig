//! Mini DFlash 2 selector walk kernel probe (3 concurrent streams, FP32 Gate 4) and planted control.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const controls = @import("metal_inkling2_dflash2_batch_controls.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const host = tf.inkling2_dflash2_host;
const sources = @import("inkling2_metal");

const block_size = 8;
const rank = 256;
const top_k = 16;
const walk_vocab = 2048;

fn bf(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    return c.fromBf16(c.toBf16(x));
}

pub const WalkProbe = struct {
    const streams: usize = 3;
    const stride: usize = block_size;
    const row0: usize = 1;
    const softcap: f32 = 30.0;
    const Sim = struct { c.Accum, g.Order };

    a: std.mem.Allocator,
    rig: rig_.Rig,
    control: bool,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    uv: []f32 = &.{},
    ui: []u32 = &.{},
    h: []u16 = &.{},
    pa: []u16 = &.{},
    pb: []u16 = &.{},
    seq: [streams * 8]u32 = undefined,
    ins: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [180]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !WalkProbe {
        const pipes = if (control) blk: {
            const txt = try controls.source(a, sources.inkling2_dflash2, .walk);
            defer a.free(txt);
            break :blk try host.Pipelines.loadSource(device, txt);
        } else try host.Pipelines.loadSource(device, sources.inkling2_dflash2);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .control = control, .pipes = pipes };
    }

    pub fn deinit(t: *WalkProbe) void {
        t.free();
        t.rig.deinit();
        t.pipes.deinit();
    }

    fn free(t: *WalkProbe) void {
        inline for (.{ "uv", "ui", "h", "pa", "pb" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    const vt = g.hooks(WalkProbe);
    pub fn kernel(t: *WalkProbe) g.Kernel {
        return .{
            .ctx = t,
            .vt = &vt,
            .control = if (t.control) .shapes else null,
            .spec = .{
                .name = if (t.control) controls.name(.walk) else "metal inkling2 dflash2 walk (3 streams)",
                .accum = .f32,
                .depth = walkDepth,
                .out = .raw,
                .nan = .{ .stated = "each stream takes the candidate with the largest softcapped score (NaN never wins; ids >= vocab skipped)" },
                .tiles = .{ .{}, .{}, .{} },
                .served = &.{ .{ .rows = 7, .n = top_k, .k = rank }, .{ .rows = 1, .n = top_k, .k = rank } },
            },
        };
    }

    fn walkDepth(_: c.Shape) usize {
        return 15;
    }

    pub fn count(t: *WalkProbe) usize {
        return streams * (t.shape.rows + 1) * 4;
    }

    pub fn prepare(t: *WalkProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 7 or shape.n != top_k or shape.k != rank) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const total_rows = streams * stride;
        t.uv = try t.a.alloc(f32, total_rows * top_k);
        t.ui = try t.a.alloc(u32, total_rows * top_k);
        t.h = try t.a.alloc(u16, total_rows * rank);
        t.pa = try t.a.alloc(u16, walk_vocab * rank);
        t.pb = try t.a.alloc(u16, walk_vocab * rank);
        for (t.uv) |*x| x.* = rng.float(f32) * 4 - 2;
        for (0..total_rows) |r| {
            for (t.ui[r * top_k ..][0..top_k], 0..) |*x, cnd| x.* = @intCast((r * 401 + cnd * 73) % walk_vocab);
            t.ui[r * top_k + top_k - 1] = 0xffffffff;
            t.ui[r * top_k + top_k - 2] = walk_vocab;
        }
        for (t.h) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.pa) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.pb) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) {
            @memset(t.h, c.toBf16(1));
            for (0..walk_vocab) |r| for (0..rank / 2) |j| {
                const lo_a: f32 = if (j < 4) @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 16))) - 8)) else @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 256))) - 128);
                t.pa[r * rank + j] = c.toBf16(lo_a);
                t.pa[r * rank + rank / 2 + j] = t.pa[r * rank + j];
                const b: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
                t.pb[r * rank + j] = c.toBf16(b);
                t.pb[r * rank + rank / 2 + j] = c.toBf16(-b + (if (j < 4) @as(f32, @floatFromInt(rng.uintLessThan(u32, 3))) - 1 else 0));
            };
            for (t.uv) |*x| x.* = @floatFromInt(rng.uintLessThan(u32, 8));
        }
        if (fill == .nonfinite) {
            const base = row0 * top_k;
            t.uv[base + 3] = std.math.nan(f32);
            if (shape.rows > 1) t.uv[base + top_k + 5] = std.math.inf(f32);
            if (shape.rows > 2) t.uv[base + 2 * top_k + 7] = -std.math.inf(f32);
        }
        @memset(&t.seq, 0);
        for (0..streams) |s| t.seq[s * (shape.rows + 1)] = @intCast(100 + s * 313);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.uv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ui));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.h));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.pa));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.pb));
        t.out = try t.rig.output(t.count());
        return true;
    }

    pub fn run(t: *WalkProbe) !void {
        const elems = streams * (t.shape.rows + 1);
        @memcpy(t.rig.bytes(t.out, t.count()), std.mem.sliceAsBytes(t.seq[0..elems]));
        const pass = t.rig.begin();
        const r = struct {
            fn of(x: rig_.Ref) host.Ref {
                return .{ .buf = x.buf, .off = x.off };
            }
        }.of;
        host.walk(&t.pipes, pass.e, r(t.ins[0]), r(t.ins[1]), r(t.ins[2]), r(t.ins[3]), r(t.ins[4]), r(t.out), .{
            .slots = @intCast(t.shape.rows),
            .rank = rank,
            .top = top_k,
            .vocab = walk_vocab,
            .stride = stride,
            .row0 = row0,
            .softcap = softcap,
        }, streams);
        try t.rig.finish(pass);
    }

    fn scores(t: *const WalkProbe, s_idx: usize, e: usize, prev: u32, sim: ?Sim, out: *[top_k]f64, bnd: *[top_k]f64) void {
        const rb = s_idx * stride + row0 + e;
        var terms: [rank + 1]f64 = undefined;
        for (0..top_k) |cnd| {
            const id = t.ui[rb * top_k + cnd];
            const vrow = @min(id, walk_vocab - 1);
            var abs: f64 = @abs(t.uv[rb * top_k + cnd]);
            var sum: f64 = t.uv[rb * top_k + cnd];
            terms[rank] = t.uv[rb * top_k + cnd];
            for (0..rank) |r| {
                const ah = bf(@as(f64, c.fromBf16(t.pa[@as(usize, prev) * rank + r])) * c.fromBf16(t.h[rb * rank + r]));
                terms[r] = ah * c.fromBf16(t.pb[@as(usize, vrow) * rank + r]);
                sum += terms[r];
                abs += @abs(terms[r]);
            }
            const raw = if (sim) |o| contract.rows.sumIn(o[0], o[1], terms[0 .. rank + 1]) else sum;
            out[cnd] = softcap * std.math.tanh(raw / softcap);
            bnd[cnd] = if (t.fill == .stress or sim != null) 0 else c.gamma(15, 0x1p-24) * abs;
        }
    }

    fn choose(t: *const WalkProbe, rb: usize, s: *const [top_k]f64) usize {
        var best = -std.math.inf(f64);
        var bi: usize = 0;
        for (0..top_k) |cnd| if (t.ui[rb * top_k + cnd] < walk_vocab and s[cnd] > best) {
            best = s[cnd];
            bi = cnd;
        };
        return bi;
    }

    fn walkRef(t: *WalkProbe, sim: ?Sim) [streams * 8]u32 {
        var seq = t.seq;
        var s: [top_k]f64 = undefined;
        var b: [top_k]f64 = undefined;
        for (0..streams) |s_idx| {
            const base = s_idx * (t.shape.rows + 1);
            for (0..t.shape.rows) |e| {
                t.scores(s_idx, e, seq[base + e], sim, &s, &b);
                seq[base + 1 + e] = t.ui[(s_idx * stride + row0 + e) * top_k + t.choose(s_idx * stride + row0 + e, &s)];
            }
        }
        return seq;
    }

    pub fn reference(t: *WalkProbe, want: []f64, mag: []f64) !void {
        const elems = streams * (t.shape.rows + 1);
        const seq = t.walkRef(null);
        for (want, std.mem.sliceAsBytes(seq[0..elems])) |*w, byte| w.* = @floatFromInt(byte);
        @memset(mag, 0);
    }

    pub fn simulate(t: *WalkProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const elems = streams * (t.shape.rows + 1);
        const seq = t.walkRef(.{ acc, order });
        for (out, std.mem.sliceAsBytes(seq[0..elems])) |*w, byte| w.* = @floatFromInt(byte);
    }

    pub fn bounds(_: *WalkProbe, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }

    pub fn check(t: *WalkProbe, _: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u32, t.rig.bytes(t.out, t.count()));
        var s: [top_k]f64 = undefined;
        var b: [top_k]f64 = undefined;
        for (0..streams) |s_idx| {
            const base = s_idx * (t.shape.rows + 1);
            if (got[base] != t.seq[base]) return try std.fmt.bufPrint(&t.msg, "stream {d} seed changed", .{s_idx});
            for (0..t.shape.rows) |e| {
                const rb = s_idx * stride + row0 + e;
                t.scores(s_idx, e, got[base + e], null, &s, &b);
                const w = t.choose(rb, &s);
                const id = got[base + 1 + e];
                const cand = for (0..top_k) |cnd| {
                    if (t.ui[rb * top_k + cnd] == id) break cnd;
                } else return try std.fmt.bufPrint(&t.msg, "stream {d} slot {d} took invalid candidate {d}", .{ s_idx, e, id });
                if (cand == w) continue;
                if (!std.math.isFinite(s[w]) or !std.math.isFinite(s[cand]) or id >= walk_vocab or !(s[cand] >= s[w] - b[cand] - b[w])) {
                    return try std.fmt.bufPrint(&t.msg, "stream {d} slot {d} mismatch: got {d} want {d}", .{ s_idx, e, cand, w });
                }
            }
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    walks: std.ArrayList(*WalkProbe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |ctrl| {
            const label = if (ctrl) controls.name(.walk) else "metal inkling2 dflash2 walk (3 streams)";
            if (only) |o| if (std.mem.indexOf(u8, label, o) == null) continue;
            const w = try s.a.create(WalkProbe);
            errdefer s.a.destroy(w);
            w.* = try WalkProbe.init(s.a, s.device, s.queue, ctrl);
            errdefer w.deinit();
            try contract.register(s.a, &s.walks, s.kernels, w, w.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.walks.items) |w| {
            w.deinit();
            s.a.destroy(w);
        }
        s.walks.deinit(s.a);
    }
};
