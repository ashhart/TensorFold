//! Cosmos3's Metal attention under the contract: the decode rows, the text model's flash tiles and the tower's.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const ref = @import("cosmos3_ref.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

pub const Which = enum { decode, flash_text, flash_tower, ctl_rescale, ctl_last_key };

/// The text model's GQA (16 query heads over 8, head dim 128) or the tower's full attention (16 heads of 72).
const Geo = struct { heads: usize, kv: usize, d: usize, causal: bool, flash: bool };

fn geo(w: Which) Geo {
    return switch (w) {
        .decode, .ctl_rescale => .{ .heads = 16, .kv = 8, .d = 128, .causal = true, .flash = false },
        .flash_text => .{ .heads = 16, .kv = 8, .d = 128, .causal = true, .flash = true },
        .flash_tower, .ctl_last_key => .{ .heads = 16, .kv = 16, .d = 72, .causal = false, .flash = true },
    };
}

fn function(w: Which) [:0]const u8 {
    const x = geo(w);
    return if (!x.flash) "cos3_attn_128" else if (x.d == 128) "cos3_flash_128" else "cos3_flash_72";
}

fn mutation(w: Which) ?[2][]const u8 {
    return switch (w) {
        .ctl_rescale => .{ "for (int i = 0; i < NS; ++i) acc[i] *= alpha;", "" },
        .ctl_last_key => .{ "(j < my_n && scol + c < count)", "(j < my_n && scol + c + 1 < count)" },
        else => null,
    };
}

fn depthOf(s: c.Shape) usize {
    return s.k + (s.k + 31) / 32 + 130;
}

const why = "all keys alike make every weight exactly 1: o is a plain sum of V's integers over l's count, exact in fp32 (any order); fp16 sums overflow, bf16 ones lose its low bits";

pub fn spec(w: Which) c.Spec {
    const keys = c.Tile{ .size = 32, .min = 8 };
    var s: c.Spec = switch (w) {
        .decode, .ctl_rescale => .{ .name = "metal cosmos3 cos3_attn_128 (decode rows, causal GQA 16/8 over a stream's cache)", .accum = .f32, .depth = depthOf, .out = .bf16, .tiles = .{ .{}, .{}, keys }, .served = &.{ .{ .rows = 1, .n = 128, .k = 40 }, .{ .rows = 4, .n = 128, .k = 333 }, .{ .rows = 7, .n = 128, .k = 1100 } } },
        .flash_text => .{ .name = "metal cosmos3 cos3_flash_128 (prompt rows, causal GQA 16/8, 8 x 8 bf16 MMAs, P as bf16)", .accum = .f32, .depth = depthOf, .out = .bf16, .tiles = .{ .{ .size = 32, .min = 8 }, .{}, .{} }, .served = &.{ .{ .rows = 40, .n = 128, .k = 40 }, .{ .rows = 8, .n = 128, .k = 8 }, .{ .rows = 33, .n = 128, .k = 600 }, .{ .rows = 96, .n = 128, .k = 96 } } },
        .flash_tower, .ctl_last_key => .{ .name = "metal cosmos3 cos3_flash_72 (the tower's full attention, 16 heads of 72, P as bf16)", .accum = .f32, .depth = depthOf, .out = .bf16, .tiles = .{ .{ .size = 32, .step = 4, .min = 4 }, .{}, .{ .size = 32, .step = 4, .min = 4 } }, .served = &.{ .{ .rows = 196, .n = 72, .k = 196 }, .{ .rows = 16, .n = 72, .k = 16 }, .{ .rows = 300, .n = 72, .k = 300 } } },
    };
    s.sums = .{ .bf16 = 0, .why = why };
    if (w == .ctl_rescale) s.name = "control: cosmos3 decode attention skipping a block's rescale of o";
    if (w == .ctl_last_key) s.name = "control: cosmos3 flash attention dropping each block's last key";
    return s;
}

pub fn control(w: Which) ?g.Gate {
    return if (mutation(w) != null) .shapes else null;
}

const Args = extern struct { first: u32, keys: u32, causal: u32, group: u32, q_row: u32, k_row: u32, k_head: u32, o_row: u32, scale: f32 };

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    which: Which,
    rows: usize = 0,
    keys: usize = 0,
    first: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    point: []f64 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    in: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which) !Probe {
        const m = mutation(which);
        if (m) |x| if (std.mem.indexOf(u8, sources.cosmos3, x[0]) == null) return error.ControlLineMissing;
        const src = if (m) |x| try std.mem.replaceOwned(u8, a, sources.cosmos3, x[0], x[1]) else try a.dupe(u8, sources.cosmos3);
        defer a.free(src);
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, function(which)), .which = which };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.pipe.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.which), .ctx = p, .vt = &vt, .control = control(p.which) };
    }

    const vt = g.hooks(Probe);

    /// Row strides: q and o [rows, heads * d]; text caches [kv][keys][d], the tower's k and v [keys, heads * d].
    fn qRow(x: Geo) usize {
        return x.heads * x.d;
    }

    fn kAt(p: *const Probe, x: Geo, h: usize, j: usize) usize {
        return if (x.causal) (h * p.keys + j) * x.d else j * x.kv * x.d + h * x.d;
    }

    /// Keys row r sees: the causal rows a stream's cache through its own position, the tower every key.
    fn seen(p: *const Probe, x: Geo, r: usize) usize {
        return if (x.causal) p.first + r + 1 else p.keys;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const x = geo(p.which);
        if (x.causal and shape.k < shape.rows) return false;
        if (!x.flash and shape.rows > 7) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const ar = p.arena.allocator();
        p.rows = shape.rows;
        p.keys = shape.k;
        p.first = if (x.causal) shape.k - shape.rows else 0;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // flash tiles load whole 32-row query tiles: q carries the engine's 32 rows of slack past the call's rows
        p.q = try ar.alloc(u16, (p.rows + (if (x.flash) @as(usize, 32) else 0)) * qRow(x));
        p.k = try ar.alloc(u16, p.keys * x.kv * x.d);
        p.v = try ar.alloc(u16, p.keys * x.kv * x.d);
        for (p.q) |*y| y.* = c.toBf16((rng.float(f32) * 2 - 1) * 1.5);
        for (p.k) |*y| y.* = c.toBf16((rng.float(f32) * 2 - 1) * 1.5);
        for (p.v) |*y| y.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..x.kv) |h| for (0..p.keys) |j| for (0..x.d) |e| {
            p.k[p.kAt(x, h, j) + e] = p.k[p.kAt(x, h, 0) + e]; // every key alike: every weight exactly 1
            const m: f32 = @floatFromInt((1 + rng.uintLessThan(u32, 255)) * 64);
            p.v[p.kAt(x, h, j) + e] = c.toBf16(if (j < p.keys / 2) m else -m);
        };
        if (fill == .nonfinite) {
            p.q[3] = c.toBf16(std.math.nan(f32)); // row 0, head 0
            const half = p.keys / 2;
            p.v[p.kAt(x, 0, half) + 5] = c.toBf16(std.math.inf(f32));
            if (half + 1 < p.keys) {
                p.v[p.kAt(x, 0, half + 1) + 6] = c.toBf16(-std.math.inf(f32));
                p.v[p.kAt(x, 0, half) + 6] = c.toBf16(std.math.inf(f32));
            }
        }
        const bytes = std.mem.sliceAsBytes;
        p.in[0] = try p.rig.input(bytes(p.q));
        p.in[1] = try p.rig.input(bytes(p.k));
        p.in[2] = try p.rig.input(bytes(p.v));
        p.out = try p.rig.output(p.rows * qRow(x) * 2);
        try p.compute();
        return true;
    }

    pub fn run(p: *Probe) !void {
        const x = geo(p.which);
        const args = Args{
            .first = @intCast(p.first),
            .keys = @intCast(p.keys),
            .causal = @intFromBool(x.causal),
            .group = @intCast(x.heads / x.kv),
            .q_row = @intCast(qRow(x)),
            .k_row = @intCast(if (x.causal) x.d else x.kv * x.d),
            .k_head = @intCast(if (x.causal) p.keys * x.d else x.d),
            .o_row = @intCast(qRow(x)),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(x.d))),
        };
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |y, i| e.setBuffer(y.buf, y.off, i);
        e.setValue(args, 3);
        e.setBuffer(p.out.buf, p.out.off, 4);
        if (x.flash) {
            e.setValue(@as(u32, @intCast(p.rows)), 5);
            e.dispatchGroups(mtl.Size.of(x.heads, (p.rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        } else e.dispatchGroups(mtl.Size.of(x.heads, p.rows, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.rows * qRow(geo(p.which));
    }

    fn compute(p: *Probe) !void {
        const x = geo(p.which);
        const ar = p.arena.allocator();
        const n = p.count();
        p.point = try ar.alloc(f64, n);
        p.lo = try ar.alloc(f64, n);
        p.hi = try ar.alloc(f64, n);
        const kf = try ar.alloc(f64, p.keys * x.d);
        const vf = try ar.alloc(f64, p.keys * x.d);
        const qf = try ar.alloc(f64, x.d);
        const w = try ar.alloc(f64, p.keys);
        const tau = try ar.alloc(f64, p.keys);
        const half = try ar.alloc(f64, x.d);
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, @floatFromInt(x.d))));
        for (0..x.kv) |kh| {
            for (0..p.keys) |j| for (0..x.d) |e| {
                kf[j * x.d + e] = c.fromBf16(p.k[p.kAt(x, kh, j) + e]);
                vf[j * x.d + e] = c.fromBf16(p.v[p.kAt(x, kh, j) + e]);
            };
            const group = x.heads / x.kv;
            for (kh * group..(kh + 1) * group) |h| for (0..p.rows) |r| {
                const nk = p.seen(x, r);
                for (qf, 0..) |*y, e| y.* = c.fromBf16(p.q[r * qRow(x) + h * x.d + e]);
                const f = if (x.flash) ref.flashForm(x.d, nk) else ref.decodeForm(x.d, nk);
                const at = r * qRow(x) + h * x.d;
                ref.head(qf, kf, vf, nk, x.d, scale, f, w, tau, p.point[at..][0..x.d], half);
                for (0..x.d) |e| ref.ends(p.point[at + e], half[e], &p.lo[at + e], &p.hi[at + e]);
            };
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        @memcpy(want, p.point);
        @memset(mag, 0);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, p.lo, p.hi) |*l, w, lo, hi| l.* = @max(@abs(lo - w), @abs(hi - w));
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }

    /// Gate 4's calls: every weight is 1, so o is V's column sum, in `acc` and `order`, over the keys a row sees.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const x = geo(p.which);
        const t = try p.a.alloc(f64, p.keys);
        defer p.a.free(t);
        for (0..p.rows) |r| for (0..x.heads) |h| for (0..x.d) |e| {
            const nk = p.seen(x, r);
            const kh = h / (x.heads / x.kv);
            for (t[0..nk], 0..) |*y, j| y.* = c.fromBf16(p.v[p.kAt(x, kh, j) + e]);
            out[r * qRow(x) + h * x.d + e] = rows_.sumIn(acc, order, t[0..nk]) / @as(f64, @floatFromInt(nk));
        };
    }
};
