//! window_attention.cu (#798) under the contract on CUDA: decode and prompt rows over a ring of keys, head by head.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const wa = cuda.window_attention;

/// Why the split decode (fp32 weights) runs attention_ref's bound: a score's rounding reaches its weight through exp.
const split_why = "a score's absolute rounding error is its weight's relative error: gates 1 and 3 hold outputs to attention_ref's bound";

/// A decode or prompt call's fixed heads, window and ring (the variant), and which kernels it runs.
pub const Cfg = struct { name: []const u8, h: usize, hk: usize, d: usize, win: usize, ring: usize, streams: usize = 1, prompt: bool = false, served: []const c.Shape };

pub const cfgs = [_]Cfg{
    // Kolibri's sliding layers (48 heads over 4, window 513): decode on tensor cores, prompt rows 4 heads a block
    .{ .name = "cuda window decode tf_window_decode_tc, 48/4 heads, window 513, ring 640", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 640, .streams = 2, .served = &.{ .{ .rows = 16, .n = 48, .k = 3000 }, .{ .rows = 64, .n = 48, .k = 1000 } } },
    .{ .name = "cuda window decode tf_window_decode_tc, 16/1 heads, no window, ring 300", .h = 16, .hk = 1, .d = 128, .win = 0, .ring = 300, .served = &.{.{ .rows = 16, .n = 16, .k = 300 }} },
    .{ .name = "cuda window decode tf_window_part + tf_window_merge, 8/2 heads of 64, window 100, ring 128", .h = 8, .hk = 2, .d = 64, .win = 100, .ring = 128, .streams = 3, .served = &.{ .{ .rows = 16, .n = 8, .k = 3000 }, .{ .rows = 64, .n = 8, .k = 1000 } } },
    .{ .name = "cuda window prompt wattn_kernel<128, 8, 4, 8>, 48/4 heads, window 513, ring 2624", .h = 48, .hk = 4, .d = 128, .win = 513, .ring = 2624, .prompt = true, .served = &.{ .{ .rows = 128, .n = 48, .k = 0 }, .{ .rows = 96, .n = 48, .k = 2600 } } },
    .{ .name = "cuda window prompt wattn_kernel<128, 8, 8, 8>, 16/2 heads, no window, a row a position", .h = 16, .hk = 2, .d = 128, .win = 0, .ring = 0, .prompt = true, .served = &.{ .{ .rows = 64, .n = 16, .k = 100 }, .{ .rows = 48, .n = 16, .k = 0 } } },
};

/// The P x V chain of a row: its window's keys, a tile at a time, then the tiles' fold.
fn keyChain(s: c.Shape) usize {
    return s.k + 64 + 64;
}

/// Decode: rows, query heads (n) and the positions' bound (k); prompt: chunk rows, query heads and the chunk's start.
pub fn spec(f: Cfg) c.Spec {
    const key_tile: usize = if (f.prompt or f.d != 128) 64 else 32;
    const row_tile: usize = if (f.prompt) 16 * (8 / (if ((f.h / f.hk) % 8 == 0) @as(usize, 8) else 4)) else 0;
    const k_tile: c.Tile = if (f.prompt) .{ .size = 64, .min = 0 } else .{ .size = key_tile };
    return .{ .name = f.name, .accum = .f32, .depth = keyChain, .ops = 1, .out = .bf16, .tiles = .{ .{ .size = row_tile }, .{}, k_tile }, .served = f.served, .sums = .{ .why = if (f.d == 128) attn.why else split_why } };
}

pub const Window = struct {
    rig: rig_.Rig,
    f: Cfg,
    g: *const wa.WindowAttention,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    pos: std.ArrayList(i32) = .empty,
    slot: std.ArrayList(i32) = .empty,
    q: std.ArrayList(u16) = .empty,
    kc: std.ArrayList(u16) = .empty,
    vc: std.ArrayList(u16) = .empty,
    dev: [5]u64 = @splat(0),
    scratch: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Window) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "pos", "slot", "q", "kc", "vc" }) |n| @field(self, n).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Window) g.Kernel {
        return .{ .spec = spec(self.f), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Window);

    fn win(self: *const Window) wa.Window {
        const f = self.f;
        return .{ .h = f.h, .hk = f.hk, .d = f.d, .win = f.win, .ring = f.ring, .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(f.d)))) };
    }

    /// A row's position and its cache's first row (stream slot * ring; a prompt's cache holds one stream).
    fn at(self: *const Window, r: usize) struct { p: usize, base: usize } {
        if (self.f.prompt) return .{ .p = self.shape.k + r, .base = 0 };
        return .{ .p = @intCast(self.pos.items[r]), .base = @as(usize, @intCast(self.slot.items[r])) * self.f.ring };
    }

    fn cacheRow(self: *const Window, base: usize, key: usize) usize {
        return base + if (self.f.ring > 0) key % self.f.ring else key;
    }

    fn first(self: *const Window, p: usize) usize {
        return if (self.f.win > 0 and p + 1 > self.f.win) p + 1 - self.f.win else 0;
    }

    pub fn prepare(self: *Window, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const f = self.f;
        if (shape.n != f.h or shape.rows == 0 or (!f.prompt and shape.k == 0)) return false;
        if (!f.prompt and f.win == 0 and shape.k > f.ring) return false; // an unwindowed row's keys all fit the ring
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        try self.pos.resize(a, n);
        try self.slot.resize(a, n);
        for (self.pos.items, self.slot.items, 0..) |*p, *sl, i| {
            // short histories, wrapped rings, the last position; gate 4: one stream's row at the bound, every row alike
            p.* = @intCast(if (fill == .stress or i == n - 1) shape.k -| 1 else if (i % 4 == 0) i % @min(40, shape.k) else (i * 7919) % shape.k);
            sl.* = @intCast(if (fill == .stress) 0 else i % f.streams);
        }
        const crow = if (f.prompt) (if (f.ring > 0) f.ring else shape.k + n) else f.streams * f.ring;
        const kv = f.hk * f.d;
        try self.q.resize(a, n * f.h * f.d);
        try self.kc.resize(a, crow * kv);
        try self.vc.resize(a, crow * kv);
        for ([_][]u16{ self.q.items, self.kc.items, self.vc.items }) |x| for (x) |*e| {
            e.* = attn.normal(rng);
        };
        var most: usize = 1;
        for (0..n) |r| {
            const w = self.at(r);
            most = @max(most, w.p + 1 - self.first(w.p));
        }
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, f.d, most);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        for ([_][]const u8{ std.mem.sliceAsBytes(self.q.items), std.mem.sliceAsBytes(self.kc.items), std.mem.sliceAsBytes(self.vc.items), std.mem.sliceAsBytes(self.pos.items), std.mem.sliceAsBytes(self.slot.items) }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.scratch = if (f.prompt) 0 else try self.rig.scratch(wa.WindowAttention.scratchBytes(self.win(), n));
        self.out = try self.rig.output(n * f.h * f.d * 2);
        return true;
    }

    fn cell(self: *Window, x: []u16, r: usize, key: usize, kh: usize, e: usize) *u16 {
        const w = self.at(r);
        return &x[(self.cacheRow(w.base, key) * self.f.hk + kh) * self.f.d + e];
    }

    /// NaN query; +inf and -inf values, both in one column; NaN key; +inf at a window's first key and at tiny weights.
    fn plant(self: *Window) void {
        const n = self.shape.rows;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const hk = self.f.hk;
        self.q.items[3] = nan; // row 0, head 0
        self.cell(self.vc.items, 0, self.at(0).p, 0, 7).* = inf;
        if (n > 1) self.cell(self.vc.items, 1, self.at(1).p, (1 % hk), 9).* = ninf;
        if (n > 2 and self.at(2).p >= self.first(self.at(2).p) + 1) {
            self.cell(self.vc.items, 2, self.at(2).p, (2 % hk), 11).* = inf;
            self.cell(self.vc.items, 2, self.at(2).p - 1, (2 % hk), 11).* = ninf;
        }
        if (n > 3) self.cell(self.kc.items, 3, self.at(3).p, (3 % hk), 0).* = nan;
        if (n > 4) self.cell(self.vc.items, 4, self.first(self.at(4).p), (4 % hk), 13).* = inf;
        // prompt rows: +inf at float64 weight 0 (row 5's own key, a causal edge: NaN), and under float32's least (inf)
        if (self.f.prompt and n > 6) {
            if (hk > 2) self.reach(5, 1);
            self.under(n - 1, 2 % hk);
        }
        // decode rows: the same over a row's window (unwindowed: row 4, whose keys stop short of row 3's NaN key)
        if (!self.f.prompt and n > 7) self.split(if (self.f.win == 0) 4 else n - 1, if (hk > 2) 2 else 0, 21);
    }

    /// Row r, key head kh: key p all 8, its other keys -8 with +inf at e; queries 2 (inf) and 8 (weight 0: NaN).
    fn split(self: *Window, r: usize, kh: usize, e: usize) void {
        const f = self.f;
        const p = self.at(r).p;
        const G = f.h / f.hk;
        @memset(self.q.items[(r * f.h + kh * G) * f.d ..][0..f.d], c.toBf16(2));
        @memset(self.q.items[(r * f.h + kh * G + 1) * f.d ..][0..f.d], c.toBf16(8));
        for (self.first(p)..p + 1) |key| {
            const k = self.cell(self.kc.items, r, key, kh, 0);
            @memset(@as([*]u16, @ptrCast(k))[0..f.d], c.toBf16(if (key == p) 8 else -8));
            if (key != p) self.cell(self.vc.items, r, key, kh, e).* = c.toBf16(std.math.inf(f32));
        }
    }

    /// Row r, key head kh's first head: query all 8, key p - 1 all 8, key p all -8 with +inf at element 17.
    fn reach(self: *Window, r: usize, kh: usize) void {
        const f = self.f;
        const p = self.at(r).p;
        @memset(self.q.items[(r * f.h + kh * (f.h / f.hk)) * f.d ..][0..f.d], c.toBf16(8));
        @memset(self.kc.items[(self.cacheRow(0, p - 1) * f.hk + kh) * f.d ..][0..f.d], c.toBf16(8));
        @memset(self.kc.items[(self.cacheRow(0, p) * f.hk + kh) * f.d ..][0..f.d], c.toBf16(-8));
        self.cell(self.vc.items, r, p, kh, 17).* = c.toBf16(std.math.inf(f32));
    }

    /// Row r, key head kh's first head: query all 2, key p - 1 all 8, its other keys all -8 with +inf at element 21.
    fn under(self: *Window, r: usize, kh: usize) void {
        const f = self.f;
        const p = self.at(r).p;
        @memset(self.q.items[(r * f.h + kh * (f.h / f.hk)) * f.d ..][0..f.d], c.toBf16(2));
        for (self.first(p)..p + 1) |key| {
            @memset(self.kc.items[(self.cacheRow(0, key) * f.hk + kh) * f.d ..][0..f.d], c.toBf16(if (key == p - 1) 8 else -8));
            if (key != p - 1) self.cell(self.vc.items, r, key, kh, 21).* = c.toBf16(std.math.inf(f32));
        }
    }

    /// Gate 4: every key alike (each p is 1); values mirrored over the keys every row sees, 0 elsewhere.
    fn stress(self: *Window, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const f = self.f;
        const kv = f.hk * f.d;
        const key = try a.alloc(u16, kv);
        defer a.free(key);
        for (key) |*e| e.* = attn.normal(rng);
        for (0..self.kc.items.len / kv) |i| @memcpy(self.kc.items[i * kv ..][0..kv], key);
        @memset(self.vc.items, 0);
        const last = self.at(self.shape.rows - 1).p;
        const lo = self.first(last);
        const hi = self.at(0).p;
        const col = try a.alloc(f32, hi + 1 - lo);
        defer a.free(col);
        for (0..f.hk) |kh| for (0..f.d) |e| {
            attn.mirrored(rng, col);
            for (col, lo..) |y, k| self.cell(self.vc.items, 0, k, kh, e).* = c.toBf16(y);
        };
    }

    pub fn run(self: *Window) !void {
        const n = self.shape.rows;
        if (self.f.prompt) {
            try self.g.prompt(self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.shape.k, n, self.win());
        } else try self.g.decode(self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.dev[3], self.dev[4], self.win(), self.scratch, n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Window) usize {
        return self.shape.rows * self.f.h * self.f.d;
    }

    /// Output head i (row i / h, head i % h): keys (p - win, p] of its stream's ring.
    pub fn load(self: *Window, i: usize, hd: *attn.Head) !void {
        const f = self.f;
        const d = f.d;
        const r = i / f.h;
        const head = i % f.h;
        const kh = head / (f.h / f.hk);
        const w = self.at(r);
        const lo = self.first(w.p);
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[(r * f.h + head) * d + e]);
        for (lo..w.p + 1, 0..) |key, j| for (0..d) |e| {
            const cr = (self.cacheRow(w.base, key) * f.hk + kh) * d + e;
            hd.k[j * d + e] = c.fromBf16(self.kc.items[cr]);
            hd.v[j * d + e] = c.fromBf16(self.vc.items[cr]);
        };
        hd.n = w.p + 1 - lo;
    }

    fn heads(self: *Window) usize {
        return self.shape.rows * self.f.h;
    }

    fn scale(self: *const Window) f64 {
        return self.win().scale;
    }

    pub fn check(self: *Window, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), self.scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Window, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), self.scale(), want, mag);
    }

    pub fn simulate(self: *Window, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), self.scale(), acc, order, out);
    }
};
