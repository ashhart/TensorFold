//! window_rows.cu under the contract on CUDA: Gemma 4's sliding layers (head dim 256), every output in float64's range.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const wc = @import("window_rows_check.zig");
const wt = @import("window_rows_tests.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const ar = core.attention_range;
const bfr = core.bf16_round;
const wa = cuda.window_attention;
const WA = wa.WindowAttention;

/// A planted fault, from contract_controls.cu (window_rows.cuh's Fault), or the served bf16-P kernel (p16).
pub const Ctl = enum { none, edge, rescale, trunc, prompt_edge, last, p16 };

/// A call's fixed heads, window and ring, and which kernels it runs.
pub const Cfg = struct {
    name: []const u8,
    h: usize,
    hk: usize,
    d: usize = 256,
    win: usize = 1024,
    ring: usize = 1536,
    streams: usize = 3,
    prompt: bool = false,
    p32: bool = false,
    image: bool = false,
    ctl: Ctl = .none,
    served: []const c.Shape,
};

const decode_served = &[_]c.Shape{ .{ .rows = 16, .n = 16, .k = 40000 }, .{ .rows = 64, .n = 16, .k = 3000 }, .{ .rows = 1, .n = 16, .k = 1100 } };
const prompt_served = &[_]c.Shape{ .{ .rows = 512, .n = 16, .k = 0 }, .{ .rows = 512, .n = 16, .k = 5000 }, .{ .rows = 96, .n = 16, .k = 1300 } };

/// Gemma 4 12B's sliding layers (16 query heads over 8 of 256, window 1024, ring 1536); controls each plant one fault.
pub const cfgs = [_]Cfg{
    .{ .name = "cuda window_rows decode tf_wrows_part_256 + tf_wrows_merge_256, 16/8 heads, window 1024, ring 1536", .h = 16, .hk = 8, .served = decode_served },
    .{ .name = "cuda window_rows prompt tf_wrows_prompt_256_p32, 16/8 heads, window 1024, ring 1536, image rows", .h = 16, .hk = 8, .prompt = true, .p32 = true, .image = true, .served = prompt_served },
    .{ .name = "cuda window_rows prompt tf_wrows_prompt_256 (P as bf16), 16/8 heads, window 1024, ring 1536", .h = 16, .hk = 8, .prompt = true, .served = prompt_served },
    .{ .name = "control: decode partials whose window starts a key early", .h = 16, .hk = 8, .ctl = .edge, .served = decode_served },
    .{ .name = "control: decode merge without each chunk's exp(m_c - M)", .h = 16, .hk = 8, .ctl = .rescale, .served = decode_served },
    .{ .name = "control: decode merge rounding its output toward zero", .h = 16, .hk = 8, .ctl = .trunc, .served = decode_served },
    .{ .name = "control: P32 prompt rows whose window starts a key early", .h = 16, .hk = 8, .prompt = true, .p32 = true, .ctl = .prompt_edge, .served = prompt_served },
    .{ .name = "control: P32 prompt rows ignoring an image's last key", .h = 16, .hk = 8, .prompt = true, .p32 = true, .image = true, .ctl = .last, .served = prompt_served },
    .{ .name = "control: P rounded to bf16 under the P32 range", .h = 16, .hk = 8, .prompt = true, .p32 = true, .ctl = .p16, .served = prompt_served },
};

/// The control kernel's symbol in contract_controls.cu, if it has one.
pub fn symbol(ctl: Ctl) ?[:0]const u8 {
    return switch (ctl) {
        .none, .p16 => null,
        .edge => "tf_ctl_wrows_part_edge",
        .rescale => "tf_ctl_wrows_merge_rescale",
        .trunc => "tf_ctl_wrows_merge_trunc",
        .prompt_edge => "tf_ctl_wrows_prompt_edge",
        .last => "tf_ctl_wrows_prompt_last",
    };
}

/// The longest chain: a chunk's keys and the chunks (decode), or the tiles' folds (prompt), as the range counts them.
fn depth(s: c.Shape) usize {
    return 256 + s.k / 32 + 16;
}

const why = "all keys alike make every weight exactly 1: o is a plain sum of V's integers, exact in fp32 (any order); fp16 sums overflow, bf16 ones lose its low bits";

pub fn spec(f: Cfg) c.Spec {
    const tiles: [3]c.Tile = if (f.prompt) .{ .{ .size = 16 }, .{}, .{ .size = 32, .min = 0 } } else .{ .{}, .{}, .{ .size = 256 } };
    return .{ .name = f.name, .accum = .f32, .depth = depth, .ops = 4, .out = .bf16, .tiles = tiles, .served = f.served, .sums = .{ .f16 = 100, .bf16 = 0, .why = why } };
}

pub const Rows = struct {
    rig: rig_.Rig,
    f: Cfg,
    g: *const WA,
    /// The controls' image (contract_controls.cu), when this registration runs one.
    ctl: ?cuda.Module = null,
    arena: std.heap.ArenaAllocator,
    call: wc.Call = undefined,
    fns: WA.Wide = undefined,
    dev: [6]u64 = @splat(0),
    scratch: u64 = 0,
    out: u64 = 0,
    half: []f64 = &.{},
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Rows) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Rows) g.Kernel {
        return .{ .spec = spec(self.f), .ctx = self, .vt = &vt, .control = if (self.f.ctl != .none) .shapes else null };
    }

    const vt = g.hooks(Rows);

    fn win(self: *const Rows) wa.Window {
        const f = self.f;
        return .{ .h = f.h, .hk = f.hk, .d = f.d, .win = f.win, .ring = f.ring, .scale = 1, .p32 = f.p32 and f.ctl != .p16 };
    }

    /// The kernels a run launches: the served ones, one of them swapped for its control.
    fn kernels(self: *Rows) !WA.Wide {
        var k = self.g.wideFns(self.f.d).*;
        const sym = symbol(self.f.ctl) orelse return k;
        const fun = try self.ctl.?.function(sym);
        switch (self.f.ctl) {
            .edge => k.part = fun,
            .rescale, .trunc => k.merge = fun,
            else => {
                try fun.allowDynamicShared(WA.promptShared(self.f.d));
                k.prompt[1] = fun;
            },
        }
        return k;
    }

    pub fn prepare(self: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const f = self.f;
        if (shape.n != f.h or shape.rows == 0 or (!f.prompt and shape.k == 0)) return false;
        if (f.prompt and shape.rows + f.win > f.ring) return false; // a chunk and its window fit the ring
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows;
        const crow = if (f.prompt) f.ring else f.streams * f.ring;
        const q = try a.alloc(u16, n * f.h * f.d);
        const kc = try a.alloc(u16, crow * f.hk * f.d);
        const vc = try a.alloc(u16, kc.len);
        wt.fill(rng, q, kc, vc, f.d);
        const pos = try a.alloc(i32, n);
        const slot = try a.alloc(i32, n);
        const hi = try a.alloc(i32, n);
        for (pos, slot, hi, 0..) |*p, *sl, *x, i| {
            // short histories, wrapped rings, the last position; gate 4: one stream's row at the bound, every row alike
            p.* = @intCast(if (fill == .stress or i == n - 1) shape.k -| 1 else if (i % 4 == 0) i % @min(40, shape.k) else (i * 7919) % shape.k);
            sl.* = @intCast(if (fill == .stress) 0 else i % f.streams);
            x.* = @intCast(shape.k + i);
        }
        if (f.image and fill != .stress and n > 8) { // an image: rows n/8 .. n/2 see each other
            for (n / 8..n / 2) |i| hi[i] = @intCast(shape.k + n / 2 - 1);
        }
        self.call = .{ .h = f.h, .hk = f.hk, .d = f.d, .win = f.win, .ring = f.ring, .scale = 1, .prompt = f.prompt, .p32 = f.p32, .rows = n, .q = q, .kc = kc, .vc = vc, .pos = pos, .slot = slot, .p0 = shape.k, .hi = if (f.image) hi else null };
        if (fill == .random) {
            var r: usize = 0;
            while (r < n) : (r += 3) {
                const w = self.call.at(r);
                const lo = self.call.first(w.p);
                for (0..f.hk) |kh| wt.sharpen(self.call, kc, r, lo + (r * 31 + kh * 7) % (w.last + 1 - lo), kh);
            }
        }
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(q, kc, vc),
            .stress => try self.stress(rng, kc, vc),
        }
        for ([_][]const u8{ std.mem.sliceAsBytes(q), std.mem.sliceAsBytes(kc), std.mem.sliceAsBytes(vc), std.mem.sliceAsBytes(pos), std.mem.sliceAsBytes(slot), std.mem.sliceAsBytes(hi) }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.scratch = if (f.prompt) 0 else try self.rig.scratch(WA.scratchBytes(self.win(), n));
        self.out = try self.rig.output(n * f.h * f.d * 2);
        self.fns = try self.kernels();
        self.half = try a.alloc(f64, self.count());
        return true;
    }

    fn cell(self: *Rows, x: []u16, r: usize, key: usize, kh: usize) []u16 {
        const w = self.call.at(r);
        return x[(self.call.cacheRow(w.base, key) * self.f.hk + kh) * self.f.d ..][0..self.f.d];
    }

    /// NaN in q and a key; +inf in a later row's value; +-inf in a column; +inf at float64 weight e^-200 (row 4).
    fn plant(self: *Rows, q: []u16, kc: []u16, vc: []u16) void {
        const f = self.f;
        const n = self.call.rows;
        const nan = bfr.bits(std.math.nan(f32));
        const inf = bfr.bits(std.math.inf(f32));
        q[3] = nan; // row 0, head 0
        self.cell(vc, 1, self.call.at(1).p, 1 % f.hk)[7] = inf;
        if (n > 2) {
            const p = self.call.at(2).p;
            self.cell(vc, 2, p, 2 % f.hk)[11] = inf;
            if (p > self.call.first(p)) self.cell(vc, 2, p - 1, 2 % f.hk)[11] = bfr.bits(-std.math.inf(f32));
        }
        if (n > 3) self.cell(kc, 3, self.call.at(3).p, 3 % f.hk)[0] = nan;
        if (n > 4) {
            const r = 4;
            const kh = 4 % f.hk;
            const w = self.call.at(r);
            const grp = f.h / f.hk;
            for (0..grp) |hh| @memset(q[(r * f.h + kh * grp + hh) * f.d ..][0..f.d], bfr.bits(0.625));
            for (self.call.first(w.p)..w.last + 1) |key| {
                const k = self.cell(kc, r, key, kh);
                @memset(k, bfr.bits(if (key == w.p) 0.625 else -0.625));
                if (key != w.p) self.cell(vc, r, key, kh)[21] = inf;
            }
        }
    }

    /// Gate 4: every key alike (each weight 1); values mirrored over the keys every row sees, 0 elsewhere.
    fn stress(self: *Rows, rng: std.Random, kc: []u16, vc: []u16) !void {
        const a = self.arena.allocator();
        const f = self.f;
        const kv = f.hk * f.d;
        for (0..kc.len / kv) |i| @memcpy(kc[i * kv ..][0..kv], kc[0..kv]);
        @memset(vc, 0);
        const n = self.call.rows;
        const lo = self.call.first(self.call.at(n - 1).p);
        const hi = self.call.at(0).last;
        const col = try a.alloc(f32, hi + 1 - lo);
        for (0..f.hk) |kh| for (0..f.d) |e| {
            mirrored(rng, col);
            for (col, lo..) |y, k| self.cell(vc, 0, k, kh)[e] = bfr.bits(y);
        };
    }

    pub fn run(self: *Rows) !void {
        const n = self.call.rows;
        const w = self.win();
        if (self.f.prompt) {
            try WA.promptWith(&self.fns, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.call.p0, n, w, if (self.f.image) self.dev[5] else null);
        } else try WA.decodeWith(&self.fns, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.dev[3], self.dev[4], w, self.scratch, n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rows) usize {
        return self.call.rows * self.f.h * self.f.d;
    }

    /// Every output in its float64 range (gates 1 and 3); at gate 1 some, not all, of float64's outputs non-finite.
    pub fn check(self: *Rows, at: g.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const out = try a.alloc(u16, self.count());
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.outputBytes());
        var st: ar.Stats = .{};
        if (try wc.verify(self.rig.gpu.gpa, self.call, out, &st, &self.msg)) |why_| return why_;
        if (at == .nonfinite and (st.nonfinite == 0 or st.nonfinite == out.len)) return try std.fmt.bufPrint(&self.msg, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ st.nonfinite, out.len });
        return null;
    }

    pub fn reference(self: *Rows, want: []f64, mag: []f64) !void {
        try wc.reference(self.rig.gpu.gpa, self.call, want, self.half);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    /// The range's half width, then the output's own rounding at its far end.
    pub fn bounds(self: *Rows, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, self.half) |*l, w, h| l.* = h + 0x1p-8 * (@abs(w) + h) + 0x1p-133;
    }

    /// Gate 4's sums in an accumulator and order: every weight is 1, so o is V's column sum over l's count of ones.
    pub fn simulate(self: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.arena.allocator();
        const call = self.call;
        const d = call.d;
        const grp = call.h / call.hk;
        for (0..call.rows) |r| {
            const w = call.at(r);
            const lo = call.first(w.p);
            const n = w.last + 1 - lo;
            const terms = try a.alloc(f64, n);
            const ones = try a.alloc(f64, n);
            @memset(ones, 1);
            const l = rows.sumIn(acc, order, ones);
            for (0..call.hk) |kh| for (0..d) |e| {
                for (terms, lo..) |*t, key| t.* = bfr.value(call.vc[(call.cacheRow(w.base, key) * call.hk + kh) * d + e]);
                const o = rows.sumIn(acc, order, terms) / l;
                for (0..grp) |hh| out[(r * call.h + kh * grp + hh) * d + e] = o;
            };
        }
    }
};

/// Gate 4's values: odd integers, every other one times 64 (bf16-exact), the second half cancelling the first.
fn mirrored(rng: std.Random, out: []f32) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt((1 + 2 * rng.uintLessThan(u32, 128)) * @as(u32, if (i % 2 == 1) 64 else 1));
        out[i] = v;
        out[h + i] = (if (i < 8) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}

test "every control names a kernel of contract_controls.cu, or runs the served bf16-P kernel" {
    for (cfgs) |f| if (f.ctl != .none) {
        try std.testing.expect(symbol(f.ctl) != null or f.ctl == .p16);
        try std.testing.expect(std.mem.startsWith(u8, f.name, "control: "));
    };
}
