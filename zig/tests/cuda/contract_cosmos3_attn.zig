//! cosmos3.cu's attention under the contract: prompt rows (tf_cos3_attn) and decode (tf_cos3_attn1) in float64's range.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_cosmos3_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const ar = core.attention_range;
const QH = ref.QH;
const KVH = ref.KVH;
const HD = ref.HD;

/// A registration: prompt rows or decode, or a control from contract_controls.cu planting one fault.
pub const Entry = struct { name: []const u8, prompt: bool, ctl: ?[:0]const u8 = null };

pub const entries = [_]Entry{
    .{ .name = "cuda cosmos3 tf_cos3_attn (causal GQA 16/8 at dim 128, prompt rows over a stream's cache)", .prompt = true },
    .{ .name = "cuda cosmos3 tf_cos3_attn1 (decode, 8 warps' online softmaxes merged)", .prompt = false },
    .{ .name = "control: cosmos3 prompt rows seeing the next row's key", .prompt = true, .ctl = "tf_ctl_cos3_attn_late" },
    .{ .name = "control: cosmos3 prompt rows not rescaling earlier blocks to a new maximum", .prompt = true, .ctl = "tf_ctl_cos3_attn_stale" },
    .{ .name = "control: cosmos3 decode merge without exp(m_w - M)", .prompt = false, .ctl = "tf_ctl_cos3_attn1_merge" },
    .{ .name = "control: cosmos3 decode leaving out the row's own key", .prompt = false, .ctl = "tf_ctl_cos3_attn1_short" },
};

/// The longest chain: a row's keys (an fma each into o), and an online-softmax step a 32-key block.
fn depth(s: c.Shape) usize {
    return s.k + s.rows + (s.k + s.rows) / 32 * 6 + 16;
}

const why = "all keys alike make every weight exactly 1: o is a plain sum of V's integers over n, exact in fp32 (any order); fp16 sums overflow, bf16 ones lose its low bits";

/// Shape: query rows of one segment, the 16 query heads, and the keys cached before it.
pub fn spec(e: Entry) c.Spec {
    const prompt = &[_]c.Shape{ .{ .rows = 8, .n = QH, .k = 0 }, .{ .rows = 160, .n = QH, .k = 0 }, .{ .rows = 100, .n = QH, .k = 1200 } };
    const decode = &[_]c.Shape{ .{ .rows = 1, .n = QH, .k = 40 }, .{ .rows = 1, .n = QH, .k = 31 }, .{ .rows = 1, .n = QH, .k = 32 }, .{ .rows = 1, .n = QH, .k = 700 }, .{ .rows = 1, .n = QH, .k = 3000 } };
    const tiles: [3]c.Tile = if (e.prompt) .{ .{ .size = 8 }, .{}, .{ .size = 32, .min = 0 } } else .{ .{}, .{}, .{ .size = 256, .min = 0 } };
    return .{ .name = e.name, .accum = .f32, .depth = depth, .ops = 4, .out = .bf16, .tiles = tiles, .served = if (e.prompt) prompt else decode, .sums = .{ .f16 = 100, .bf16 = 0, .why = why } };
}

pub const Attn = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    cap: usize = 0,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    dev: [6]u64 = @splat(0),
    msg: [220]u8 = undefined,

    pub fn deinit(self: *Attn) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.ctl != null) .shapes else null };
    }

    const vt = g.hooks(Attn);
    const scale: f32 = 1.0 / @sqrt(@as(f32, HD));

    /// Keys row r sees: every cached one, and the segment's own through its own.
    fn seen(self: *const Attn, r: usize) usize {
        return self.shape.k + r + 1;
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != QH or shape.rows == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        self.fill = fill;
        const n = shape.rows;
        const total = shape.k + n;
        self.cap = total + 5;
        self.q = try a.alloc(u16, n * QH * HD);
        self.kc = try a.alloc(u16, KVH * self.cap * HD);
        self.vc = try a.alloc(u16, self.kc.len);
        for (self.q) |*x| x.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.kc, self.vc) |*k, *v| {
            k.* = c.toBf16(rng.float(f32) * 4 - 2);
            v.* = c.toBf16(rng.float(f32) * 2 - 1);
        }
        if (fill == .random) { // a few keys equal to a query: sharp weights, and maxima that move between blocks
            var j: usize = 0;
            while (j < total) : (j += 37) for (0..KVH) |kh| {
                const r = j % n;
                @memcpy(self.kc[(kh * self.cap + j) * HD ..][0..HD], self.q[(r * QH + 2 * kh) * HD ..][0..HD]);
            };
        }
        if (fill == .stress) { // every key alike; values: integers whose first half the second half mirrors (8 nudged)
            const half = shape.k / 2;
            const key = try a.dupe(u16, self.kc[0..HD]);
            for (0..KVH) |kh| for (0..total) |j| {
                @memcpy(self.kc[(kh * self.cap + j) * HD ..][0..HD], key);
                for (0..HD) |e| {
                    const at = (kh * self.cap + j) * HD + e;
                    const v: f32 = if (j < half) @floatFromInt(128 + (at * 2654435761) % 129) else if (j < 2 * half) blk: {
                        const m = (kh * self.cap + j - half) * HD + e;
                        break :blk -@as(f32, @floatFromInt(128 + (m * 2654435761) % 129)) + @as(f32, if (j - half < 8) 1 else 0);
                    } else 0;
                    self.vc[at] = c.toBf16(v);
                }
            };
        }
        if (fill == .nonfinite) { // NaN in a query and a key, -Inf and +Inf in two values
            const nan = c.toBf16(std.math.nan(f32));
            self.q[3] = nan;
            self.kc[(1 * self.cap + 1) * HD + 4] = nan;
            self.vc[(3 * self.cap + 2) * HD + 7] = c.toBf16(-std.math.inf(f32));
            self.vc[(5 * self.cap + 0) * HD + 9] = c.toBf16(std.math.inf(f32));
        }
        const slots = try a.alloc(i32, n);
        for (slots, 0..) |*s, r| s.* = @intCast(shape.k + r);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.kc));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.vc));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(slots));
        self.dev[4] = try self.rig.output(n * QH * HD * 2);
        self.dev[5] = try self.rig.output(n * QH * HD / 64 * 4);
        return true;
    }

    /// Prompt rows in one launch; decode one launch a row, row r a stream whose cache holds k + r keys before it.
    pub fn run(self: *Attn) !void {
        const v = self.dev;
        const n = self.shape.rows;
        const launches = if (self.entry.prompt) 1 else n;
        for (0..launches) |r| {
            const row: u64 = if (self.entry.prompt) 0 else r;
            var a: cuda.Args = .{};
            a.add(v[0] + row * QH * HD * 2);
            a.add(v[1]);
            a.add(v[2]);
            a.add(v[4] + row * QH * HD * 2);
            a.add(v[3] + row * 4);
            if (self.entry.prompt) a.add(@as(c_int, @intCast(n)));
            a.add(@as(c_int, @intCast(self.cap)));
            a.add(scale);
            a.add(v[5] + row * QH * HD / 64 * 4);
            const blocks: u32 = if (self.entry.prompt) @intCast((n + 7) / 8) else 1;
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = QH, .y = blocks, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        return self.shape.rows * QH * HD;
    }

    /// One (row, head)'s float64 point and fp32 half-width over its keys.
    fn head(self: *Attn, r: usize, h: usize, point: []f64, half: []f64) !void {
        const a = self.arena.allocator();
        const n = self.seen(r);
        const kh = h / (QH / KVH);
        const q = try a.alloc(f64, HD);
        const k = try a.alloc(f64, n * HD);
        const v = try a.alloc(f64, n * HD);
        const w = try a.alloc(f64, 2 * n);
        defer {
            a.free(w);
            a.free(v);
            a.free(k);
            a.free(q);
        }
        for (q, self.q[(r * QH + h) * HD ..][0..HD]) |*x, b| x.* = ref.val(b);
        for (k, v, self.kc[kh * self.cap * HD ..][0 .. n * HD], self.vc[kh * self.cap * HD ..][0 .. n * HD]) |*x, *y, kb, vb| {
            x.* = ref.val(kb);
            y.* = ref.val(vb);
        }
        const form = if (self.entry.prompt) ref.promptForm(n) else ref.decodeForm(n);
        ar.head(q, k, v, n, HD, scale, form, w[0..n], w[n..], point, half);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        var half: [HD]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..QH) |h| try self.head(r, h, want[(r * QH + h) * HD ..][0..HD], &half);
        @memset(mag, 0);
    }

    /// Gate 4's outputs with o's sums in `acc` and `order`: every weight is exactly 1 there, so o = sum v / n.
    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.fill != .stress) return error.NoSums;
        const t = try self.arena.allocator().alloc(f64, self.seen(self.shape.rows - 1));
        for (0..self.shape.rows) |r| for (0..QH) |h| for (0..HD) |e| {
            const n = self.seen(r);
            const kh = h / (QH / KVH);
            for (t[0..n], 0..) |*x, j| x.* = ref.val(self.vc[(kh * self.cap + j) * HD + e]);
            out[(r * QH + h) * HD + e] = rows.sumIn(acc, order, t[0..n]) / @as(f64, @floatFromInt(n));
        };
    }

    fn fail(self: *Attn, comptime fmt: []const u8, args: anytype) ?[]const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args) catch "check failed";
    }

    /// Every output within float64's range over its fp32 spread, and its group sums the fp32 chain of the outputs.
    pub fn check(self: *Attn, at: g.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const n = self.count();
        const out = try a.alloc(u16, n);
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.bytes(self.dev[4], n * 2));
        const xs = try a.alloc(f32, n / 64);
        @memcpy(std.mem.sliceAsBytes(xs), try self.rig.bytes(self.dev[5], n / 64 * 4));
        var point: [HD]f64 = undefined;
        var half: [HD]f64 = undefined;
        var st: ar.Stats = .{};
        for (0..self.shape.rows) |r| for (0..QH) |h| {
            try self.head(r, h, &point, &half);
            for (out[(r * QH + h) * HD ..][0..HD], point, half, 0..) |got, p, hw, e|
                if (!ref.within(p, hw, got, &st)) return self.fail("row {d} head {d} dim {d}: {x} outside {e:.6} +- {e:.3}", .{ r, h, e, got, p, hw });
        };
        var v: [64]f32 = undefined;
        for (xs, 0..) |got, j| {
            for (&v, out[64 * j ..][0..64]) |*o, b| o.* = ref.val(b);
            const want = ref.group64(&v);
            if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(want)) and !(std.math.isNan(got) and std.math.isNan(want)))
                return self.fail("group sum {d}: {e} where the fp32 chain of the outputs gives {e}", .{ j, got, want });
        }
        if (at == .nonfinite and st.nonfinite == 0) return self.fail("the planted values reach no output: no proof", .{});
        return null;
    }
};

pub const Group = struct {
    list: [entries.len]Attn = undefined,
    made: usize = 0,

    pub fn add(self: *Group, gpu: check.Gpu, s: cuda.Stream, mod: cuda.Module, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        for (entries) |e| {
            const f = if (e.ctl) |sym| try ctl.function(sym) else try mod.function(if (e.prompt) "tf_cos3_attn" else "tf_cos3_attn1");
            self.list[self.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = f, .arena = .init(gpu.gpa) };
            self.made += 1;
        }
        for (self.list[0..self.made]) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Group) void {
        for (self.list[0..self.made]) |*x| x.deinit();
    }
};
