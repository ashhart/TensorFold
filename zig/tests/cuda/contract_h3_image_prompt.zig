//! MiniMax-H3-Image's prompt CUDA kernels under the contract, each against its float64 formula.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_h3_image_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

/// The prompt kernels: each served form — Qwen (mode 0) and H3 (mode 1) rounding where both run.
pub const Which = enum {
    norm_qwen,
    norm_h3,
    heads_rotary,
    heads_fused,
    attention_causal,
    attention_fused,
    glu_qwen,
    glu_h3,
    add,
    scatter_add,
    embed,
    splice,
    linear,
    round,
};

/// norm's row sum of squares, sequential over the width.
fn rowChain(s: c.Shape) usize {
    return s.k + 2;
}
/// heads' 128 squares, sequential.
fn headChain(_: c.Shape) usize {
    return 128 + 2;
}
/// attention's sequential key dots of 128, then the weight and value sums over the keys in order.
fn attnChain(s: c.Shape) usize {
    return 128 + s.k + 2;
}
/// linear's sequential dot over k.
fn dotK(s: c.Shape) usize {
    return s.k;
}
fn none(_: c.Shape) usize {
    return 0;
}

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const att_why = "softmax weights in [0, 1] with fp32 scores: the bf16 output hides a bf16 sum's error";

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .norm_qwen => .{ .name = "cuda h3 image prompt h3p_norm qwen", .accum = .f32, .depth = rowChain, .ops = 6, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 256, .k = 256 }, .{ .rows = 77, .n = 1024, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .norm_h3 => .{ .name = "cuda h3 image prompt h3p_norm h3", .accum = .f32, .depth = rowChain, .ops = 6, .mids = 0, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 256, .k = 256 }, .{ .rows = 77, .n = 1024, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .heads_rotary => .{ .name = "cuda h3 image prompt h3p_heads rotary", .accum = .f32, .depth = headChain, .ops = 8, .mids = 3, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 8, .n = 128, .k = 128 }, .{ .rows = 77, .n = 128, .k = 128 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .heads_fused => .{ .name = "cuda h3 image prompt h3p_heads fused qkv", .accum = .f32, .depth = headChain, .ops = 8, .mids = 1, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 8, .n = 128, .k = 128 }, .{ .rows = 77, .n = 128, .k = 128 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = norm_why } },
        .attention_causal => .{ .name = "cuda h3 image prompt h3p_attention causal", .accum = .f32, .depth = attnChain, .ops = 8, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 128, .k = 8 }, .{ .rows = 77, .n = 128, .k = 77 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = att_why } },
        .attention_fused => .{ .name = "cuda h3 image prompt h3p_attention all rows", .accum = .f32, .depth = attnChain, .ops = 8, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 128, .k = 8 }, .{ .rows = 77, .n = 128, .k = 77 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = att_why } },
        .glu_qwen => .{ .name = "cuda h3 image prompt h3p_glu qwen", .accum = .none, .depth = none, .ops = 4, .mids = 1, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 256, .k = 1 }, .{ .rows = 77, .n = 1024, .k = 1 } } },
        .glu_h3 => .{ .name = "cuda h3 image prompt h3p_glu h3", .accum = .none, .depth = none, .ops = 5, .mids = 2, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 256, .k = 1 }, .{ .rows = 77, .n = 1024, .k = 1 } } },
        .add => .{ .name = "cuda h3 image prompt h3p_add", .accum = .none, .depth = none, .ops = 1, .mids = 1, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 8, .n = 1024, .k = 1 }, .{ .rows = 77, .n = 1024, .k = 1 } } },
        .scatter_add => .{ .name = "cuda h3 image prompt h3p_scatter_add", .accum = .none, .depth = none, .ops = 1, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 4, .n = 256, .k = 8 }, .{ .rows = 8, .n = 1024, .k = 16 } } },
        .embed => .{ .name = "cuda h3 image prompt h3p_embed", .accum = .none, .depth = none, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 256, .k = 32 }, .{ .rows = 77, .n = 1024, .k = 128 } } },
        .splice => .{ .name = "cuda h3 image prompt h3p_splice", .accum = .none, .depth = none, .out = .bf16, .in_place = true, .served = &.{ .{ .rows = 4, .n = 256, .k = 8 }, .{ .rows = 8, .n = 1024, .k = 16 } } },
        .linear => .{ .name = "cuda h3 image prompt h3p_linear", .accum = .f32, .depth = dotK, .ops = 0, .out = .f32, .served = &.{ .{ .rows = 4, .n = 128, .k = 256 }, .{ .rows = 8, .n = 256, .k = 1024 } }, .sums = .{ .bf16 = 0, .f16 = 0, .exact = false, .why = "a k-length sequential fp32 dot is the reference's own order" } },
        .round => .{ .name = "cuda h3 image prompt h3p_round", .accum = .none, .depth = none, .out = .bf16, .served = &.{ .{ .rows = 8, .n = 1024, .k = 1 }, .{ .rows = 77, .n = 1024, .k = 1 } } },
    };
}

pub fn symbol(w: Which) [:0]const u8 {
    return switch (w) {
        .norm_qwen, .norm_h3 => "h3p_norm",
        .heads_rotary, .heads_fused => "h3p_heads",
        .attention_causal, .attention_fused => "h3p_attention",
        .glu_qwen, .glu_h3 => "h3p_glu",
        .add => "h3p_add",
        .scatter_add => "h3p_scatter_add",
        .embed => "h3p_embed",
        .splice => "h3p_splice",
        .linear => "h3p_linear",
        .round => "h3p_round",
    };
}

pub const Prm = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    in: ref.In = .{},
    bufs: [10]u64 = undefined,
    nbufs: usize = 0,
    /// In-place starting bytes: buffer index and payload, re-uploaded before every launch.
    fresh: [2]struct { at: usize, bytes: []const u8 } = .{ .{ .at = 0, .bytes = &.{} }, .{ .at = 0, .bytes = &.{} } },
    nfresh: usize = 0,
    outs: usize = 0,
    /// Launch geometry the prepared inputs imply.
    rows: usize = 0,
    heads: usize = 0,
    kv_heads: usize = 0,
    width: usize = 0,
    q_row: usize = 0,
    k_row: usize = 0,
    fused: bool = false,

    pub fn deinit(self: *Prm) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Prm) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Prm);

    fn values(self: *Prm, n: usize, lo: f64, hi: f64, fill: g.Fill, rng: std.Random, half: bool, plant: bool) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        for (v, 0..) |*x, i| {
            const r = lo + (hi - lo) * rng.float(f64);
            x.* = switch (fill) {
                .stress => @floatFromInt(@as(i64, @intFromFloat(@round(r * 8)))),
                else => r,
            };
            if (i % 61 == 7 and fill != .stress) x.* *= 6;
            x.* = if (half) ref.bf(x.*) else @as(f32, @floatCast(x.*));
        }
        if (fill == .nonfinite and plant) {
            const at = [_]usize{ n / 3, n / 2, n - 1 };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |at_, x| v[at_] = x;
        }
        return v;
    }

    fn upload(self: *Prm, v: []const f64, half: bool) !u64 {
        return self.bytes(try self.encode(v, half));
    }

    fn encode(self: *Prm, v: []const f64, half: bool) ![]const u8 {
        const al = self.arena.allocator();
        if (half) {
            const h = try al.alloc(u16, v.len);
            for (h, v) |*o, x| o.* = c.toBf16(@floatCast(x));
            return std.mem.sliceAsBytes(h);
        }
        const f = try al.alloc(f32, v.len);
        for (f, v) |*o, x| o.* = @floatCast(x);
        return std.mem.sliceAsBytes(f);
    }

    fn bytes(self: *Prm, b: []const u8) !u64 {
        self.bufs[self.nbufs] = try self.rig.input(b);
        self.nbufs += 1;
        return self.bufs[self.nbufs - 1];
    }

    fn ints(self: *Prm, v: []const u32) !u64 {
        return self.bytes(std.mem.sliceAsBytes(v));
    }

    fn output(self: *Prm, len: usize) !u64 {
        self.bufs[self.nbufs] = try self.rig.output(len);
        self.nbufs += 1;
        return self.bufs[self.nbufs - 1];
    }

    /// An in-place buffer: an output slot whose starting bytes are restored before every launch.
    fn inPlace(self: *Prm, v: []const f64, half: bool) !u64 {
        const enc = try self.encode(v, half);
        self.bufs[self.nbufs] = try self.rig.output(enc.len);
        const ptr = self.bufs[self.nbufs];
        self.fresh[self.nfresh] = .{ .at = self.nbufs, .bytes = enc };
        self.nfresh += 1;
        self.nbufs += 1;
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = enc.len };
        try b.upload(0, enc);
        return ptr;
    }

    pub fn prepare(self: *Prm, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.nbufs = 0;
        self.nfresh = 0;
        self.shape = s;
        self.fused = false;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const al = self.arena.allocator();
        switch (self.which) {
            .norm_qwen, .norm_h3 => {
                if (s.n != s.k or s.n > 4096) return false;
                const mode: u32 = if (self.which == .norm_qwen) 0 else 1;
                const x = try self.values(s.rows * s.n, -2, 2, fill, rng, true, true);
                const w = try self.values(s.n, 0.5, 1.5, .random, rng, false, false);
                self.in = .{ .a = x, .b = w, .rows = s.rows, .n = s.n, .k = s.n, .mode = mode, .eps = 1e-5 };
                self.width = s.n;
                _ = try self.upload(x, true);
                _ = try self.upload(w, false);
                _ = try self.output(s.rows * s.n * 2);
                self.outs = s.rows * s.n;
            },
            .heads_rotary => {
                // The text model: q [rows, 8*128], k [rows, 2*128], rotated by cos/sin [rows, 64]; mode 0.
                const heads = 8;
                const kv = 2;
                const q = try self.values(s.rows * heads * 128, -2, 2, fill, rng, true, true);
                const k = try self.values(s.rows * kv * 128, -2, 2, .random, rng, true, false);
                const wq = try self.values(128, 0.5, 1.5, .random, rng, false, false);
                const wk = try self.values(128, 0.5, 1.5, .random, rng, false, false);
                // The served tables: row r's angle for channel pair i is r / 10000^(i/64), so sine spans [-1, 1].
                const cs = try al.alloc(f64, s.rows * 64);
                const sn = try al.alloc(f64, s.rows * 64);
                for (cs, sn, 0..) |*co, *si, at| {
                    const angle: f32 = @floatCast(@as(f64, @floatFromInt(at / 64)) / std.math.pow(f64, 10000, @as(f64, @floatFromInt(at % 64)) / 64));
                    co.* = @cos(angle);
                    si.* = @sin(angle);
                }
                self.in = .{ .a = q, .b = k, .d = wq, .e = wk, .f = cs, .g = sn, .rows = s.rows, .n = s.n, .k = s.k, .heads = heads, .kv_heads = kv, .head = 128, .q_row = heads * 128, .k_row = kv * 128, .mode = 0, .eps = 1e-5, .rotary = true };
                self.rows = s.rows;
                self.heads = heads;
                self.kv_heads = kv;
                self.q_row = heads * 128;
                self.k_row = kv * 128;
                _ = try self.inPlace(q, true);
                _ = try self.inPlace(k, true);
                _ = try self.upload(wq, false);
                _ = try self.upload(wk, false);
                _ = try self.upload(cs, false);
                _ = try self.upload(sn, false);
                self.outs = s.rows * (heads + kv) * 128;
            },
            .heads_fused => {
                // The refiner: qkv rows of 3 * 128 (one q, k, v triple), no rotary; mode 1.
                const heads = 1;
                const kv = 1;
                const q_row = 3 * 128;
                const qkv = try self.values(s.rows * q_row, -2, 2, fill, rng, true, true);
                const wq = try self.values(128, 0.5, 1.5, .random, rng, false, false);
                const wk = try self.values(128, 0.5, 1.5, .random, rng, false, false);
                self.in = .{ .a = qkv, .d = wq, .e = wk, .rows = s.rows, .n = s.n, .k = s.k, .heads = heads, .kv_heads = kv, .head = 128, .stride = 3 * 128, .q_row = q_row, .k_row = q_row, .mode = 1, .eps = 1e-5 };
                self.rows = s.rows;
                self.heads = heads;
                self.kv_heads = kv;
                self.q_row = q_row;
                self.k_row = q_row;
                self.fused = true;
                _ = try self.inPlace(qkv, true);
                _ = try self.upload(wq, false);
                _ = try self.upload(wk, false);
                self.outs = s.rows * q_row;
            },
            .attention_causal, .attention_fused => {
                // q [k, 4*128], k/v [k, 2*128]: s.k rows; causal for the text model, all rows for the refiner.
                const n = s.k;
                const heads = 4;
                const kv = 2;
                const causal = self.which == .attention_causal;
                const q = try self.values(n * heads * 128, -2, 2, fill, rng, true, true);
                const k = try self.values(n * kv * 128, -2, 2, .random, rng, true, false);
                const v = try self.values(n * kv * 128, 0, 2, .random, rng, true, false);
                self.in = .{ .a = q, .b = k, .d = v, .rows = n, .n = s.n, .k = n, .heads = heads, .kv_heads = kv, .head = 128, .q_row = heads * 128, .k_row = kv * 128, .scale = 1.0 / @sqrt(128.0), .causal = causal };
                self.rows = n;
                self.heads = heads;
                self.kv_heads = kv;
                self.q_row = heads * 128;
                self.k_row = kv * 128;
                _ = try self.upload(q, true);
                _ = try self.upload(k, true);
                _ = try self.upload(v, true);
                _ = try self.output(n * heads * 128 * 2);
                self.outs = n * heads * 128;
            },
            .glu_qwen, .glu_h3 => {
                const mode: u32 = if (self.which == .glu_qwen) 0 else 1;
                const gg = try self.values(s.rows * s.n, -4, 4, fill, rng, true, true);
                const v = try self.values(s.rows * s.n, -2, 2, .random, rng, true, false);
                self.in = .{ .a = gg, .b = v, .rows = s.rows, .n = s.n, .mode = mode };
                self.width = s.n;
                _ = try self.upload(gg, true);
                _ = try self.upload(v, true);
                _ = try self.output(s.rows * s.n * 2);
                self.outs = s.rows * s.n;
            },
            .add => {
                const x = try self.values(s.rows * s.n, -2, 2, fill, rng, true, true);
                const y = try self.values(s.rows * s.n, -2, 2, .random, rng, false, false);
                self.in = .{ .a = x, .b = y, .rows = s.rows, .n = s.n };
                _ = try self.inPlace(x, true);
                _ = try self.upload(y, false);
                self.outs = s.rows * s.n;
            },
            .scatter_add => {
                // s.rows scattered rows of width s.n into an x buffer of s.k rows.
                if (s.k < s.rows or s.rows == 0) return false;
                const x = try self.values(s.k * s.n, -2, 2, fill, rng, true, true);
                const y = try self.values(s.rows * s.n, -2, 2, .random, rng, true, false);
                const idx = try al.alloc(u32, s.rows);
                for (idx, 0..) |*v, i| v.* = @intCast((i * s.k) / s.rows);
                rng.shuffle(u32, idx);
                self.in = .{ .a = x, .b = y, .ints = idx, .rows = s.k, .n = s.n, .k = s.rows };
                _ = try self.inPlace(x, true);
                _ = try self.upload(y, true);
                _ = try self.ints(idx);
                self.outs = s.k * s.n;
            },
            .embed => {
                // s.rows gathered rows of width s.n from a table of s.k rows.
                const table = try self.values(s.k * s.n, -2, 2, fill, rng, true, true);
                const ids = try al.alloc(u32, s.rows);
                for (ids, 0..) |*v, i| v.* = @intCast((i * s.k) / s.rows);
                rng.shuffle(u32, ids);
                self.in = .{ .a = table, .ints = ids, .rows = s.rows, .n = s.n, .k = s.k };
                _ = try self.upload(table, true);
                _ = try self.ints(ids);
                _ = try self.output(s.rows * s.n * 2);
                self.outs = s.rows * s.n;
            },
            .splice => {
                // s.rows replacement rows spliced into an x buffer of s.k rows.
                if (s.k < s.rows or s.rows == 0) return false;
                const x = try self.values(s.k * s.n, -2, 2, fill, rng, true, true);
                const rows_ = try self.values(s.rows * s.n, -2, 2, .random, rng, true, false);
                const pads = try al.alloc(u32, s.rows);
                for (pads, 0..) |*v, i| v.* = @intCast((i * s.k) / s.rows);
                rng.shuffle(u32, pads);
                self.in = .{ .a = x, .b = rows_, .ints = pads, .rows = s.k, .n = s.n, .k = s.rows };
                _ = try self.inPlace(x, true);
                _ = try self.upload(rows_, true);
                _ = try self.ints(pads);
                self.outs = s.k * s.n;
            },
            .linear => {
                if (s.k > 8192) return false;
                const x = try self.values(s.rows * s.k, -1, 1, fill, rng, true, true);
                const w = try self.values(s.n * s.k, -1, 1, .random, rng, true, false);
                self.in = .{ .a = x, .b = w, .rows = s.rows, .n = s.n, .k = s.k };
                _ = try self.upload(x, true);
                _ = try self.upload(w, true);
                _ = try self.output(s.rows * s.n * 4);
                self.outs = s.rows * s.n;
            },
            .round => {
                const x = try self.values(s.rows * s.n, -8, 8, fill, rng, false, true);
                self.in = .{ .a = x, .rows = s.rows, .n = s.n };
                _ = try self.upload(x, false);
                _ = try self.output(s.rows * s.n * 2);
                self.outs = s.rows * s.n;
            },
        }
        return true;
    }

    pub fn run(self: *Prm) !void {
        for (self.fresh[0..self.nfresh]) |f| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.bufs[f.at], .len = f.bytes.len };
            try b.upload(0, f.bytes);
        }
        const s = self.shape;
        const b = self.bufs;
        var a: cuda.Args = .{};
        const ci = struct {
            fn of(n: usize) c_int {
                return @intCast(n);
            }
        }.of;
        switch (self.which) {
            .norm_qwen, .norm_h3 => {
                for (b[0..3]) |p| a.add(p);
                a.add(ci(s.n));
                a.add(@as(c_int, @intCast(self.in.mode)));
                a.add(@as(f32, 1e-5));
                try self.go(.{ .x = @intCast(s.rows) }, &a);
            },
            .heads_rotary => {
                for (b[0..6]) |p| a.add(p);
                for ([_]usize{ self.heads, self.kv_heads, self.q_row, self.k_row, 128, 0, 1 }) |v| a.add(ci(v));
                a.add(@as(f32, 1e-5));
                try self.go(.{ .x = @intCast(self.rows), .y = @intCast((self.heads + self.kv_heads + 7) / 8) }, &a);
            },
            .heads_fused => {
                // q and k point into the one qkv buffer (k at +128 elements).
                for ([_]u64{ b[0], b[0] + 128 * 2, b[1], b[2], 0, 0 }) |p| a.add(p);
                for ([_]usize{ self.heads, self.kv_heads, self.q_row, self.k_row, 3 * 128, 1, 0 }) |v| a.add(ci(v));
                a.add(@as(f32, 1e-5));
                try self.go(.{ .x = @intCast(self.rows), .y = @intCast((self.heads + self.kv_heads + 7) / 8) }, &a);
            },
            .attention_causal, .attention_fused => {
                for (b[0..4]) |p| a.add(p);
                for ([_]usize{ self.rows, self.heads, self.kv_heads, self.q_row, self.k_row, self.k_row, 128, @intFromBool(self.in.causal) }) |v| a.add(ci(v));
                a.add(@as(f32, 1.0 / @sqrt(128.0)));
                try self.go(.{ .x = @intCast(self.rows), .y = @intCast((self.heads + 7) / 8) }, &a);
            },
            .glu_qwen, .glu_h3 => {
                for (b[0..3]) |p| a.add(p);
                for ([_]usize{ self.width, self.width, self.width, self.in.mode }) |v| a.add(ci(v));
                try self.go(.{ .x = @intCast((self.width + 255) / 256), .y = @intCast(s.rows) }, &a);
            },
            .add => {
                for (b[0..2]) |p| a.add(p);
                a.add(ci(s.rows * s.n));
                try self.go(.{ .x = @intCast((s.rows * s.n + 255) / 256) }, &a);
            },
            .scatter_add => {
                for (b[0..3]) |p| a.add(p);
                a.add(ci(s.n));
                try self.go(.{ .x = @intCast((s.n + 255) / 256), .y = @intCast(s.rows) }, &a);
            },
            .embed => {
                for (b[0..3]) |p| a.add(p);
                a.add(ci(s.n));
                try self.go(.{ .x = @intCast((s.n + 255) / 256), .y = @intCast(s.rows) }, &a);
            },
            .splice => {
                for (b[0..3]) |p| a.add(p);
                a.add(ci(s.n));
                try self.go(.{ .x = @intCast((s.n + 255) / 256), .y = @intCast(s.rows) }, &a);
            },
            .linear => {
                for (b[0..3]) |p| a.add(p);
                for ([_]usize{ s.rows, s.n, s.k }) |v| a.add(ci(v));
                try self.go(.{ .x = @intCast((s.rows * s.n + 255) / 256) }, &a);
            },
            .round => {
                for (b[0..2]) |p| a.add(p);
                a.add(ci(s.rows * s.n));
                try self.go(.{ .x = @intCast((s.rows * s.n + 255) / 256) }, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    fn go(self: *Prm, grid: cuda.Dim3, args: *cuda.Args) !void {
        try cuda.launch.launch(self.f, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, args);
    }

    pub fn count(self: *Prm) usize {
        return self.outs;
    }

    fn compute(self: *Prm, s_: ref.Sum, want: []f64, mag: []f64) !void {
        switch (self.which) {
            .norm_qwen, .norm_h3 => ref.norm(self.in, s_, want, mag),
            .heads_rotary, .heads_fused => ref.pheads(self.in, s_, want, mag, self.fused),
            .attention_causal, .attention_fused => ref.pattention(self.in, s_, want, mag),
            .glu_qwen, .glu_h3 => ref.glu(self.in, want, mag),
            .add => ref.add(self.in, want, mag),
            .scatter_add => ref.scatterAdd(self.in, want, mag),
            .embed => ref.gather(self.in, want, mag),
            .splice => ref.splice(self.in, want, mag),
            .linear => ref.linear(self.in, s_, want, mag),
            .round => ref.round(self.in, want, mag),
        }
    }

    pub fn reference(self: *Prm, want: []f64, mag: []f64) !void {
        return self.compute(.{}, want, mag);
    }

    pub fn simulate(self: *Prm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        return self.compute(.{ .acc = acc, .order = order }, out, mag);
    }
};
