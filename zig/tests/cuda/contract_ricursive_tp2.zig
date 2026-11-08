//! Ripple's two-rank glue (ricursive_tp2.cu) under the contract, each kernel with a planted control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig"); // the family-neutral Gated and Checked wrappers, fills and intervals
const rows = @import("contract_ricursive_tp2_rows.zig"); // gate, swiglu and norm_add against float64

const c = core.contract.spec;
const g = core.contract.gates;
const Function = cuda.Function;

const hidden = 5120;
const embed_scale: f32 = 71.55417527999327;
const vocab = 100352;
pub const elementwise: c.Tile = .{ .size = 256, .step = 1, .min = 1 };

/// The forward's row counts at width n (k: the reduction): decode rows 1 to 16, a tail and a 128-row prompt chunk.
pub fn rowsAt(comptime n: usize, comptime k: usize) [5]c.Shape {
    var out: [5]c.Shape = undefined;
    for ([_]usize{ 16, 1, 3, 77, 128 }, &out) |r, *s| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

const hidden_rows = rowsAt(hidden, 1);
const head_rows = rowsAt(4 * 10 * 128, 1); // a rank's 4 key heads' 10 query heads a row

pub fn halves(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]f16 {
    const v = try a.alloc(f16, n);
    for (v) |*x| x.* = @floatCast(lo + rng.float(f32) * (hi - lo));
    return v;
}

pub fn launch(f: Function, s: cuda.Stream, blocks: usize, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn grid(n: usize) usize {
    return (n + 255) / 256;
}

fn sameBits(comptime T: type, have: T, want: T) bool {
    if (std.math.isNan(have) and std.math.isNan(want)) return true;
    return @as(std.meta.Int(.unsigned, @bitSizeOf(T)), @bitCast(have)) == @as(std.meta.Int(.unsigned, @bitSizeOf(T)), @bitCast(want));
}

fn read(comptime T: type, bytes: []const u8, i: usize) T {
    const U = std.meta.Int(.unsigned, @bitSizeOf(T));
    return @bitCast(std.mem.readInt(U, bytes[i * @sizeOf(T) ..][0..@sizeOf(T)], .little));
}

fn toBf(x: f32) f32 {
    return @floatCast(ref.bf(c.toBf16(x)));
}

/// x = bf16(e * scale) in fp32 words and seed = fp16(x), each id's row: the bits fp32 and the two roundings give.
pub const Embed = struct {
    f: Function,
    ctl: bool = false, // control: the product left unrounded
    shape: c.Shape = undefined,
    table: []u16 = &.{},
    ids: []i32 = &.{},
    dev: [2]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Embed) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_embed without bf16(e * scale)" else "cuda ricursive_tp2 tf_rtp2_embed (x fp32 of bf16(e * scale), seed fp16)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "a table's NaN and infinities times the scale, as fp32 gives them" }, .tiles = .{ .{}, elementwise, .{} }, .served = &hidden_rows };
    }

    pub fn control(self: *const Embed) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Embed, a: std.mem.Allocator) void {
        a.free(self.table);
        a.free(self.ids);
        self.table = &.{};
        self.ids = &.{};
    }

    pub fn prepare(self: *Embed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const rows_ = @min(shape.rows, 64); // distinct table rows: ids repeat past 64
        self.ids = try a.alloc(i32, shape.rows);
        for (self.ids, 0..) |*id, r| id.* = @intCast(if (r == 0) vocab - 1 else r % rows_);
        self.table = try ref.bfs(a, rng, vocab * shape.n, -1, 1);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |x, r| if (r < shape.rows) {
            self.table[@as(usize, @intCast(self.ids[r])) * shape.n + (7 * r + 1) % shape.n] = c.toBf16(x);
        };
        self.dev = .{ try ref.input(rig, self.ids), try ref.input(rig, self.table) };
        self.out = .{ try rig.output(shape.rows * shape.n * 4), try rig.output(shape.rows * shape.n * 2) };
        return true;
    }

    pub fn run(self: *Embed, rig: *rig_.Rig) !void {
        try launch(self.f, rig.s, self.shape.rows, .{ self.dev[0], self.dev[1], self.out[0], self.out[1], @as(i32, @intCast(self.shape.n)), embed_scale });
    }

    pub fn count(self: *const Embed) usize {
        return self.shape.rows * self.shape.n * 6;
    }

    pub fn check(self: *const Embed, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const half = got[self.shape.rows * n * 4 ..];
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const e: f32 = @floatCast(ref.bf(self.table[@as(usize, @intCast(self.ids[r])) * n + i]));
            const x = toBf(e * embed_scale);
            const hx = read(f32, got, r * n + i);
            const hs = read(f16, half, r * n + i);
            if (!sameBits(f32, hx, x) or !sameBits(f16, hs, @floatCast(x))) return try std.fmt.bufPrint(msg, "row {d} col {d}: x {e} seed {e} where {e} is due", .{ r, i, hx, hs, x });
        };
        return null;
    }
};

/// out = fp16(lerp(seed, y, w)) in fp32 as torch.lerp forms it: start-anchored below w = 0.5, end-anchored from it.
pub const Lerp = struct {
    f: Function,
    w: f32,
    ctl: bool = false, // control: start-anchored at every w
    shape: c.Shape = undefined,
    seed: []f16 = &.{},
    y: []f16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Lerp) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_lerp start-anchored past w = 0.5" else if (self.w < 0.5) "cuda ricursive_tp2 tf_rtp2_lerp (w below one half)" else "cuda ricursive_tp2 tf_rtp2_lerp (w from one half)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "NaN and infinities through fp32's sub, mul and add" }, .tiles = .{ .{}, elementwise, .{} }, .served = &hidden_rows };
    }

    pub fn control(self: *const Lerp) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Lerp, a: std.mem.Allocator) void {
        a.free(self.seed);
        a.free(self.y);
        self.seed = &.{};
        self.y = &.{};
    }

    pub fn prepare(self: *Lerp, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.seed = try halves(a, rng, n, -8, 8);
        self.y = try halves(a, rng, n, -8, 8);
        if (fill == .nonfinite) for ([_]f16{ std.math.nan(f16), std.math.inf(f16), -std.math.inf(f16) }, 0..) |x, i| if (3 * i + 2 < n) {
            self.seed[3 * i] = x;
            self.y[3 * i + 2] = x;
        };
        self.dev = .{ try ref.input(rig, self.seed), try ref.input(rig, self.y) };
        self.out = try rig.output(n * 2);
        return true;
    }

    pub fn run(self: *Lerp, rig: *rig_.Rig) !void {
        const n = self.shape.rows * self.shape.n;
        try launch(self.f, rig.s, grid(n), .{ self.dev[0], self.dev[1], self.out, @as(i64, @intCast(n)), self.w });
    }

    pub fn count(self: *const Lerp) usize {
        return self.shape.rows * self.shape.n * 2;
    }

    pub fn check(self: *const Lerp, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        for (self.seed, self.y, 0..) |sh, yh, i| {
            const s: f32 = sh;
            const e: f32 = yh;
            const d = e - s;
            const v: f16 = @floatCast(if (self.w < 0.5) s + self.w * d else e - d * (1.0 - self.w));
            const have = read(f16, got, i);
            if (!sameBits(f16, have, v)) return try std.fmt.bufPrint(msg, "value {d}: {e} where {e} is due", .{ i, have, v });
        }
        return null;
    }
};

/// out = parts[0] + parts[1] in rank order, fp32: the bits one fp32 add gives.
pub const RankSum = struct {
    f: Function,
    ctl: bool = false, // control: rank 1's partial rounded to bf16 first
    shape: c.Shape = undefined,
    parts: []f32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const RankSum) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_rank_sum, rank 1's partial in bf16" else "cuda ricursive_tp2 tf_rtp2_rank_sum (two ranks' fp32 partials, added once in rank order)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "NaN and infinities through fp32's add" }, .tiles = .{ .{}, elementwise, .{} }, .served = &hidden_rows };
    }

    pub fn control(self: *const RankSum) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *RankSum, a: std.mem.Allocator) void {
        a.free(self.parts);
        self.parts = &.{};
    }

    pub fn prepare(self: *RankSum, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.parts = try ref.f32s(a, rng, 2 * n, -64, 64);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |x, i| if (i < n) {
            self.parts[n + i] = x;
        };
        self.dev = try ref.input(rig, self.parts);
        self.out = try rig.output(n * 4);
        return true;
    }

    pub fn run(self: *RankSum, rig: *rig_.Rig) !void {
        const n = self.shape.rows * self.shape.n;
        try launch(self.f, rig.s, grid(n), .{ self.dev, self.out, @as(i64, @intCast(n)), @as(i32, 2) });
    }

    pub fn count(self: *const RankSum) usize {
        return self.shape.rows * self.shape.n * 4;
    }

    pub fn check(self: *const RankSum, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.rows * self.shape.n;
        for (0..n) |i| {
            const want = self.parts[i] + self.parts[n + i];
            const have = read(f32, got, i);
            if (!sameBits(f32, have, want)) return try std.fmt.bufPrint(msg, "value {d}: {e} where {e} is due", .{ i, have, want });
        }
        return null;
    }
};

/// The first g heads of each group's run, 16-byte moves: bf16 [groups, from, 128] -> [groups, to, 128], words as given.
pub const Heads = struct {
    f: Function,
    pad: bool, // in: 10 -> 12 heads a group (pads untouched); out: 12 -> 10
    ctl: bool = false, // control: each head read from its neighbour
    shape: c.Shape = undefined,
    src: []u32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    const g_ = 10;

    fn widths(self: *const Heads) [2]usize {
        return if (self.pad) .{ 10, 12 } else .{ 12, 10 };
    }

    pub fn spec(self: *const Heads) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_heads, each head read from its neighbour" else if (self.pad) "cuda ricursive_tp2 tf_rtp2_heads (10 query heads a key head into 12)" else "cuda ricursive_tp2 tf_rtp2_heads (12 padded heads back to 10)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "every word's bits as given, NaN patterns included; pad heads untouched" }, .served = &head_rows };
    }

    pub fn control(self: *const Heads) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Heads, a: std.mem.Allocator) void {
        a.free(self.src);
        self.src = &.{};
    }

    /// shape.n is a row's 10 heads a key head over its key heads; the groups are rows x key heads.
    fn groups(self: *const Heads) usize {
        return self.shape.rows * self.shape.n / (g_ * 128);
    }

    pub fn prepare(self: *Heads, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % (g_ * 128) != 0) return false;
        self.free(a);
        self.shape = shape;
        const w = self.widths();
        self.src = try a.alloc(u32, self.groups() * w[0] * 64);
        for (self.src) |*v| v.* = rng.int(u32);
        if (fill == .nonfinite) self.src[1] = @bitCast(ref.nan);
        self.dev = try ref.input(rig, self.src);
        self.out = try rig.output(self.groups() * w[1] * 256);
        return true;
    }

    pub fn run(self: *Heads, rig: *rig_.Rig) !void {
        const w = self.widths();
        const n = self.groups() * g_ * 16;
        try launch(self.f, rig.s, grid(n), .{ self.dev, self.out, @as(i64, @intCast(n)), @as(i32, g_), @as(i32, @intCast(w[0])), @as(i32, @intCast(w[1])) });
    }

    pub fn count(self: *const Heads) usize {
        return self.groups() * self.widths()[1] * 256;
    }

    pub fn check(self: *const Heads, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const w = self.widths();
        for (0..self.groups()) |grp| for (0..w[1]) |j| for (0..64) |i| {
            const have = read(u32, got, (grp * w[1] + j) * 64 + i);
            const want: u32 = if (j < g_) self.src[(grp * w[0] + j) * 64 + i] else 0xffff_ffff;
            if (have != want) return try std.fmt.bufPrint(msg, "group {d} head {d} word {d}: {x} where {x} is due", .{ grp, j, i, have, want });
        };
        return null;
    }
};

/// Rows' fp32 residuals into a tap ring as bf16 (nearest even) at ring[slot][pos % 256][tap]; other rows untouched.
pub const Taps = struct {
    f: Function,
    ctl: bool = false, // control: rounded toward zero
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    pos: []i32 = &.{},
    slot: []i32 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    const ring = 256;
    const ntaps = 3;
    const tap = 1;
    const slots = 2;

    pub fn spec(self: *const Taps) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_taps, bf16 toward zero" else "cuda ricursive_tp2 tf_rtp2_taps (fp32 rows into a drafter's bf16 tap ring)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "NaN and infinities as bf16 gives them; ring rows no call names stay untouched" }, .tiles = .{ .{}, elementwise, .{} }, .served = &hidden_rows };
    }

    pub fn control(self: *const Taps) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Taps, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.pos);
        a.free(self.slot);
        self.x = &.{};
        self.pos = &.{};
        self.slot = &.{};
    }

    pub fn prepare(self: *Taps, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.rows > ring) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.f32s(a, rng, shape.rows * shape.n, -40, 40);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |v, i| if (i < self.x.len) {
            self.x[i] = v;
        };
        self.pos = try a.alloc(i32, shape.rows);
        self.slot = try a.alloc(i32, shape.rows);
        const start = rng.uintLessThan(u32, 4096);
        for (self.pos, self.slot, 0..) |*p, *s, r| {
            p.* = @intCast(start + r); // positions wrap the ring past 256
            s.* = @intCast(r % slots);
        }
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.pos), try ref.input(rig, self.slot) };
        self.out = try rig.output(slots * ring * ntaps * shape.n * 2);
        return true;
    }

    pub fn run(self: *Taps, rig: *rig_.Rig) !void {
        try launch(self.f, rig.s, self.shape.rows, .{ self.dev[0], self.dev[1], self.dev[2], self.out, @as(i32, ring), @as(i32, ntaps), @as(i32, tap), @as(i32, @intCast(self.shape.n)) });
    }

    pub fn count(self: *const Taps) usize {
        return slots * ring * ntaps * self.shape.n * 2;
    }

    pub fn check(self: *const Taps, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            const row = @as(usize, @intCast(self.slot[r])) * ring + @as(usize, @intCast(self.pos[r])) % ring;
            for (0..n) |i| {
                const have = read(u16, got, (row * ntaps + tap) * n + i);
                const want = c.toBf16(self.x[r * n + i]);
                const nan = std.math.isNan(self.x[r * n + i]) and (have & 0x7fff) > 0x7f80;
                if (have != want and !nan) return try std.fmt.bufPrint(msg, "row {d} col {d}: {x} where {x} is due", .{ r, i, have, want });
            }
        }
        return null;
    }
};

/// Every registration and its control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mod: cuda.Module,
    embeds: [2]ref.Checked(Embed) = undefined,
    lerps: [3]ref.Checked(Lerp) = undefined,
    sums_: [2]ref.Checked(RankSum) = undefined,
    heads: [3]ref.Checked(Heads) = undefined,
    taps: [2]ref.Checked(Taps) = undefined,
    gates: [2]ref.Gated(rows.Gate) = undefined,
    swiglus: [2]ref.Gated(rows.SwiGlu) = undefined,
    norms: [2]ref.Gated(rows.NormAdd) = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .mod = try cuda.Module.load(gpu.d, cuda.kernels.ricursive_tp2) };
        errdefer set.mod.unload();
        const m = &set.mod;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        set.embeds = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_embed") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_embed"), .ctl = true } } };
        set.lerps = .{
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_lerp"), .w = 0.30078125 } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_lerp"), .w = 0.69921875 } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_lerp"), .w = 0.69921875, .ctl = true } },
        };
        set.sums_ = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_rank_sum") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_rank_sum"), .ctl = true } } };
        set.heads = .{
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_heads"), .pad = true } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_heads"), .pad = false } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_heads"), .pad = true, .ctl = true } },
        };
        set.taps = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_taps") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_taps"), .ctl = true } } };
        set.gates = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_gate") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_gate"), .ctl = true } } };
        set.swiglus = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_swiglu") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_swiglu"), .ctl = true } } };
        set.norms = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_norm_add") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2_ctl_norm_add"), .ctl = true } } };
        inline for (.{ &set.embeds, &set.lerps, &set.sums_, &set.heads, &set.taps, &set.gates, &set.swiglus, &set.norms }) |xs| for (xs) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        inline for (.{ &set.embeds, &set.lerps, &set.sums_, &set.heads, &set.taps, &set.gates, &set.swiglus, &set.norms }) |xs| for (xs) |*x| x.deinit();
        set.mod.unload();
        set.gpa.destroy(set);
    }
};

/// The two-rank glue's gates 1-4 and its controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x727470, &t);
    std.debug.print("SUMMARY ricursive_tp2: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "ricursive-tp2-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
