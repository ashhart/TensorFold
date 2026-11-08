//! Ripple's DFlash2 drafter glue (ricursive_tp2_draft.cu) under the contract, each kernel with a planted control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const rows = @import("contract_ricursive_tp2_draft_rows.zig"); // norms, SwiGLU, convolutions, head norms
const Attn = @import("contract_ricursive_tp2_draft_attn.zig").Attn;

const c = core.contract.spec;
const g = core.contract.gates;
const Function = cuda.Function;
const go = rows.go;
const at = rows.at;
const elementwise: c.Tile = .{ .size = 256, .step = 1, .min = 1 };

const tap_rows = 256; // the target's tap ring
const tap_width = 5 * 5120; // a tap row: five target layers' residuals
const vocab = 100352;
const slice = 2048; // logits a top-k block takes
const topk = 32;

fn read(comptime T: type, bytes: []const u8, i: usize) T {
    const U = @Int(.unsigned, @bitSizeOf(T));
    return @bitCast(std.mem.readInt(U, bytes[i * @sizeOf(T) ..][0..@sizeOf(T)], .little));
}

/// OUT row i = bf16 columns [col, col + width) of tap ring row (start + i) % 256, words as stored.
pub const Gather = struct {
    f: Function,
    col: usize,
    ctl: bool = false, // control: each row read from the ring's next row
    shape: c.Shape = undefined,
    ring: []u16 = &.{},
    start: i64 = 0,
    dev: u64 = 0,
    out: u64 = 0,

    const served = at(&.{ 16, 1, 3, 77, 128 }, tap_width / 2, 1); // a rank's half of each tap row

    pub fn spec(self: *const Gather) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_gather, each row from the ring's next row" else if (self.col == 0) "cuda ricursive_tp2_draft tf_rtp2d_gather (rank 0's tap columns)" else "cuda ricursive_tp2_draft tf_rtp2d_gather (rank 1's tap columns)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "every bf16 word as stored, NaN patterns included" }, .served = &served };
    }

    pub fn control(self: *const Gather) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Gather, a: std.mem.Allocator) void {
        a.free(self.ring);
        self.ring = &.{};
    }

    pub fn prepare(self: *Gather, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.rows > tap_rows or self.col + shape.n > tap_width) return false;
        self.free(a);
        self.shape = shape;
        self.ring = try a.alloc(u16, tap_rows * tap_width);
        for (self.ring) |*v| v.* = rng.int(u16);
        self.start = rng.uintLessThan(u32, 4096); // rows wrap the ring past 256
        if (fill == .nonfinite) for ([_]u16{ 0x7fc1, 0x7f80, 0xff80 }, 0..) |x, r| if (r < shape.rows) {
            const row: usize = @intCast(@mod(self.start + @as(i64, @intCast(r)), tap_rows));
            self.ring[row * tap_width + self.col + (7 * r + 1) % shape.n] = x;
        };
        self.dev = try ref.input(rig, self.ring);
        self.out = try rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Gather, rig: *rig_.Rig) !void {
        const args = .{ self.dev, @as(i32, tap_rows), self.start, @as(i32, tap_width), @as(i32, @intCast(self.col)), @as(i32, @intCast(self.shape.n)), self.out };
        try go(self.f, rig.s, .{ self.shape.rows, 1, 1 }, 256, 0, args);
    }

    pub fn count(self: *const Gather) usize {
        return self.shape.rows * self.shape.n * 2;
    }

    pub fn check(self: *const Gather, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            const row: usize = @intCast(@mod(self.start + @as(i64, @intCast(r)), tap_rows));
            for (0..n) |i| {
                const have = read(u16, got, r * n + i);
                const want = self.ring[row * tap_width + self.col + i];
                if (have != want) return try std.fmt.bufPrint(msg, "row {d} col {d}: {x} where {x} is due", .{ r, i, have, want });
            }
        }
        return null;
    }
};

/// OUT fp32 [rows, H] = each id's bf16 table row as stored (no scale), id vocab - 1 included.
pub const Embed = struct {
    f: Function,
    ctl: bool = false, // control: the table read as fp16
    shape: c.Shape = undefined,
    table: []u16 = &.{},
    ids: []i32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    const served = at(&rows.hold, 5120, 1);

    pub fn spec(self: *const Embed) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_embed reading the table as fp16" else "cuda ricursive_tp2_draft tf_rtp2d_embed (the target's bf16 rows, unscaled, into fp32)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "each bf16 word widened to fp32 bit for bit, NaN payloads included" }, .tiles = .{ .{}, elementwise, .{} }, .served = &served };
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

    /// Only the rows the ids name are filled (64 distinct and vocab - 1); the rest of the table stays zero.
    pub fn prepare(self: *Embed, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        const distinct: usize = @min(shape.rows, 64);
        self.ids = try a.alloc(i32, shape.rows);
        for (self.ids, 0..) |*id, r| id.* = @intCast(if (r == 0) vocab - 1 else r % distinct);
        self.table = try a.alloc(u16, vocab * n);
        @memset(self.table, 0);
        for (0..distinct) |r| for (self.table[r * n ..][0..n]) |*v| {
            v.* = c.toBf16(-1 + 2 * rng.float(f32));
        };
        for (self.table[(vocab - 1) * n ..][0..n]) |*v| v.* = c.toBf16(-1 + 2 * rng.float(f32));
        if (fill == .nonfinite) for ([_]u16{ 0x7fc1, 0x7f80, 0xff80 }, 0..) |x, r| if (r < shape.rows) {
            self.table[@as(usize, @intCast(self.ids[r])) * n + (7 * r + 1) % n] = x;
        };
        self.dev = .{ try ref.input(rig, self.ids), try ref.input(rig, self.table) };
        self.out = try rig.output(shape.rows * n * 4);
        return true;
    }

    pub fn run(self: *Embed, rig: *rig_.Rig) !void {
        try go(self.f, rig.s, .{ self.shape.rows, 1, 1 }, 256, 0, .{ self.dev[0], self.dev[1], self.out, @as(i32, @intCast(self.shape.n)) });
    }

    pub fn count(self: *const Embed) usize {
        return self.shape.rows * self.shape.n * 4;
    }

    pub fn check(self: *const Embed, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const want = @as(u32, self.table[@as(usize, @intCast(self.ids[r])) * n + i]) << 16;
            const have = read(u32, got, r * n + i);
            if (have != want) return try std.fmt.bufPrint(msg, "row {d} col {d}: {x} where {x} is due", .{ r, i, have, want });
        };
        return null;
    }
};

/// OUT bf16 [n] = X fp32 [n], nearest even: the value rows into bf16 for the attention.
pub const F2b = struct {
    f: Function,
    ctl: bool = false, // control: rounded toward zero
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    const served = at(&rows.hold, 4 * 128, 1); // a rank's 4 value heads a row

    pub fn spec(self: *const F2b) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_f2b toward zero" else "cuda ricursive_tp2_draft tf_rtp2d_f2b (fp32 values to bf16, nearest even)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "NaN stays NaN, infinities stay infinite, as bf16 gives them" }, .tiles = .{ .{}, elementwise, .{} }, .served = &served };
    }

    pub fn control(self: *const F2b) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *F2b, a: std.mem.Allocator) void {
        a.free(self.x);
        self.x = &.{};
    }

    pub fn prepare(self: *F2b, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.f32s(a, rng, shape.rows * shape.n, -40, 40);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |v, i| if (i < self.x.len) {
            self.x[3 * i % self.x.len] = v;
        };
        self.dev = try ref.input(rig, self.x);
        self.out = try rig.output(self.x.len * 2);
        return true;
    }

    pub fn run(self: *F2b, rig: *rig_.Rig) !void {
        const n = self.x.len;
        try go(self.f, rig.s, .{ (n + 255) / 256, 1, 1 }, 256, 0, .{ self.dev, self.out, @as(i64, @intCast(n)) });
    }

    pub fn count(self: *const F2b) usize {
        return self.shape.rows * self.shape.n * 2;
    }

    pub fn check(self: *const F2b, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        for (self.x, 0..) |x, i| {
            const have = read(u16, got, i);
            const want = c.toBf16(x);
            const nan = std.math.isNan(x) and (have & 0x7fff) > 0x7f80;
            if (have != want and !nan) return try std.fmt.bufPrint(msg, "value {d}: {x} where {x} is due", .{ i, have, want });
        }
        return null;
    }
};

/// Each row's 2,048-logit slices: the 32 best (value, id), ties to the lower id, into PV / PI [rows, slices, 32].
pub const TopK = struct {
    f: Function,
    ctl: bool = false, // control: a lane tie goes to the higher id
    shape: c.Shape = undefined,
    l: []f32 = &.{},
    dev: u64 = 0,
    out: [2]u64 = @splat(0),

    const served = at(&.{ 15, 30, 45, 60 }, vocab / 2, topk); // a rank's half of the head's columns

    pub fn spec(self: *const TopK) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_topk, lane ties to the higher id" else "cuda ricursive_tp2_draft tf_rtp2d_topk (each 2,048-logit slice's 32 best, ties to the lower id)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = "NaN reads as -Inf and -Inf is never chosen: a slice short of 32 ends in (-Inf, -1)" }, .tiles = .{ .{}, .{ .size = slice, .step = 1, .min = 1 }, .{} }, .served = &served };
    }

    pub fn control(self: *const TopK) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *TopK, a: std.mem.Allocator) void {
        a.free(self.l);
        self.l = &.{};
    }

    fn slices(self: *const TopK) usize {
        return (self.shape.n + slice - 1) / slice;
    }

    /// Whole logits in [-24, 24]: every slice holds many ties.
    pub fn prepare(self: *TopK, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        self.l = try a.alloc(f32, shape.rows * shape.n);
        for (self.l) |*v| v.* = ref.int(rng, -24, 24);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf, ref.nan, ref.inf }, 0..) |v, i| {
            self.l[(5 * i + 2) % self.l.len] = v;
        };
        self.dev = try ref.input(rig, self.l);
        const words = shape.rows * self.slices() * topk;
        self.out = .{ try rig.output(words * 4), try rig.output(words * 4) };
        return true;
    }

    pub fn run(self: *TopK, rig: *rig_.Rig) !void {
        const args = .{ self.dev, @as(i32, @intCast(self.shape.n)), @as(i32, slice), @as(i32, topk), self.out[0], self.out[1] };
        try go(self.f, rig.s, .{ self.slices(), self.shape.rows, 1 }, 256, slice * 4, args);
    }

    pub fn count(self: *const TopK) usize {
        return self.shape.rows * self.slices() * topk * 8;
    }

    fn before(l: []const f32, x: u32, y: u32) bool {
        return l[x] > l[y] or (l[x] == l[y] and x < y);
    }

    pub fn check(self: *const TopK, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const words = self.shape.rows * self.slices() * topk;
        const vals = got[0 .. words * 4];
        const ids = got[words * 4 ..];
        const order = try a.alloc(u32, slice);
        defer a.free(order);
        const l = try a.alloc(f32, n);
        defer a.free(l);
        for (0..self.shape.rows) |r| {
            for (l, self.l[r * n ..][0..n]) |*o, v| o.* = if (std.math.isNan(v)) -ref.inf else v;
            for (0..self.slices()) |sl| {
                const s0 = sl * slice;
                var live: usize = 0;
                for (s0..@min(n, s0 + slice)) |i| if (l[i] > -ref.inf) {
                    order[live] = @intCast(i);
                    live += 1;
                };
                std.mem.sort(u32, order[0..live], l, before);
                for (0..topk) |k| {
                    const o = (r * self.slices() + sl) * topk + k;
                    const want_v = if (k < live) l[order[k]] else -ref.inf;
                    const want_i: i32 = if (k < live) @intCast(order[k]) else -1;
                    const hv = read(f32, vals, o);
                    const hi = read(i32, ids, o);
                    if (hi != want_i or @as(u32, @bitCast(hv)) != @as(u32, @bitCast(want_v))) return try std.fmt.bufPrint(msg, "row {d} slice {d} place {d}: ({e}, {d}) where ({e}, {d}) is due", .{ r, sl, k, hv, hi, want_v, want_i });
                }
            }
        }
        return null;
    }
};

/// Every registration and its control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mod: cuda.Module,
    gathers: [3]ref.Checked(Gather) = undefined,
    embeds: [2]ref.Checked(Embed) = undefined,
    f2bs: [2]ref.Checked(F2b) = undefined,
    topks: [2]ref.Checked(TopK) = undefined,
    rmss: [4]ref.Gated(rows.Rms) = undefined,
    silus: [2]ref.Gated(rows.Silu) = undefined,
    convs: [4]ref.Gated(rows.Conv) = undefined,
    heads: [4]ref.Gated(rows.Heads) = undefined,
    kvs: [3]ref.Gated(rows.KvPut) = undefined,
    attns: [4]ref.Gated(Attn) = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa, .mod = try cuda.Module.load(gpu.d, cuda.kernels.ricursive_tp2_draft) };
        errdefer set.mod.unload();
        const m = &set.mod;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const half = tap_width / 2;
        set.gathers = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_gather"), .col = 0 } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_gather"), .col = half } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_gather"), .col = half, .ctl = true } } };
        set.embeds = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_embed") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_embed"), .ctl = true } } };
        set.f2bs = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_f2b") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_f2b"), .ctl = true } } };
        set.topks = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_topk") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_topk"), .ctl = true } } };
        const rms = try m.function("tf_rtp2d_rms");
        set.rmss = .{ .{ .rig = rig, .k = .{ .f = rms } }, .{ .rig = rig, .k = .{ .f = rms, .half = true, .blocks = true } }, .{ .rig = rig, .k = .{ .f = rms, .blocks = true } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_rms"), .ctl = true } } };
        set.silus = .{ .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_silu") } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_silu"), .ctl = true } } };
        set.convs = .{
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_conv_in"), .second = false } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_conv_out"), .second = true } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_conv_in"), .second = false, .ctl = true } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_conv_out"), .second = true, .ctl = true } },
        };
        const hf = try m.function("tf_rtp2d_heads");
        set.heads = .{
            .{ .rig = rig, .k = .{ .f = hf, .served = &rows.Heads.q_rows } },
            .{ .rig = rig, .k = .{ .f = hf, .served = &rows.Heads.k_rows } },
            .{ .rig = rig, .k = .{ .f = hf, .served = &rows.Heads.q_rows, .rope = false } },
            .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_heads"), .served = &rows.Heads.q_rows, .ctl = true } },
        };
        const kf = try m.function("tf_rtp2d_kv_put");
        set.kvs = .{ .{ .rig = rig, .k = .{ .f = kf } }, .{ .rig = rig, .k = .{ .f = kf, .rope = false } }, .{ .rig = rig, .k = .{ .f = try m.function("tf_rtp2d_ctl_kv_put"), .ctl = true } } };
        const af = try m.function("tf_rtp2d_attn");
        const mf = try m.function("tf_rtp2d_merge");
        set.attns = .{
            .{ .rig = rig, .k = .{ .attn = af, .merge = mf, .window = 4096 } },
            .{ .rig = rig, .k = .{ .attn = af, .merge = mf, .window = 16384 } },
            .{ .rig = rig, .k = .{ .attn = try m.function("tf_rtp2d_ctl_attn"), .merge = mf, .window = 4096, .ctl = .attn } },
            .{ .rig = rig, .k = .{ .attn = af, .merge = try m.function("tf_rtp2d_ctl_merge"), .window = 4096, .ctl = .merge } },
        };
        inline for (.{ &set.gathers, &set.embeds, &set.f2bs, &set.topks, &set.rmss, &set.silus, &set.convs, &set.heads, &set.kvs, &set.attns }) |xs| for (xs) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        inline for (.{ &set.gathers, &set.embeds, &set.f2bs, &set.topks, &set.rmss, &set.silus, &set.convs, &set.heads, &set.kvs, &set.attns }) |xs| for (xs) |*x| x.deinit();
        set.mod.unload();
        set.gpa.destroy(set);
    }
};

/// The drafter glue's gates 1-4 and its controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x727464, &t);
    std.debug.print("SUMMARY ricursive_tp2_draft: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "ricursive-tp2-draft-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
