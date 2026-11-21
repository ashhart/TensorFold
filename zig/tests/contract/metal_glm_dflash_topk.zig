//! GLM's DFlash2 drafter top-k under the contract: each row's bytes equal core bf16_topk's CPU oracle (its bytes).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const cpu = tf.bf16_topk;
const dtopk = tf.glm.metal_dflash_topk;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "a row holding NaN or an infinity returns no entries and the mask of what it held (1 NaN, 2 +Inf, 4 -Inf)";

/// Eighths in [-8, 8): hundreds of ties at the k-th key (the ordered path); spread: few ties (the sorted picks).
const Data = enum { ties, spread };
/// Served; a tie broken toward the higher token; the k-th key's own tokens left out.
const Case = enum { plain, tie, kth };
const Variant = struct { data: Data, case: Case = .plain };
pub const registered = [_]Variant{ .{ .data = .ties }, .{ .data = .spread }, .{ .data = .ties, .case = .tie }, .{ .data = .spread, .case = .kth } };

pub fn name(v: Variant) []const u8 {
    return switch (v.case) {
        .plain => if (v.data == .ties) "metal glm dflash2 top-k, ties at the k-th key" else "metal glm dflash2 top-k, spread logits",
        .tie => "control: metal glm dflash2 top-k, a tie broken toward the higher token",
        .kth => "control: metal glm dflash2 top-k, the k-th key's tokens left out",
    };
}

fn edit(case: Case) ?[2][]const u8 {
    return switch (case) {
        .plain => null,
        .tie => .{ "(key == other_key && id < other_id)", "(key == other_key && id > other_id)" },
        .kth => .{ "if (key < kth) continue;", "if (key <= kth) continue;" },
    };
}

/// Rows, entries a row (k) and vocabulary: a stream's 15 or 7 block rows, three streams' 48, over 154,880 tokens.
pub fn spec(v: Variant) c.Spec {
    return .{
        .name = name(v),
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .replaces = "core tf_bf16_topk (its spans and merge)",
        .tiles = .{ .{ .size = 16 }, .{ .size = 16 }, .{ .size = dtopk.threads, .min = 16 } },
        .served = &.{ .{ .rows = 15, .n = 16, .k = 154880 }, .{ .rows = 48, .n = 16, .k = 154880 }, .{ .rows = 7, .n = 16, .k = 154880 }, .{ .rows = 15, .n = 4, .k = 4095 } },
    };
}

pub const TopK = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    top: dtopk.TopK,
    core: tf.bf16_topk_gpu.Ops,
    p: cpu.Params = undefined,
    words: []u16 = &.{},
    x: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !TopK {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .top = try dtopk.TopK.init(a, device, edit(v.case)), .core = try tf.bf16_topk_gpu.Ops.init(device) };
    }

    pub fn deinit(t: *TopK) void {
        t.a.free(t.words);
        t.top.deinit();
        t.core.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *TopK) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt, .control = if (t.v.case == .plain) null else .shapes };
    }

    const vt = g.hooks(TopK);

    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        t.p = .{ .rows = @intCast(shape.rows), .vocab = @intCast(shape.k), .k = @intCast(shape.n), .stride = @intCast(shape.k) };
        if (!dtopk.takes(t.p)) return false;
        t.a.free(t.words);
        t.words = &.{};
        t.words = try t.a.alloc(u16, shape.rows * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.words) |*w| w.* = switch (t.v.data) {
            .ties => c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 128))) / 8 - 8),
            .spread => c.toBf16((rng.float(f32) * 2 - 1) * 40),
        };
        // a negative zero beside a positive one: both are one key, ordered by token
        if (shape.k > 3) {
            t.words[1] = 0x8000;
            t.words[2] = 0x0000;
        }
        if (fill == .nonfinite) {
            const k = shape.k;
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]u16{ 0x7fc1, 0x7f80, 0xff80, 0x7f80, 0xff80 };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                t.words[p[0] * k + p[1]] = v;
            };
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.words));
        t.out = try t.rig.output(shape.rows * @sizeOf(cpu.Row));
        t.base = try t.rig.scratch(shape.rows * @sizeOf(cpu.Row));
        return true;
    }

    pub fn run(t: *TopK) !void {
        const pass = t.rig.begin();
        try t.top.encode(pass.e, .{ .buf = t.x.buf, .off = t.x.off }, .{ .buf = t.out.buf, .off = t.out.off }, t.p);
        try t.rig.finish(pass);
    }

    pub fn count(t: *TopK) usize {
        return t.p.rows * @sizeOf(cpu.Row);
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every row's bytes against the oracle; at the non-finite gate, also the rule on the rows that were planted.
    pub fn check(t: *TopK, at: g.Gate) !?[]const u8 {
        const v = t.p.vocab;
        const got = t.rig.bytes(t.out, t.count());
        var rejected: usize = 0;
        for (0..t.p.rows) |r| {
            const row = t.words[r * v ..][0..v];
            const want = try cpu.oracle(row, t.p.k);
            if (!std.mem.eql(u8, std.mem.asBytes(&want), got[r * @sizeOf(cpu.Row) ..][0..@sizeOf(cpu.Row)]))
                return try std.fmt.bufPrint(&t.msg, "row {d} of {d} (vocab {d}, k {d}) differs from the CPU oracle", .{ r, t.p.rows, v, t.p.k });
            if (want.nonfinite != 0) rejected += 1;
        }
        if (at == .nonfinite and (rejected == 0 or rejected == t.p.rows)) return try std.fmt.bufPrint(&t.msg, "{d} of {d} rows planted: no proof", .{ rejected, t.p.rows });
        return null;
    }

    /// Core's tf_bf16_topk on the same rows: the bytes gate 3 requires.
    pub fn replaced(t: *TopK) ![]const u8 {
        const pass = t.rig.begin();
        // core takes 16 rows a call; a row's entries do not depend on the rows beside it
        var first: u32 = 0;
        while (first < t.p.rows) : (first += 16) {
            var p = t.p;
            p.rows = @min(16, t.p.rows - first);
            try t.core.encode(pass.e, .{ .buf = t.x.buf, .off = t.x.off + @as(usize, first) * t.p.stride * 2 }, .{ .buf = t.base.buf, .off = t.base.off + @as(usize, first) * @sizeOf(cpu.Row) }, p);
            pass.e.barrier();
        }
        try t.rig.finish(pass);
        return t.rig.bytes(t.base, t.count());
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tops: std.ArrayList(*TopK) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (registered) |v| {
            if (only) |o| if (std.mem.indexOf(u8, name(v), o) == null) continue;
            const p = try s.a.create(TopK);
            errdefer s.a.destroy(p);
            p.* = try TopK.init(s.a, s.device, s.queue, v);
            errdefer p.deinit();
            try s.tops.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.tops.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.tops.deinit(s.a);
    }
};
