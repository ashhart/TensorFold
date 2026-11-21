//! Hy4-flash's index kernels under the contract: the key norm here, the others in metal_hy4_index_*.zig.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const pool = @import("metal_hy4_index_groups.zig");
const score = @import("metal_hy4_index_scores.zig");
const recall = @import("metal_hy4_index_recall.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const bf = fl.bf;
const Range = fl.Range;
const u = fl.u32_ulp;

/// The indexer's widths as its kernels fix them; `Suite.collect` checks them against the checkpoint's config.
pub const width = 128;
pub const heads = 16;
pub const slots = 4;
pub const budget = h.hy.metal_index.budget_groups;

pub const Source = enum { build, recall };

fn text(src: Source) []const u8 {
    return switch (src) {
        .build => sources.hy4_flash_index_build,
        .recall => sources.hy4_flash_index_recall,
    };
}

/// A planted control: its gate-5 word, the source and function it edits, the fault, what it drops, its gate.
pub const Control = struct { word: []const u8, src: Source, function: []const u8, edit: fl.Edit, drops: []const u8, gate: g.Gate };

pub const controls = [_]Control{
    .{ .word = "unrounded", .src = .build, .function = "tf_hy4_index_keys", .edit = .{ .kernel = "tf_hy4_index_keys", .from = "hy4_ix_value(hy4_ix_round(x * inv))", .to = "(x * inv)" }, .drops = "the normed key's bf16 rounding dropped", .gate = .shapes },
    .{ .word = "slot0-bias", .src = .build, .function = "tf_hy4_index_groups", .edit = .{ .kernel = "tf_hy4_index_groups", .from = "bias[s * 128 + c]", .to = "bias[c]" }, .drops = "every slot takes slot 0's ape bias", .gate = .shapes },
    .{ .word = "head15", .src = .recall, .function = "tf_hy4_group_scores", .edit = .{ .kernel = "tf_hy4_group_scores", .from = "head < 16", .to = "head < 15" }, .drops = "the last index head dropped", .gate = .shapes },
    .{ .word = "signed-zero", .src = .recall, .function = "tf_hy4_recall_groups", .edit = .{ .kernel = "hy4_order", .from = "x + 0.0f", .to = "x" }, .drops = "no + 0.0f in its order: -0 below +0, a negative NaN below -inf", .gate = .nonfinite },
    .{ .word = "own-token", .src = .recall, .function = "tf_hy4_recall_groups", .edit = .{ .kernel = "tf_hy4_recall_groups", .from = "position % 4 + 1", .to = "position % 4" }, .drops = "the row's own token dropped", .gate = .shapes },
};

/// One function of an index source compiled as metal_index.zig's Kernels.load compiles it (MLX's options).
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, src: Source, function: []const u8, e: ?fl.Edit) !mtl.Pipeline {
    const body = try fl.edited(a, text(src), e);
    defer a.free(body);
    const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, function, false);
}

/// Every spec name: "metal hy4 <source stem> <function>" and a variant.
pub fn baseName(src: Source, function: []const u8, variant: []const u8, buf: []u8) []const u8 {
    const stem = if (src == .build) "index_build" else "index_recall";
    return std.fmt.bufPrint(buf, "metal hy4 {s} {s}{s}{s}", .{ stem, function, if (variant.len > 0) " " else "", variant }) catch unreachable;
}

// --- tf_hy4_index_keys: each row's key RMS-normed in fp32, rounded to bf16, times its gain, at its slot ------------

/// metal_index.zig's KeysArgs (index_build.metal's Hy4IndexKeys).
const KeysArgs = extern struct { rows: u32, capacity: u32, epsilon: f32 };

/// Any order of a simdgroup's 32 squares (31 roundings), then the 3 adds of the four partials.
fn keysDepth(_: c.Shape) usize {
    return 34;
}

/// Metal's division to 2.5 ulps and the epsilon's add move rsqrt's argument by 6 u, its 2 ulps 4 u more: 7 u in all.
const inv_slack = 7 * u;

const norm_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub const Keys = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    eps: f32,
    fault: ?Control,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    capacity: usize = 0,
    x: []u16 = &.{},
    gain: []u16 = &.{},
    slot: []i32 = &.{},
    /// The cache before the launch, restored before every run: a slot no row writes keeps it.
    old: []u16 = &.{},
    /// Each cache slot's writing row, or -1.
    writer: []i32 = &.{},
    ins: [3]rig_.Ref = undefined,
    cache: rig_.Ref = undefined,

    const base = "metal hy4 index_build tf_hy4_index_keys";
    /// Rows: the pass's rows; n: the key's 128 columns; k: the 128 squares each norm sums.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = keysDepth,
            .ops = 4,
            .mids = 1,
            .out = .bf16,
            // DFly2's window first (the single-shape gates run it), a decode row, a prompt chunk
            .served = &.{ .{ .rows = h.window_rows, .n = width, .k = width }, .{ .rows = 1, .n = width, .k = width }, .{ .rows = h.prompt_rows, .n = width, .k = width } },
            .sums = .{ .bf16 = 0, .why = norm_why },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, eps: f32, fault: ?Control) !Keys {
        const pipe = try pipeline(a, device, .build, "tf_hy4_index_keys", if (fault) |f| f.edit else null);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .eps = eps, .fault = fault };
    }
    pub fn deinit(t: *Keys) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Keys) void {
        inline for (.{ "x", "gain", "slot", "old", "writer" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Keys) g.Kernel {
        t.name = fl.named(base, if (t.fault) |f| f.drops else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = if (t.fault) |f| f.gate else null };
    }
    const vt = g.hooks(Keys);

    /// Slots from 3 on, one a row; from six rows row 2's is -1 and the last row's the capacity, which the kernel skips.
    pub fn prepare(t: *Keys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != width or shape.k != width) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        t.capacity = rows + 6;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, rows * width);
        t.gain = try t.a.alloc(u16, width);
        t.slot = try t.a.alloc(i32, rows);
        t.old = try t.a.alloc(u16, t.capacity * width);
        t.writer = try t.a.alloc(i32, t.capacity);
        for (t.old) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) {
            // integers 24 to 40: squares sum exactly in fp32 (at most 128 x 1,600) and past fp16's range in any order
            for (t.x) |*v| v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(24 + rng.uintLessThan(u32, 17))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            @memset(t.gain, c.toBf16(1));
        } else {
            for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
            for (t.gain) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        }
        if (fill == .nonfinite and rows >= 5) {
            // NaN in row 0, +Inf in row 1, -Inf in row 3, both in row 4 (row 2 writes nothing at six rows)
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 17 }, .{ 3, 40 }, .{ 4, 2 }, .{ 4, 100 } };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[2], fl.planted[1], fl.planted[2] };
            for (at, vals) |p, v| t.x[p[0] * width + p[1]] = c.toBf16(v);
        }
        @memset(t.writer, -1);
        for (t.slot, 0..) |*s, r| s.* = @intCast(3 + r);
        if (rows >= 6) {
            t.slot[2] = -1;
            t.slot[rows - 1] = @intCast(t.capacity);
        }
        for (t.slot, 0..) |s, r| if (s >= 0 and s < t.capacity) {
            t.writer[@intCast(s)] = @intCast(r);
        };
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.gain));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.slot));
        t.cache = try t.rig.output(t.old.len * 2);
        return true;
    }

    pub fn run(t: *Keys) !void {
        @memcpy(t.rig.bytes(t.cache, t.old.len * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(t.cache.buf, t.cache.off, 3);
        e.setValue(KeysArgs{ .rows = @intCast(t.shape.rows), .capacity = @intCast(t.capacity), .epsilon = t.eps }, 4);
        // as metal_index.zig: a threadgroup of 128 a row
        e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(width, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Keys) usize {
        return t.capacity * width;
    }

    fn xv(t: *const Keys, r: usize, j: usize) f64 {
        return c.fromBf16(t.x[r * width + j]);
    }
    fn inv(t: *const Keys, ss: f64) f64 {
        return 1 / @sqrt(ss / width + @as(f64, t.eps));
    }
    fn squares(t: *const Keys, r: usize, terms: *[width]f64) f64 {
        var ss: f64 = 0;
        for (terms, 0..) |*q, j| {
            q.* = t.xv(r, j) * t.xv(r, j);
            ss += q.*;
        }
        return ss;
    }

    /// Float64 at the bf16 points: bf16(gain * bf16(x / rms)); a slot no row writes keeps the old cache's value.
    pub fn reference(t: *Keys, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        var terms: [width]f64 = undefined;
        for (0..t.capacity) |s| {
            const out = want[s * width ..][0..width];
            if (t.writer[s] < 0) {
                for (out, 0..) |*w, j| w.* = c.fromBf16(t.old[s * width + j]);
                continue;
            }
            const r: usize = @intCast(t.writer[s]);
            const iv = t.inv(t.squares(r, &terms));
            for (out, 0..) |*w, j| w.* = bf(@as(f64, c.fromBf16(t.gain[j])) * bf(t.xv(r, j) * iv));
        }
    }

    /// An output's range: squares within gamma(34) (none on gate 4's integers), rsqrt's 7 u, x * inv, bf16, gain, bf16.
    pub fn intervals(t: *Keys, low: []f64, high: []f64) !void {
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(keysDepth(t.shape)), u);
        var terms: [width]f64 = undefined;
        for (0..t.capacity) |s| {
            if (t.writer[s] < 0) {
                for (0..width) |j| {
                    low[s * width + j] = c.fromBf16(t.old[s * width + j]);
                    high[s * width + j] = low[s * width + j];
                }
                continue;
            }
            const r: usize = @intCast(t.writer[s]);
            const ss = t.squares(r, &terms);
            const iv = Range{ .lo = t.inv(ss * (1 + gam)) * (1 - inv_slack), .hi = t.inv(ss * (1 - gam)) * (1 + inv_slack) };
            for (0..width) |j| {
                const o = Range.point(t.xv(r, j)).mul(iv).fp32().bf16().times(c.fromBf16(t.gain[j])).bf16();
                low[s * width + j] = o.lo;
                high[s * width + j] = o.hi;
            }
        }
    }

    pub fn bounds(t: *Keys, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Keys, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [width]f64 = undefined;
        for (0..t.capacity) |s| {
            if (t.writer[s] < 0) {
                for (0..width) |j| out[s * width + j] = c.fromBf16(t.old[s * width + j]);
                continue;
            }
            const r: usize = @intCast(t.writer[s]);
            _ = t.squares(r, &terms);
            const iv = t.inv(contract.rows.sumIn(acc, order, &terms));
            for (0..width) |j| out[s * width + j] = bf(@as(f64, c.fromBf16(t.gain[j])) * bf(t.xv(r, j) * iv));
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    keys: fl.List(Keys) = .{},
    groups: fl.List(pool.Groups) = .{},
    scores: fl.List(score.Scores) = .{},
    recalls: fl.List(recall.Recall) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const parsed = try h.text(s.a);
        defer parsed.deinit();
        const t = parsed.value.text_config;
        if (t.compressed_indexer_num_attention_heads != heads or t.compressed_indexer_head_dim != width or
            t.compressed_indexer_compress_ratio != slots or t.compressed_indexer_group_topk != budget or
            t.compressed_indexer_candidate_topk != budget * slots or h.hy.index_recall.heads != heads or
            h.hy.index_recall.width != width) return error.Hy4IndexWidthsMoved;
        const eps = t.rms_norm_eps;
        var buf: [256]u8 = undefined;
        if (fl.wanted(only, Keys.base)) try s.keys.add(s.a, s.kernels, try Keys.init(s.a, s.device, s.queue, eps, null));
        if (fl.wanted(only, pool.Groups.base)) try s.groups.add(s.a, s.kernels, try pool.Groups.init(s.a, s.device, s.queue, eps, null));
        for ([_]bool{ false, true }) |nan_dot| {
            if (fl.wanted(only, score.Scores.base(nan_dot, &buf))) try s.scores.add(s.a, s.kernels, try score.Scores.init(s.a, s.device, s.queue, nan_dot, null));
        }
        if (fl.wanted(only, recall.Recall.base)) try s.recalls.add(s.a, s.kernels, try recall.Recall.init(s.a, s.device, s.queue, null));
        for (controls) |f| {
            const stem = baseName(f.src, f.function, "", &buf);
            var named_buf: [256]u8 = undefined;
            if (!fl.wanted(only, fl.named(stem, f.drops, &named_buf))) continue;
            if (std.mem.eql(u8, f.function, "tf_hy4_index_keys")) {
                try s.keys.add(s.a, s.kernels, try Keys.init(s.a, s.device, s.queue, eps, f));
            } else if (std.mem.eql(u8, f.function, "tf_hy4_index_groups")) {
                try s.groups.add(s.a, s.kernels, try pool.Groups.init(s.a, s.device, s.queue, eps, f));
            } else if (std.mem.eql(u8, f.function, "tf_hy4_group_scores")) {
                try s.scores.add(s.a, s.kernels, try score.Scores.init(s.a, s.device, s.queue, false, f));
            } else try s.recalls.add(s.a, s.kernels, try recall.Recall.init(s.a, s.device, s.queue, f));
        }
    }
    pub fn deinit(s: *Suite) void {
        s.keys.deinit(s.a);
        s.groups.deinit(s.a);
        s.scores.deinit(s.a);
        s.recalls.deinit(s.a);
    }
};

/// Gate 5's texts for `hy4 index_build ...` and `hy4 index_recall ...`: the loader's source, or a control word's edit.
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, name: []const u8) !?fl.Texts {
    const src: Source = if (std.mem.startsWith(u8, name, "hy4 index_build ")) .build else if (std.mem.startsWith(u8, name, "hy4 index_recall ")) .recall else return null;
    var words = std.mem.tokenizeScalar(u8, name, ' ');
    _ = words.next();
    _ = words.next();
    const first = words.next() orelse return error.UnknownLaunch;
    const function = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    if (std.mem.eql(u8, first, function)) return .{ .here = text(src), .pre_m5 = text(src) };
    for (controls) |f| if (f.src == src and std.mem.eql(u8, f.word, first) and std.mem.eql(u8, f.function, function)) {
        const body = try fl.edited(a, text(src), f.edit);
        return .{ .here = body, .pre_m5 = body };
    };
    return error.UnknownLaunch;
}

test "every control's edit lands in its own function's body" {
    for (controls) |f| {
        const body = try fl.edited(std.testing.allocator, text(f.src), f.edit);
        defer std.testing.allocator.free(body);
        try std.testing.expect(!std.mem.eql(u8, body, text(f.src)));
    }
}
