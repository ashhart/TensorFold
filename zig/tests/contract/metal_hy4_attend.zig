//! Hy4-flash's MLA attention kernels (mla_attention.metal) under the contract, launched by mla_metal.zig's encoder.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const h = @import("metal_hy4.zig");
const rig_ = @import("rig.zig");
const inputs = @import("metal_hy4_attend_inputs.zig");
const g = contract.gates;
const c = contract.spec;
const ar = contract.attention_range;
const fl = h.fl;
const mla = h.hy.mla_metal;
const Lists = inputs.Lists;
const lat = inputs.lat;

const stem = "hy4 mla_attention ";
const sliding_heads = 64;
const sparse_heads = 32;
/// A row deep in a long stream: past the sliding window and the 2,048 keys a sparse layer attends unrecalled.
const deep = 9000;

/// The one-step pass (chunks) or the two-step one (scores, then each chunk's softmax over them).
pub const Path = enum { chunks, scored };

const one_step = "metal hy4 mla_attention tf_hy4_attend_chunks + tf_hy4_attend_merge";
const two_step = "metal hy4 mla_attention tf_hy4_attend_scores + tf_hy4_attend_scored + tf_hy4_attend_merge";

/// A planted control: one kernel's edit, the gate that must catch it, what it breaks, and its launch's word.
pub const Fault = enum { drop_last, no_sink, half_sums, skip_nan, shuffle };
const Control = struct { edit: fl.Edit, gate: c.Gate, what: []const u8, word: []const u8 };

fn control(f: Fault) Control {
    return switch (f) {
        .drop_last => .{ .edit = .{ .kernel = "tf_hy4_attend_chunks", .from = "counts[row]);", .to = "counts[row] - 1);" }, .gate = .shapes, .what = "each row's last key dropped", .word = "drop-last" },
        .no_sink => .{ .edit = .{ .kernel = "tf_hy4_attend_merge", .from = "float denom = precise::exp(sinks[head] - top)", .to = "float denom = 0.0f" }, .gate = .shapes, .what = "the sink dropped from the merge's denominator", .word = "no-sink" },
        .half_sums => .{ .edit = .{ .kernel = "tf_hy4_attend_chunks", .from = "accum[i] = fma(p, kv[i], accum[i] * old);", .to = "accum[i] = float(half(fma(p, kv[i], accum[i] * old)));" }, .gate = .sums, .what = "each chunk's values summed in half", .word = "half-sums" },
        .skip_nan => .{ .edit = .{ .kernel = "tf_hy4_attend_merge", .from = "const float p = precise::exp(part[lat] - top);", .to = "if (isnan(part[lat + 1])) continue;\n        const float p = precise::exp(part[lat] - top);" }, .gate = .nonfinite, .what = "the merge skipping a chunk whose mass is NaN", .word = "skip-nan" },
        .shuffle => .{ .edit = .{ .kernel = "tf_hy4_attend_scored", .from = "simd_shuffle(p, ushort(h + t))", .to = "simd_shuffle(p, ushort(t))" }, .gate = .shapes, .what = "each 32-key step's last 16 keys weighed as its first 16", .word = "shuffle" },
    };
}

/// Gate 1's finding: carried by an entry of its own that plants it, so the other entries keep a clean gate 1.
const first_inf_finding = "a key whose score is -Inf at the start of a 512-key chunk gives its row's head NaN where " ++
    "float64 leaves the key out: the running max is still -Inf there, so the first rescale is exp(-Inf - -Inf)";

/// What one registration runs: its pass, its lists, and a control's fault or the known finding's plant.
pub const Case = struct { path: Path, lists: Lists, fault: ?Fault = null, first_inf: bool = false };

const cases = [_]Case{
    .{ .path = .chunks, .lists = .window },
    .{ .path = .scored, .lists = .sparse },
    .{ .path = .scored, .lists = .draft },
    .{ .path = .chunks, .lists = .window, .first_inf = true },
    .{ .path = .scored, .lists = .sparse, .first_inf = true },
    .{ .path = .chunks, .lists = .window, .fault = .drop_last },
    .{ .path = .chunks, .lists = .window, .fault = .no_sink },
    .{ .path = .chunks, .lists = .window, .fault = .half_sums },
    .{ .path = .chunks, .lists = .window, .fault = .skip_nan },
    .{ .path = .scored, .lists = .sparse, .fault = .shuffle },
};

/// Rows of n heads, the last at position k - 1: sliding layers (64 heads) a 512-key window, sparse ones (32) every key.
const window_served = [_]c.Shape{
    .{ .rows = h.window_rows, .n = sliding_heads, .k = deep }, // DFly2's window, 512 keys a row
    .{ .rows = h.decode_rows, .n = sliding_heads, .k = deep },
    .{ .rows = h.prompt_rows, .n = sliding_heads, .k = h.prompt_rows }, // a prompt's first chunk: 1 to 512 keys
    .{ .rows = h.prompt_rows, .n = sliding_heads, .k = 2 * h.prompt_rows }, // its next: 512 keys a row
    .{ .rows = h.decode_rows, .n = sparse_heads, .k = 1 }, // a sparse layer's first key
    .{ .rows = h.window_rows, .n = sparse_heads, .k = 300 },
    .{ .rows = h.prompt_rows, .n = sparse_heads, .k = h.prompt_rows },
};
/// Past 2,048 keys each row attends its 2,048 recalled slots (a few dropped), in the recall's order.
const sparse_served = [_]c.Shape{
    .{ .rows = h.window_rows, .n = sparse_heads, .k = 1100 }, // a stride of 1,536
    .{ .rows = h.decode_rows, .n = sparse_heads, .k = 1024 },
    .{ .rows = h.prompt_rows, .n = sparse_heads, .k = 2 * h.prompt_rows },
    .{ .rows = h.prompt_rows, .n = sparse_heads, .k = 4 * h.prompt_rows },
    .{ .rows = h.decode_rows, .n = sparse_heads, .k = deep },
    .{ .rows = h.window_rows, .n = sparse_heads, .k = deep },
    .{ .rows = h.prompt_rows, .n = sparse_heads, .k = deep },
};
/// DFly2's draft rows at a stride of 2,560: past 2,048 the anchor's recall then the window's own rows, else every key.
const draft_served = [_]c.Shape{
    .{ .rows = h.window_rows, .n = sparse_heads, .k = deep },
    .{ .rows = h.decode_rows, .n = sparse_heads, .k = deep },
    .{ .rows = h.window_rows, .n = sparse_heads, .k = 2052 },
    .{ .rows = h.window_rows, .n = sparse_heads, .k = 700 },
};

/// A score's chain and a chunk's rescales and sums, then the merge's chunks (bounds() derives the real allowance).
fn depth(s: c.Shape) usize {
    return 23 + 2 * inputs.chunk + inputs.strideOf(.sparse, s.k) / inputs.chunk + 2;
}

pub fn spec(v: Case, name: []const u8) c.Spec {
    const served: []const c.Shape = switch (v.lists) {
        .window => &window_served,
        .sparse => &sparse_served,
        .draft => &draft_served,
    };
    // heads come four a threadgroup; each row's keys in 512-key chunks (the draft's stride stays 2,560)
    var s: c.Spec = .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = 1, // gate 4's exact sums meet the merge's fp32 division before the bf16 rounding
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 4, .step = 1, .min = 1 }, if (v.lists == .draft) .{} else .{ .size = inputs.chunk, .step = 1, .min = 1 } },
        .served = served,
        .replaces = if (v.path == .scored) one_step else null,
    };
    if (v.first_inf) {
        s.served = served[0..1];
        s.tiles = .{ .{}, .{}, .{} };
        s.known = .{ .gate = .nonfinite, .finding = first_inf_finding };
    }
    return s;
}

fn baseName(v: Case, buf: []u8) []const u8 {
    const path = if (v.path == .chunks) one_step else two_step;
    const extra = if (v.first_inf) ", -Inf score at a chunk's first key" else "";
    return std.fmt.bufPrint(buf, "{s} {t}{s}", .{ path, v.lists, extra }) catch path;
}

/// The control's own pipeline: mla_attention.metal with its edit, compiled as Attention.load compiles the source.
fn ownPipeline(a: std.mem.Allocator, device: mtl.Device, f: Fault) !mtl.Pipeline {
    const e = control(f).edit;
    const text = try fl.edited(a, sources.hy4_flash_mla_attention, e);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, e.kernel, false);
}

fn ref(r: rig_.Ref) mla.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    plain: *const mla.Attention,
    own: ?mtl.Pipeline = null,
    v: Case,
    base: [192]u8 = undefined,
    label: [320]u8 = undefined,
    name: []const u8 = "",
    in: inputs.Inputs,
    ins: [5]rig_.Ref = undefined,
    parts: rig_.Ref = undefined,
    scores: ?rig_.Ref = null,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, plain: *const mla.Attention, v: Case) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .plain = plain, .v = v, .in = .{ .a = a }, .own = if (v.fault) |f| try ownPipeline(a, device, f) else null };
    }

    fn drop(t: *Probe) void {
        t.rig.reset();
        t.in.deinit();
    }

    pub fn deinit(t: *Probe) void {
        t.drop();
        t.rig.deinit();
        if (t.own) |o| o.deinit();
    }

    /// The harness's release frees the host inputs and ranges too, so only one entry's are held at a time.
    const vt: g.VTable = blk: {
        var x = g.hooks(Probe);
        x.release = release;
        break :blk x;
    };
    fn release(ctx: *anyopaque) void {
        const t: *Probe = @ptrCast(@alignCast(ctx));
        t.drop();
    }

    pub fn kernel(t: *Probe) g.Kernel {
        const b = baseName(t.v, &t.base);
        const what = if (t.v.fault) |f| control(f).what else null;
        t.name = fl.named(b, what, &t.label);
        return .{ .spec = spec(t.v, t.name), .ctx = t, .vt = &vt, .control = if (t.v.fault) |f| control(f).gate else null };
    }

    /// The loader's pipelines, with a control's edited one in its kernel's place.
    fn pipes(t: *const Probe) mla.Attention {
        var p = t.plain.*;
        const o = t.own orelse return p;
        const name = control(t.v.fault.?).edit.kernel;
        inline for (.{ "chunks", "merge", "scores", "scored" }) |f| {
            if (std.mem.eql(u8, name, "tf_hy4_attend_" ++ f)) @field(p, f) = o;
        }
        return p;
    }

    fn shape(t: *const Probe) !h.hy.mla_attention.Shape {
        return h.hy.mla_attention.Shape.init(@intCast(t.in.rows), @intCast(t.in.heads), @intCast(t.in.stride));
    }

    pub fn prepare(t: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        t.in = (try inputs.build(t.a, t.v.lists, s, fill, seed, t.v.first_inf)) orelse return false;
        const in = &t.in;
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(in.q), std.mem.sliceAsBytes(in.cache), std.mem.sliceAsBytes(in.lists), std.mem.sliceAsBytes(in.counts), std.mem.sliceAsBytes(in.sinks) };
        for (bytes, &t.ins) |b, *r| r.* = try t.rig.input(b);
        t.parts = try t.rig.scratch((try t.shape()).scratchBytes());
        t.scores = if (t.v.path == .scored) try t.rig.scratch(in.rows * in.heads * in.stride * 4) else null;
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    /// One pass through mla_metal.zig's Attention.attend: the two-step path with key scores bound, else chunks.
    fn launch(t: *Probe, p: mla.Attention, two_step_path: bool) !void {
        const b: mla.Attention.Buffers = .{
            .query = ref(t.ins[0]),
            .cache = ref(t.ins[1]),
            .slots = ref(t.ins[2]),
            .counts = ref(t.ins[3]),
            .sinks = ref(t.ins[4]),
            .scratch = ref(t.parts),
            .output = ref(t.out),
            .key_scores = if (two_step_path) ref(t.scores.?) else null,
        };
        const sh = try t.shape();
        const pass = t.rig.begin();
        const encoded = p.attend(pass.e, b, sh);
        try t.rig.finish(pass);
        return encoded;
    }

    pub fn run(t: *Probe) !void {
        return t.launch(t.pipes(), t.v.path == .scored);
    }

    /// The one-step pass on the same inputs: the bytes the two-step pass must give.
    pub fn replaced(t: *Probe) ![]const u8 {
        try t.launch(t.plain.*, false);
        return t.rig.bytes(t.out, t.count() * 2);
    }

    pub fn count(t: *Probe) usize {
        return t.in.outputs();
    }

    pub fn reference(t: *Probe, want: []f64, mag: []f64) !void {
        try t.in.ranges();
        @memcpy(want, t.in.point);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    /// The range's allowance (attention_range.allowBf16); gate 4's exact sums keep the exact bound.
    pub fn bounds(t: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        try t.in.ranges();
        const s = spec(t.v, t.name);
        for (allow, want, t.in.point, t.in.half) |*l, w, p, hf| l.* = if (t.in.fill == .stress) c.boundExact(s, w) else ar.allowBf16(p, hf, 0, 0);
    }

    /// Each bf16 output within its range's rounded ends (attention_range.intervalBf16); gate 4's are open.
    pub fn intervals(t: *Probe, low: []f64, high: []f64) !void {
        try t.in.ranges();
        for (low, high, t.in.point, t.in.half) |*l, *hi, p, hf| {
            const e = if (t.in.fill == .stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(p, hf, 0, 0);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    pub fn simulate(t: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.in.simulate(acc, order, out);
    }
};

/// The engine's widths, so these shapes move with its config: 64 sliding heads, 32 sparse, a 576-word slot.
fn checkConfig(a: std.mem.Allocator) !void {
    const parsed = try h.text(a);
    defer parsed.deinit();
    const t = parsed.value.text_config;
    if (t.swa_num_attention_heads != sliding_heads or t.num_attention_heads != sparse_heads or t.sliding_window != inputs.window or
        t.kv_lora_rank != lat or lat + t.qk_rope_head_dim != inputs.key or lat + t.swa_qk_rope_head_dim != inputs.key)
        return error.Hy4AttentionGeometryMoved;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    /// The loader's own four pipelines (Attention.load), shared by every entry.
    plain: ?*mla.Attention = null,
    tests: fl.List(Probe) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (cases) |v| {
            var base: [192]u8 = undefined;
            var label: [320]u8 = undefined;
            const name = fl.named(baseName(v, &base), if (v.fault) |f| control(f).what else null, &label);
            if (!fl.wanted(only, name)) continue;
            if (s.plain == null) {
                try checkConfig(s.a);
                const p = try s.a.create(mla.Attention);
                errdefer s.a.destroy(p);
                p.* = try mla.Attention.load(s.device);
                s.plain = p;
            }
            try s.tests.add(s.a, s.kernels, try Probe.init(s.a, s.device, s.queue, s.plain.?, v));
        }
    }

    pub fn deinit(s: *Suite) void {
        s.tests.deinit(s.a);
        if (s.plain) |p| {
            p.deinit();
            s.a.destroy(p);
        }
    }
};

/// Gate 5's text for an `hy4 mla_attention ` launch: the source as Attention.load compiles it, or a control's edit.
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, name: []const u8) !?fl.Texts {
    if (!std.mem.startsWith(u8, name, stem)) return null;
    for (std.enums.values(Fault)) |f| {
        const ctl = control(f);
        const want = try std.fmt.allocPrint(a, stem ++ "control {s} {s}", .{ ctl.word, ctl.edit.kernel });
        defer a.free(want);
        if (!std.mem.eql(u8, name, want)) continue;
        const text = try fl.edited(a, sources.hy4_flash_mla_attention, ctl.edit);
        return .{ .here = text, .pre_m5 = text };
    }
    return .{ .here = sources.hy4_flash_mla_attention, .pre_m5 = sources.hy4_flash_mla_attention };
}
