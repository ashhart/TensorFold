//! DeepSeek V4.1 routed experts over EXL3 trellises on Metal: main's EXL3 rotations around the slots' dots.
const mtl = @import("metal");
const sources = @import("kernel_sources");
const exl3 = @import("../../../core/exl3_projection.zig");
const experts = @import("experts.zig");
pub const ref = @import("experts_exl3_ref.zig");

pub const Ref = experts.Ref;
pub const Route = experts.Route;
pub const Call = experts.Call;

/// main's EXL3 projection source (the rotations, the decoder), then the slots' dots and the gate/up epilogue.
pub const text = sources.core_exl3_projection ++ sources.dsv41.experts_exl3;

/// Each matrix's `exl3_projection.ExpertDescriptor` by expert id, one for each of the call's experts.
pub const Tables = struct { gate: Ref, up: Ref, down: Ref, count: u32 };

const Args = extern struct { k: u32, n: u32, pairs: u32, experts: u32, limit: f32 = 0 };

pub const Pipelines = struct {
    rotate_in: mtl.Pipeline,
    rotate_out: mtl.Pipeline,
    /// The slots' gate and up dots together, and the down's
    up: mtl.Pipeline,
    down: mtl.Pipeline,
    act: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, text);
    }

    /// The same kernels from other text: the contract's planted controls edit it.
    pub fn loadSource(device: mtl.Device, src: []const u8) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const ri = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_in", false);
        errdefer ri.deinit();
        const ro = try mtl.Pipeline.init(device, lib, "tf_exl3_rotate_out", false);
        errdefer ro.deinit();
        const u = try mtl.Pipeline.init(device, lib, "tf_dsv41_x3_up", false);
        errdefer u.deinit();
        const d = try mtl.Pipeline.init(device, lib, "tf_dsv41_x3_down", false);
        errdefer d.deinit();
        const a = try mtl.Pipeline.init(device, lib, "tf_dsv41_x3_act", false);
        errdefer a.deinit();
        for ([_]mtl.Pipeline{ ri, ro, u, d, a }) |q|
            if (q.simdWidth() != 32 or q.maxThreads() < 32) return error.UnsupportedExl3Simd;
        return .{ .rotate_in = ri, .rotate_out = ro, .up = u, .down = d, .act = a };
    }

    pub fn deinit(p: *Pipelines) void {
        inline for (.{ "rotate_in", "rotate_out", "up", "down", "act" }) |f| @field(p, f).deinit();
    }
};

/// A table shorter than the call's experts is refused: its pairs would leave rows the combine reads unwritten.
pub fn check(c: Call, t: Tables) !void {
    try experts.check(c);
    if (c.k % 128 != 0 or c.n % 128 != 0 or t.count > c.experts) return error.BadExpertShape;
    if (t.count < c.experts) return error.ShortExpertTable;
}

fn pairs(c: Call) u32 {
    return c.rows * c.topk;
}

fn align16(v: usize) usize {
    return (v + 15) & ~@as(usize, 15);
}

/// Scratch offsets: rotated rows (fp16 [matrices][pairs][k]), then their fp32 sums ([matrices][pairs][n]).
pub const Layout = struct { rotated: usize = 0, sums: usize, end: usize };

/// The up's scratch (gate and up of every pair, by pair index).
pub fn upLayout(c: Call) Layout {
    const sums = align16(2 * @as(usize, pairs(c)) * c.k * 2);
    return .{ .sums = sums, .end = sums + 2 * @as(usize, pairs(c)) * c.n * 4 };
}

/// The down's scratch (its rotated act and sums, by pair index).
pub fn downLayout(c: Call) Layout {
    const sums = align16(@as(usize, pairs(c)) * c.n * 2);
    return .{ .sums = sums, .end = sums + @as(usize, pairs(c)) * c.k * 4 };
}

pub fn scratchBytes(c: Call) usize {
    return @max(upLayout(c).end, downLayout(c).end);
}

fn at(r: Ref, extra: usize) Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

/// main's routed tf_exl3_rotate_in: pair p takes row p / repeat and its expert's scales.
fn rotateIn(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, out: Ref, table: Ref, r: Route, k: u32, repeat: u32, c: Call, count: u32) void {
    e.setPipeline(p.rotate_in);
    bind(e, &.{ x, x, out });
    e.setValue(exl3.Args{ .k = k, .n = k, .rows = pairs(c), .halves = 0, .codebook = 0, .su_signs = 0, .sv_signs = 0, .has_bias = 0, .output_f32 = 0, .input_ids = 0, .output_ids = 0, .routed = 0, .expert_id = 0, .repeat = repeat, .indexed = 1, .experts = count, .x_stride = k, .y_stride = 0, .tile_k_stride = 0, .tile_n_stride = 0 }, 3);
    for ([_]Ref{ r.ids, r.ids, table }, 4..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.dispatchGroups(.{ .width = k / 128, .height = pairs(c) }, .{ .width = 32 });
}

fn slots(c: Call) u32 {
    return @min(c.experts, pairs(c));
}

/// Gate and up of every routed pair, the clamped SwiGLU and route weight: act [pair][n] bf16, by pair index.
pub fn up(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, t: Tables, r: Route, act: Ref, scratch: Ref, c: Call, limit: f32) void {
    const l = upLayout(c);
    const half = @as(usize, pairs(c)) * c.k * 2;
    rotateIn(p, e, x, at(scratch, 0), t.gate, r, c.k, c.topk, c, t.count);
    rotateIn(p, e, x, at(scratch, half), t.up, r, c.k, c.topk, c, t.count);
    e.barrier();
    const a: Args = .{ .k = c.k, .n = c.n, .pairs = pairs(c), .experts = t.count, .limit = limit };
    e.setPipeline(p.up);
    bind(e, &.{ scratch, t.gate, t.up, r.slots, r.nslots, r.pairs, at(scratch, l.sums) });
    e.setValue(a, 7);
    e.dispatchGroups(.{ .width = c.n / 32, .height = slots(c) }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.act);
    bind(e, &.{ at(scratch, l.sums), t.gate, t.up, r.ids, r.weights, act });
    e.setValue(a, 6);
    e.dispatchGroups(.{ .width = c.n / 128, .height = pairs(c) }, .{ .width = 32 });
    e.barrier();
}

/// The down of act [pair][n] into out [pair position][k] bf16: experts.zig's layout, for its combine.
pub fn down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, t: Tables, r: Route, out: Ref, scratch: Ref, c: Call) void {
    const l = downLayout(c);
    rotateIn(p, e, act, scratch, t.down, r, c.n, 1, c, t.count);
    e.barrier();
    e.setPipeline(p.down);
    bind(e, &.{ scratch, t.down, t.down, r.slots, r.nslots, r.pairs, at(scratch, l.sums) });
    e.setValue(Args{ .k = c.n, .n = c.k, .pairs = pairs(c), .experts = t.count }, 7);
    e.dispatchGroups(.{ .width = c.k / 32, .height = slots(c) }, .{ .width = 32 });
    e.barrier();
    e.setPipeline(p.rotate_out);
    bind(e, &.{ at(scratch, l.sums), scratch, scratch, out });
    e.setValue(exl3.Args{ .k = c.n, .n = c.k, .rows = pairs(c), .halves = 0, .codebook = 0, .su_signs = 0, .sv_signs = 0, .has_bias = 0, .output_f32 = 0, .input_ids = 0, .output_ids = 1, .routed = 0, .expert_id = 0, .repeat = 1, .indexed = 1, .experts = t.count, .x_stride = 0, .y_stride = c.k, .tile_k_stride = 0, .tile_n_stride = 0 }, 4);
    for ([_]Ref{ r.where, r.ids, t.down }, 5..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.dispatchGroups(.{ .width = c.k / 128, .height = pairs(c) }, .{ .width = 32 });
    e.barrier();
}
