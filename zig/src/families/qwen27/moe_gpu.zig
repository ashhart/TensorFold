//! The routed block on the GPU: router, route, experts by id (row kernels) or sorted (core.affine_mm), SwiGLU, combine.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const row = core.row_projection;
const am = core.affine_mm;
const source = @import("qwen_runtime_sources");
const profile = core.gpu_profile;
const Config = @import("config.zig").Config;
const wts = @import("gpu_weights.zig");
const Frame = @import("gpu_frame.zig").Frame;
const Ref = @import("projection.zig").Ref;

const RouteArgs = extern struct { experts: i32, topk: i32, ld: i32, renorm: i32 };
const SortArgs = extern struct { slots: i32, experts: i32 };
const GluArgs = extern struct { width: i32 };
const CombineArgs = extern struct { hidden: i32, topk: i32, shared_in_out: i32, pad: i32 = 0 };
const R = struct { buf: mtl.Buffer, off: usize = 0 };
const Matrix = struct { w: R, s: R, b: R, n: u32, k: u32 };
const Key = struct { bits: u8, group: u16 };

pub const Kernels = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    route: mtl.Pipeline,
    sort: mtl.Pipeline,
    gather: mtl.Pipeline,
    glu: mtl.Pipeline,
    combine: mtl.Pipeline,
    rows: ?row.Pipelines = null,
    /// core.affine_mm's pipelines by the experts' width, built when a prompt first takes the tensor units.
    nax: std.AutoHashMapUnmanaged(Key, am.Pipes) = .empty,
    profiler: ?*profile.Trace = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) !Kernels {
        const lib = try mtl.Library.fromSource(device, source.moe, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [5]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |p| p.deinit();
        for ([_][:0]const u8{ "q27_route", "q27_route_sort", "q27_route_gather", "q27_expert_glu", "q27_expert_combine" }, &pipes) |name, *p| {
            p.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return .{ .allocator = a, .device = device, .route = pipes[0], .sort = pipes[1], .gather = pipes[2], .glu = pipes[3], .combine = pipes[4] };
    }

    pub fn deinit(k: *Kernels) void {
        inline for (.{ k.route, k.sort, k.gather, k.glu, k.combine }) |p| p.deinit();
        if (k.rows) |*p| p.deinit();
        var it = k.nax.valueIterator();
        while (it.next()) |pipes| for (pipes) |p| p.deinit();
        k.nax.deinit(k.allocator);
    }

    fn rowPipelines(k: *Kernels) !*const row.Pipelines {
        if (k.rows == null) k.rows = try row.Pipelines.load(k.device);
        return &k.rows.?;
    }

    fn naxPipes(k: *Kernels, key: Key) !am.Pipes {
        if (k.nax.get(key)) |p| return p;
        const src = try am.source(k.device, k.allocator, .{ .bits = key.bits, .group = key.group });
        defer k.allocator.free(src);
        const lib = try mtl.Library.fromSource(k.device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: am.Pipes = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |p| p.deinit();
        for (am.names, &pipes) |name, *p| {
            p.* = try mtl.Pipeline.init(k.device, lib, name, false);
            made += 1;
        }
        try k.nax.put(k.allocator, key, pipes);
        return pipes;
    }

    /// The sorted tensor-unit path: prompt chunks of whole tiles only; decode windows keep the row kernels (exact).
    pub fn sorted(k: *Kernels, mo: *const wts.Moe, rows: u32, prompt: bool) bool {
        if (!prompt or rows < 16 or !k.device.tensorUnits()) return false;
        const g = mo.gate.one;
        return g.bits == mo.up.one.bits and g.group == mo.up.one.group and g.bits == mo.down.one.bits and g.group == mo.down.one.group;
    }

    /// f.output becomes the weighted experts plus the gated shared expert (f.output for a prompt chunk, else a slot).
    pub fn block(k: *Kernels, e: mtl.ComputeEncoder, c: Config, mo: *const wts.Moe, f: *const Frame, rows: u32, prompt: bool) !void {
        var span = profile.begin(k.profiler, e, .route);
        errdefer span.cancel();
        const enc = span.encoder;
        const slots = f.slotsOf(rows); // each row's picks and its shared expert
        const experts: u32 = @intCast(c.experts);
        const topk: u32 = @intCast(c.top_k);
        const hidden: u32 = @intCast(c.hidden);
        const inter: u32 = @intCast(c.moe_intermediate);
        const routed = rows * topk; // the sorted path's slots: the picks alone
        const units = k.sorted(mo, rows, prompt);
        // the router's bf16 logits [rows, ld] (column `experts`: the shared gate), on tensor units or the row kernel
        const ld: u32 = @intCast(mo.router.n); // a multiple of 64: both paths write that pitch
        const pipes = try k.rowPipelines();
        if (units) {
            const p = try k.naxPipes(.{ .bits = mo.router.bits, .group = @intCast(mo.router.group) });
            const x = R{ .buf = f.get(.input).buffer };
            const sums = R{ .buf = f.get(.expert_sums).buffer };
            am.rowSums(enc, p, @intCast(mo.router.group), x, sums, rows, hidden);
            enc.barrier();
            am.dense(enc, p, x, sums, matrix(mo.router), R{ .buf = f.get(.router).buffer }, rows);
        } else {
            const call = try row.call(pipes, mo.router, rows, false);
            enc.setPipeline(call.pipeline);
            enc.setBuffer(f.get(.input).buffer, 0, 0);
            enc.setBuffer(mo.router.w, mo.router.w_off, 1);
            enc.setBuffer(mo.router.scales, mo.router.s_off, 2);
            enc.setBuffer(mo.router.biases, mo.router.b_off, 3);
            enc.setValue(call.dims, 4);
            enc.setBuffer(f.get(.router).buffer, 0, 5);
            enc.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
        }
        enc.barrier();
        enc.setPipeline(k.route);
        enc.setBuffer(f.get(.router).buffer, 0, 0);
        enc.setBuffer(f.get(.picks).buffer, 0, 1);
        enc.setBuffer(f.get(.route_weights).buffer, 0, 2);
        enc.setValue(RouteArgs{ .experts = @intCast(experts), .topk = @intCast(topk), .ld = @intCast(ld), .renorm = @intFromBool(c.renorm) }, 3);
        enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(@max(32, (experts + 31) / 32 * 32), 1, 1));
        enc.barrier();
        try span.end();
        var products = profile.begin(k.profiler, e, .experts);
        errdefer products.cancel();
        const enc2 = products.encoder;
        if (units) {
            const p = try k.naxPipes(.{ .bits = mo.gate.one.bits, .group = @intCast(mo.gate.one.group) });
            enc2.setPipeline(k.sort);
            enc2.setBuffer(f.get(.picks).buffer, 0, 0);
            enc2.setBuffer(f.get(.route_weights).buffer, 0, 1);
            enc2.setBuffer(f.get(.offsets).buffer, 0, 2);
            enc2.setBuffer(f.get(.order).buffer, 0, 3);
            enc2.setBuffer(f.get(.sorted_weights).buffer, 0, 4);
            enc2.setValue(SortArgs{ .slots = @intCast(slots), .experts = @intCast(experts) }, 5); // the shared slots skipped
            enc2.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            enc2.barrier();
            enc2.setPipeline(k.gather);
            enc2.setBuffer(f.get(.input).buffer, 0, 0);
            enc2.setBuffer(f.get(.order).buffer, 0, 1);
            enc2.setBuffer(f.get(.sorted_rows).buffer, 0, 2);
            enc2.setValue([2]i32{ @intCast(hidden), @intCast(topk + 1) }, 3);
            enc2.dispatchThreads(mtl.Size.of(hidden, routed, 1), mtl.Size.of(256, 1, 1));
            enc2.barrier();
            const sums = R{ .buf = f.get(.expert_sums).buffer };
            const offsets = R{ .buf = f.get(.offsets).buffer };
            const x = R{ .buf = f.get(.sorted_rows).buffer };
            am.rowSums(enc2, p, @intCast(mo.gate.one.group), x, sums, routed, hidden);
            enc2.barrier();
            am.gather(enc2, p, x, sums, matrix(mo.gate.one), offsets, R{ .buf = f.get(.expert_gate).buffer }, routed, experts);
            am.gather(enc2, p, x, sums, matrix(mo.up.one), offsets, R{ .buf = f.get(.expert_up).buffer }, routed, experts);
            enc2.barrier();
            k.activation(enc2, f, routed, inter);
            enc2.barrier();
            const act = R{ .buf = f.get(.expert_act).buffer };
            am.rowSums(enc2, p, @intCast(mo.down.one.group), act, sums, routed, inter);
            enc2.barrier();
            am.gatherTo(enc2, p, act, sums, matrix(mo.down.one), offsets, R{ .buf = f.get(.expert_out).buffer }, R{ .buf = f.get(.order).buffer }, routed, experts);
        } else {
            // every slot through the experts' kernel, the shared expert as the extra matrix of each product
            try k.indexed(enc2, pipes, mo.gate, mo.shared_gate, slots, f.get(.input), f.get(.picks), f.get(.expert_gate));
            try k.indexed(enc2, pipes, mo.up, mo.shared_up, slots, f.get(.input), f.get(.picks), f.get(.expert_up));
            enc2.barrier();
            k.activation(enc2, f, slots, inter);
            enc2.barrier();
            try k.indexed(enc2, pipes, mo.down, mo.shared_down, slots, f.get(.expert_act), f.get(.picks), f.get(.expert_out));
        }
        enc2.barrier();
        enc2.setPipeline(k.combine);
        enc2.setBuffer(f.get(.expert_out).buffer, 0, 0);
        enc2.setBuffer(f.get(.route_weights).buffer, 0, 1);
        enc2.setBuffer(f.get(.output).buffer, 0, 2);
        enc2.setValue(CombineArgs{ .hidden = @intCast(hidden), .topk = @intCast(topk), .shared_in_out = @intFromBool(units) }, 3);
        enc2.dispatchThreads(mtl.Size.of(hidden, rows, 1), mtl.Size.of(256, 1, 1));
        try products.end();
    }

    fn activation(k: *Kernels, e: mtl.ComputeEncoder, f: *const Frame, slots: u32, inter: u32) void {
        e.setPipeline(k.glu);
        e.setBuffer(f.get(.expert_gate).buffer, 0, 0);
        e.setBuffer(f.get(.expert_up).buffer, 0, 1);
        e.setBuffer(f.get(.expert_act).buffer, 0, 2);
        e.setValue(GluArgs{ .width = @intCast(inter) }, 3);
        e.dispatchThreads(mtl.Size.of(inter, slots, 1), mtl.Size.of(256, 1, 1));
    }

    /// Slot i takes expert ids[i] on input row i / repeat; id `experts` is the shared expert's `extra` matrix.
    fn indexed(_: *Kernels, e: mtl.ComputeEncoder, pipes: *const row.Pipelines, x: row.Experts, extra: row.Weights, slots: u32, in: Ref, ids: Ref, out: Ref) !void {
        const call = try row.callIndexedExtra(pipes, x, extra, slots);
        e.setPipeline(call.pipeline);
        e.setBuffer(in.buffer, in.offset, 0);
        e.setBuffer(x.one.w, x.one.w_off, 1);
        e.setBuffer(x.one.scales, x.one.s_off, 2);
        e.setBuffer(x.one.biases, x.one.b_off, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(out.buffer, out.offset, 5);
        e.setBuffer(ids.buffer, ids.offset, 6);
        e.setBuffer(extra.w, extra.w_off, 7);
        e.setBuffer(extra.scales, extra.s_off, 8);
        e.setBuffer(extra.biases, extra.b_off, 9);
        e.dispatchThreads(mtl.Size.of(call.groups * call.threads, call.slots, 1), mtl.Size.of(call.threads, 1, 1));
    }
};

fn matrix(w: row.Weights) Matrix {
    return .{ .w = .{ .buf = w.w, .off = w.w_off }, .s = .{ .buf = w.scales, .off = w.s_off }, .b = .{ .buf = w.biases, .off = w.b_off }, .n = @intCast(w.n), .k = @intCast(w.k) };
}
