//! A Metal routed Hy4-flash feed-forward block, including the shared branch and residual merge.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const affine = @import("row_projection");
const route = @import("moe_route");
const experts = @import("affine_experts_metal.zig");
const dense = @import("metal_dense.zig");
const weights = @import("model_weights.zig");
const config = @import("config.zig");
const gpu_experts = @import("metal_tensor_experts.zig");
const lane = @import("lane_projection");

const Ref = dense.Ref;
const RouteRef = @import("draft_ops").Ref;
const RouteOutputs = route.Outputs(RouteRef);
pub const max_rows = 16;

const ExpertWeights = union(enum) {
    portable: struct { up: experts.Weights, down: experts.Weights },
    tensor: gpu_experts.Weights,
};

pub const Layer = struct {
    own: dense.Buffers,
    merge: dense.Merge,
    input_norm: mtl.Buffer,
    output_norm: mtl.Buffer,
    router: RouteRef,
    correction: RouteRef,
    down: affine.Weights,
    up: affine.Weights,
    expert_weights: ExpertWeights,
    shared_gate: affine.Weights,
    shared_up: affine.Weights,
    shared_down: affine.Weights,
    precision: experts.Precision,
    size: experts.Shape,
    scale: f32,

    pub fn upload(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, t: config.Text) !Layer {
        return uploadForModel(a, device, w, t, false);
    }

    pub fn uploadForModel(a: std.mem.Allocator, device: mtl.Device, w: *const weights.TargetLayer, t: config.Text, use_tensors: bool) !Layer {
        if (w.feed != .routed) return error.Hy4RoutedFeedRequired;
        const f = w.feed.routed;
        if (f.experts.ids.first != 0 or f.experts.ids.count != t.n_routed_experts) return error.Hy4MetalNeedsWholeExpertSet;
        var b: dense.Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const packed_router = try b.alloc(f.router.bytes.len);
        const router = try a.alloc(u16, f.router.bytes.len / @sizeOf(u16));
        defer a.free(router);
        @memcpy(std.mem.sliceAsBytes(router), f.router.bytes);
        route.pack(router, packed_router.slice(u16, router.len), t.n_routed_experts, t.hidden_size);
        const precision: experts.Precision = switch (f.experts.gate_up.bits) {
            4 => .q4,
            8 => .q8,
            else => return error.BadHy4ExpertBits,
        };
        var out: Layer = .{
            .own = undefined,
            .merge = try dense.Merge.upload(&b, w.mlp_hc),
            .input_norm = try b.upload(w.mlp_norm.bytes),
            .output_norm = try b.upload(f.norm.bytes),
            .router = .{ .buf = packed_router },
            .correction = .{ .buf = try b.upload(f.correction.bytes) },
            .down = try b.matrix(f.down),
            .up = try b.matrix(f.up),
            .expert_weights = if (use_tensors)
                .{ .tensor = try gpu_experts.Weights.upload(&b, f.experts.gate_up, f.experts.down) }
            else
                .{ .portable = .{ .up = try uploadExperts(&b, f.experts.gate_up), .down = try uploadExperts(&b, f.experts.down) } },
            .shared_gate = try b.matrix(f.shared.gate),
            .shared_up = try b.matrix(f.shared.up),
            .shared_down = try b.matrix(f.shared.down),
            .precision = precision,
            .size = .{
                .width = t.routed_expert_hidden_size,
                .inner = t.moe_intermediate_size,
                .first = 0,
                .experts = t.n_routed_experts,
                .rows = 1,
                .pairs = t.num_experts_per_tok,
                .swiglu_limit = t.swiglu_limit,
            },
            .scale = t.routed_scaling_factor,
        };
        try out.size.check();
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Layer) void {
        self.own.clear();
        self.* = undefined;
    }
};

fn uploadExperts(b: *dense.Buffers, w: @import("core").quant.mlx.Host) !experts.Weights {
    return .{
        .words = .{ .buffer = try b.upload(w.words.bytes) },
        .scales = .{ .buffer = try b.upload(w.scales.bytes) },
        .biases = .{ .buffer = try b.upload(w.biases.bytes) },
    };
}

fn raw(b: *dense.Buffers, elements: usize) !RouteRef {
    return .{ .buf = try b.alloc(elements * @sizeOf(u32)) };
}

pub const Scratch = struct {
    own: dense.Buffers,
    rows: u32,
    picks: u32,
    logits: RouteRef,
    select: RouteOutputs,
    expert_rows: ?mtl.Buffer,
    tensors: ?gpu_experts.Scratch,
    latent: Ref,
    weighted: Ref,
    pair_activations: Ref,
    pair_outputs: mtl.Buffer,
    routed: Ref,
    shared: Ref,

    pub fn init(a: std.mem.Allocator, d: mtl.Device, t: config.Text, rows: u32) !Scratch {
        return initForModel(a, d, t, rows, false);
    }

    pub fn initForModel(a: std.mem.Allocator, d: mtl.Device, t: config.Text, rows: u32, use_tensors: bool) !Scratch {
        if (rows == 0 or rows > max_rows) return error.BadHy4RouteRows;
        var b: dense.Buffers = .{ .a = a, .d = d };
        errdefer b.clear();
        const slots = t.num_experts_per_tok;
        const total = rows * slots;
        const e = t.n_routed_experts;
        const width = t.routed_expert_hidden_size;
        const si = if (use_tensors) null else (try raw(&b, total)).buf;
        if (si) |ids| for (ids.slice(i32, total), 0..) |*dst, i| {
            dst.* = @intCast(i / slots);
        };
        var out: Scratch = .{
            .own = undefined,
            .rows = rows,
            .picks = slots,
            .logits = try raw(&b, rows * e),
            .select = .{
                .pick = try raw(&b, total),
                .wts = try raw(&b, total),
                .ids = try raw(&b, e),
                .members = try raw(&b, e * max_rows),
                .count = try raw(&b, 1),
                .mine = try raw(&b, total),
                .theirs = try raw(&b, total),
                .counts = try raw(&b, 2),
                .word = try raw(&b, 1),
            },
            .expert_rows = si,
            .tensors = if (use_tensors) try gpu_experts.Scratch.init(&b, rows, slots, width, t.moe_intermediate_size) else null,
            .latent = try b.rows(rows, width),
            .weighted = try b.rows(rows, width),
            .pair_activations = try b.rows(total, t.moe_intermediate_size),
            .pair_outputs = (try raw(&b, @as(usize, total) * width)).buf,
            .routed = try b.rows(rows, t.hidden_size),
            .shared = try b.rows(rows, t.hidden_size),
        };
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.own.clear();
        self.* = undefined;
    }
};

pub const Kernels = struct {
    shape: route.Shape,
    logits: mtl.Pipeline,
    select: mtl.Pipeline,
    experts: ?experts.Kernels,
    tensors: ?gpu_experts.Kernels,
    sum: mtl.Pipeline,
    add: mtl.Pipeline,

    pub fn load(a: std.mem.Allocator, device: mtl.Device, t: config.Text) !Kernels {
        return loadImpl(a, device, t, null);
    }

    pub fn loadForModel(a: std.mem.Allocator, device: mtl.Device, t: config.Text, bits: u8) !Kernels {
        return loadImpl(a, device, t, if (device.tensorUnits()) bits else null);
    }

    fn loadImpl(a: std.mem.Allocator, device: mtl.Device, t: config.Text, tensor_bits: ?u8) !Kernels {
        const shape: route.Shape = .{ .hidden = t.hidden_size, .experts = t.n_routed_experts, .topk = t.num_experts_per_tok, .max_rows = max_rows };
        const src = try route.source(a, shape);
        defer a.free(src);
        const routes = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer routes.deinit();
        const sums = try mtl.Library.fromSource(device, sources.hy4_flash_affine_experts, mtl.CompileOptions.mlx());
        defer sums.deinit();
        const adds = try mtl.Library.fromSource(device, sources.elementwise, mtl.CompileOptions.mlx());
        defer adds.deinit();
        var logits = try mtl.Pipeline.init(device, routes, route.names[0], false);
        errdefer logits.deinit();
        var select = try mtl.Pipeline.init(device, routes, route.names[1], false);
        errdefer select.deinit();
        var expert = if (tensor_bits == null) try experts.Kernels.load(device) else null;
        errdefer if (expert) |*op| op.deinit();
        const tensors = if (tensor_bits) |bits| try gpu_experts.Kernels.load(a, device, bits) else null;
        errdefer if (tensors) |op| op.deinit();
        var sum = try mtl.Pipeline.init(device, sums, "tf_hy4_routed_sum", false);
        errdefer sum.deinit();
        return .{
            .shape = shape,
            .logits = logits,
            .select = select,
            .experts = expert,
            .tensors = tensors,
            .sum = sum,
            .add = try mtl.Pipeline.init(device, adds, "tf_add_bf16", false),
        };
    }

    pub fn deinit(self: *Kernels) void {
        self.logits.deinit();
        self.select.deinit();
        if (self.experts) |*op| op.deinit();
        if (self.tensors) |op| op.deinit();
        self.sum.deinit();
        self.add.deinit();
    }

    pub fn forward(self: *const Kernels, shared: *const dense.Kernels, e: mtl.ComputeEncoder, w: *const Layer, d: *const dense.Scratch, s: *const Scratch, x: Ref, out: Ref) !void {
        const rows = d.shape.rows;
        const inner: u32 = @intCast(w.shared_gate.n);
        if (rows > s.rows or s.picks != self.shape.topk or inner > d.inner_capacity) return error.ShortHy4Scratch;
        const merge = try shared.collapse(e, x, w.merge, w.input_norm, d);
        try shared.project(e, w.down, d.normalized, s.latent, rows);
        try shared.project(e, w.shared_gate, d.normalized, d.gate, rows);
        try shared.project(e, w.shared_up, d.normalized, d.up, rows);
        route.logits(e, self.logits, self.shape, dense.Kernels.dref(d.normalized), w.router, s.logits, rows);
        e.barrier();
        route.select(e, self.select, s.logits, w.correction, w.scale, rows, .{ 0, w.size.experts }, s.select);
        try shared.dense.swiglu(e, dense.Kernels.dref(d.gate), dense.Kernels.dref(d.up), dense.Kernels.dref(d.inner), .{
            .rows = rows,
            .width = inner,
            .gate_stride = inner,
            .up_stride = inner,
            .y_stride = inner,
        });
        e.barrier();
        var shape = w.size;
        shape.rows = rows;
        shape.pairs = rows * s.picks;
        switch (w.expert_weights) {
            .portable => |mat| {
                const ops = self.experts orelse return error.Hy4ExpertKernelsMissing;
                const ids: experts.Ref = .{ .buffer = s.select.pick.buf };
                try ops.run(e, w.precision, .up, .{
                    .input = .{ .buffer = s.latent.buffer },
                    .ids = ids,
                    .src_rows = .{ .buffer = s.expert_rows orelse return error.ShortHy4Scratch },
                    .weights = mat.up,
                    .output = .{ .buffer = s.pair_activations.buffer },
                }, shape);
                try shared.project(e, w.shared_down, d.inner, s.shared, rows);
                e.barrier();
                try ops.run(e, w.precision, .down, .{
                    .input = .{ .buffer = s.pair_activations.buffer },
                    .ids = ids,
                    .weights = mat.down,
                    .output = .{ .buffer = s.pair_outputs },
                }, shape);
            },
            .tensor => |mat| {
                const ops = self.tensors orelse return error.Hy4ExpertKernelsMissing;
                const tensor_scratch = s.tensors orelse return error.ShortHy4Scratch;
                const latent: lane.Ref = .{ .buf = s.latent.buffer, .off = s.latent.offset };
                const jobs = ops.prepare(e, tensor_scratch, .{
                    .pick = tensorRef(s.select.pick),
                    .ids = tensorRef(s.select.ids),
                    .members = tensorRef(s.select.members),
                    .count = tensorRef(s.select.count),
                }, latent, rows, shape.pairs, max_rows, shape.width);
                try shared.project(e, w.shared_down, d.inner, s.shared, rows);
                e.barrier();
                ops.apply(e, mat, tensor_scratch, jobs, latent, .{ .buf = s.pair_activations.buffer, .off = s.pair_activations.offset }, .{ .buf = s.pair_outputs }, shape.pairs, s.picks, shape.swiglu_limit);
            },
        }
        e.barrier();
        e.setPipeline(self.sum);
        e.setBuffer(s.pair_outputs, 0, 0);
        e.setBuffer(s.select.wts.buf, 0, 1);
        e.setBuffer(s.weighted.buffer, 0, 2);
        e.setValue([2]u32{ shape.width, s.picks }, 3);
        e.dispatchThreads(mtl.Size.of(shape.width, rows, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        try shared.project(e, w.up, s.weighted, s.routed, rows);
        e.barrier();
        e.setPipeline(self.add);
        e.setBuffer(s.routed.buffer, 0, 0);
        e.setBuffer(s.shared.buffer, 0, 1);
        e.setValue(@as(i32, @intCast(rows * merge.width)), 2);
        e.setBuffer(d.branch.buffer, 0, 3);
        e.dispatchThreads(mtl.Size.of(rows * merge.width, 1, 1), mtl.Size.of(128, 1, 1));
        e.barrier();
        try shared.norm(e, d.branch, w.output_norm, d.normalized, rows, merge.width, merge.eps);
        e.barrier();
        try shared.residual(e, x, d.normalized, d.post, out, merge);
    }
};

fn tensorRef(ref: RouteRef) lane.Ref {
    return .{ .buf = ref.buf, .off = ref.off };
}
