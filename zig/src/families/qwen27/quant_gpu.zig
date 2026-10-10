//! Runtime specialization preserves the original M5 group arithmetic and GPU-resident argument buffers.
const std = @import("std");
const mtl = @import("metal");
const lane = @import("core").lane_projection;
const row = @import("core").row_projection;
const sources = @import("qwen_runtime_sources");
const projection = @import("projection.zig");
const profile = @import("core").gpu_profile;
pub const Ref = projection.Ref;
/// Group sums the input's producer wrote: XS[group * stride + row], in the lane projection's order.
pub const Sums = struct { ref: Ref, stride: u32 };

const Key = struct { kind: Kind, n: u32 = 0, k: u32, slices: u32 = 0, tile: u32 = 0, tmr: u32 = 0, edge: bool = false, skip: u32 = 0, cut: [2]u32 = .{ 0, 0 } };
pub const Kind = enum { projection, xsum, norm, input_norm, mlp, ane_input, mlp_ane, add_ane };

pub const Kernels = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    profiler: ?*profile.Trace = null,
    row_pipelines: ?row.Pipelines = null,
    simd: ?row.simd.Pipelines = null,
    /// Per [n, k] shape: whether the scalar twin's 1-2 row bits equal the tiles' (checked once on this chip).
    twins: std.AutoHashMapUnmanaged([2]usize, bool) = .empty,
    /// Set around prompt chunks: prompt rows take projection.promptSplit, decoded rows the decode split.
    prompt: bool = false,
    lanes: std.AutoHashMapUnmanaged(Key, lane.Projection) = .empty,
    cache: std.AutoHashMapUnmanaged(Key, mtl.Pipeline) = .empty,

    pub fn deinit(k: *Kernels) void {
        if (k.row_pipelines) |*pipelines| pipelines.deinit();
        if (k.simd) |*pipelines| pipelines.deinit();
        k.twins.deinit(k.allocator);
        var lane_it = k.lanes.valueIterator();
        while (lane_it.next()) |p| p.deinit();
        k.lanes.deinit(k.allocator);
        var it = k.cache.valueIterator();
        while (it.next()) |p| p.deinit();
        k.cache.deinit(k.allocator);
    }

    fn get(k: *Kernels, key: Key) !mtl.Pipeline {
        if (k.cache.get(key)) |p| return p;
        const template = switch (key.kind) {
            .projection => unreachable,
            .xsum => sources.xsum,
            .norm => sources.norm_xs,
            .input_norm => sources.norm_input,
            .mlp => sources.mlp_xs,
            .ane_input, .mlp_ane, .add_ane => sources.mlp_ane,
        };
        const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
        const definitions = switch (key.kind) {
            .projection => unreachable,
            .xsum => try std.fmt.allocPrint(k.allocator, "constexpr int K={d}, GS=64;", .{key.k}),
            .norm, .input_norm => try std.fmt.allocPrint(k.allocator, "constexpr int K={d};", .{key.k}),
            .mlp => try std.fmt.allocPrint(k.allocator, "constexpr int N={d};", .{key.n}),
            .ane_input, .mlp_ane, .add_ane => try std.fmt.allocPrint(k.allocator, "constant constexpr int H={d}, N={d}, A={d};", .{ key.k, key.n, key.tile }),
        };
        defer k.allocator.free(definitions);
        const source = try std.mem.concat(k.allocator, u8, &.{ template[0..mark], definitions, template[mark + "Q27_SPECIALIZE".len ..] });
        defer k.allocator.free(source);
        const library = try mtl.Library.fromSource(k.device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const name: [:0]const u8 = switch (key.kind) {
            .projection => unreachable,
            .xsum => "custom_kernel_q27_xsum_bfloat16_t_int32_t_float",
            .norm => "custom_kernel_q27_norm_xs_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_bfloat16_t_float",
            .input_norm => "custom_kernel_q27_norm_input_bfloat16_t_bfloat16_t_floatc_int32_t_bfloat16_t_float",
            .mlp => "custom_kernel_q27_mlp_xs_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_float",
            .ane_input => "q27_ane_input",
            .mlp_ane => "q27_mlp_ane",
            .add_ane => "q27_add_ane",
        };
        const p = try mtl.Pipeline.init(k.device, library, name, false);
        errdefer p.deinit();
        try k.cache.put(k.allocator, key, p);
        return p;
    }

    pub fn quant(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32) !void {
        return k.quantSkip(input_encoder, p, x, xs, dims, y, rows, 0);
    }

    /// `quant` over input groups [cut[0], cut[0] + cut[1]) only, into fp32 partial sums.
    pub fn quantCut(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: Sums, dims: Ref, y: Ref, rows: u32, cut: [2]u32) !void {
        return k.quantImpl(input_encoder, p, x, xs, dims, y, rows, 0, cut);
    }

    /// `quant` over a fused two-half projection (the MLP's gate_up) without the first `skip` tiles of each half.
    pub fn quantSkip(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, skip: u32) !void {
        return k.quantImpl(input_encoder, p, x, xs, dims, y, rows, skip, .{ 0, 0 });
    }

    fn quantImpl(k: *Kernels, input_encoder: mtl.ComputeEncoder, p: projection.Linear, x: Ref, xs: ?Sums, dims: Ref, y: Ref, rows: u32, skip: u32, cut: [2]u32) !void {
        if ((skip > 0 or cut[1] > 0) and p.raw != null) return error.BadCoreProjectionRows;
        if (skip > 0 and (p.raw != null or skip >= p.n / 64)) return error.BadCoreProjectionRows;
        var span = profile.begin(k.profiler, input_encoder, .dense);
        const sk = if (k.prompt) projection.promptSplit(p.n, p.k) orelse p.slices else p.slices;
        span.shape = .{ .n = p.n, .k = p.k, .sk = sk };
        errdefer span.cancel();
        const e = span.encoder;

        _ = dims;
        if (rows == 0 or rows > 128 or p.tile != 32) return error.BadCoreProjectionRows;
        if (p.raw) |raw| if (raw.bits == 4 and raw.group == 64 and raw.sum == .f32 and raw.n % 8 == 0) {
            const m = row.simd.Matrix{ .w = raw.w, .w_off = raw.w_off, .scales = raw.scales, .s_off = raw.s_off, .biases = raw.biases, .b_off = raw.b_off, .n = raw.n, .k = raw.k };
            const scalar = try k.twin(m); // loads the tiles before they're read
            const call = try row.simd.call(&k.simd.?, raw.n, raw.k, rows, scalar);
            row.simd.encode(e, call, m, x.buffer, x.offset, y.buffer, y.offset);
            e.barrier();
            try span.end();
            return;
        };
        if (p.raw) |raw| {
            if (k.row_pipelines == null) k.row_pipelines = try row.Pipelines.load(k.device);
            const call = try row.call(&k.row_pipelines.?, raw, rows, false);
            e.setPipeline(call.pipeline);
            e.setBuffer(x.buffer, x.offset, 0);
            e.setBuffer(raw.w, raw.w_off, 1);
            e.setBuffer(raw.scales, raw.s_off, 2);
            e.setBuffer(raw.biases, raw.b_off, 3);
            e.setValue(call.dims, 4);
            e.setBuffer(y.buffer, y.offset, 5);
            e.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
            e.barrier();
            try span.end();
            return;
        }
        const key = Key{ .kind = .projection, .n = p.n, .k = p.k, .slices = sk, .skip = skip, .cut = cut };
        const half = p.n / 64;
        const ranges = [_][2]usize{ .{ skip, half - skip }, .{ half + skip, half - skip } };
        const entry = try k.lanes.getOrPut(k.allocator, key);
        if (!entry.found_existing) {
            errdefer _ = k.lanes.remove(key);
            entry.value_ptr.* = try lane.Projection.init(k.allocator, k.device, .{ .n = p.n, .k = p.k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .ranges = if (skip > 0) &ranges else &.{}, .groups = if (cut[1] > 0) .{ cut[0], cut[1] } else null, .output = if (cut[1] > 0) .f32 else .bf16, .precompute_sums = true, .reg = true });
        }
        const step: u32 = if (entry.value_ptr.reg) lane.reg_rows else 16;
        var first: u32 = 0;
        while (first < rows) : (first += step) {
            const xr = lane.Ref{ .buf = x.buffer, .off = x.offset + @as(usize, first) * p.k * 2 };
            const w = lane.Ref{ .buf = p.words.buffer, .off = p.words.offset };
            const sb = lane.Ref{ .buf = p.pairs.buffer, .off = p.pairs.offset };
            const yr = lane.Ref{ .buf = y.buffer, .off = y.offset + @as(usize, first) * p.n * @as(usize, if (cut[1] > 0) 4 else 2) };
            if (xs) |s| try entry.value_ptr.encodeSums(e, xr, w, sb, .{ .ref = .{ .buf = s.ref.buffer, .off = s.ref.offset + @as(usize, first) * 4 }, .stride = s.stride }, yr, @min(rows - first, step)) else try entry.value_ptr.encode(e, xr, w, sb, yr, @min(rows - first, step));
        }
        try span.end();
    }

    /// Loads the tiles on first use and checks the scalar twin against them once per shape.
    fn twin(k: *Kernels, m: row.simd.Matrix) !bool {
        if (k.simd == null) k.simd = try row.simd.Pipelines.load(k.device);
        if (k.twins.get(.{ m.n, m.k })) |ok| return ok;
        const ok = try row.simd.twinExact(&k.simd.?, k.device, m);
        if (!ok) std.log.info("qwen27: {d}x{d} rows take simdgroup tiles at every width (the scalar twin's bits differ on this chip)", .{ m.n, m.k });
        try k.twins.put(k.allocator, .{ m.n, m.k }, ok);
        return ok;
    }

    pub fn sum(k: *Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, dims: Ref, xs: Ref, width: u32, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .xsum);
        errdefer span.cancel();
        const e = span.encoder;

        e.setPipeline(try k.get(.{ .kind = .xsum, .k = width }));
        bind(e, .{ x, dims, xs });
        e.dispatchThreads(mtl.Size.of(width / 64, padded(rows), 1), mtl.Size.of(@min(width / 64, 256), 1, 1));
        try span.end();
    }

    pub fn norm(k: *Kernels, input_encoder: mtl.ComputeEncoder, h: Ref, residual: ?Ref, weight: Ref, eps: Ref, dims: Ref, combined: Ref, out: Ref, xs: Ref, width: u32, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .norm);
        errdefer span.cancel();
        const e = span.encoder;

        e.setPipeline(try k.get(.{ .kind = if (residual != null) .norm else .input_norm, .k = width }));
        if (residual) |r| bind(e, .{ h, r, weight, eps, dims, combined, out, xs }) else bind(e, .{ h, weight, eps, dims, out, xs });
        e.dispatchGroups(mtl.Size.of(1, padded(rows), 1), mtl.Size.of(width / 16, 1, 1));
        try span.end();
    }

    /// Prompt rows into the Neural Engine's fp16 input planes [hidden][128].
    pub fn aneInput(k: *Kernels, input_encoder: mtl.ComputeEncoder, x: Ref, dims: Ref, planes: Ref, hidden: u32, width: u32, columns: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .copy);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .ane_input, .k = hidden, .n = width, .tile = columns }));
        bind(e, .{ x, dims, planes });
        e.dispatchThreads(mtl.Size.of(128, hidden, 1), mtl.Size.of(128, 1, 1));
        try span.end();
    }

    /// `mlp` over 64-column groups from `first`; gate/up below `columns` come from the Neural Engine's planes.
    pub fn mlpAne(k: *Kernels, input_encoder: mtl.ComputeEncoder, gate: Ref, up: Ref, dims: Ref, out: Ref, xs: Ref, planes: Ref, hidden: u32, width: u32, columns: u32, rows: u32, first: u32, groups: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .activation);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .mlp_ane, .k = hidden, .n = width, .tile = columns }));
        bind(e, .{ gate, up, dims, out, xs, planes });
        e.setValue(first, 6);
        e.dispatchGroups(mtl.Size.of(groups, padded(rows), 1), mtl.Size.of(64, 1, 1));
        try span.end();
    }

    /// The down projection: bf16(fp32 partial + the Neural Engine's fp16 partial planes [hidden][128]) for `rows` rows.
    pub fn addAne(k: *Kernels, input_encoder: mtl.ComputeEncoder, partial: Ref, planes: Ref, y: Ref, hidden: u32, width: u32, columns: u32, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .copy);
        errdefer span.cancel();
        const e = span.encoder;
        e.setPipeline(try k.get(.{ .kind = .add_ane, .k = hidden, .n = width, .tile = columns }));
        bind(e, .{ partial, planes, y });
        e.dispatchThreads(mtl.Size.of(hidden, rows, 1), mtl.Size.of(256, 1, 1));
        try span.end();
    }

    pub fn mlp(k: *Kernels, input_encoder: mtl.ComputeEncoder, gate: Ref, up: Ref, dims: Ref, out: Ref, xs: Ref, width: u32, rows: u32) !void {
        var span = profile.begin(k.profiler, input_encoder, .activation);
        errdefer span.cancel();
        const e = span.encoder;

        e.setPipeline(try k.get(.{ .kind = .mlp, .k = width, .n = width }));
        bind(e, .{ gate, up, dims, out, xs });
        e.dispatchGroups(mtl.Size.of(width / 64, padded(rows), 1), mtl.Size.of(64, 1, 1));
        try span.end();
    }
};

pub fn padded(rows: u32) u32 {
    return 16 * ((rows + 15) / 16);
}

fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
}
