//! GGUF calls existing Metal operators with FP32 metadata; source kernels and native MLX arithmetic stay unchanged.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const sources = @import("kernel_sources");
const qsource = @import("qwen_runtime_sources");
const wo = core.weight_only;
const api = core.weight_only_metal_adapter(wo, mtl);
const projection = @import("projection.zig");
const Ref = projection.Ref;
const abi = @import("glue.zig");
const Frame = @import("gpu_frame.zig");
pub const Kernels = struct {
    weight: api.Kernels,
    norm_pipe: mtl.Pipeline,
    narrow_pipe: mtl.Pipeline,
    copy_pipe: mtl.Pipeline,
    embedding_temp: mtl.Buffer,
    product_temp: mtl.Buffer,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), hidden: usize, intermediate: usize) !Kernels {
        const text = try api.source(a, sources.core_weight_only_body, sources.core_weight_only, false);
        defer a.free(text);
        var weight = try api.Kernels.load(device, text, false);
        errdefer weight.deinit();
        const glue = try mtl.Library.fromSource(device, qsource.glue, mtl.CompileOptions.mlx());
        defer glue.deinit();
        const train = try mtl.Library.fromSource(device, sources.train, mtl.CompileOptions.mlx());
        defer train.deinit();
        const norm_pipe = try mtl.Pipeline.init(device, glue, "q27_norm", false);
        errdefer norm_pipe.deinit();
        const narrow = try mtl.Pipeline.init(device, train, "tf_train_narrow", false);
        errdefer narrow.deinit();
        const copy = try mtl.Pipeline.init(device, train, "tf_train_copy_cols", false);
        errdefer copy.deinit();
        const embedding_temp = try own(a, device, owned, Frame.max_rows * hidden * 4);
        const product_temp = try own(a, device, owned, 128 * @max(2 * intermediate, 2 * hidden + 256) * 2);
        return .{ .weight = weight, .norm_pipe = norm_pipe, .narrow_pipe = narrow, .copy_pipe = copy, .embedding_temp = embedding_temp, .product_temp = product_temp };
    }
    pub fn deinit(k: *Kernels) void {
        k.weight.deinit();
        k.norm_pipe.deinit();
        k.narrow_pipe.deinit();
        k.copy_pipe.deinit();
    }
    fn view(r: Ref) api.View {
        return .{ .buffer = r.buffer, .offset = r.offset, .len = r.buffer.length() - r.offset };
    }
    fn one(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear, x: Ref, y: Ref, rows: u32, stride: u32, first: u32) !void {
        const weights = p.gguf orelse return error.MissingGgufProjection;
        if (p.n != weights.params.n or p.k != weights.params.k or p.n > stride or first > stride - p.n or rows == 0 or rows > Frame.max_rows) return error.BadGgufProjection;
        var done: u32 = 0;
        while (done < rows) : (done += 128) {
            var params = weights.params;
            const count = @min(rows - done, 128);
            params.rows = @intCast(count);
            const input: Ref = .{ .buffer = x.buffer, .offset = x.offset + @as(usize, done) * p.k * 2 };
            const output: Ref = .{ .buffer = y.buffer, .offset = y.offset + (@as(usize, done) * stride + first) * 2 };
            const direct = stride == p.n and first == 0;
            const target: Ref = if (direct) output else .{ .buffer = k.product_temp };
            try k.weight.encode(e, params, view(input), view(weights.words), view(weights.scales), view(weights.biases), view(target));
            if (!direct) {
                e.setPipeline(k.copy_pipe);
                e.setBuffer(target.buffer, target.offset, 0);
                e.setBuffer(output.buffer, output.offset, 1);
                e.setValue([4]u32{ p.n, stride, p.n, count }, 2);
                e.dispatchThreads(mtl.Size.of(p.n, count, 1), mtl.Size.of(256, 1, 1));
                e.barrier();
            }
        }
    }
    pub fn quant(k: *Kernels, e: mtl.ComputeEncoder, p: projection.Linear, x: Ref, y: Ref, rows: u32) !void {
        if (p.runs.len == 0) return k.one(e, p, x, y, rows, p.n, 0);
        for (p.runs) |run| try k.one(e, run.linear, x, y, rows, p.n, run.first);
    }
    pub fn norm(k: *Kernels, e: mtl.ComputeEncoder, h: Ref, residual: ?Ref, gain: Ref, combined: Ref, out: Ref, width: u32, rows: u32, eps: f32) !void {
        const params: abi.Norm = .{ .width = width, .rows = rows, .activation = .bf16, .gain = .f32, .residual = @intFromBool(residual != null), .eps = eps };
        const threads = try abi.normThreads(params);
        e.setPipeline(k.norm_pipe);
        for ([_]Ref{ h, residual orelse h, gain }, 0..) |r, i| e.setBuffer(r.buffer, r.offset, i);
        e.setValue(params, 3);
        e.setBuffer(combined.buffer, combined.offset, 4);
        e.setBuffer(out.buffer, out.offset, 5);
        e.dispatchGroups(mtl.Size.of(1, rows, 1), mtl.Size.of(threads, 1, 1));
    }
    pub fn embedding(k: *Kernels, e: mtl.ComputeEncoder, pipeline: mtl.Pipeline, c: @import("config.zig").Config, p: projection.Triple, ids: Ref, out: Ref, rows: u32) !void {
        const params: abi.Embed = .{ .width = @intCast(c.hidden), .rows = rows, .vocab = @intCast(c.vocab), .bits = p.bits, .group = p.group, .metadata = .f32 };
        if (rows == 0 or rows > Frame.max_rows or c.hidden % p.group != 0) return error.BadGgufEmbedding;
        e.setPipeline(pipeline);
        e.setBuffer(ids.buffer, ids.offset, 0);
        inline for (.{ p.weight, p.scales, p.biases }, 1..) |t, i| e.setBuffer(t.buffer, t.offset, i);
        e.setValue(params, 4);
        e.setBuffer(k.embedding_temp, 0, 5);
        e.dispatchThreads(mtl.Size.of(c.hidden, rows, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.setPipeline(k.narrow_pipe);
        e.setBuffer(k.embedding_temp, 0, 0);
        e.setBuffer(out.buffer, out.offset, 1);
        e.dispatchThreads(mtl.Size.of(c.hidden * rows, 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
    }
};
fn own(a: std.mem.Allocator, device: mtl.Device, list: *std.ArrayList(mtl.Buffer), bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer b.deinit();
    try list.append(a, b);
    return b;
}
