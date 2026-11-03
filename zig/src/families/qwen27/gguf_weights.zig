//! GGUF-owned Metal tensors preserve FP32 metadata and use the existing layer graph's row identities.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const adapter = @import("gguf.zig");
const preparation = @import("gguf_projection.zig");
const weights = @import("gpu_weights.zig");
const projection = @import("projection.zig");
const Builder = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    source: *const adapter.Adapter,
    owned: *std.ArrayList(mtl.Buffer),
    runs: *std.ArrayList([]projection.Run),
    problem: *adapter.Problem,
    fn upload(b: Builder, bytes: []const u8) !mtl.Buffer {
        const buffer = try b.device.buffer(bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        @memcpy(buffer.contents()[0..bytes.len], bytes);
        try b.owned.append(b.a, buffer);
        return buffer;
    }
    fn get(b: Builder, name: []const u8) !*const adapter.Tensor {
        return b.source.get(name) orelse b.problem.fail("{s}: missing prepared tensor", .{name});
    }
    fn scalar(b: Builder, name: []const u8) !cp.Tensor {
        const t = try b.get(name);
        if (t.source.kind != .f32 or t.data != .dense) return b.problem.fail("{s}: expected F32 GGUF scalar tensor", .{name});
        var arena = std.heap.ArenaAllocator.init(b.a);
        defer arena.deinit();
        var bytes = try preparation.layout.scalar(arena.allocator(), b.source.config, name, t.data.dense);
        if (t.name.role == .negative_exp_decay) {
            const log = try arena.allocator().alloc(u8, bytes.len);
            for (0..bytes.len / 4) |i| {
                const v: f32 = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
                if (!std.math.isFinite(v) or v >= 0) return b.problem.fail("{s}: expected finite negative exponential decay", .{t.source.name});
                const value: f32 = @log(-v);
                std.mem.writeInt(u32, log[i * 4 ..][0..4], @bitCast(value), .little);
            }
            bytes = log;
        }
        var shape: [5]usize = @splat(1);
        @memcpy(shape[0..t.name.rank], t.name.dims[0..t.name.rank]);
        return .{ .buffer = try b.upload(bytes), .offset = 0, .bytes = bytes.len, .dtype = .f32, .shape = shape, .rank = t.name.rank };
    }
    fn matrix(b: Builder, name: []const u8) !projection.Linear {
        var arena = std.heap.ArenaAllocator.init(b.a);
        defer arena.deinit();
        const p = try preparation.prepare(arena.allocator(), b.source.config, (try b.get(name)).*, b.problem);
        const word = try b.upload(p.words);
        const scale = try b.upload(p.scales);
        const bias = try b.upload(p.biases);
        return .{ .gguf = .{ .params = p.params, .words = .{ .buffer = word }, .scales = .{ .buffer = scale }, .biases = .{ .buffer = bias } }, .words = .{ .buffer = word }, .pairs = .{ .buffer = scale }, .n = @intCast(p.params.n), .k = @intCast(p.params.k), .tile = 32, .slices = 1, .bits = if (p.params.format == .mlx4) 4 else 8, .group = @intCast(p.params.group) };
    }
    fn stack(b: Builder, names: []const []const u8) !projection.Linear {
        if (names.len == 1) return b.matrix(names[0]);
        const runs = try b.a.alloc(projection.Run, names.len);
        b.runs.append(b.a, runs) catch |err| {
            b.a.free(runs);
            return err;
        };
        var first: u32 = 0;
        for (names, runs) |name, *run| {
            run.* = .{ .linear = try b.matrix(name), .first = first };
            first += run.linear.n;
        }
        var combined = runs[0].linear;
        combined.gguf = null;
        combined.n = first;
        combined.runs = runs;
        return combined;
    }
    fn layer(b: Builder, i: usize) !weights.Layer {
        var arena = std.heap.ArenaAllocator.init(b.a);
        defer arena.deinit();
        const a = arena.allocator();
        const N = struct {
            fn of(mem: std.mem.Allocator, layer_id: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.allocPrint(mem, "model.layers.{d}.{s}", .{ layer_id, suffix });
            }
        };
        const c = b.source.config;
        const mixer: @FieldType(weights.Layer, "mixer") = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.matrix(try N.of(a, i, "linear_attn.in_proj_qkv.weight")),
                .zba = try b.stack(&.{ try N.of(a, i, "linear_attn.in_proj_z.weight"), try N.of(a, i, "linear_attn.in_proj_b.weight"), try N.of(a, i, "linear_attn.in_proj_a.weight") }),
                .out = try b.matrix(try N.of(a, i, "linear_attn.out_proj.weight")),
                .conv = try b.scalar(try N.of(a, i, "linear_attn.conv1d.weight")),
                .a_log = try b.scalar(try N.of(a, i, "linear_attn.A_log")),
                .dt = try b.scalar(try N.of(a, i, "linear_attn.dt_bias")),
                .norm = try b.scalar(try N.of(a, i, "linear_attn.norm.weight")),
            } },
            .attention => .{ .attention = .{
                .q = try b.matrix(try N.of(a, i, "self_attn.q_proj.weight")),
                .kv = try b.stack(&.{ try N.of(a, i, "self_attn.k_proj.weight"), try N.of(a, i, "self_attn.v_proj.weight") }),
                .out = try b.matrix(try N.of(a, i, "self_attn.o_proj.weight")),
                .q_norm = try b.scalar(try N.of(a, i, "self_attn.q_norm.weight")),
                .k_norm = try b.scalar(try N.of(a, i, "self_attn.k_norm.weight")),
            } },
        };
        return .{ .norm = try b.scalar(try N.of(a, i, "input_layernorm.weight")), .post_norm = try b.scalar(try N.of(a, i, "post_attention_layernorm.weight")), .gu = try b.stack(&.{ try N.of(a, i, "mlp.gate_proj.weight"), try N.of(a, i, "mlp.up_proj.weight") }), .down = try b.matrix(try N.of(a, i, "mlp.down_proj.weight")), .mixer = mixer };
    }
};
pub fn load(a: std.mem.Allocator, device: mtl.Device, source: *const adapter.Adapter, problem: *adapter.Problem) !weights.Weights {
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (owned.items) |buffer| buffer.deinit();
        owned.deinit(a);
    }
    var runs: std.ArrayList([]projection.Run) = .empty;
    errdefer {
        for (runs.items) |run| a.free(run);
        runs.deinit(a);
    }
    const b: Builder = .{ .a = a, .device = device, .source = source, .owned = &owned, .runs = &runs, .problem = problem };
    const layers = try a.alloc(weights.Layer, source.config.layers);
    errdefer a.free(layers);
    for (layers, 0..) |*layer, i| layer.* = try b.layer(i);
    const embed = try b.matrix("model.embed_tokens.weight");
    const exact = embed.gguf.?;
    const n: usize = embed.n;
    const k: usize = embed.k;
    const shape: [5]usize = .{ n, k * embed.bits / 32, 1, 1, 1 };
    const meta_shape: [5]usize = .{ n, k / embed.group, 1, 1, 1 };
    return .{ .allocator = a, .layers = layers, .embed = .{
        .weight = .{ .buffer = exact.words.buffer, .offset = 0, .bytes = exact.words.buffer.length(), .dtype = .u32, .rank = 2, .shape = shape },
        .scales = .{ .buffer = exact.scales.buffer, .offset = 0, .bytes = exact.scales.buffer.length(), .dtype = .f32, .rank = 2, .shape = meta_shape },
        .biases = .{ .buffer = exact.biases.buffer, .offset = 0, .bytes = exact.biases.buffer.length(), .dtype = .f32, .rank = 2, .shape = meta_shape },
        .bits = embed.bits,
        .group = embed.group,
    }, .norm = try b.scalar("model.norm.weight"), .head = try b.matrix("lm_head.weight"), .owned = owned, .runs = runs };
}
