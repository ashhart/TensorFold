//! Real GLM weights.load metadata checks must precede CUDA weight allocation.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const shard = @import("cuda_shard.zig");

const plain_nvfp4 =
    \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","config_groups":
    \\{"g":{"weights":{"num_bits":4,"type":"float","group_size":16}}}}}
;

const Target = enum { q_a, kv_a, wq_b, all_bf16 };
const layer = "model.language_model.layers.3.";

const Recording = struct {
    var allocations: usize = 0;
    var live: usize = 0;

    fn reset() void {
        allocations = 0;
        live = 0;
    }

    fn alloc(out: *u64, _: usize) callconv(.c) cuda.abi.Result {
        allocations += 1;
        live += 1;
        out.* = 0x1000 + allocations * 256;
        return cuda.abi.success;
    }

    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        return cuda.abi.success;
    }

    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn context() callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};

const Fixture = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    entries: std.ArrayList(core.checkpoint_fixture.Entry) = .empty,

    fn init(gpa: std.mem.Allocator) Fixture {
        return .{ .gpa = gpa, .arena = .init(gpa) };
    }

    fn deinit(f: *Fixture) void {
        f.entries.deinit(f.gpa);
        f.arena.deinit();
    }

    fn tensor(f: *Fixture, name: []const u8, dtype: []const u8, shape: []const usize, len: usize, fill: u8) ![]u8 {
        const a = f.arena.allocator();
        const bytes = try a.alloc(u8, len);
        @memset(bytes, fill);
        try f.entries.append(f.gpa, .{
            .name = name,
            .dtype = dtype,
            .shape = try a.dupe(usize, shape),
            .bytes = bytes,
        });
        return bytes;
    }

    fn bf16(f: *Fixture, name: []const u8, shape: []const usize) !void {
        var count: usize = 1;
        for (shape) |n| count *= n;
        _ = try f.tensor(name, "BF16", shape, count * 2, 0);
    }

    fn scalar(f: *Fixture, name: []const u8) !void {
        const bytes = try f.tensor(name, "F32", &.{}, 4, 0);
        std.mem.writeInt(u32, bytes[0..4], @bitCast(@as(f32, 1.0)), .little);
    }

    fn suffixed(f: *Fixture, base: []const u8, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(f.arena.allocator(), "{s}{s}", .{ base, suffix });
    }

    // Full plain-NVFP4 metadata makes the refusal specifically the BF16 requirement.
    fn nvfp4(f: *Fixture, base: []const u8, n: usize, k: usize) !void {
        _ = try f.tensor(try f.suffixed(base, ".weight"), "U8", &.{ n, k / 2 }, n * k / 2, 0x11);
        _ = try f.tensor(try f.suffixed(base, ".weight_scale"), "F8_E4M3", &.{ n, k / 16 }, n * k / 16, 0x38);
        try f.scalar(try f.suffixed(base, ".weight_scale_2"));
        try f.scalar(try f.suffixed(base, ".input_scale"));
    }
};

fn modelConfig() @import("config.zig").Config {
    return .{
        .hidden = 64,
        .layers = 4,
        .vocab = 128,
        .dense_layers = 3,
        .dense_inter = 128,
        .experts = 8,
        .topk = 2,
        .moe_inter = 128,
        .shared_inter = 128,
        .kda_heads = 2,
        .kda_dim = 16,
        .mla_heads = 2,
        .nope = 16,
        .v_dim = 16,
        .q_lora = 32,
        .kv_lora = 16,
        .i_heads = 2,
        .i_dim = 16,
        .i_topk = 64,
        .kpool = 4,
        .mtp = 0,
    };
}

fn makeFixture(f: *Fixture, target: Target, c: @import("config.zig").Config) !void {
    try f.bf16(layer ++ "input_layernorm.weight", &.{c.hidden});
    try f.bf16(layer ++ "post_attention_layernorm.weight", &.{c.hidden});
    inline for (.{ "attn", "ffn" }) |site| {
        try f.bf16(layer ++ "hc_" ++ site ++ "_fn", &.{1});
        try f.bf16(layer ++ "hc_" ++ site ++ "_base", &.{1});
        try f.bf16(layer ++ "hc_" ++ site ++ "_scale", &.{1});
    }
    try f.bf16(layer ++ "self_attn.kv_b_proj.weight", &.{ c.mla_heads * (c.nope + c.v_dim), c.kv_lora });
    const qa = layer ++ "self_attn.q_a_proj";
    if (target == .q_a) try f.nvfp4(qa, c.q_lora, c.hidden) else try f.bf16(qa ++ ".weight", &.{ c.q_lora, c.hidden });
    const kva = layer ++ "self_attn.kv_a_proj_with_mqa";
    if (target == .kv_a) try f.nvfp4(kva, c.kv_lora, c.hidden) else try f.bf16(kva ++ ".weight", &.{ c.kv_lora, c.hidden });
    try f.bf16(layer ++ "self_attn.q_a_layernorm.weight", &.{c.q_lora});
    try f.bf16(layer ++ "self_attn.kv_a_layernorm.weight", &.{c.kv_lora});
    try f.bf16(layer ++ "self_attn.q_b_proj.weight", &.{ c.mla_heads * c.nope, c.q_lora });
    try f.bf16(layer ++ "self_attn.o_proj.weight", &.{ c.hidden, c.mla_heads * c.v_dim });
    try f.bf16(layer ++ "self_attn.indexer.wk.weight", &.{ c.i_dim, c.hidden });
    try f.bf16(layer ++ "self_attn.indexer.weights_proj.weight", &.{ c.i_heads, c.hidden });
    const qb = layer ++ "self_attn.indexer.wq_b";
    if (target == .wq_b) try f.nvfp4(qb, c.i_heads * c.i_dim, c.q_lora) else try f.bf16(qb ++ ".weight", &.{ c.i_heads * c.i_dim, c.q_lora });
}

fn expectLoad(target: Target, expected: anyerror, allocation_limit: ?usize) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = Fixture.init(gpa);
    defer fixture.deinit();
    const c = modelConfig();
    try makeFixture(&fixture, target, c);
    try core.checkpoint_fixture.write(gpa, io, tmp.dir, "model.safetensors", fixture.entries.items);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = plain_nvfp4 });
    var path_buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var share = try shard.Share.of(&c, 1, 2);
    share.dsa_columns = true;
    Recording.reset();
    var d = Recording.driver();
    const tools: weights.Tools = .{ .pack_dense = .{ .d = &d, .handle = null }, .lane = undefined };
    const result = weights.load(gpa, io, &d, .{ .d = &d, .handle = null }, tools, dir, c, share, &.{3}, false);
    try std.testing.expectError(expected, result);
    try std.testing.expectEqual(@as(usize, 0), Recording.live);
    if (allocation_limit) |want| {
        try std.testing.expectEqual(want, Recording.allocations);
    } else {
        try std.testing.expect(Recording.allocations > 0);
    }
}

// K179 first allocates the layer's norms, so each old-path refusal has allocations > 0.
test "GLM DSA preflight allocation: NVFP4 q_a_proj refuses before any weight allocation" {
    try expectLoad(.q_a, error.UnreadableFormat, 0);
}

test "GLM DSA preflight allocation: NVFP4 kv_a_proj_with_mqa refuses before any weight allocation" {
    try expectLoad(.kv_a, error.UnreadableFormat, 0);
}

test "GLM DSA preflight allocation: NVFP4 backbone indexer wq_b refuses before any weight allocation" {
    try expectLoad(.wq_b, error.UnreadableFormat, 0);
}

test "GLM DSA preflight allocation: BF16 targets pass preflight and reach later layer metadata" {
    try expectLoad(.all_bf16, error.MissingTensor, null);
}
