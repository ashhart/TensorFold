//! The `info` command: one checkpoint's family, format, context, weights size and memory floor.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const core = @import("core");
const checkpoint = if (@hasDecl(core, "checkpoint_host")) core.checkpoint_host else core.checkpoint;

pub const GiB: u64 = 1 << 30;

/// The same reserve the native residency and preflight bounds keep.
pub const runtime_margin = @import("sys").memory_policy.margin;

/// Prints what a Zig family would serve from `model` (a directory or a cached repo id).
/// Returns the process exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, model: []const u8) !u8 {
    const dir: []const u8 = blk: {
        if (hub.isDir(io, model)) break :blk model;
        if (hub.isRepoIdLike(model)) {
            const root = try hub.cacheDir(a, env, override);
            if (try hub.cachedSnapshot(a, io, root, model)) |snapshot| break :blk snapshot;
            try out.print("{s} is not in the Hugging Face cache; run: tensorfold pull {s}\n", .{ model, model });
            return 1;
        }
        try out.print("{s} is neither a directory nor a Hugging Face repo id (owner/name)\n", .{model});
        return 1;
    };
    var problem: []const u8 = "";
    const bytes = @import("checkpoint_config").read(a, io, dir, &problem) catch |e| {
        if (e == error.OutOfMemory) return e;
        try out.print("{s}: {s}\n", .{ model, problem });
        return 1;
    };
    if (bytes) |b| a.free(b);
    const model_type = hub.modelType(a, io, dir);
    const family = hub.family(model_type) orelse {
        try out.print("{s}: no registered Zig family serves model_type {s}; the 0.6 line may: tensorfold@0.6\n", .{ model, model_type });
        return 1;
    };
    const flash_metal = @import("builtin").os.tag == .macos and std.mem.eql(u8, model_type, "qwen4_exp");
    const weights = if (flash_metal) try @import("flashnext_memory").weightBytes(checkpoint, a, io, dir) else try hub.sizeOf(a, io, dir);
    const details = hub.configDetails(a, io, dir);
    defer if (details.modelopt_formats) |formats| {
        for (formats) |format| a.free(format);
        a.free(formats);
    };
    const title = if (details.is_bolt) "Nemotron 3.5 Bolt" else family.title;
    try out.print("family: {s} ({s})\n", .{ title, family.model_type });
    if (hub.quantization(a, io, dir)) |q| {
        try out.print("format: {d}-bit, group {d}\n", .{ q.bits, q.group });
    } else if (details.modelopt_formats) |formats| {
        if (formats.len == 1) {
            try out.print("format: modelopt {s}\n", .{formats[0]});
        } else {
            try out.writeAll("format: modelopt mixed precision (");
            for (formats, 0..) |format, i| {
                if (i != 0) try out.writeAll(", ");
                try out.writeAll(format);
            }
            try out.writeAll(")\n");
        }
    } else {
        try out.print("format: full precision\n", .{});
    }
    if (hub.configInt(a, io, dir, "max_position_embeddings")) |context| {
        try out.print("context: {d}\n", .{context});
    }
    try out.print("weights: {d:.2} GiB ({d} bytes)\n", .{ @as(f64, @floatFromInt(weights)) / GiB, weights });
    if (flash_metal) {
        const memory = @import("flashnext_memory");
        const inventory = try memory.weights_plan.read(core.Checkpoint, core.quant.mlx, a, io, dir);
        const plan = try memory.budget(weights, memory.trained, inventory, 0);
        try out.print("startup budget: {d:.2} GiB (conservative raw/packed text weights, both prompt planes/selectors, KV/state, keepalive and {d} GiB runtime reserve; trained window)\n", .{ @as(f64, @floatFromInt(plan.steady)) / GiB, runtime_margin >> 30 });
        try out.print("cold pack budget: {d:.2} GiB (source pages, one output phase and conversion scratch; skipped with a validated pack; vision and requested storage are additional at load)\n", .{@as(f64, @floatFromInt(plan.building)) / GiB});
    } else try out.print("memory floor: {d:.2} GiB (weights plus a {d:.2} GiB runtime margin)\n", .{
        @as(f64, @floatFromInt(weights + runtime_margin)) / GiB,
        @as(f64, @floatFromInt(runtime_margin)) / GiB,
    });
    return 0;
}

test "info prints family, format, context, weights and floor for a cached checkpoint" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-info-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = std.Io.Dir.cwd();
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1" }));
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/refs/main" }), .data = "rev1" });
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/config.json" }), .data = "{\"model_type\": \"qwen4_exp\", \"quantization\": {\"bits\": 6, \"group_size\": 32}, \"max_position_embeddings\": 65536}" });
    var weights: [8192]u8 = undefined;
    @memset(&weights, 'x');
    const header = "{\"language_model.fixture\":{\"dtype\":\"U8\",\"shape\":[8192],\"data_offsets\":[0,8192]}}";
    const model_bytes = try a.alloc(u8, 8 + header.len + weights.len);
    std.mem.writeInt(u64, model_bytes[0..8], header.len, .little);
    @memcpy(model_bytes[8..][0..header.len], header);
    @memcpy(model_bytes[8 + header.len ..], &weights);
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Flash/snapshots/rev1/model.safetensors" }), .data = model_bytes });

    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, root, "Org/Flash"));
    try std.testing.expect(has(out.written(), "family: Qwen3.8 Flash Next (qwen4_exp)"));
    try std.testing.expect(has(out.written(), "format: 6-bit, group 32"));
    try std.testing.expect(has(out.written(), "context: 65536"));
    try std.testing.expect(has(out.written(), try std.fmt.allocPrint(a, "weights: 0.00 GiB ({d} bytes)", .{model_bytes.len + (if (@import("builtin").os.tag == .macos) @as(usize, 0) else @as(usize, 108))})));
    if (@import("builtin").os.tag == .macos) {
        try std.testing.expect(has(out.written(), "startup budget:"));
        try std.testing.expect(has(out.written(), "conservative raw/packed text weights"));
        try std.testing.expect(has(out.written(), "cold pack budget:"));
        try std.testing.expect(has(out.written(), "skipped with a validated pack"));
        try std.testing.expect(has(out.written(), "both prompt planes/selectors, KV/state"));
    } else try std.testing.expect(has(out.written(), "memory floor: 2.00 GiB"));

    var refused: std.Io.Writer.Allocating = .init(a);
    try w.createDirPath(io, try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2" }));
    try w.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, "models--Org--Draft/snapshots/rev2/config.json" }), .data = "{\"model_type\": \"gemma3n\"}" });
    try std.testing.expectEqual(@as(u8, 1), try run(a, io, &refused.writer, null, root, "Org/Draft"));
    try std.testing.expect(has(refused.written(), "no registered Zig family serves model_type gemma3n"));
}

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}

fn runConfig(a: Allocator, io: std.Io, config: []const u8) ![]const u8 {
    if (hub.family("nemotron_h_omni") == null) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    var out: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, null, dir));
    return out.toOwnedSlice();
}

test "info names Bolt and its sorted modelopt formats" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const output = try runConfig(arena.allocator(), std.testing.io,
        \\{
        \\  "model_type": "nemotron_h_omni",
        \\  "llm_config": {"model_type": "nemotron_h_puzzle"},
        \\  "quantization_config": {
        \\    "quant_algo": "MIXED_PRECISION",
        \\    "quantized_layers": {
        \\      "language_model.layers.0": {"quant_algo": "Q8_0"},
        \\      "language_model.layers.1": {"quant_algo": "IQ2_XXS"},
        \\      "language_model.layers.2": {"quant_algo": "W4A16_NVFP4"},
        \\      "language_model.lm_head": {"quant_algo": "Q8_0"}
        \\    },
        \\    "config_groups": {
        \\      "group_0": {"quant_algo": "Q8_0", "targets": ["*lm_head*"]},
        \\      "group_1": {"weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16}, "targets": ["*mlp*"]},
        \\      "group_2": {"quant_algo": "IQ2_XXS", "targets": ["*experts*"]}
        \\    }
        \\  }
        \\}
    );
    try std.testing.expect(has(output, "family: Nemotron 3.5 Bolt (nemotron_h_omni)"));
    try std.testing.expect(has(output, "format: modelopt mixed precision (IQ2_XXS, Q8_0, W4A16_NVFP4)"));
}

test "info reads one modelopt algorithm without quantized layers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const output = try runConfig(arena.allocator(), std.testing.io,
        \\{"model_type":"nemotron_h_omni","quantization_config":{"quant_algo":"NVFP4"}}
    );
    try std.testing.expect(has(output, "format: modelopt NVFP4"));
}

test "info treats an empty modelopt layer algorithm as full precision" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const output = try runConfig(arena.allocator(), std.testing.io,
        \\{"model_type":"nemotron_h_omni","quantization_config":{"quantized_layers":{"layer":{"quant_algo":""}}}}
    );
    try std.testing.expect(has(output, "format: full precision"));
}

test "info treats an empty modelopt config algorithm as full precision" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const output = try runConfig(arena.allocator(), std.testing.io,
        \\{"model_type":"nemotron_h_omni","quantization_config":{"quant_algo":""}}
    );
    try std.testing.expect(has(output, "format: full precision"));
}

test "info treats malformed modelopt metadata as full precision" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    for ([_][]const u8{
        \\{"model_type":"nemotron_h_omni","quantization_config":{"quantized_layers":[]}}
        ,
        \\{"model_type":"nemotron_h_omni","quantization_config":{"quantized_layers":{"layer":{}}}}
        ,
        \\{"model_type":"nemotron_h_omni","quantization_config":{"quant_algo":4}}
        ,
    }) |config| {
        const output = try runConfig(a, io, config);
        try std.testing.expect(has(output, "format: full precision"));
    }
}

test "info keeps the Super VL title" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const output = try runConfig(arena.allocator(), std.testing.io,
        \\{"model_type":"nemotron_h_omni","llm_config":{"model_type":"nemotron_h"}}
    );
    try std.testing.expect(has(output, "family: Nemotron 3.5 Super VL (nemotron_h_omni)"));
    try std.testing.expect(has(output, "format: full precision"));
}

test "info keeps MLX format precedence over modelopt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const output = try runConfig(arena.allocator(), std.testing.io,
        \\{"model_type":"nemotron_h_omni","quantization":{"bits":4,"group_size":64},"quantization_config":{"quant_algo":"NVFP4"}}
    );
    try std.testing.expect(has(output, "format: 4-bit, group 64"));
}

test "issue 934: info reads the same large config and reports an oversized config with its limit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    const bytes = try a.alloc(u8, 8_668_620);
    @memset(bytes, ' ');
    const prefix = "{\"model_type\":\"nemotron_h_omni\",\"max_position_embeddings\":65536}";
    @memcpy(bytes[0..prefix.len], prefix);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = bytes });
    var out: std.Io.Writer.Allocating = .init(a);
    if (hub.family("nemotron_h_omni") != null) {
        try std.testing.expectEqual(@as(u8, 0), try run(a, io, &out.writer, null, null, dir));
        try std.testing.expect(has(out.written(), "family: Nemotron 3.5 Super VL (nemotron_h_omni)"));
        try std.testing.expect(has(out.written(), "context: 65536"));
    } else { // a platform that serves no Super VL: the model type still came from the whole large config
        try std.testing.expectEqual(@as(u8, 1), try run(a, io, &out.writer, null, null, dir));
        try std.testing.expect(has(out.written(), "serves model_type nemotron_h_omni"));
    }
    const file = try tmp.dir.createFile(io, "config.json", .{});
    try file.setLength(io, 67_108_865);
    file.close(io);
    out.clearRetainingCapacity();
    try std.testing.expectEqual(@as(u8, 1), try run(a, io, &out.writer, null, null, dir));
    try std.testing.expect(has(out.written(), "config.json is 67108865 bytes, over the 67108864-byte limit"));
}
