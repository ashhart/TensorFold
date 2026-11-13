//! The `info` command: one checkpoint's family, format, context, weights size and memory floor or startup budget.
const std = @import("std");
const Allocator = std.mem.Allocator;
const hub = @import("hub.zig");
const core = @import("core");
const checkpoint = if (@hasDecl(core, "checkpoint_host")) core.checkpoint_host else core.checkpoint;

pub const GiB: u64 = 1 << 30;

/// The same reserve the native residency and preflight bounds keep.
pub const runtime_margin = @import("sys").memory_policy.margin;

/// A start `info` prices: --context and --parallel as `serve` reads them.
pub const Start = struct { context: ?i64 = null, streams: u32 = 8, fixed: bool = false };

/// `[--context N] [--parallel N|auto]`; null on anything else.
pub fn parseStart(args: []const []const u8) ?Start {
    var s: Start = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 == args.len) return null;
        const v = std.mem.trim(u8, args[i + 1], " ");
        if (std.mem.eql(u8, args[i], "--context")) {
            s.context = std.fmt.parseInt(i64, v, 10) catch return null;
        } else if (!std.mem.eql(u8, args[i], "--parallel")) {
            return null;
        } else if (std.ascii.eqlIgnoreCase(v, "auto")) {
            s.streams = 8;
            s.fixed = false;
        } else {
            const n = std.fmt.parseInt(i64, v, 10) catch return null;
            s.streams = @intCast(@max(1, @min(n, 4096)));
            s.fixed = true;
        }
    }
    return s;
}

/// Prints what a Zig family would serve from `model` (a directory or a cached repo id); returns the process exit code.
pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, model: []const u8) !u8 {
    return runStart(a, io, out, env, override, model, .{});
}

/// `run`, with `serve`'s startup charge priced at `start` where the engine has one.
pub fn runStart(a: Allocator, io: std.Io, out: *std.Io.Writer, env: ?*const std.process.Environ.Map, override: ?[]const u8, model: []const u8, start: Start) !u8 {
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
    } else if (try qwen27(a, io, out, model_type, dir, weights, start)) |code| return code else try out.print("memory floor: {d:.2} GiB (weights plus a {d:.2} GiB runtime margin)\n", .{
        @as(f64, @floatFromInt(weights + runtime_margin)) / GiB,
        @as(f64, @floatFromInt(runtime_margin)) / GiB,
    });
    return 0;
}

/// The Qwen3.8-27B-class startup lines from `serve`'s own charge; null where this build's engines have none.
fn qwen27(a: Allocator, io: std.Io, out: *std.Io.Writer, model_type: []const u8, dir: []const u8, weights: u64, s: Start) !?u8 {
    const engines = @import("native_engines");
    if (comptime !@hasDecl(engines, "qwen27Budget")) return null;
    var problem: []const u8 = "";
    const b = (engines.qwen27Budget(a, io, model_type, dir, weights, s.context, s.streams, s.fixed, &problem) catch |err| {
        if (err == error.OutOfMemory) return err;
        if (err == error.ContextFull) {
            try out.writeAll("startup budget: not priced (--context is past this checkpoint's max_position_embeddings)\n");
            return 1;
        }
        // a checkpoint the engine can't price keeps the floor
        try out.print("startup budget: not priced ({s})\n", .{@errorName(err)});
        return null;
    }) orelse {
        if (problem.len == 0) return null;
        try out.print("startup budget: not priced ({s})\n", .{problem});
        return 1;
    };
    const charged = if (b.slots == 0) b.streams else b.slots;
    try out.print("startup budget: {d:.2} GiB at --context {d}: weights, {d} stream{s} of KV and state, {d}-row prompt scratch (no drafter or images)\n", .{ gib(b.total), b.context, charged, if (charged == 1) "" else "s", b.rows });
    const set = gib(b.working_set);
    const margin = engines.qwen27_margin >> 30;
    if (b.slots == b.streams) {
        try out.print("this Mac: fits beside its {d:.1} GiB Metal working set and an {d} GiB margin\n", .{ set, margin });
    } else if (b.slots == 0) {
        try out.print("this Mac: not one stream fits its {d:.1} GiB Metal working set with an {d} GiB margin; serve refuses: lower --context\n", .{ set, margin });
    } else if (s.fixed) {
        try out.print("this Mac: {d} of --parallel {d} fit its {d:.1} GiB Metal working set with an {d} GiB margin; serve refuses: lower --parallel or --context\n", .{ b.slots, b.streams, set, margin });
    } else try out.print("this Mac: {d} of {d} streams fit its {d:.1} GiB Metal working set with an {d} GiB margin; serve starts {d}\n", .{ b.slots, b.streams, set, margin, b.slots });
    return 0;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / GiB;
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

test "info --context and --parallel parse as serve reads them" {
    const t = std.testing;
    try t.expectEqualDeep(Start{}, parseStart(&.{}).?);
    try t.expectEqualDeep(Start{ .context = 65536 }, parseStart(&.{ "--context", "65536" }).?);
    try t.expectEqualDeep(Start{ .context = 8192, .streams = 2, .fixed = true }, parseStart(&.{ "--parallel", "2", "--context", "8192" }).?);
    try t.expectEqualDeep(Start{ .streams = 1, .fixed = true }, parseStart(&.{ "--parallel", "0" }).?);
    try t.expectEqualDeep(Start{}, parseStart(&.{ "--parallel", "AUTO" }).?);
    for ([_][]const []const u8{ &.{"--context"}, &.{ "--context", "lots" }, &.{ "--parallel", "two" }, &.{ "--drafter", "x" } }) |bad| try t.expect(parseStart(bad) == null);
}

/// Prices the 27B's config at `start` with the working set capped at `cap_gib`.
fn price27(a: Allocator, io: std.Io, dir: []const u8, cap_gib: u64, start: Start) !struct { code: u8, text: []const u8 } {
    const env = @import("sys").env;
    const cap = try std.fmt.allocPrintSentinel(a, "{d}", .{cap_gib << 30}, 0);
    _ = env.setenv("TF_QWEN27_WORKING_SET_BYTES", cap, 1);
    defer _ = env.unsetenv("TF_QWEN27_WORKING_SET_BYTES");
    var out: std.Io.Writer.Allocating = .init(a);
    const code = try runStart(a, io, &out.writer, null, null, dir, start);
    return .{ .code = code, .text = out.written() };
}

test "info prices a Qwen3.8-27B start from serve's charge: context, streams and the Mac's working set" {
    if (comptime !@hasDecl(@import("native_engines"), "qwen27Budget")) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"qwen3_5","hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":0.000001,"partial_rotary_factor":0.25,"rope_theta":10000000,"full_attention_interval":4,"max_position_embeddings":262144,"eos_token_id":248046}
    });
    const small = try price27(a, io, dir, 4096, .{ .context = 8192, .streams = 1, .fixed = true });
    try std.testing.expectEqual(@as(u8, 0), small.code);
    try std.testing.expect(has(small.text, "startup budget: "));
    try std.testing.expect(has(small.text, " at --context 8192: weights, 1 stream of KV and state, "));
    try std.testing.expect(has(small.text, "this Mac: fits beside its "));
    try std.testing.expect(std.mem.indexOf(u8, small.text, "memory floor") == null);
    const big = try price27(a, io, dir, 4096, .{ .context = 262144, .streams = 1, .fixed = true });
    try std.testing.expect(gibOf(big.text) > gibOf(small.text) + 10); // 16 attention layers of 4x256 K and V at 262,144 tokens: 16 GiB
    const two = try price27(a, io, dir, 4096, .{ .context = 8192, .streams = 2, .fixed = true });
    try std.testing.expect(has(two.text, "2 streams of KV and state"));
    try std.testing.expect(gibOf(two.text) > gibOf(small.text));
    const tight = try price27(a, io, dir, 8, .{ .context = 8192, .streams = 1, .fixed = true });
    try std.testing.expect(has(tight.text, "not one stream fits its 8.0 GiB Metal working set with an 8 GiB margin; serve refuses"));
    const auto = try price27(a, io, dir, 4096, .{ .context = 8192 });
    try std.testing.expect(has(auto.text, "8 streams of KV and state"));
    const past = try price27(a, io, dir, 4096, .{ .context = 300000 });
    try std.testing.expectEqual(@as(u8, 1), past.code);
    try std.testing.expect(has(past.text, "--context 300000 is outside this Qwen checkpoint's 1 to 262,144 tokens"));
    // a shape the engine refuses (16 query heads over 2 KV heads) keeps the floor
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"qwen3_5","hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"vocab_size":248320,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"rms_norm_eps":0.000001,"partial_rotary_factor":0.25,"rope_theta":10000000,"full_attention_interval":4,"max_position_embeddings":262144,"eos_token_id":248046}
    });
    const refused = try price27(a, io, dir, 4096, .{});
    try std.testing.expectEqual(@as(u8, 0), refused.code);
    try std.testing.expect(has(refused.text, "startup budget: not priced ("));
    try std.testing.expect(has(refused.text, "memory floor: "));
}

/// The GiB figure of a "startup budget: X GiB" line.
fn gibOf(text: []const u8) f64 {
    const at = (std.mem.indexOf(u8, text, "startup budget: ") orelse return 0) + "startup budget: ".len;
    const end = std.mem.indexOfScalarPos(u8, text, at, ' ') orelse return 0;
    return std.fmt.parseFloat(f64, text[at..end]) catch 0;
}
