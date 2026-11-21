//! Direct GGUF metadata, tensor-plan, clamp, KV-head, and EOS refusal tests for Ling-3.0-tiny CPU loading.
const std = @import("std");
const g = @import("core").gguf;
const cpu_gguf = @import("cpu_gguf.zig");
const cfg = @import("cpu_gguf_config.zig");
const names = @import("gguf_names.zig");

const MetaOpts = struct {
    arch: ?g.Value = .{ .string = "bailingmoe3" },
    block_count: g.Value = .{ .uint32 = 24 },
    rope_freq_base: g.Value = .{ .float32 = 6000000.0 },
    safe_gate: g.Value = .{ .boolean = true },
    tokens_len: u64 = 157184,
    kv_mode: enum { valid, missing, scalar, short_len, bad_pattern, string_elems } = .valid,
    clamp_exp: bool = false,
    clamp_shexp: bool = false,
    eos: ?g.Value = .{ .uint32 = 157152 },
    bos: ?g.Value = null,
};

fn appendInt(buf: *std.ArrayList(u8), a: std.mem.Allocator, comptime T: type, v: T) !void {
    var raw: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &raw, v, .little);
    try buf.appendSlice(a, &raw);
}

fn appendStr(buf: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    try appendInt(buf, a, u64, s.len);
    try buf.appendSlice(a, s);
}

fn appendKv(buf: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, v: g.Value) !void {
    try appendStr(buf, a, key);
    try appendInt(buf, a, u32, @backingInt(v));
    switch (v) {
        .string => |s| try appendStr(buf, a, s),
        .float32 => |x| try appendInt(buf, a, u32, @bitCast(x)),
        .float64 => |x| try appendInt(buf, a, u64, @bitCast(x)),
        .boolean => |b| try appendInt(buf, a, u8, @intFromBool(b)),
        .array => unreachable,
        inline else => |x| try appendInt(buf, a, @TypeOf(x), x),
    }
}

fn buildMetadataGguf(a: std.mem.Allocator, opts: MetaOpts) ![]u8 {
    var kvs: std.ArrayList(u8) = .empty;
    defer kvs.deinit(a);
    var count: u64 = 0;

    if (opts.arch) |v| {
        try appendKv(&kvs, a, "general.architecture", v);
        count += 1;
    }
    try appendKv(&kvs, a, "bailingmoe3.block_count", opts.block_count);
    count += 1;

    const ints = .{
        .{ "embedding_length", 1536 },
        .{ "feed_forward_length", 4608 },
        .{ "attention.head_count", 16 },
        .{ "expert_count", 128 },
        .{ "expert_used_count", 8 },
        .{ "expert_group_count", 8 },
        .{ "expert_group_used_count", 4 },
        .{ "expert_gating_func", 2 },
        .{ "vocab_size", 157184 },
        .{ "ssm.conv_kernel", 4 },
        .{ "kda.head_dim", 128 },
        .{ "attention.q_lora_rank", 256 },
        .{ "attention.kv_lora_rank", 512 },
        .{ "rope.dimension_count", 64 },
        .{ "expert_feed_forward_length", 512 },
        .{ "expert_shared_feed_forward_length", 512 },
        .{ "leading_dense_block_count", 1 },
        .{ "context_length", 131072 },
        .{ "attention.key_length_mla", 192 },
        .{ "attention.value_length_mla", 128 },
        .{ "expert_shared_count", 1 },
    };
    inline for (ints) |item| {
        try appendKv(&kvs, a, "bailingmoe3." ++ item[0], .{ .uint32 = item[1] });
        count += 1;
    }

    try appendKv(&kvs, a, "bailingmoe3.rope.freq_base", opts.rope_freq_base);
    count += 1;
    const floats = .{
        .{ "attention.layer_norm_rms_epsilon", @as(f32, 0.000001) },
        .{ "kda.gate_lower_bound", @as(f32, -5.0) },
        .{ "expert_weights_scale", @as(f32, 2.5) },
    };
    inline for (floats) |item| {
        try appendKv(&kvs, a, "bailingmoe3." ++ item[0], .{ .float32 = item[1] });
        count += 1;
    }

    try appendKv(&kvs, a, "bailingmoe3.kda.safe_gate", opts.safe_gate);
    try appendKv(&kvs, a, "bailingmoe3.expert_weights_norm", .{ .boolean = true });
    count += 2;

    switch (opts.kv_mode) {
        .missing => {},
        .scalar => {
            try appendKv(&kvs, a, "bailingmoe3.attention.head_count_kv", .{ .uint32 = 16 });
            count += 1;
        },
        .valid, .short_len, .bad_pattern => {
            const len: u64 = if (opts.kv_mode == .short_len) 23 else 24;
            try appendStr(&kvs, a, "bailingmoe3.attention.head_count_kv");
            try appendInt(&kvs, a, u32, @backingInt(g.MetadataKind.array));
            try appendInt(&kvs, a, u32, @backingInt(g.MetadataKind.uint32));
            try appendInt(&kvs, a, u64, len);
            for (0..@intCast(len)) |layer| {
                var val: u32 = if (layer % 4 == 3) 1 else 0;
                if (opts.kv_mode == .bad_pattern and layer == 0) val = 1;
                try appendInt(&kvs, a, u32, val);
            }
            count += 1;
        },
        .string_elems => {
            try appendStr(&kvs, a, "bailingmoe3.attention.head_count_kv");
            try appendInt(&kvs, a, u32, @backingInt(g.MetadataKind.array));
            try appendInt(&kvs, a, u32, @backingInt(g.MetadataKind.string));
            try appendInt(&kvs, a, u64, 24);
            for (0..24) |_| try appendStr(&kvs, a, "0");
            count += 1;
        },
    }

    try appendStr(&kvs, a, "tokenizer.ggml.tokens");
    try appendInt(&kvs, a, u32, @backingInt(g.MetadataKind.array));
    try appendInt(&kvs, a, u32, @backingInt(g.MetadataKind.string));
    try appendInt(&kvs, a, u64, opts.tokens_len);
    // Empty strings are 8-byte zero lengths, keeping the 157184-token array at ~1.2 MiB in memory.
    try kvs.appendNTimes(a, 0, @as(usize, @intCast(opts.tokens_len)) * 8);
    count += 1;

    if (opts.clamp_exp) {
        try appendKv(&kvs, a, "bailingmoe3.swiglu_clamp_exp", .{ .float32 = 7.0 });
        count += 1;
    }
    if (opts.clamp_shexp) {
        try appendKv(&kvs, a, "bailingmoe3.swiglu_clamp_shexp", .{ .float32 = 7.0 });
        count += 1;
    }
    if (opts.eos) |v| {
        try appendKv(&kvs, a, "tokenizer.ggml.eos_token_id", v);
        count += 1;
    }
    if (opts.bos) |v| {
        try appendKv(&kvs, a, "tokenizer.ggml.bos_token_id", v);
        count += 1;
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "GGUF");
    try appendInt(&out, a, u32, 3);
    try appendInt(&out, a, u64, 0);
    try appendInt(&out, a, u64, count);
    try out.appendSlice(a, kvs.items);
    const aligned = std.mem.alignForward(usize, out.items.len, 32);
    try out.appendNTimes(a, 0, aligned - out.items.len);
    return out.toOwnedSlice(a);
}

fn parseAndCheckMeta(a: std.mem.Allocator, opts: MetaOpts) !void {
    const bytes = try buildMetadataGguf(a, opts);
    defer a.free(bytes);
    var doc = try g.Document.parse(a, bytes, .{});
    defer doc.deinit();
    try cpu_gguf.validateMetadata(&doc);
}

fn parseAndCheckConfig(a: std.mem.Allocator, opts: MetaOpts) !void {
    const bytes = try buildMetadataGguf(a, opts);
    defer a.free(bytes);
    var doc = try g.Document.parse(a, bytes, .{});
    defer doc.deinit();
    try cpu_gguf.validateMetadata(&doc);
    _ = try cfg.tiny(&doc);
}

test "Ling GGUF refuses wrong architecture and wrong geometry" {
    const a = std.testing.allocator;
    try parseAndCheckConfig(a, .{});

    try std.testing.expectError(error.MissingLingGgufMetadata, parseAndCheckMeta(a, .{ .arch = null }));
    try std.testing.expectError(error.UnsupportedLingGgufArchitecture, parseAndCheckMeta(a, .{ .arch = .{ .string = "llama" } }));
    try std.testing.expectError(error.UnsupportedLingGgufArchitecture, parseAndCheckMeta(a, .{ .arch = .{ .uint32 = 1 } }));

    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .block_count = .{ .uint32 = 23 } }));
    try std.testing.expectError(error.BadLingGgufMetadata, parseAndCheckMeta(a, .{ .block_count = .{ .int32 = 24 } }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .rope_freq_base = .{ .float32 = 10000.0 } }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .rope_freq_base = .{ .float32 = std.math.nan(f32) } }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .safe_gate = .{ .boolean = false } }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .tokens_len = 157183 }));

    // Verify File.open itself refuses wrong architecture end-to-end on disk.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bad = try buildMetadataGguf(a, .{ .arch = .{ .string = "qwen2" } });
    defer a.free(bad);
    try tmp.dir.writeFile(io, .{ .sub_path = "bad_arch.gguf", .data = bad });
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/bad_arch.gguf", .{tmp.sub_path});
    defer a.free(path);
    try std.testing.expectError(error.UnsupportedLingGgufArchitecture, cpu_gguf.File.open(a, io, path));
}

test "Ling GGUF refuses malformed KV-head metadata and forbidden clamps" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.MissingLingGgufMetadata, parseAndCheckMeta(a, .{ .kv_mode = .missing }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .kv_mode = .scalar }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .kv_mode = .short_len }));
    try std.testing.expectError(error.UnsupportedLingGgufGeometry, parseAndCheckMeta(a, .{ .kv_mode = .bad_pattern }));
    try std.testing.expectError(error.MetadataTypeMismatch, parseAndCheckMeta(a, .{ .kv_mode = .string_elems }));

    try std.testing.expectError(error.UnsupportedLingGgufClamp, parseAndCheckMeta(a, .{ .clamp_exp = true }));
    try std.testing.expectError(error.UnsupportedLingGgufClamp, parseAndCheckMeta(a, .{ .clamp_shexp = true }));
}

test "Ling GGUF config refuses missing or invalid EOS and BOS tokens" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.MissingLingGgufEos, parseAndCheckConfig(a, .{ .eos = null }));
    try std.testing.expectError(error.BadLingGgufToken, parseAndCheckConfig(a, .{ .eos = .{ .uint32 = 157184 } }));
    try std.testing.expectError(error.MetadataTypeMismatch, parseAndCheckConfig(a, .{ .eos = .{ .string = "eos" } }));
    try std.testing.expectError(error.MetadataTypeMismatch, parseAndCheckConfig(a, .{ .eos = .{ .int32 = -1 } }));
    try std.testing.expectError(error.BadLingGgufToken, parseAndCheckConfig(a, .{ .bos = .{ .uint32 = 157184 } }));
}

fn makeValidTensor(n: names.Name, decay_bytes: []const u8) g.Tensor {
    const kind: g.TensorKind = switch (n.role) {
        .positive_exp_decay, .vector, .convolution => .f32,
        else => .q4_k,
    };
    return .{
        .name = n.source,
        .kind = kind,
        .rank = n.rank,
        .shape = .{ n.dims[0], n.dims[1], n.dims[2], 1 },
        .offset = 0,
        .len = if (n.role == .positive_exp_decay) decay_bytes.len else 0,
        .bytes = if (n.role == .positive_exp_decay) decay_bytes else &.{},
    };
}

test "Ling GGUF refuses missing and extra tensors against the tiny tensor plan" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const plan = try names.tiny(arena.allocator());

    var decay_raw: [64]u8 = undefined;
    for (0..16) |i| std.mem.writeInt(u32, decay_raw[i * 4 ..][0..4], @bitCast(@as(f32, 0.5)), .little);

    // Baseline: exact plan passes.
    var doc = g.Document{
        .gpa = a,
        .bytes = &.{},
        .metadata = &.{},
        .tensors = try a.alloc(g.Tensor, plan.len),
        .data_offset = 0,
        .alignment = 32,
    };
    defer {
        doc.names.deinit(a);
        a.free(doc.tensors);
    }
    for (plan, 0..) |n, i| {
        doc.tensors[i] = makeValidTensor(n, &decay_raw);
        try doc.names.put(a, n.source, i);
    }
    try cpu_gguf.validateTensors(&doc, plan);

    // Missing tensor: drop the last tensor from the document.
    var missing = g.Document{
        .gpa = a,
        .bytes = &.{},
        .metadata = &.{},
        .tensors = doc.tensors[0 .. plan.len - 1],
        .data_offset = 0,
        .alignment = 32,
    };
    defer missing.names.deinit(a);
    for (missing.tensors, 0..) |t, i| try missing.names.put(a, t.name, i);
    try std.testing.expectError(error.MissingLingGgufTensor, cpu_gguf.validateTensors(&missing, plan));

    // Extra tensor: all expected tensors plus one unexpected tensor.
    const extra_tensors = try a.alloc(g.Tensor, plan.len + 1);
    defer a.free(extra_tensors);
    @memcpy(extra_tensors[0..plan.len], doc.tensors);
    extra_tensors[plan.len] = .{ .name = "blk.0.extra.weight", .kind = .f32, .rank = 1, .shape = .{ 16, 1, 1, 1 }, .offset = 0, .len = 0 };
    var extra = g.Document{
        .gpa = a,
        .bytes = &.{},
        .metadata = &.{},
        .tensors = extra_tensors,
        .data_offset = 0,
        .alignment = 32,
    };
    defer extra.names.deinit(a);
    for (extra.tensors, 0..) |t, i| try extra.names.put(a, t.name, i);
    try std.testing.expectError(error.UnexpectedLingGgufTensor, cpu_gguf.validateTensors(&extra, plan));
}
