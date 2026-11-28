//! Host-side GGUF inspection against the Gemma 4 E4B IT Q4_0 export's text tensor layout.
const std = @import("std");
const gguf = @import("core").gguf;

const layers = 42;
const width = 2560;
const ff_width = 10240;
const heads = 8;
const kv_heads = 2;
const vocab = 262144;
const layer_input = 256;
const shared_kv = 18;
const full_head = 512;
const swa_head = 256;

const integers = .{
    .{ "general.quantization_version", 2 },
    .{ "general.file_type", 2 },
    .{ "gemma4.block_count", layers },
    .{ "gemma4.context_length", 131072 },
    .{ "gemma4.embedding_length", width },
    .{ "gemma4.feed_forward_length", ff_width },
    .{ "gemma4.attention.head_count", heads },
    .{ "gemma4.attention.head_count_kv", kv_heads },
    .{ "gemma4.attention.key_length", full_head },
    .{ "gemma4.attention.value_length", full_head },
    .{ "gemma4.attention.sliding_window", 512 },
    .{ "gemma4.attention.shared_kv_layers", shared_kv },
    .{ "gemma4.embedding_length_per_layer_input", layer_input },
    .{ "gemma4.attention.key_length_swa", swa_head },
    .{ "gemma4.attention.value_length_swa", swa_head },
    .{ "gemma4.rope.dimension_count", full_head },
    .{ "gemma4.rope.dimension_count_swa", swa_head },
};
const floats = .{
    .{ "gemma4.rope.freq_base", 1_000_000.0 },
    .{ "gemma4.rope.freq_base_swa", 10_000.0 },
    .{ "gemma4.attention.layer_norm_rms_epsilon", 0.000001 },
    .{ "gemma4.final_logit_softcapping", 30.0 },
};
const swa_key = "gemma4.attention.sliding_window_pattern";
const tokens_key = "tokenizer.ggml.tokens";

const Spec = struct {
    name: []const u8,
    kind: gguf.TensorKind,
    shape: []const u64,
};

const globals: []const Spec = &.{
    .{ .name = "output_norm.weight", .kind = .f32, .shape = &.{width} },
    .{ .name = "per_layer_model_proj.weight", .kind = .f16, .shape = &.{ width, layers * layer_input } },
    .{ .name = "per_layer_proj_norm.weight", .kind = .f32, .shape = &.{layer_input} },
    .{ .name = "per_layer_token_embd.weight", .kind = .q6_k, .shape = &.{ layers * layer_input, vocab } },
    .{ .name = "rope_freqs.weight", .kind = .f32, .shape = &.{full_head / 2} },
    .{ .name = "token_embd.weight", .kind = .q6_k, .shape = &.{ width, vocab } },
};

const layer_weights: []const Spec = &.{
    .{ .name = "attn_norm.weight", .kind = .f32, .shape = &.{width} },
    .{ .name = "ffn_down.weight", .kind = .q4_0, .shape = &.{ ff_width, width } },
    .{ .name = "ffn_gate.weight", .kind = .q4_0, .shape = &.{ width, ff_width } },
    .{ .name = "ffn_norm.weight", .kind = .f32, .shape = &.{width} },
    .{ .name = "ffn_up.weight", .kind = .q4_0, .shape = &.{ width, ff_width } },
    .{ .name = "inp_gate.weight", .kind = .q4_0, .shape = &.{ width, layer_input } },
    .{ .name = "layer_output_scale.weight", .kind = .f32, .shape = &.{1} },
    .{ .name = "post_attention_norm.weight", .kind = .f32, .shape = &.{width} },
    .{ .name = "post_ffw_norm.weight", .kind = .f32, .shape = &.{width} },
    .{ .name = "post_norm.weight", .kind = .f32, .shape = &.{width} },
    .{ .name = "proj.weight", .kind = .q4_0, .shape = &.{ layer_input, width } },
};

fn isSwa(layer: usize) bool {
    return layer % 6 != 5;
}

fn attention(comptime head: u64) [6]Spec {
    return .{
        .{ .name = "attn_output.weight", .kind = .q4_0, .shape = &.{ heads * head, width } },
        .{ .name = "attn_q.weight", .kind = .q4_0, .shape = &.{ width, heads * head } },
        .{ .name = "attn_q_norm.weight", .kind = .f32, .shape = &.{head} },
        .{ .name = "attn_k.weight", .kind = .q4_0, .shape = &.{ width, kv_heads * head } },
        .{ .name = "attn_k_norm.weight", .kind = .f32, .shape = &.{head} },
        .{ .name = "attn_v.weight", .kind = .q4_0, .shape = &.{ width, kv_heads * head } },
    };
}

const Refusal = struct { reason: []const u8, field: []const u8 };

fn matchesString(doc: *const gguf.Document, key: []const u8, text: []const u8) bool {
    const v = doc.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, text);
}

fn metadata(doc: *const gguf.Document) !?Refusal {
    if (!matchesString(doc, "general.architecture", "gemma4"))
        return .{ .reason = "UnsupportedArchitecture", .field = "general.architecture" };
    if (!matchesString(doc, "general.type", "model"))
        return .{ .reason = "UnsupportedCheckpointType", .field = "general.type" };
    if (doc.get("split.no") != null or doc.get("split.count") != null or doc.get("split.tensors.count") != null)
        return .{ .reason = "UnsupportedSplitCheckpoint", .field = "split" };
    inline for (integers) |check| {
        const v = doc.get(check[0]) orelse return .{ .reason = "MissingMetadata", .field = check[0] };
        if ((v.unsigned() catch null) != @as(?u64, check[1]))
            return .{ .reason = "UnsupportedGeometry", .field = check[0] };
    }
    inline for (floats) |check| {
        const v = doc.get(check[0]) orelse return .{ .reason = "MissingMetadata", .field = check[0] };
        if (v != .float32 or v.float32 != @as(f32, check[1]))
            return .{ .reason = "UnsupportedGeometry", .field = check[0] };
    }
    const pattern = doc.get(swa_key) orelse return .{ .reason = "MissingMetadata", .field = swa_key };
    if (pattern != .array or pattern.array.kind != .boolean or pattern.array.len != layers)
        return .{ .reason = "UnsupportedGeometry", .field = swa_key };
    var it = pattern.array.iterator();
    for (0..layers) |layer| {
        if ((try it.next()).?.boolean != isSwa(layer))
            return .{ .reason = "UnsupportedGeometry", .field = swa_key };
    }
    const tokens = doc.get(tokens_key) orelse return .{ .reason = "MissingMetadata", .field = tokens_key };
    if (tokens != .array or tokens.array.kind != .string or tokens.array.len != vocab)
        return .{ .reason = "UnsupportedGeometry", .field = tokens_key };
    return null;
}

fn checkTensor(doc: *const gguf.Document, name: []const u8, spec: Spec, err: *std.Io.Writer) !bool {
    const got = doc.getTensor(name) orelse return refuse(err, "MissingTensor", name);
    if (got.kind != spec.kind) {
        try err.print("Hexagon inspection refused (UnsupportedTensorKind): {s} is {s}, expected {s}\n", .{
            name, @tagName(got.kind), @tagName(spec.kind),
        });
        return false;
    }
    if (!std.mem.eql(u64, got.shape[0..got.rank], spec.shape)) return refuse(err, "UnsupportedTensorShape", name);
    return true;
}

fn refuse(err: *std.Io.Writer, reason: []const u8, field: []const u8) !bool {
    try err.print("Hexagon inspection refused ({s}): {s}\n", .{ reason, field });
    return false;
}

fn validate(doc: *const gguf.Document, err: *std.Io.Writer) !bool {
    if (try metadata(doc)) |failure| return refuse(err, failure.reason, failure.field);
    for (globals) |spec| if (!try checkTensor(doc, spec.name, spec, err)) return false;
    var count = globals.len;
    for (0..layers) |layer| {
        const layer_attention = if (isSwa(layer)) attention(swa_head) else attention(full_head);
        const used_attention = layer_attention[0..if (layer < layers - shared_kv) 6 else 3];
        for ([_][]const Spec{ layer_weights, used_attention }) |specs| {
            for (specs) |spec| {
                var buf: [64]u8 = undefined;
                const name = try std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, spec.name });
                if (!try checkTensor(doc, name, spec, err)) return false;
                count += 1;
            }
        }
    }
    if (doc.tensors.len != count) {
        try err.print("Hexagon inspection refused (UnexpectedTensor): expected {d} tensors, found {d}\n", .{
            count, doc.tensors.len,
        });
        return false;
    }
    return true;
}

pub fn run(a: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, path: []const u8) !u8 {
    var file = gguf.File.open(a, io, path, .{}) catch |problem| {
        _ = try refuse(err, @errorName(problem), "could not read a valid GGUF");
        return 2;
    };
    defer file.close(io);
    if (!try validate(&file.document, err)) return 2;
    try out.writeAll(
        "Hexagon inspection: accepted Gemma 4 E4B IT Q4_0 text layout (GGUF)\n" ++
            "Device execution: unverified\n",
    );
    return 0;
}

const FixtureCase = enum { accepted, architecture, geometry, missing, extra, kind, shape, truncated };

fn writeInt(w: *std.Io.Writer, comptime T: type, value: T) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try w.writeAll(&bytes);
}

fn writeString(w: *std.Io.Writer, s: []const u8) !void {
    try writeInt(w, u64, s.len);
    try w.writeAll(s);
}

fn writeKey(w: *std.Io.Writer, s: []const u8, kind: gguf.MetadataKind) !void {
    try writeString(w, s);
    try writeInt(w, u32, @backingInt(kind));
}

fn writeTensor(w: *std.Io.Writer, offset: *u64, name: []const u8, spec: Spec) !void {
    try writeString(w, name);
    try writeInt(w, u32, @intCast(spec.shape.len));
    var size: u64 = 1;
    for (spec.shape) |dim| {
        try writeInt(w, u64, dim);
        size *= dim;
    }
    try writeInt(w, u32, @backingInt(spec.kind));
    offset.* = std.mem.alignForward(u64, offset.*, 32);
    try writeInt(w, u64, offset.*);
    const block = spec.kind.block();
    offset.* += size / block.values * block.bytes;
}

fn fixture(w: *std.Io.Writer, which: FixtureCase) !u64 {
    try w.writeAll("GGUF");
    try writeInt(w, u32, 3);
    try writeInt(w, u64, switch (which) {
        .missing => 665,
        .extra => 667,
        else => 666,
    });
    try writeInt(w, u64, 4 + integers.len + floats.len);
    try writeKey(w, "general.architecture", .string);
    try writeString(w, if (which == .architecture) "gemma3" else "gemma4");
    try writeKey(w, "general.type", .string);
    try writeString(w, "model");
    inline for (integers) |check| {
        try writeKey(w, check[0], .uint32);
        try writeInt(w, u32, if (which == .geometry and std.mem.eql(u8, check[0], "gemma4.block_count")) 41 else check[1]);
    }
    inline for (floats) |check| {
        try writeKey(w, check[0], .float32);
        try writeInt(w, u32, @bitCast(@as(f32, check[1])));
    }
    try writeKey(w, swa_key, .array);
    try writeInt(w, u32, @backingInt(gguf.MetadataKind.boolean));
    try writeInt(w, u64, layers);
    for (0..layers) |layer| try writeInt(w, u8, @intFromBool(isSwa(layer)));
    try writeKey(w, tokens_key, .array);
    try writeInt(w, u32, @backingInt(gguf.MetadataKind.string));
    try writeInt(w, u64, vocab);
    try w.splatByteAll(0, vocab * @sizeOf(u64));
    var offset: u64 = 0;
    switch (which) {
        .missing => {},
        .kind => try writeTensor(w, &offset, "output_norm.weight", .{ .name = "", .kind = .f16, .shape = &.{width} }),
        .shape => try writeTensor(w, &offset, "output_norm.weight", .{ .name = "", .kind = .f32, .shape = &.{ width, 1 } }),
        else => try writeTensor(w, &offset, globals[0].name, globals[0]),
    }
    for (globals[1..]) |spec| try writeTensor(w, &offset, spec.name, spec);
    for (0..layers) |layer| {
        const layer_attention = if (isSwa(layer)) attention(swa_head) else attention(full_head);
        for ([_][]const Spec{ layer_weights, layer_attention[0..if (layer < layers - shared_kv) 6 else 3] }) |specs| {
            for (specs) |spec| {
                var buf: [64]u8 = undefined;
                try writeTensor(w, &offset, try std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, spec.name }), spec);
            }
        }
    }
    if (which == .extra)
        try writeTensor(w, &offset, "extra.weight", .{ .name = "", .kind = .f32, .shape = &.{1} });
    return offset;
}

test "info prints family for an accepted Hexagon GGUF and names each layout refusal" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "layout.gguf" });
    defer a.free(path);
    const file = try tmp.dir.createFile(io, "layout.gguf", .{});
    defer file.close(io);
    const cases = [_]struct { FixtureCase, []const u8 }{
        .{ .accepted, "" },
        .{ .architecture, "(UnsupportedArchitecture): general.architecture" },
        .{ .geometry, "(UnsupportedGeometry): gemma4.block_count" },
        .{ .missing, "(MissingTensor): output_norm.weight" },
        .{ .extra, "(UnexpectedTensor): expected 666 tensors, found 667" },
        .{ .kind, "(UnsupportedTensorKind): output_norm.weight is f16, expected f32" },
        .{ .shape, "(UnsupportedTensorShape): output_norm.weight" },
        .{ .truncated, "(TensorOutsideFile)" },
    };
    for (cases) |case| {
        var bytes: std.Io.Writer.Allocating = .init(a);
        defer bytes.deinit();
        const payload = try fixture(&bytes.writer, case[0]);
        try file.writePositionalAll(io, bytes.written(), 0);
        const length = std.mem.alignForward(u64, bytes.written().len, 32) + payload;
        try file.setLength(io, if (case[0] == .truncated) length - 1 else length);
        var out_buf: [256]u8 = undefined;
        var err_buf: [256]u8 = undefined;
        var out = std.Io.Writer.fixed(&out_buf);
        var err = std.Io.Writer.fixed(&err_buf);
        const accepted = case[0] == .accepted;
        try std.testing.expectEqual(@as(u8, if (accepted) 0 else 2), try run(a, io, &out, &err, path));
        if (accepted) {
            try std.testing.expectEqualStrings("", err.buffered());
            try std.testing.expect(std.mem.indexOf(u8, out.buffered(), "Device execution: unverified") != null);
        } else {
            try std.testing.expectEqualStrings("", out.buffered());
            try std.testing.expect(std.mem.indexOf(u8, err.buffered(), case[1]) != null);
        }
    }
}
