//! tf-drama-backbone-replay's fixtures: both stage manifests, their captures and the contract prompt, checked on CPU.
const std = @import("std");
const drama = @import("drama");
const Allocator = std.mem.Allocator;
const Stage = drama.backbone.Stage;
const backbone_replay = @import("drama_backbone_replay.zig");
const width = backbone_replay.width;
const max_rows = backbone_replay.max_rows;
const context = backbone_replay.context;
const rows = backbone_replay.rows;
const tensor_bytes = backbone_replay.tensor_bytes;
const layer_count = backbone_replay.layer_count;
const contract_tokens_sha256 = backbone_replay.contract_tokens_sha256;
const replay_layers = backbone_replay.replay_layers;
const chunks = backbone_replay.chunks;
const Origin = backbone_replay.Origin;
const used_stages = backbone_replay.used_stages;
const usedIndex = backbone_replay.usedIndex;
const stageFileName = backbone_replay.stageFileName;
const Errors = backbone_replay.Errors;
const nonfiniteBf16 = backbone_replay.nonfiniteBf16;
const sha256Hex = backbone_replay.sha256Hex;

/// One stage directory's validated fixtures. A null tensor failed validation (and recorded why).
pub const Source = struct {
    origin: Origin,
    path: []const u8,
    manifest_sha: ?[64]u8 = null,
    manifest_evidence_valid: bool = false,
    tensors: [replay_layers.len][used_stages.len]?[]align(16) const u8 = @splat(@splat(null)),
    shas: [replay_layers.len][used_stages.len][64]u8 = undefined,
    /// Native only: the prompt and the production run's final outputs the trace recorded.
    tokens_file: ?[]const u8 = null,
    final_hidden_sha: ?[]const u8 = null,
    semantic_logits_sha: ?[]const u8 = null,

    pub fn tensor(s: *const Source, li: usize, stage: Stage) []align(16) const u8 {
        return s.tensors[li][usedIndex(stage)].?;
    }

    pub fn sha(s: *const Source, li: usize, stage: Stage) []const u8 {
        return &s.shas[li][usedIndex(stage)];
    }
};

fn jsonUint(v: ?std.json.Value, expected: u64) bool {
    const value = v orelse return false;
    return value == .integer and value.integer >= 0 and @as(u64, @intCast(value.integer)) == expected;
}

fn jsonText(v: ?std.json.Value) ?[]const u8 {
    const value = v orelse return null;
    return if (value == .string) value.string else null;
}

fn isSha256Hex(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| switch (c) {
        '0'...'9', 'a'...'f' => {},
        else => return false,
    };
    return true;
}

fn expectUint(errors: *Errors, origin: Origin, o: std.json.ObjectMap, key: []const u8, expected: u64) !void {
    if (!jsonUint(o.get(key), expected)) try errors.add("{s} manifest: {s} is not {d}", .{ @tagName(origin), key, expected });
}

fn expectText(errors: *Errors, origin: Origin, o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const got = jsonText(o.get(key)) orelse "";
    if (!std.mem.eql(u8, got, expected)) try errors.add("{s} manifest: {s} is not {s}", .{ @tagName(origin), key, expected });
}

/// A nested object's string field, or null after recording the failure.
fn nestedText(errors: *Errors, origin: Origin, o: std.json.ObjectMap, outer: []const u8, inner: []const u8) !?[]const u8 {
    if (o.get(outer)) |v| {
        if (v == .object) {
            if (jsonText(v.object.get(inner))) |s| return s;
        }
    }
    try errors.add("{s} manifest: {s}.{s} missing or not a string", .{ @tagName(origin), outer, inner });
    return null;
}

fn jsonChunks(v: std.json.Value) bool {
    if (v != .array or v.array.items.len != chunks.len) return false;
    for (v.array.items, chunks) |item, c| {
        if (!jsonUint(item, c)) return false;
    }
    return true;
}

pub fn loadSource(arena: Allocator, io: std.Io, origin: Origin, path: []const u8, errors: *Errors) !Source {
    var src: Source = .{ .origin = origin, .path = path };
    const dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| {
        try errors.add("{s}: stage directory {s} unreadable: {s}", .{ @tagName(origin), path, @errorName(err) });
        return src;
    };
    defer dir.close(io);
    const text = dir.readFileAlloc(io, "manifest.json", arena, .limited(1 << 24)) catch |err| {
        try errors.add("{s}: manifest.json unreadable: {s}", .{ @tagName(origin), @errorName(err) });
        return src;
    };
    src.manifest_sha = sha256Hex(text);
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |err| {
        try errors.add("{s}: manifest.json is not JSON: {s}", .{ @tagName(origin), @errorName(err) });
        return src;
    };
    if (doc != .object) {
        try errors.add("{s}: manifest.json is not an object", .{@tagName(origin)});
        return src;
    }
    const root = doc.object;
    try expectUint(errors, origin, root, "schema_version", 1);
    try expectUint(errors, origin, root, "rows", rows);
    try expectUint(errors, origin, root, "width", width);
    try expectUint(errors, origin, root, "layer_count", layer_count);
    try expectText(errors, origin, root, "dtype", "bf16le");
    try expectText(errors, origin, root, "tokens_sha256", contract_tokens_sha256);
    src.manifest_evidence_valid = if (root.get("evidence_valid")) |v| v == .bool and v.bool else false;
    if (!src.manifest_evidence_valid) try errors.add("{s} manifest: evidence_valid is not true", .{@tagName(origin)});
    if (origin == .native) {
        const options_ok = if (root.get("model_options")) |v| v == .object and jsonUint(v.object.get("context"), context) and jsonUint(v.object.get("max_rows"), max_rows) else false;
        if (!options_ok) try errors.add("native manifest: model_options is not context {d} max_rows {d}", .{ context, max_rows });
        const chunks_ok = if (root.get("chunks")) |v| jsonChunks(v) else false;
        if (!chunks_ok) try errors.add("native manifest: chunks is not [128, 54]", .{});
        src.tokens_file = jsonText(root.get("tokens_file"));
        if (src.tokens_file == null) try errors.add("native manifest: tokens_file missing", .{});
        src.final_hidden_sha = try nestedText(errors, origin, root, "final_hidden", "sha256");
        src.semantic_logits_sha = try nestedText(errors, origin, root, "semantic_logits", "sha256");
    }
    const captures = root.get("captures") orelse std.json.Value{ .null = {} };
    if (captures != .array) {
        try errors.add("{s} manifest: captures is not an array", .{@tagName(origin)});
        return src;
    }
    for (replay_layers, 0..) |layer, li| {
        for (used_stages, 0..) |stage, si| {
            src.tensors[li][si] = try loadCapture(arena, io, dir, origin, captures.array.items, layer, stage, errors, &src.shas[li][si]);
        }
    }
    return src;
}

/// One stage file checked against its unique manifest entry; null after recording any failure.
fn loadCapture(arena: Allocator, io: std.Io, dir: std.Io.Dir, origin: Origin, captures: []const std.json.Value, layer: usize, stage: Stage, errors: *Errors, sha_out: *[64]u8) !?[]align(16) const u8 {
    const tag = @tagName(origin);
    var name_buffer: [64]u8 = undefined;
    const name = try stageFileName(&name_buffer, layer, stage);
    var entry: ?std.json.ObjectMap = null;
    var matches: usize = 0;
    for (captures) |c| {
        if (c != .object) continue;
        const stage_name = jsonText(c.object.get("stage")) orelse continue;
        if (jsonUint(c.object.get("layer"), layer) and std.mem.eql(u8, stage_name, @tagName(stage))) {
            entry = c.object;
            matches += 1;
        }
    }
    if (matches != 1) {
        try errors.add("{s} {s}: {d} manifest captures, expected exactly 1", .{ tag, name, matches });
        return null;
    }
    const e = entry.?;
    const file = jsonText(e.get("file")) orelse "";
    if (!std.mem.eql(u8, file, name)) {
        try errors.add("{s} {s}: manifest file is {s}", .{ tag, name, file });
        return null;
    }
    const want = jsonText(e.get("sha256")) orelse "";
    if (!isSha256Hex(want)) {
        try errors.add("{s} {s}: manifest sha256 is not 64 lowercase hex digits", .{ tag, name });
        return null;
    }
    const shape_ok = if (e.get("shape")) |v| v == .array and v.array.items.len == 2 and jsonUint(v.array.items[0], rows) and jsonUint(v.array.items[1], width) else false;
    if (!shape_ok) {
        try errors.add("{s} {s}: manifest shape is not [{d}, {d}]", .{ tag, name, rows, width });
        return null;
    }
    const bytes = dir.readFileAllocOptions(io, name, arena, .limited(tensor_bytes + 1), .@"16", null) catch |err| {
        try errors.add("{s} {s}: unreadable or larger than {d} bytes: {s}", .{ tag, name, tensor_bytes, @errorName(err) });
        return null;
    };
    if (bytes.len != tensor_bytes) {
        try errors.add("{s} {s}: {d} bytes, expected {d}", .{ tag, name, bytes.len, tensor_bytes });
        return null;
    }
    const got = sha256Hex(bytes);
    if (!std.mem.eql(u8, &got, want)) {
        try errors.add("{s} {s}: sha256 {s} does not match manifest {s}", .{ tag, name, got[0..], want });
        return null;
    }
    const nonfinite = nonfiniteBf16(std.mem.bytesAsSlice(u16, bytes));
    if (nonfinite != 0) {
        try errors.add("{s} {s}: {d} nonfinite values", .{ tag, name, nonfinite });
        return null;
    }
    sha_out.* = got;
    return bytes;
}

/// The contract prompt, from the native manifest's tokens_file; null after recording any failure.
pub fn loadTokens(arena: Allocator, io: std.Io, native: *const Source, errors: *Errors) !?[]const u32 {
    const path = native.tokens_file orelse return null;
    const bytes = std.Io.Dir.cwd().readFileAllocOptions(io, path, arena, .limited(context * 4 + 1), .@"16", null) catch |err| {
        try errors.add("tokens file {s} unreadable: {s}", .{ path, @errorName(err) });
        return null;
    };
    if (bytes.len != rows * 4) {
        try errors.add("tokens file {s}: {d} bytes, expected {d}", .{ path, bytes.len, rows * 4 });
        return null;
    }
    const got = sha256Hex(bytes);
    if (!std.mem.eql(u8, &got, contract_tokens_sha256)) {
        try errors.add("tokens file {s}: sha256 {s} is not the contract prompt", .{ path, got[0..] });
        return null;
    }
    return std.mem.bytesAsSlice(u32, bytes);
}
