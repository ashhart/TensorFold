//! EXL3 admission for the Flash Next CUDA family: config.json's quantization_config names quant_method "exl3" and the
//! "mul1" codebook. The geometry and semantics are the ones cuda_config.zig fixes for every mode. The bit widths are
//! never read from `bits` (3.05 here is an average): each matrix's width comes from its own trellis (16 * bits words).

const std = @import("std");

/// The one codebook the family reads (the kernels' `mul1`, cuda.exl3.Codebook.mul1).
pub const codebook = "mul1";
/// The weight format the family lists for an EXL3 checkpoint.
pub const format = "exl3-mul1";

fn object(o: std.json.ObjectMap, name: []const u8) ?std.json.ObjectMap {
    const v = o.get(name) orelse return null;
    return if (v == .object) v.object else null;
}

fn word(o: std.json.ObjectMap, name: []const u8) ?[]const u8 {
    const v = o.get(name) orelse return null;
    return if (v == .string) v.string else null;
}

/// config.json's quantization_config: top level, else the text config's.
pub fn block(root: std.json.ObjectMap) ?std.json.ObjectMap {
    if (object(root, "quantization_config")) |q| return q;
    if (object(root, "text_config")) |t| if (object(t, "quantization_config")) |q| return q;
    return null;
}

/// Whether config.json declares an EXL3 checkpoint (quant_method "exl3"), whatever its codebook.
pub fn named(root: std.json.ObjectMap) bool {
    const q = block(root) orelse return false;
    const m = word(q, "quant_method") orelse return false;
    return std.mem.eql(u8, m, "exl3");
}

/// The first field an EXL3 checkpoint differs in from what the family serves, else null. `geometry` is
/// cuda_config.geometry (shapes, then semantics), passed in so this file stays out of cuda_config's privates.
pub fn mismatch(root: std.json.ObjectMap, text: std.json.ObjectMap, comptime geometry: fn (std.json.ObjectMap) ?[]const u8) ?[]const u8 {
    const q = block(root) orelse return "quantization_config";
    const m = word(q, "quant_method") orelse return "quantization_config.quant_method";
    if (!std.mem.eql(u8, m, "exl3")) return "quantization_config.quant_method";
    const book = word(q, "codebook") orelse return "quantization_config.codebook";
    if (!std.mem.eql(u8, book, codebook)) return "quantization_config.codebook";
    return geometry(text);
}

const fixture = @embedFile("fixtures/exl3_config.json");

test "exl3: the Qwen3.8-Flash-Next EXL3 pack's config.json is admitted as Mode.exl3" {
    const config = @import("cuda_config.zig");
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = fixture });
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    try std.testing.expectEqual(config.Mode.exl3, try config.mode(a, std.testing.io, dir));
}

test "exl3: other codebooks and other methods are refused by name; bits are never read" {
    const config = @import("cuda_config.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { field: []const u8, value: std.json.Value, refused: ?[]const u8 };
    const cases = [_]Case{
        .{ .field = "codebook", .value = .{ .string = "mcg" }, .refused = "quantization_config.codebook" },
        .{ .field = "codebook", .value = .{ .string = "inst3" }, .refused = "quantization_config.codebook" },
        .{ .field = "codebook", .value = .null, .refused = "quantization_config.codebook" },
        .{ .field = "quant_method", .value = .{ .string = "gptq" }, .refused = "quantization_config.quant_method" },
        // the average width and the head's are not a contract: the trellis says each matrix's own width
        .{ .field = "bits", .value = .{ .float = 4.25 }, .refused = null },
        .{ .field = "bits", .value = .{ .integer = 2 }, .refused = null },
        .{ .field = "head_bits", .value = .{ .integer = 8 }, .refused = null },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
        const root = parsed.value.object;
        const text = root.get("text_config").?.object;
        try std.testing.expect(named(root));
        try std.testing.expectEqual(@as(?[]const u8, null), mismatch(root, text, config.geometry));
        try root.getPtr("quantization_config").?.object.put(a, c.field, c.value);
        const got = mismatch(parsed.value.object, text, config.geometry);
        if (c.refused) |want| try std.testing.expectEqualStrings(want, got orelse "accepted") else try std.testing.expectEqual(@as(?[]const u8, null), got);
    }
}

test "exl3: the shapes and semantics checks still apply to an EXL3 checkpoint" {
    const config = @import("cuda_config.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { field: []const u8, value: std.json.Value, refused: []const u8 };
    const cases = [_]Case{
        .{ .field = "hidden_size", .value = .{ .integer = 4096 }, .refused = "hidden_size" },
        .{ .field = "num_experts_per_tok", .value = .{ .integer = 8 }, .refused = "num_experts_per_tok" },
        .{ .field = "rms_norm_eps", .value = .{ .float = 1e-5 }, .refused = "rms_norm_eps" },
        .{ .field = "output_gate_type", .value = .{ .string = "silu" }, .refused = "output_gate_type" },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
        const root = parsed.value.object;
        const text = &root.getPtr("text_config").?.object;
        try text.put(a, c.field, c.value);
        try std.testing.expectEqualStrings(c.refused, mismatch(root, text.*, config.geometry) orelse "accepted");
    }
}

test "exl3: a non-EXL3 config is not named EXL3" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), "{\"quantization_config\":{\"quant_method\":\"modelopt\"}}", .{});
    try std.testing.expect(!named(parsed.value.object));
    const bare = try std.json.parseFromSlice(std.json.Value, arena.allocator(), "{\"text_config\":{}}", .{});
    try std.testing.expect(!named(bare.value.object));
}
