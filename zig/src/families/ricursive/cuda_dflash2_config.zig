//! Ripple's DFlash2 drafter config on CUDA: #2260's qualified parse of the published config.json, and the CUDA gate.
const std = @import("std");
const spec = @import("dflash2_spec.zig");
const dw = @import("cuda_dflash2_weights.zig");
const dk = @import("cuda_dflash2_kernels.zig");
const da = @import("cuda_dflash2_attn.zig");
const cfg = @import("config.zig");

const Config = spec.Config;

/// The kernels' shapes besides the packer's (B's gate): 128-wide heads in D's groups, 16-row blocks, one top-32 merge.
pub fn gate(c: *const Config) !void {
    try dw.gate(c);
    if (c.block != da.block or c.heads == 0 or c.kv_heads == 0 or c.heads % c.kv_heads != 0) return error.UnsupportedRicursiveDFlash2Shape;
    if (c.hidden % 8 != 0 or c.hidden > dk.most_width or c.top != da.top or c.vocab > 64 * da.top_chunk) return error.UnsupportedRicursiveDFlash2Shape;
}

/// The drafter's config.json in `dir` through PR 2260's qualified parse beside `target`, and the CUDA gate.
pub fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, target: *const cfg.Config) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    const c = try spec.parse(gpa, bytes, target);
    try gate(&c);
    return c;
}

const testing = std.testing;

fn release() !Config {
    const target = try cfg.parse(testing.allocator, @embedFile("fixtures/config.json"));
    return spec.parse(testing.allocator, @embedFile("fixtures/dflash2_config.json"), &target);
}

test "ricursive dflash2 cuda: the gate takes the release and refuses a block D's attention does not hold" {
    var c = try release();
    try gate(&c);
    c.block = 8;
    try testing.expectError(error.UnsupportedRicursiveDFlash2Shape, gate(&c));
    c = try release();
    c.vocab = 64 * da.top_chunk + 1;
    try testing.expectError(error.UnsupportedRicursiveDFlash2Shape, gate(&c));
}

test "ricursive dflash2 cuda: the config read takes the published drafter and refuses a clamped MLP, as #2260 does" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const target = try cfg.parse(a, @embedFile("fixtures/config.json"));
    const published = @embedFile("fixtures/dflash2_config.json");
    const clamped = try std.mem.replaceOwned(u8, a, published, "\"no_rope_layers\"", "\"ffn_clamp_threshold\": 20, \"no_rope_layers\"");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = published });
    _ = try readConfig(testing.allocator, testing.io, dir, &target);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = clamped });
    try testing.expectError(error.UnqualifiedRippleDFlash2, readConfig(testing.allocator, testing.io, dir, &target));
}
