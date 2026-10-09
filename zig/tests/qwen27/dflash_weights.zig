//! The target and drafter headers qualify the stock graph without reading tensor payloads or using a GPU.
const std = @import("std");
const core = @import("core");
const q = @import("qwen27");
const df = q.dflash;
const a = std.heap.page_allocator;
fn read(io: std.Io, dir: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
}
fn bind(source: q.checkpoint.Source, arena: std.mem.Allocator, prefix: []const u8, name: []const u8, c: df.config.Config) !q.checkpoint.Linear {
    const module = try std.mem.concat(arena, u8, &.{ prefix, name });
    const w = try source.get(try std.mem.concat(arena, u8, &.{ module, ".weight" }));
    const s = try source.get(try std.mem.concat(arena, u8, &.{ module, ".scales" }));
    const b = try source.get(try std.mem.concat(arena, u8, &.{ module, ".biases" }));
    return q.checkpoint.validate(w, s, b, .{ .bits = 4, .group_size = 64 }, c.vocab, c.hidden);
}
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4 or !std.mem.eql(u8, args[3], "--check")) return error.CheckModeRequired;
    const text = try read(io, args[2]);
    defer a.free(text);
    const c = try df.config.parse(a, text);
    var target = try core.Checkpoint.openModelPrefix(a, io, args[1], "language_model.");
    defer target.close();
    var draft = try core.Checkpoint.openModel(a, io, args[2]);
    defer draft.close();
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(a);
    for (draft.files.items) |file| for (file.names.keys()) |name| try names.append(a, name);
    const source = q.checkpoint.Source.from(&target);
    var graph = try df.weights.load(a, .{ .source = q.checkpoint.Source.from(&draft), .names = names.items }, c, try bind(source, arena, "language_model.", "model.embed_tokens", c), try bind(source, arena, "language_model.", "lm_head", c));
    defer graph.deinit();
    var config_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &config_hash, .{});
    const hex = std.fmt.bytesToHex(config_hash, .lower);
    const record = try std.json.Stringify.valueAlloc(arena, .{ .kind = "qwen27_dflash2_metadata", .block = c.block, .layers = c.layers, .heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .taps = c.taps, .window = c.window, .tensor_count = names.items.len, .native_bytes = try graph.nativeBytes(), .linears = c.linearCount(), .config_sha256 = @as([]const u8, &hex), .native_storage = "bf16", .target_binding = "mlx-q4g64", .tensor_payload_read = false, .gpu = false, .model_ready = false, .default_quantization_verified = false }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, record);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}
