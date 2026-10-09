//! Validate the complete target graph and report metadata pins without touching tensor payload bytes.
const std = @import("std");
const core = @import("core");
const qwen = @import("qwen27");

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit));
}

fn digest(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 or !std.mem.eql(u8, args[2], "--check")) return error.CheckModeRequired;
    const config = try read(a, io, args[1], "config.json", 1 << 20);
    const generation = read(a, io, args[1], "generation_config.json", 1 << 20) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    const index = try read(a, io, args[1], "model.safetensors.index.json", 1 << 26);
    const c = try qwen.config.parse(a, config, generation);
    var formats = try qwen.affine.Formats.init(a, config);
    defer formats.deinit();
    var ck = try core.Checkpoint.openModelPrefix(a, io, args[1], "language_model.");
    defer ck.close();
    var w = try qwen.weights.load(init.gpa, qwen.checkpoint.Source.from(&ck), &formats, c);
    defer w.deinit();
    var recurrent: usize = 0;
    for (c.kinds[0..c.layers]) |kind| recurrent += @intFromBool(kind == .linear);
    const cs = digest(config);
    const is = digest(index);
    const record = try std.json.Stringify.valueAlloc(a, .{ .kind = "qwen27_weight_graph", .version = 1, .exact_metadata = true, .namespace = w.prefix, .layers = c.layers, .recurrent_layers = recurrent, .attention_layers = c.layers - recurrent, .native_tensor_count = ck.used.count(), .native_tensor_bytes = try w.tensorBytes(), .config_sha256 = @as([]const u8, &cs), .index_sha256 = @as([]const u8, &is), .tensor_payload_read = false, .gpu = false, .model_ready = false }, .{});
    std.debug.print("{s}\n", .{record});
}
