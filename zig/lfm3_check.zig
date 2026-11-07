//! Check saved header metadata from tools/lfm3/inspect_checkpoint.py without loading model data or running a GPU.
const std = @import("std");
const lfm3 = struct {
    const config = @import("src/families/lfm3/config.zig");
    const weights = @import("src/families/lfm3/weights.zig");
};
const st = @import("safetensors");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) {
        std.debug.print("usage: tf-lfm3-check CHECKPOINT_METADATA.json\n", .{});
        return error.ExpectedMetadata;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(32 << 20));
    const value = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (value != .object) return error.BadMetadata;
    const root = value.object;
    const cfg_bytes = try std.json.Stringify.valueAlloc(a, root.get("config") orelse return error.BadMetadata, .{});
    var cfg = try lfm3.config.parse(a, cfg_bytes);
    defer cfg.deinit();
    const tensors = root.get("tensors") orelse return error.BadMetadata;
    const shards = root.get("shards") orelse return error.BadMetadata;
    const lengths = root.get("shard_data_lengths") orelse return error.BadMetadata;
    if (tensors != .object or shards != .array or shards.array.items.len == 0 or lengths != .object)
        return error.BadMetadata;
    var header: st.Header = .empty;
    for (shards.array.items) |shard| {
        if (shard != .string) return error.BadMetadata;
        var selected: std.json.ObjectMap = .empty;
        const length = lengths.object.get(shard.string) orelse return error.BadMetadata;
        if (length != .integer or length.integer < 0) return error.BadMetadata;
        const data_len = std.math.cast(usize, length.integer) orelse return error.BadMetadata;
        var it = tensors.object.iterator();
        while (it.next()) |item| {
            const tensor = item.value_ptr.*;
            if (tensor != .object) return error.BadMetadata;
            const file = tensor.object.get("shard") orelse return error.BadMetadata;
            if (file != .string) return error.BadMetadata;
            if (!std.mem.eql(u8, file.string, shard.string)) continue;
            try selected.put(a, item.key_ptr.*, tensor);
        }
        if (selected.count() == 0) return error.BadMetadata;
        const text = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = selected }, .{});
        const parsed = try st.parseHeader(a, text, data_len);
        for (parsed.keys(), parsed.values()) |key, entry| {
            if (header.contains(key)) return error.DuplicateTensor;
            try header.put(a, key, entry);
        }
    }
    if (header.count() != tensors.object.count()) return error.BadMetadata;
    const manifest = try lfm3.weights.validate(a, &cfg, &header);
    std.debug.print("LFM3 text admission passed: {s}, {d} text tensors, {d} stored bytes; ", .{
        @tagName(cfg.kind), manifest.tensors, manifest.bytes,
    });
    std.debug.print("{d} other tensors not qualified; GPU inference not tested\n", .{header.count() - manifest.tensors});
}
