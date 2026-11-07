//! Check the real dots.mocr upload after its mappings close, then verify device ownership returns to its baseline.
const std = @import("std");
const hip = @import("hip");
const family = @import("dots_mocr");
const st = @import("core").safetensors;

test "dots HIP device weights: all stored tensors upload unchanged and release their device bytes" {
    const a = std.testing.allocator;
    const dir = @import("sys").getenv("TF_DOTS_DIR") orelse return error.MissingDotsCheckpoint;
    const c = try family.config.Config.read(a, std.testing.io, dir);
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const before = hip.usage(false).device;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const text = try family.config.readFile(scratch, std.testing.io, dir, "model.safetensors.index.json");
    const root = try std.json.parseFromSliceLeaky(std.json.Value, scratch, text, .{});
    const index = try st.index.weightMap(root);
    const chunk = try a.alloc(u8, 1 << 20);
    defer a.free(chunk);
    {
        var w = try family.weights.Weights.load(a, &r, std.testing.io, dir, c, 7 << 30);
        defer w.deinit();
        try std.testing.expectEqual(before + 6078358528, hip.usage(false).device);
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        for (index.values()) |v| {
            if (seen.contains(v.string)) continue;
            try seen.put(scratch, v.string, {});
            const path = try std.fs.path.join(scratch, &.{ dir, v.string });
            var file = try st.File.open(a, std.testing.io, path);
            defer file.close(std.testing.io);
            for (file.names.keys()) |name| {
                const tensor = file.get(name).?;
                const place = w.layout.tensors.get(name).?;
                try std.testing.expectEqual(w.buffer.base() + place.offset, try w.get(name));
                var offset: usize = 0;
                while (offset < tensor.bytes.len) {
                    const n = @min(chunk.len, tensor.bytes.len - offset);
                    try w.buffer.download(place.offset + offset, chunk[0..n]);
                    try std.testing.expectEqualSlices(u8, tensor.bytes[offset..][0..n], chunk[0..n]);
                    offset += n;
                }
            }
        }
        try std.testing.expectError(error.DotsMissingTensor, w.get("absent"));
    }
    try std.testing.expectEqual(before, hip.usage(false).device);
}
