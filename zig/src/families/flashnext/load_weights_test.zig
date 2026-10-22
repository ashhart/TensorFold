//! The metadata budget bounds actual pack payloads, including the optional gathered draft vocabulary.
const std = @import("std");
const fixture = @import("pack_test.zig");
const pack = @import("pack.zig");
const checkpoint = @import("../../core/checkpoint.zig");
const st = @import("../../core/safetensors.zig");
const memory = @import("flashnext_memory");

test "Flash Next memory: the real pack builder stays inside the metadata payload budget" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeCheckpoint(tmp, arena.allocator());
    const dir = try fixture.tmpPath(arena.allocator(), tmp, ".");
    const out = try fixture.tmpPath(arena.allocator(), tmp, "out");
    try tmp.dir.createDirPath(io, "out");
    try tmp.dir.writeFile(io, .{ .sub_path = "draft.txt", .data = "0 1 2 3" });
    const vocab = try fixture.tmpPath(arena.allocator(), tmp, "draft.txt");
    const inventory = try memory.weights_plan.read(checkpoint.Checkpoint, @import("../../core/quant/mlx.zig"), a, io, dir);
    for ([_]?[]const u8{ null, vocab }) |draft| {
        _ = try pack.build(a, io, dir, out, draft);
        var payload: u64 = 0;
        for ([_][]const u8{ "pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors" }) |name| {
            const path = try std.fs.path.join(a, &.{ out, name });
            defer a.free(path);
            var file = try st.File.open(a, io, path);
            defer file.close(io);
            for (file.names.values()) |e| payload += e.end - e.begin;
        }
        try std.testing.expect(inventory.source > 0);
        try std.testing.expect(payload <= try inventory.resident());
        try std.testing.expect(try inventory.building() >= try inventory.resident());
    }
}

test {
    _ = @import("pack_peak_test.zig");
}
