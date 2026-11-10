//! A dots.mocr checkpoint dir for host tests: the released configs, and optionally a sparse shard of each tensor.
const std = @import("std");
const cfg = @import("config.zig");
const fixtures = @import("config_test.zig");
const inventory = @import("weights.zig");
const t = std.testing;

fn config() !cfg.Config {
    return cfg.Config.parse(t.allocator, fixtures.sample, fixtures.generation);
}

/// A checkpoint dir in the test's tmp dir: the released configs, and with `shards` a sparse shard of each tensor.
pub const Dir = struct {
    tmp: std.testing.TmpDir,
    path: []u8,

    pub fn init(shards: bool, drop: ?[]const u8) !Dir {
        const a = t.allocator;
        const io = t.io;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = fixtures.sample });
        try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = fixtures.generation });
        if (shards) try shard(tmp, drop);
        return .{ .tmp = tmp, .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path}) };
    }

    pub fn deinit(d: *Dir) void {
        t.allocator.free(d.path);
        d.tmp.cleanup();
    }

    /// One shard's header naming every inventory tensor but `drop`, its data left sparse, and its index.
    fn shard(tmp: std.testing.TmpDir, drop: ?[]const u8) !void {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const io = t.io;
        var header: std.Io.Writer.Allocating = .init(a);
        var index: std.Io.Writer.Allocating = .init(a);
        try header.writer.writeAll("{");
        try index.writer.writeAll("{\"weight_map\":{");
        var at: usize = 0;
        var first = true;
        for (try inventory.inventory(a, try config())) |s| {
            if (drop) |name| if (std.mem.eql(u8, name, s.name)) continue;
            const comma = if (first) "" else ",";
            first = false;
            try header.writer.print("{s}\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{ comma, s.name });
            for (s.shape[0..s.rank], 0..) |x, i| try header.writer.print("{s}{d}", .{ if (i == 0) "" else ",", x });
            try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + s.bytes() });
            try index.writer.print("{s}\"{s}\":\"model.safetensors\"", .{ comma, s.name });
            at += s.bytes();
        }
        try header.writer.writeAll("}");
        try index.writer.writeAll("}}");
        const h = header.written();
        var f = try tmp.dir.createFile(io, "model.safetensors", .{});
        defer f.close(io);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, h.len, .little);
        try f.writePositionalAll(io, &len, 0);
        try f.writePositionalAll(io, h, 8);
        try f.setLength(io, 8 + h.len + at);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
    }
};
