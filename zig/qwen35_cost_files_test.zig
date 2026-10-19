//! Exercise the shared file publisher in a workspace-local directory, including overlapping writers.
const std = @import("std");
const shared = @import("lanes").cost_cache;
const costs = @import("qwen35").native.window_costs;
const a = std.testing.allocator;
const io = std.testing.io;
const key: [32]u8 = @splat(42);
const filename = std.fmt.bytesToHex(key, .lower) ++ ".bin";
const Fault = struct {
    threadlocal var creations: usize = 0;
    threadlocal var private: bool = true;
    fn create(ctx: ?*anyopaque, dir: std.Io.Dir, name: []const u8, options: std.Io.Dir.CreateFileAtomicOptions) std.Io.Dir.CreateFileAtomicError!std.Io.File.Atomic {
        private = private and options.replace and options.permissions.toMode() == 0o600;
        creations += 1;
        return io.vtable.dirCreateFileAtomic(ctx, dir, name, options);
    }
    fn rename(_: ?*anyopaque, _: std.Io.Dir, _: []const u8, _: std.Io.Dir, _: []const u8) std.Io.Dir.RenameError!void {
        return error.AccessDenied;
    }
};

fn onlyDestination(dir: std.Io.Dir) !void {
    var it = dir.iterate();
    var count: usize = 0;
    while (try it.next(io)) |entry| {
        try std.testing.expectEqualStrings(filename, entry.name);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), count);
}

pub fn failure() !void {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var vtable = io.vtable.*;
    vtable.dirCreateFileAtomic = Fault.create;
    const checked: std.Io = .{ .userdata = io.userdata, .vtable = &vtable };
    Fault.creations = 0;
    Fault.private = true;
    const old = costs.record(key, 64, @splat(1));
    shared.saveAt(costs.Record, a, checked, key, old, root);
    try std.testing.expectEqual(@as(usize, 1), Fault.creations);
    vtable.dirRename = Fault.rename;
    shared.saveAt(costs.Record, a, checked, key, costs.record(key, 64, @splat(2)), root);
    try std.testing.expectEqual(@as(usize, 2), Fault.creations);
    try std.testing.expectEqualDeep(old, shared.loadAt(costs.Record, a, io, key, root).?);
    try onlyDestination(tmp.dir);
    try std.testing.expect(Fault.private);
}

const Writer = struct {
    root: []const u8,
    value: f64,
    failed: bool = false,
    fn run(w: *Writer) void {
        for (0..32) |_| {
            shared.saveAt(costs.Record, a, io, key, costs.record(key, 64, @splat(w.value)), w.root);
            const read = shared.loadAt(costs.Record, a, io, key, w.root) orelse {
                w.failed = true;
                return;
            };
            if (!costs.accepted(read, key, 64) or (read.ms[0] != 1 and read.ms[0] != 2)) {
                w.failed = true;
                return;
            }
            for (read.ms) |ms| if (ms != read.ms[0]) {
                w.failed = true;
                return;
            };
        }
    }
};
pub fn concurrent() !void {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var writers: [4]Writer = undefined;
    var threads: [4]std.Thread = undefined;
    for (&writers, &threads, 0..) |*writer, *thread, i| {
        writer.* = .{ .root = root, .value = @floatFromInt(1 + i % 2) };
        thread.* = try std.Thread.spawn(.{}, Writer.run, .{writer});
    }
    for (&threads) |thread| thread.join();
    for (writers) |writer| try std.testing.expect(!writer.failed);
    try onlyDestination(tmp.dir);
}
