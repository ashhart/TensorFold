//! Exercise the shared file publisher in a workspace-local directory, including overlapping writers.
const std = @import("std");
const shared = @import("lanes").cost_cache;
const costs = @import("qwen35").native.window_costs;
const a = std.testing.allocator;
const io = std.testing.io;
const key: [32]u8 = @splat(42);
const filename = std.fmt.bytesToHex(key, .lower) ++ ".bin";
const windows = @import("builtin").os.tag == .windows;
const Fault = struct {
    threadlocal var creations: usize = 0;
    threadlocal var private: bool = true;
    fn create(ctx: ?*anyopaque, dir: std.Io.Dir, name: []const u8, options: std.Io.Dir.CreateFileAtomicOptions) std.Io.Dir.CreateFileAtomicError!std.Io.File.Atomic {
        private = private and options.replace and (windows or options.permissions.toMode() == 0o600);
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

const Tally = struct {
    whole: usize = 0,
    missed: usize = 0,
    refused: usize = 0,
    fn add(t: *Tally, got: shared.Kept(costs.Record)) void {
        switch (got) {
            .missing => t.missed += 1,
            .refused => t.refused += 1,
            .value => |read| if (complete(read)) {
                t.whole += 1;
            } else {
                t.refused += 1;
            },
        }
    }
};

fn complete(read: costs.Record) bool {
    if (!costs.accepted(read, key, 64) or (read.ms[0] != 1 and read.ms[0] != 2)) return false;
    for (read.ms) |ms| if (ms != read.ms[0]) return false;
    return true;
}

const Writer = struct {
    root: []const u8,
    value: f64,
    io: std.Io,
    tally: Tally = .{},
    fn run(w: *Writer) void {
        for (0..32) |_| {
            shared.saveAt(costs.Record, a, w.io, key, costs.record(key, 64, @splat(w.value)), w.root);
            w.tally.add(shared.readAt(costs.Record, a, w.io, key, w.root));
        }
    }
};

/// Four writers publish through `with` and read back at once; `after` is one plain read once all have joined.
fn race(with: std.Io) !struct { during: Tally, after: Tally } {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var writers: [4]Writer = undefined;
    var threads: [4]std.Thread = undefined;
    for (&writers, &threads, 0..) |*writer, *thread, i| {
        writer.* = .{ .root = root, .value = @floatFromInt(1 + i % 2), .io = with };
        thread.* = try std.Thread.spawn(.{}, Writer.run, .{writer});
    }
    for (&threads) |thread| thread.join();
    var during: Tally = .{};
    for (writers) |writer| {
        during.whole += writer.tally.whole;
        during.missed += writer.tally.missed;
        during.refused += writer.tally.refused;
    }
    try std.testing.expectEqual(@as(usize, 4 * 32), during.whole + during.missed + during.refused);
    var after: Tally = .{};
    after.add(shared.readAt(costs.Record, a, io, key, root));
    try onlyDestination(tmp.dir);
    return .{ .during = during, .after = after };
}

pub fn concurrent() !void {
    const got = try race(io);
    try std.testing.expectEqual(@as(usize, 0), got.during.refused);
    // Windows may refuse to open a table while another writer's replacement settles: a miss, never a torn table.
    if (!windows) try std.testing.expectEqual(@as(usize, 0), got.during.missed);
    try std.testing.expectEqualDeep(Tally{ .whole = 1 }, got.after);
}

/// Windows's replace window, planted: each odd open of the table just after this thread replaced it is refused.
const Window = struct {
    threadlocal var renames: usize = 0;
    threadlocal var settling: bool = false;
    var planted: std.atomic.Value(usize) = .init(0);
    fn rename(
        ctx: ?*anyopaque,
        old_dir: std.Io.Dir,
        old_path: []const u8,
        new_dir: std.Io.Dir,
        new_path: []const u8,
    ) std.Io.Dir.RenameError!void {
        try io.vtable.dirRename(ctx, old_dir, old_path, new_dir, new_path);
        renames += 1;
        settling = renames % 2 == 1;
    }
    fn open(
        ctx: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenFileOptions,
    ) std.Io.File.OpenError!std.Io.File {
        if (settling and std.mem.endsWith(u8, sub_path, filename)) {
            settling = false;
            _ = planted.fetchAdd(1, .monotonic);
            return error.AccessDenied;
        }
        return io.vtable.dirOpenFile(ctx, dir, sub_path, options);
    }
};

pub fn replaceWindow() !void {
    // Windows reads tables through Win32 rather than std.Io, so the plant cannot reach it; `concurrent` races there.
    if (windows) return error.SkipZigTest;
    var vtable = io.vtable.*;
    vtable.dirRename = Window.rename;
    vtable.dirOpenFile = Window.open;
    Window.planted.store(0, .monotonic);
    const got = try race(.{ .userdata = io.userdata, .vtable = &vtable });
    try std.testing.expectEqual(@as(usize, 4 * 16), Window.planted.load(.monotonic));
    try std.testing.expectEqualDeep(Tally{ .whole = 4 * 16, .missed = 4 * 16 }, got.during);
    try std.testing.expectEqualDeep(Tally{ .whole = 1 }, got.after);
}

/// A publisher that renames half-written tables into place, planted so the race's refusal count is shown to bite.
fn tornWrite(
    ctx: ?*anyopaque,
    file: std.Io.File,
    header: []const u8,
    data: []const []const u8,
    splat: usize,
    offset: u64,
) std.Io.File.WritePositionalError!usize {
    _ = try io.vtable.fileWritePositional(ctx, file, header, &.{data[0][0 .. data[0].len / 2]}, splat, offset);
    return data[0].len;
}

pub fn tornPublication() !void {
    var vtable = io.vtable.*;
    vtable.fileWritePositional = tornWrite;
    const got = try race(.{ .userdata = io.userdata, .vtable = &vtable });
    try std.testing.expectEqual(@as(usize, 0), got.during.whole);
    if (!windows) try std.testing.expectEqual(@as(usize, 4 * 32), got.during.refused);
    try std.testing.expectEqualDeep(Tally{ .refused = 1 }, got.after);
}
