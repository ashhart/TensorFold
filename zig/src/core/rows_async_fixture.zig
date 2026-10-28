//! Host fixtures for spill concurrency and fault tests; every store has independent filesystem hooks.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("rows_fake.zig").Fake;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
pub const a = std.testing.allocator;

pub const Fixture = struct {
    root: []u8,
    f: Fake,
    s: pc.Store,
    r: *rows.Rows,

    pub fn open(tag: []const u8, spill: rows.Spill, synchronous: bool, limit: u64) !*Fixture {
        const x = try a.create(Fixture);
        errdefer a.destroy(x);
        const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
        const root = try std.fmt.allocPrint(a, "{s}/tf-rows-{s}-{d}", .{ tmp, tag, platform.c.getpid() });
        errdefer a.free(root);
        x.* = .{ .root = root, .f = .{ .gpa = a }, .s = undefined, .r = undefined };
        x.s = pc.Store.init(a, x.f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 30);
        x.r = try rows.attach(a, &x.s, .{ .root = root, .identity = 7, .cap = 1 << 30, .floor = 0, .daily = 1 << 30, .lookahead = 1, .spill = spill, .sync_writes = synchronous, .staging_bytes = limit });
        return x;
    }

    pub fn close(x: *Fixture) void {
        x.s.deinit();
        x.r.deinit();
        rmTree(x.root);
        a.free(x.root);
        a.destroy(x);
    }

    pub fn keep(x: *Fixture, tokens: []const u32, at: u32) !void {
        x.f.prompt = tokens;
        x.f.at = at;
        try std.testing.expect(x.s.keep(tokens, at, null, x.f.starts, x.f.spans));
        if (!x.r.sync_writes and x.r.spill != .evicted) _ = rows.write.step(x.r);
    }

    pub fn settle(x: *Fixture) !void {
        const end = rows.write.nowNs() + 3 * std.time.ns_per_s;
        while (x.r.worker) |w| {
            rows.write.reap(x.r);
            if (w.pending() == 0) return;
            if (rows.write.nowNs() >= end) return error.WriterTimedOut;
            rows.write.io.nap();
        }
    }
};

pub fn prompt(salt: u32) [1100]u32 {
    var p: [1100]u32 = undefined;
    for (&p, 0..) |*t, i| t.* = @as(u32, @intCast(i)) *% 7919 +% salt;
    return p;
}

pub fn files(dir: [:0]const u8, ext: []const u8) usize {
    const d = platform.c.opendir(dir) orelse return 0;
    defer _ = platform.c.closedir(d);
    var n: usize = 0;
    while (platform.c.readdir(d)) |ent| {
        if (std.mem.endsWith(u8, std.mem.sliceTo(&ent.name, 0), ext)) n += 1;
    }
    return n;
}

pub fn digest(dir: [:0]const u8, name: []const u8) ![32]u8 {
    var buf: [1400]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = platform.c.close(fd);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: [4096]u8 = undefined;
    while (true) {
        const n = platform.c.read(fd, &bytes, bytes.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        h.update(bytes[0..@intCast(n)]);
    }
    return h.finalResult();
}

pub const Gate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    left: std.atomic.Value(bool) = .init(false),
    mode: enum { good, full, short, sync_failure, first_fails } = .good,
    calls: usize = 0,
    sync_entered: std.atomic.Value(bool) = .init(false),
    sync_release: std.atomic.Value(bool) = .init(true),

    pub fn io(g: *Gate) rows.file.IO {
        return .{ .ctx = g, .write = write, .sync = sync };
    }

    fn write(ctx: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        const g: *Gate = @ptrCast(@alignCast(ctx.?));
        g.entered.store(true, .release);
        while (!g.release.load(.acquire)) rows.write.io.nap();
        defer g.left.store(true, .release);
        g.calls += 1;
        if (g.mode == .full or (g.mode == .first_fails and g.calls == 1)) return error.NoSpaceLeft;
        if (g.mode == .short and g.calls > 1) return 0;
        const n = platform.c.write(fd, bytes.ptr, if (g.mode == .short) @min(bytes.len, 7) else bytes.len);
        if (n < 0) return error.RowsWrite;
        return @intCast(n);
    }

    fn sync(ctx: ?*anyopaque, fd: c_int) !void {
        const g: *Gate = @ptrCast(@alignCast(ctx.?));
        g.sync_entered.store(true, .release);
        while (!g.sync_release.load(.acquire)) rows.write.io.nap();
        if (g.mode == .sync_failure) return error.SyncFailed;
        if (platform.c.fsync(fd) != 0) return error.RowsWrite;
    }

    pub fn wait(flag: *std.atomic.Value(bool)) !void {
        const end = rows.write.nowNs() + 3 * std.time.ns_per_s;
        while (!flag.load(.acquire)) {
            if (rows.write.nowNs() >= end) return error.GateTimedOut;
            rows.write.io.nap();
        }
    }
};
