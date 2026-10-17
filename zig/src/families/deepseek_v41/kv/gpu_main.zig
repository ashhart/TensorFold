//! tf-kv-gpu-test DIR [tokens]: device tables, NVMe park/restore, RAM copy-on-write, views; exit 0: bytes equal.

const std = @import("std");
const cuda = @import("cuda");
const sessions = @import("sessions");
const layout = @import("layout.zig");
const DevicePool = @import("device.zig").DevicePool;
const Io = std.Io;

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

fn secs(a: i96, b: i96) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e9;
}

/// The bytes logical page k of family fam holds (8-byte words), whatever physical page it sits on.
fn word(k: u64, fam: u64, i: u64) u64 {
    return (k << 40) ^ (i *% 0x9E3779B97F4A7C15) ^ (fam << 32);
}

const Bounded = struct {
    gpa: std.mem.Allocator,
    const vt: sessions.store.Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };
    fn make(gpa: std.mem.Allocator) !sessions.store.Bounded {
        const b = try gpa.create(Bounded);
        b.* = .{ .gpa = gpa };
        return .{ .ptr = b, .vtable = &vt };
    }
    fn nbytes(_: *anyopaque) u64 {
        return 4096;
    }
    fn blobs(_: *anyopaque, gpa: std.mem.Allocator) anyerror![]sessions.disk.Blob {
        const out = try gpa.alloc(sessions.disk.Blob, 1);
        const bytes = try gpa.alloc(u8, 4096);
        @memset(bytes, 0x5a);
        out[0] = .{ .name = try gpa.dupe(u8, "rings"), .bytes = bytes };
        return out;
    }
    fn release(p: *anyopaque) void {
        const b: *Bounded = @ptrCast(@alignCast(p));
        b.gpa.destroy(b);
    }
};

/// Every owned page of a slot read back from the device and compared with word(); returns mismatching words.
fn compare(gpa: std.mem.Allocator, pool: *sessions.Pool, dp: *DevicePool, s: *const sessions.Slot, fams: []const sessions.Family) !u64 {
    var bad: u64 = 0;
    const st = dp.store();
    for (fams, 0..) |f, fi| {
        const pb = f.pageBytes(pool.page);
        const buf = try gpa.alloc(u8, pb);
        defer gpa.free(buf);
        for (s.mapped(), 0..) |pg, k| {
            if (f.split and !pool.owns(pg)) continue;
            try st.read(@intCast(fi), &.{pool.familyPage(f, pg)}, buf);
            for (std.mem.bytesAsSlice(u64, buf), 0..) |w, i| bad += @intFromBool(w != word(k, fi, i));
        }
    }
    return bad;
}

fn run(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, d: *const cuda.Driver, ctx: *const cuda.Context, dir: []const u8, tokens: u64, world: u32) !u64 {
    var l = try layout.Layout.fromConfig(layout.Release{}, .{ .split = world > 1 });
    const fams = l.families();
    const pool_tokens = (2 * tokens + 65536) / 512 * 512;
    var pool = try sessions.Pool.init(gpa, l.pool(), pool_tokens, world, 0);
    defer pool.deinit();
    var compute = try cuda.Stream.init(d, true);
    defer compute.deinit();
    const max_pages = sessions.pool.pagesFor(tokens + 16384, 256);
    var dp = try DevicePool.init(gpa, d, ctx, &pool, compute, 4, max_pages);
    defer dp.deinit();
    var compat: [16]u8 = @splat(0);
    compat[0] = @intCast(world);
    compat[1] = 0x6b;
    const disk = try sessions.Disk.open(gpa, io, .{ .root = dir, .compat = compat, .rank = 0, .world = world, .min_tokens = 1 });
    defer disk.close();
    var store = try sessions.Store.init(gpa, io, &pool, dp.store(), disk, .{ .async_io = true });
    defer store.deinit();
    var bad: u64 = 0;

    // a slot's rows, written page by page through the page store (what kv_store would write)
    const a = try pool.newSlot(tokens + 16384);
    try a.ensure(tokens);
    for (fams, 0..) |f, fi| {
        const pb = f.pageBytes(pool.page);
        const buf = try gpa.alloc(u8, pb);
        defer gpa.free(buf);
        for (a.mapped(), 0..) |pg, k| {
            if (f.split and !pool.owns(pg)) continue;
            for (std.mem.bytesAsSlice(u64, buf), 0..) |*w, i| w.* = word(k, fi, i);
            try dp.store().write(@intCast(fi), &.{pool.familyPage(f, pg)}, buf);
        }
    }
    // page tables: the dirty range uploaded, read back equal to the host book
    try dp.syncTables();
    try compute.synchronize();
    const tab = try gpa.alloc(i32, max_pages);
    defer gpa.free(tab);
    try dp.table.download(0, std.mem.sliceAsBytes(tab));
    for (tab, 0..) |v, i| bad += @intFromBool(v != @as(i32, @intCast(a.tableAt(@intCast(i)))));
    try dp.local.download(0, std.mem.sliceAsBytes(tab));
    for (tab, 0..) |v, i| bad += @intFromBool(v != @as(i32, @intCast(a.localTableAt(@intCast(i)))));
    const v20 = dp.view(l.compOf(20).?, 0);
    const v2 = dp.view(l.compOf(2).?, 1);
    if (v20.psh != 8 or v2.psh != 7 or v20.csc != v20.cv + 576 or v2.pt != (if (world > 1) dp.local.ptr else dp.table.ptr) + @as(u64, max_pages) * 4) bad += 1;

    // save, park on the tier thread behind the fence, restore onto other pages, compare
    const ids = try gpa.alloc(i32, tokens);
    defer gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast(i % 129280);
    const id = (try store.save(a, try Bounded.make(gpa), ids, 1, 0)).?;
    _ = try a.releaseAll();
    const t0 = now(io);
    _ = try store.settle((try store.beginPark(id)).?);
    const t1 = now(io);
    const pad = try pool.newSlot(65536);
    try pad.ensure(32768); // the restore lands on other pages
    const b = try pool.newSlot(tokens + 16384);
    const t2 = now(io);
    var got = (try store.settle(try store.beginRestore(id, b))).?;
    got.deinit(gpa);
    const t3 = now(io);
    bad += try compare(gpa, &pool, &dp, b, fams);
    // a RAM resume of a partial entry: full pages shared, the last copied on the compute stream
    const short = tokens / 2 + 100;
    const id2 = (try store.save(b, try Bounded.make(gpa), ids[0..short], 1, 0)).?;
    const c = try pool.newSlot(tokens + 16384);
    _ = try store.restoreRam(id2, c);
    try compute.synchronize();
    bad += try compare(gpa, &pool, &dp, c, fams);
    const gb = @as(f64, @floatFromInt(disk.used)) / 1e9;
    try out.print("world {d} rank 0, {d} tokens: file {d:.3} GB; park {d:.3} s ({d:.2} GB/s), restore {d:.3} s ({d:.2} GB/s) through device memory; mismatches {d}\n", .{ world, tokens, gb, secs(t0, t1), gb / secs(t0, t1), secs(t2, t3), gb / secs(t2, t3), bad });
    return bad;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-kv-gpu-test DIR [tokens]\n", .{});
        return 2;
    }
    const tokens = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 1 << 20;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name: [256]u8 = undefined;
    var buf: [4096]u8 = undefined;
    var w = Io.File.stdout().writer(io, &buf);
    const out = &w.interface;
    try out.print("device: {s}\n", .{try ctx.name(&name)});
    var bad: u64 = 0;
    bad += try run(gpa, io, out, &driver, &ctx, args[1], tokens, 1);
    try out.flush();
    bad += try run(gpa, io, out, &driver, &ctx, args[1], tokens, 2);
    try out.print("{s}\n", .{if (bad == 0) "PASS" else "FAIL"});
    try out.flush();
    return if (bad == 0) 0 else 1;
}
