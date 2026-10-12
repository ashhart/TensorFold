//! tf-kv-bench DIR [tokens] [lanes] [staging MiB]: pool page ops, a 1M-token session parked/restored on NVMe.

const std = @import("std");
const sessions = @import("sessions");
const layout = @import("layout.zig");
const Io = std.Io;

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

fn secs(a: i96, b: i96) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e9;
}

/// Bounded state of a realistic size (rings, taps: ~11 MiB) held in host memory.
const Blob = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,
    const vt: sessions.store.Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };
    fn make(gpa: std.mem.Allocator) !sessions.store.Bounded {
        const b = try gpa.create(Blob);
        b.* = .{ .gpa = gpa, .bytes = try gpa.alloc(u8, 11 << 20) };
        for (b.bytes, 0..) |*x, i| x.* = @truncate(i *% 2654435761 >> 13);
        return .{ .ptr = b, .vtable = &vt };
    }
    fn nbytes(p: *anyopaque) u64 {
        return @as(*Blob, @ptrCast(@alignCast(p))).bytes.len;
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]sessions.disk.Blob {
        const b: *Blob = @ptrCast(@alignCast(p));
        const out = try gpa.alloc(sessions.disk.Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "bounded"), .bytes = try gpa.dupe(u8, b.bytes) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const b: *Blob = @ptrCast(@alignCast(p));
        b.gpa.free(b.bytes);
        b.gpa.destroy(b);
    }
};

fn pageOps(gpa: std.mem.Allocator, io: Io, out: *Io.Writer) !void {
    var l = try layout.Layout.fromConfig(layout.Release{}, .{ .split = true });
    for ([_]u64{ 2_240_000, 9_000_000 }) |tokens| {
        var pool = try sessions.Pool.init(gpa, l.pool(), tokens / 512 * 512, 2, 0);
        defer pool.deinit();
        const cap = pool.npages / 4 / 16 * 16 * 256;
        const slots = [_]*sessions.Slot{ try pool.newSlot(cap), try pool.newSlot(cap), try pool.newSlot(cap), try pool.newSlot(cap) };
        var ops: u64 = 0;
        const t0 = now(io);
        for (0..20) |_| {
            for (slots) |s| {
                var pos: u64 = 0;
                while (pos < cap) : (pos += 4096) { // a 16-page prefill segment at a time
                    try s.ensure(pos + 4096);
                    ops += 16;
                }
            }
            for (slots) |s| {
                pool.share(s.mapped()); // a session entry keeps the slot's pages
                ops += s.len;
                _ = try s.truncate(s.tokens() / 2); // a rollback
                ops += s.len;
                _ = try s.releaseAll();
            }
            for (slots) |s| {
                try pool.drop(s.pages[0 .. cap / 256]); // the entries go
                ops += cap / 256;
            }
        }
        const dt = secs(t0, now(io));
        try out.print("page ops, pool {d} pages ({d} tokens), split over 2: {d:.1} ns a page op ({d} ops, {d:.3} s); available() {d}\n", .{ pool.npages, pool.npages * 256, dt * 1e9 / @as(f64, @floatFromInt(ops)), ops, dt, pool.available() });
    }
}

fn prefixOps(gpa: std.mem.Allocator, io: Io, out: *Io.Writer) !void {
    const n = 1 << 20;
    const ids = try gpa.alloc(i32, n);
    defer gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast(i *% 7919 % 129280);
    const d = try gpa.alloc(sessions.prefix.Digest, n / 256);
    defer gpa.free(d);
    const t0 = now(io);
    sessions.prefix.chain(1, ids, 256, d);
    const t1 = now(io);
    var x = sessions.prefix.Index(u64).init(gpa, 256);
    defer x.deinit();
    for (0..64) |k| _ = try x.insert(1, ids[0 .. n - 1 - k * 9000], d, k % 2 == 0, k);
    const t2 = now(io);
    var hits: u64 = 0;
    for (0..1000) |_| hits += @intFromBool(x.find(1, ids, d, .any) != null);
    const t3 = now(io);
    try out.print("prefix: chain digests {d:.2} GB/s of ids ({d:.1} ms a 1M prompt); 64 entries of ~1M inserted in {d:.1} ms; find over them {d:.1} us ({d} hits)\n", .{ 4.0 * n / secs(t0, t1) / 1e9, secs(t0, t1) * 1e3, secs(t1, t2) * 1e3, secs(t2, t3) * 1e6 / 1000, hits });
}

fn session(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, dir: []const u8, tokens: u64, world: u32, rank: u32, lanes: u32, staging_mib: u32) !void {
    var l = try layout.Layout.fromConfig(layout.Release{}, .{ .split = world > 1 });
    const pool_tokens = (2 * tokens + 4096) / 512 * 512 + 512;
    var pool = try sessions.Pool.init(gpa, l.pool(), pool_tokens, world, rank);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    var compat: [16]u8 = @splat(0);
    compat[0] = @intCast(world);
    const d = try sessions.Disk.open(gpa, io, .{ .root = dir, .compat = compat, .rank = rank, .world = world, .lanes = lanes, .staging = staging_mib / 8, .min_tokens = 1 });
    defer d.close();
    var st = try sessions.Store.init(gpa, io, &pool, host.store(), d, .{});
    defer st.deinit();
    const a = try pool.newSlot(tokens + 16384);
    try a.ensure(tokens);
    for (a.mapped()) |pg| for (l.families(), 0..) |f, fi| {
        if (f.split and !pool.owns(pg)) continue;
        const v = host.pageOf(@intCast(fi), pool.familyPage(f, pg));
        for (std.mem.bytesAsSlice(u64, v), 0..) |*w, i| w.* = (@as(u64, pg) << 40) ^ (i *% 0x9E3779B97F4A7C15) ^ fi;
    };
    const ids = try gpa.alloc(i32, tokens);
    defer gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast(i % 129280);
    const id = (try st.save(a, try Blob.make(gpa), ids, 1, 0)).?;
    const src_pages = try gpa.dupe(u32, a.mapped());
    defer gpa.free(src_pages);
    _ = try a.releaseAll();
    const t0 = now(io);
    const j = (try st.beginPark(id)).?;
    _ = try st.settle(j);
    const t1 = now(io);
    const file = d.used;
    const hash0 = d.lanes.hash_ns.load(.monotonic);
    const io0 = d.lanes.io_ns.load(.monotonic);
    // restore onto other pages: the source pages are free again, so take a pad first
    const pad = try pool.newSlot(tokens + 16384);
    try pad.ensure(4096);
    const b = try pool.newSlot(tokens + 16384);
    const t2 = now(io);
    const rj = try st.beginRestore(id, b);
    var got = (try st.settle(rj)).?;
    got.deinit(gpa);
    const t3 = now(io);
    // the restored rows equal the parked ones, owned rows of every family
    var bad: u64 = 0;
    var cmp: u64 = 0;
    for (b.mapped(), 0..) |pg, k| for (l.families(), 0..) |f, fi| {
        if (f.split and !pool.owns(pg)) continue;
        const v = host.pageOf(@intCast(fi), pool.familyPage(f, pg));
        const sp = src_pages[k];
        for (std.mem.bytesAsSlice(u64, v), 0..) |w, i| bad += @intFromBool(w != ((@as(u64, sp) << 40) ^ (i *% 0x9E3779B97F4A7C15) ^ fi));
        cmp += v.len;
    };
    const gb = @as(f64, @floatFromInt(file)) / 1e9;
    try out.print("{d} tokens, world {d} rank {d}: file {d:.3} GB ({d:.0} B a token); park {d:.3} s ({d:.2} GB/s), restore {d:.3} s ({d:.2} GB/s); lanes {d}: hash {d:.2} s, I/O {d:.2} s summed; staging {d} MiB; restored bytes compared {d:.2} GB, mismatches {d}\n", .{ tokens, world, rank, gb, @as(f64, @floatFromInt(file)) / @as(f64, @floatFromInt(tokens)), secs(t0, t1), gb / secs(t0, t1), secs(t2, t3), gb / secs(t2, t3), lanes, @as(f64, @floatFromInt(d.lanes.hash_ns.load(.monotonic))) / 1e9, @as(f64, @floatFromInt(d.lanes.io_ns.load(.monotonic))) / 1e9, staging_mib, @as(f64, @floatFromInt(cmp)) / 1e9, bad });
    _ = hash0;
    _ = io0;
    // the next turn (+8K tokens) parked as a delta on the restored file
    try b.ensure(tokens + 8192);
    const ids2 = try gpa.alloc(i32, tokens + 8192);
    defer gpa.free(ids2);
    for (ids2, 0..) |*t, i| t.* = @intCast(i % 129280);
    const id2 = (try st.save(b, try Blob.make(gpa), ids2, 1, 0)).?;
    _ = try b.releaseAll();
    const before = d.used;
    const t4 = now(io);
    _ = try st.settle((try st.beginPark(id2)).?);
    const t5 = now(io);
    try out.print("  next turn +8192 tokens parked as a delta: {d:.1} MB written in {d:.3} s (a whole file would be {d:.3} GB)\n", .{ @as(f64, @floatFromInt(d.used - before)) / 1e6, secs(t4, t5), gb * @as(f64, @floatFromInt(tokens + 8192)) / @as(f64, @floatFromInt(tokens)) });
    if (bad != 0) return error.RestoreMismatch;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const tokens = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 1 << 20;
    const lanes = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 4;
    const staging = if (args.len > 4) try std.fmt.parseInt(u32, args[4], 10) else 128;
    var buf: [4096]u8 = undefined;
    var w = Io.File.stdout().writer(io, &buf);
    const out = &w.interface;
    try pageOps(gpa, io, out);
    try prefixOps(gpa, io, out);
    try out.flush();
    try session(gpa, io, out, dir, tokens, 1, 0, lanes, staging);
    try out.flush();
    try session(gpa, io, out, dir, tokens, 2, 0, lanes, staging);
    try out.flush();
}
