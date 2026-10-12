//! `tf-dsv41 engram-probe DIR RANK WORLD`: no-GPU, read-only probe of the Engram read path on a node's shards.

const std = @import("std");
const linux = std.os.linux;
const er = @import("engram_rows.zig");

const layers = [_]u32{ 1, 14 };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Shard = struct { fd: linux.fd_t, lo: u64, hi: u64, size: u64, direct: bool };

fn openShard(dir: []const u8, layer: u32, rank: u32, world: u32) !Shard {
    var nb: [4096]u8 = undefined;
    const q = try std.fmt.bufPrint(nb[0 .. nb.len - 1], "{s}/engram-l{d}-r{d}of{d}.bin", .{ dir, layer, rank, world });
    nb[q.len] = 0;
    const p = nb[0..q.len :0];
    var direct = true;
    var rc = linux.openat(linux.AT.FDCWD, p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true }, 0);
    if (linux.errno(rc) != .SUCCESS) {
        direct = false;
        rc = linux.openat(linux.AT.FDCWD, p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
        if (linux.errno(rc) != .SUCCESS) return error.EngramShard;
    }
    const fd: linux.fd_t = @intCast(rc);
    var hb: [4096]u8 align(4096) = undefined;
    if (linux.pread(fd, &hb, hb.len, 0) != hb.len) return error.EngramShard;
    var st: linux.Statx = undefined;
    if (linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &st)) != .SUCCESS) return error.EngramShard;
    const h = try er.Header.parse(hb[0..48], st.size);
    return .{ .fd = fd, .lo = h.lo, .hi = h.hi, .size = st.size, .direct = direct };
}

const RawArgs = struct { s: Shard, n: usize, seed: u64, lat: []u64 };

fn rawWorker(a: RawArgs) void {
    var prng = std.Random.DefaultPrng.init(a.seed);
    const rnd = prng.random();
    var buf: [2048]u8 align(4096) = undefined;
    for (a.lat[0..a.n]) |*l| {
        const row = rnd.intRangeLessThan(u64, a.s.lo, a.s.hi);
        const b0 = er.header_bytes + (row - a.s.lo) * er.row_bytes;
        const off = b0 / er.sector * er.sector;
        const len = std.mem.alignForward(u64, b0 + er.row_bytes - off, er.sector);
        const t0 = nowNs();
        _ = linux.pread(a.s.fd, &buf, len, @intCast(off));
        l.* = nowNs() - t0;
    }
}

fn pct(xs: []u64, q: f64) f64 {
    const i: usize = @min(xs.len - 1, @as(usize, @intFromFloat(q * @as(f64, @floatFromInt(xs.len)))));
    return @as(f64, @floatFromInt(xs[i])) / 1e3;
}

fn raw(gpa: std.mem.Allocator, s: Shard, depth: usize, w: *std.Io.Writer) !void {
    const per: usize = @max(200, 6000 / depth);
    const lat = try gpa.alloc(u64, per * depth);
    defer gpa.free(lat);
    const ts = try gpa.alloc(std.Thread, depth);
    defer gpa.free(ts);
    const t0 = nowNs();
    for (ts, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, rawWorker, .{RawArgs{ .s = s, .n = per, .seed = 0x5eed + i * 7919 + depth, .lat = lat[i * per ..][0..per] }});
    for (ts) |t| t.join();
    const wall = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    std.mem.sort(u64, lat, {}, std.sort.asc(u64));
    try w.print("device depth {d:2}: {d:8.0} reads/s, latency p50 {d:7.1} us p99 {d:7.1} us max {d:8.1} us ({d} reads)\n", .{ depth, @as(f64, @floatFromInt(lat.len)) / wall, pct(lat, 0.5), pct(lat, 0.99), @as(f64, @floatFromInt(lat[lat.len - 1])) / 1e3, lat.len });
}

const Spin = struct { stop: *std.atomic.Value(bool) };

fn spinner(s: Spin) void {
    var x: u64 = 0;
    while (!s.stop.load(.monotonic)) x +%= 1;
    std.mem.doNotOptimizeAway(x);
}

var seed_calls: u64 = 0;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn pool(gpa: std.mem.Allocator, rows: *er.Rows, shards: []const Shard, batch: usize, trials: usize, mode: []const u8, label: []const u8, w: *std.Io.Writer) !void {
    // fresh rows each measurement: a batch size's second run must not re-read its first's (cached) rows
    seed_calls += 1;
    var prng = std.Random.DefaultPrng.init(0xbeef + batch + seed_calls * 1_000_003);
    const rnd = prng.random();
    const idx = try gpa.alloc(i64, batch * shards.len);
    defer gpa.free(idx);
    const out = try gpa.alloc(u16, batch * 256);
    defer gpa.free(out);
    const times = try gpa.alloc(u64, trials);
    defer gpa.free(times);
    const issued0 = rows.stats.issued;
    for (times) |*t| {
        for (shards, 0..) |s, li| for (idx[li * batch ..][0..batch]) |*x| {
            x.* = @intCast(rnd.intRangeLessThan(u64, s.lo, s.hi));
        };
        const t0 = nowNs();
        for (shards, 0..) |_, li| try rows.issue(li, idx[li * batch ..][0..batch]);
        for (shards, 0..) |_, li| try rows.rows(li, idx[li * batch ..][0..batch], 256, out);
        t.* = nowNs() - t0;
    }
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    // cold reads: rows `issue` had to start; fewer than asked = rows the cache already held
    const asked = trials * batch * shards.len;
    const issued = rows.stats.issued - issued0;
    try w.print("{s}{s} {d:3} rows a layer x {d} layers: p50 {d:7.2} ms p90 {d:7.2} ms max {d:7.2} ms ({d} trials; {d} of {d} rows read cold{s})\n", .{ mode, label, batch, shards.len, pct(times, 0.5) / 1e3, pct(times, 0.9) / 1e3, @as(f64, @floatFromInt(times[times.len - 1])) / 1e6, trials, issued, asked, if (issued < asked) ": NOT all cold" else "" });
}

pub fn main(gpa: std.mem.Allocator, dir: []const u8, rank: u32, world: u32, w: *std.Io.Writer) !void {
    var shards: [layers.len]Shard = undefined;
    for (layers, &shards) |l, *s| {
        s.* = try openShard(dir, l, rank, world);
        try w.print("shard layer {d}: rows {d} .. {d} ({d} rows, {d:.2} GiB), {s}\n", .{ l, s.lo, s.hi, s.hi - s.lo, @as(f64, @floatFromInt(s.size)) / (1 << 30), if (s.direct) "O_DIRECT" else "buffered (O_DIRECT refused)" });
    }
    try w.flush();
    for ([_]usize{ 1, 8, 32, 64 }) |d| {
        try raw(gpa, shards[0], d, w);
        try w.flush();
    }
    // the serving read paths at the gate's cache size, every batch a cold read: reader threads, native AIO, io_uring
    const n = @max(1, (std.Thread.getCpuCount() catch 4) -| 2);
    for ([_][]const u8{ "threads", "aio", "aio-poll", "io_uring" }) |mode| {
        const aio = std.mem.startsWith(u8, mode, "aio");
        _ = setenv("TF_DSV41_ENGRAM_URING", if (std.mem.eql(u8, mode, "io_uring")) "1" else "0", 1);
        _ = setenv("TF_DSV41_ENGRAM_AIO", if (aio) "1" else "0", 1);
        _ = setenv("TF_DSV41_ENGRAM_POLL", if (std.mem.eql(u8, mode, "aio-poll")) "1" else "0", 1);
        const rows = try er.Rows.open(gpa, dir, rank, world, &layers, 65536);
        defer rows.close();
        const got: []const u8 = if (rows.ring != null) "io_uring" else if (rows.aio) |x| (if (x.poll) "aio-poll" else "aio") else if (rows.pool) |_| "threads" else "serial";
        if (!std.mem.eql(u8, got, mode)) {
            try w.print("{s}: not available here (the reader runs {s}): skipped\n", .{ mode, got });
            continue;
        }
        if (rows.pool) |p| try w.print("{s}: {d} reader threads (TF_DSV41_ENGRAM_QD)\n", .{ mode, p.threads.len });
        for ([_]usize{ 1, 32, 128, 256 }) |b| {
            try pool(gpa, rows, &shards, b, if (b <= 32) 200 else 60, mode, "", w);
            try w.flush();
        }
        var stop = std.atomic.Value(bool).init(false);
        const spins = try gpa.alloc(std.Thread, n);
        defer gpa.free(spins);
        for (spins) |*t| t.* = try std.Thread.spawn(.{}, spinner, .{Spin{ .stop = &stop }});
        for ([_]usize{ 128, 256 }) |b| {
            try pool(gpa, rows, &shards, b, 60, mode, " (spinning cores)", w);
            try w.flush();
        }
        stop.store(true, .monotonic);
        for (spins) |t| t.join();
    }
    try w.print("spinning cores: {d}\n", .{n});
    for (shards) |s| _ = linux.close(s.fd);
    try w.flush();
}
