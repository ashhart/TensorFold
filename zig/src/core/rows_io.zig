//! One disk worker owns staged images; it never sees a cache entry, family, device buffer or store.
const std = @import("std");
const platform = @import("sys");
const file = @import("rows_file.zig");
const chain = @import("rows_chain.zig");
const Span = @import("cache_modes.zig").Span;
const Want = @import("rows_write_sync.zig").Want;
pub const allocator = std.heap.page_allocator;
pub const max_jobs = 8;

pub const File = struct {
    trailer: file.Trailer,
    payload: []u8,
    meta: []const []const u8 = &.{},
    writer: ?file.Writer = null,
    bytes: u64 = 0,
};

pub const Job = struct {
    arena: std.heap.ArenaAllocator,
    id: u64,
    generation: u64,
    name: chain.Name,
    at: u32,
    shared: bool,
    tokens: []u32,
    spans: []Span,
    starts: []u32,
    wants: []Want,
    files: []File,
    bytes: u64,
    staged: u64,
    verify: bool = false,
    began: i128 = 0,
    done: std.atomic.Value(bool) = .init(false),
    taken: bool = false,
    failure: ?anyerror = null,

    pub fn destroy(j: *Job) void {
        for (j.files) |*f| if (f.writer) |*w| w.abort();
        j.arena.deinit();
        allocator.destroy(j);
    }
};

pub const Worker = struct {
    dir: [:0]u8,
    io: file.IO,
    limit: u64,
    used: u64 = 0,
    slots: [max_jobs]?*Job = @splat(null),
    mutex: std.atomic.Mutex = .unlocked,
    generation: std.atomic.Value(u64) = .init(1),
    stopped: std.atomic.Value(bool) = .init(false),
    exited: std.atomic.Value(bool) = .init(false),
    refs: std.atomic.Value(u32) = .init(2),
    thread: std.Thread,

    pub fn open(dir: []const u8, limit: u64, io: file.IO) !*Worker {
        const w = try allocator.create(Worker);
        errdefer allocator.destroy(w);
        const owned = try allocator.dupeSentinel(u8, dir, 0);
        errdefer allocator.free(owned);
        w.* = .{ .dir = owned, .io = io, .limit = limit, .thread = undefined };
        w.thread = try std.Thread.spawn(.{}, run, .{w});
        return w;
    }

    fn lock(w: *Worker) void {
        while (!w.mutex.tryLock()) std.Thread.yield() catch {};
    }

    pub fn fits(w: *Worker, amount: u64) bool {
        w.lock();
        defer w.mutex.unlock();
        if (w.stopped.load(.acquire) or amount > w.limit -| w.used) return false;
        for (w.slots) |s| if (s == null) return true;
        return false;
    }

    pub fn submit(w: *Worker, j: *Job) void {
        w.lock();
        defer w.mutex.unlock();
        std.debug.assert(j.staged <= w.limit -| w.used);
        for (&w.slots) |*s| if (s.* == null) {
            s.* = j;
            w.used += j.staged;
            return;
        };
        unreachable;
    }

    /// Completed jobs stay owned here until the engine takes them, retaining their permits meanwhile.
    pub fn take(w: *Worker) ?*Job {
        w.lock();
        defer w.mutex.unlock();
        for (&w.slots) |*s| if (s.*) |j| if (j.done.load(.acquire)) {
            s.* = null;
            w.used -= j.staged;
            return j;
        };
        return null;
    }

    pub fn names(w: *Worker, name: chain.Name) bool {
        w.lock();
        defer w.mutex.unlock();
        for (w.slots) |s| if (s) |j| if (std.mem.eql(u8, &j.name, &name)) return true;
        return false;
    }

    pub fn wanted(w: *Worker, stem: chain.Stem) bool {
        w.lock();
        defer w.mutex.unlock();
        for (w.slots) |s| if (s) |j| for (j.wants) |want| {
            if (std.mem.eql(u8, want.name[0..16], &stem)) return true;
        };
        return false;
    }

    pub fn ready(w: *Worker) bool {
        w.lock();
        defer w.mutex.unlock();
        for (w.slots) |s| if (s) |j| if (j.done.load(.acquire)) return true;
        return false;
    }

    pub fn cancellations(w: *Worker, out: *[max_jobs]?*Job) void {
        w.lock();
        defer w.mutex.unlock();
        out.* = w.slots;
    }

    pub fn pending(w: *Worker) usize {
        w.lock();
        defer w.mutex.unlock();
        var n: usize = 0;
        for (w.slots) |s| n += @intFromBool(s != null);
        return n;
    }

    pub fn bytes(w: *Worker) u64 {
        w.lock();
        defer w.mutex.unlock();
        return w.used;
    }

    pub fn reserved(w: *Worker) u64 {
        w.lock();
        defer w.mutex.unlock();
        var amount: u64 = 0;
        for (w.slots) |s| if (s) |j| {
            amount += j.bytes;
        };
        return amount;
    }

    /// No old job can publish after this returns; publication belongs only to the engine.
    pub fn fence(w: *Worker) void {
        _ = w.generation.fetchAdd(1, .acq_rel);
    }

    fn valid(w: *Worker, j: *const Job) bool {
        return !w.stopped.load(.acquire) and j.generation == w.generation.load(.acquire);
    }

    fn next(w: *Worker) ?*Job {
        w.lock();
        defer w.mutex.unlock();
        for (w.slots) |s| if (s) |j| if (!j.taken) {
            j.taken = true;
            return j;
        };
        return null;
    }

    fn run(w: *Worker) void {
        while (!w.stopped.load(.acquire)) {
            const j = w.next() orelse {
                nap();
                continue;
            };
            w.write(j) catch |err| {
                j.failure = err;
            };
            j.done.store(true, .release);
        }
        for (w.slots) |s| if (s) |j| j.destroy();
        w.exited.store(true, .release);
        w.release();
    }

    fn write(w: *Worker, j: *Job) !void {
        for (j.files) |*f| {
            if (!w.valid(j)) return error.Cancelled;
            const ext: []const u8 = if (f.trailer.kind == @backingInt(file.Kind.state)) ".st" else ".blk";
            f.writer = try file.Writer.openUnique(w.dir, &chain.hex(f.trailer.name), ext, j.id);
            const out = &f.writer.?;
            out.io = w.io;
            var off: usize = 0;
            while (off < f.payload.len) {
                if (!w.valid(j)) return error.Cancelled;
                const end = @min(f.payload.len, off + (4 << 20));
                try out.append(f.payload[off..end]);
                off = end;
            }
            if (!w.valid(j)) return error.Cancelled;
            f.bytes = try out.seal(f.trailer, f.meta, true);
            if (j.verify) try verifyFile(f);
        }
        if (!w.valid(j)) return error.Cancelled;
    }

    /// A blocked syscall may outlive the deadline; only private allocations and temporary files outlive the owner.
    pub fn close(w: *Worker, deadline: i128) void {
        w.fence();
        w.stopped.store(true, .release);
        while (!w.exited.load(.acquire) and nowNs() < deadline) nap();
        if (w.exited.load(.acquire)) w.thread.join() else w.thread.detach();
        w.release();
    }

    fn release(w: *Worker) void {
        if (w.refs.fetchSub(1, .acq_rel) != 1) return;
        allocator.free(w.dir);
        allocator.destroy(w);
    }
};

pub fn nap() void {
    var t: platform.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
    _ = platform.c.nanosleep(&t, null);
}

pub fn nowNs() i128 {
    var t: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &t);
    return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
}

fn verifyFile(f: *const File) !void {
    const w = &f.writer.?;
    const path = w.tmp[0..w.tmp_len :0];
    var o = file.open(path, if (f.trailer.kind == @backingInt(file.Kind.state)) .state else .block, f.trailer.name) catch return error.VerifyFailed;
    defer o.close();
    var expected = f.trailer;
    expected.payload = f.payload.len;
    expected.payload_hash = std.hash.Wyhash.hash(0, f.payload);
    var hash = std.hash.Wyhash.init(1);
    for (f.meta) |m| {
        expected.meta += m.len;
        hash.update(m);
    }
    expected.meta_hash = hash.final();
    if (!std.mem.eql(u8, std.mem.asBytes(&expected), std.mem.asBytes(&o.trailer))) return error.VerifyFailed;
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.VerifyFailed;
    defer _ = platform.c.close(fd);
    var offset: u64 = 0;
    try compare(fd, f.payload, &offset);
    for (f.meta) |m| try compare(fd, m, &offset);
}

fn compare(fd: c_int, bytes: []const u8, offset: *u64) !void {
    var buf: [64 << 10]u8 = undefined;
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.pread(fd, &buf, @min(buf.len, bytes.len - done), @intCast(offset.*));
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.VerifyFailed;
        const got: usize = @intCast(n);
        if (!std.mem.eql(u8, bytes[done..][0..got], buf[0..got])) return error.VerifyFailed;
        done += got;
        offset.* += got;
    }
}
