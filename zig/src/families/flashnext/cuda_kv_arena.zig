//! The paged pool's arena as one reserved range of 256 MiB physical pieces, each mapped when a page in it is first used
const std = @import("std");
const cuda = @import("cuda");

/// Bytes of one physical piece.
pub const piece_bytes: usize = 256 << 20;

/// Maps pieces at chosen places through a backend whose next chunk lands where `seek` put it (cuda.arena.Vmm).
pub const Mapper = struct {
    be: cuda.arena.Backend,
    seek: *const fn (ctx: *anyopaque, at: u64) void,

    fn vmm(v: *cuda.arena.Vmm) Mapper {
        return .{ .be = v.backend(), .seek = seekVmm };
    }
    fn seekVmm(ctx: *anyopaque, at: u64) void {
        const v: *cuda.arena.Vmm = @ptrCast(@alignCast(ctx));
        v.next = at;
    }
};

/// A reserved range of `len` bytes whose pieces of `piece` bytes are mapped on first use; `mapped` one flag a piece.
pub const Lazy = struct {
    base: u64,
    len: usize,
    piece: usize,
    mapped: []bool,

    /// The pieces over [off, off + n) not mapped yet, mapped now; `zero` clears each new one. LaneMemory: no memory.
    pub fn ensure(z: *Lazy, m: Mapper, off: u64, n: u64, zero: anytype) !void {
        if (n == 0) return;
        if (off + n > z.len) return error.LaneMemory;
        var k: usize = @intCast(off / z.piece);
        while (k * z.piece < off + n) : (k += 1) {
            if (z.mapped[k]) continue;
            try z.mapPiece(m, k, zero);
            z.mapped[k] = true;
        }
    }

    /// Piece `k` mapped at its place and zeroed, not marked; whatever fails, it goes back.
    fn mapPiece(z: *const Lazy, m: Mapper, k: usize, zero: anytype) !void {
        const at = z.base + k * z.piece;
        const len = @min(z.piece, z.len - k * z.piece);
        m.seek(m.be.ctx, at);
        const got = m.be.chunk(m.be.ctx, len) orelse return error.LaneMemory;
        if (got != at) { // not at its place: the range would not hold the page
            m.be.release(m.be.ctx, got, len);
            return error.LaneMemory;
        }
        zero.piece(at, len) catch |err| { // not zeroed: it goes back, unmarked, as a piece that would not map
            m.be.release(m.be.ctx, at, len);
            return err;
        };
    }

    /// Every mapped piece goes back.
    pub fn unmapAll(z: *Lazy, m: Mapper) void {
        for (z.mapped, 0..) |*f, k| if (f.*) {
            m.be.release(m.be.ctx, z.base + k * z.piece, @min(z.piece, z.len - k * z.piece));
            f.* = false;
        };
    }
};

/// One thread mapping the pieces `post` names off the caller's path; a piece it fails to map goes back unmarked.
pub const Ahead = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    z: *Lazy,
    m: Mapper,
    hooks: Hooks,
    queued: []bool,
    busy: ?usize = null, // the piece the thread maps now
    stop: bool = false,
    mu: std.Io.Mutex = .init, // the flags, the queue, `busy` and `stop`
    changed: std.Io.Condition = .init,
    maps: std.Io.Mutex = .init, // one map at a time: the mapper's next address is shared
    thread: ?std.Thread = null,

    /// The thread's side: `enter` once on it (false: it maps nothing), `zero` clears a new piece and waits, `leave`.
    pub const Hooks = struct {
        ctx: *anyopaque,
        enter: *const fn (*anyopaque) bool,
        zero: *const fn (*anyopaque, u64, usize) anyerror!void,
        leave: *const fn (*anyopaque) void,

        fn piece(h: Hooks, at: u64, len: usize) !void {
            return h.zero(h.ctx, at, len);
        }
    };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, z: *Lazy, m: Mapper, hooks: Hooks) !Ahead {
        const queued = try gpa.alloc(bool, z.mapped.len);
        @memset(queued, false);
        return .{ .gpa = gpa, .io = io, .z = z, .m = m, .hooks = hooks, .queued = queued };
    }

    /// The thread stopped and joined, what it had queued left unmapped; before the pieces go back.
    pub fn deinit(a: *Ahead) void {
        if (a.thread) |t| {
            a.mu.lockUncancelable(a.io);
            a.stop = true;
            a.changed.broadcast(a.io);
            a.mu.unlock(a.io);
            t.join();
        }
        a.gpa.free(a.queued);
        a.* = undefined;
    }

    /// The pieces over [off, off + n) not mapped nor on their way, queued for the thread (started on the first post).
    pub fn post(a: *Ahead, off: u64, n: u64) void {
        if (n == 0 or off + n > a.z.len) return;
        a.mu.lockUncancelable(a.io);
        defer a.mu.unlock(a.io);
        var any = false;
        var k: usize = @intCast(off / a.z.piece);
        while (k * a.z.piece < off + n) : (k += 1) {
            if (a.z.mapped[k] or a.queued[k] or a.busy == k) continue;
            a.queued[k] = true;
            any = true;
        }
        if (!any) return;
        if (a.thread == null) a.thread = std.Thread.spawn(.{}, run, .{a}) catch {
            @memset(a.queued, false);
            return;
        };
        a.changed.broadcast(a.io);
    }

    /// As Lazy.ensure on the caller's thread: a piece the thread has in hand waited for, one still queued taken back.
    pub fn ensure(a: *Ahead, off: u64, n: u64, zero: anytype) !void {
        if (n == 0) return;
        if (off + n > a.z.len) return error.LaneMemory;
        a.mu.lockUncancelable(a.io);
        defer a.mu.unlock(a.io);
        var k: usize = @intCast(off / a.z.piece);
        while (k * a.z.piece < off + n) : (k += 1) {
            while (a.busy == k) a.changed.waitUncancelable(a.io, &a.mu);
            if (a.z.mapped[k]) continue;
            a.queued[k] = false;
            a.maps.lockUncancelable(a.io);
            defer a.maps.unlock(a.io);
            try a.z.mapPiece(a.m, k, zero);
            a.z.mapped[k] = true;
        }
    }

    /// Waits until the thread has nothing queued or in hand.
    pub fn settle(a: *Ahead) void {
        a.mu.lockUncancelable(a.io);
        defer a.mu.unlock(a.io);
        while (a.busy != null or std.mem.indexOfScalar(bool, a.queued, true) != null) a.changed.waitUncancelable(a.io, &a.mu);
    }

    fn run(a: *Ahead) void {
        const bound = a.hooks.enter(a.hooks.ctx);
        defer if (bound) a.hooks.leave(a.hooks.ctx);
        a.mu.lockUncancelable(a.io);
        defer a.mu.unlock(a.io);
        while (!a.stop) {
            const k = std.mem.indexOfScalar(bool, a.queued, true) orelse {
                a.changed.waitUncancelable(a.io, &a.mu);
                continue;
            };
            a.queued[k] = false;
            if (!bound) { // no context on this thread: the caller maps it when it needs it
                a.changed.broadcast(a.io);
                continue;
            }
            a.busy = k;
            a.mu.unlock(a.io);
            const ok = blk: {
                a.maps.lockUncancelable(a.io);
                defer a.maps.unlock(a.io);
                a.z.mapPiece(a.m, k, a.hooks) catch break :blk false;
                break :blk true;
            };
            a.mu.lockUncancelable(a.io);
            a.busy = null;
            if (ok) a.z.mapped[k] = true;
            a.changed.broadcast(a.io);
        }
    }
};

/// The pool's arena: `buf` a DeviceBuffer counted and freed as any other, a range mapped as used if `pieces` holds it.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    pieces: ?*Pieces = null,

    const Pieces = struct {
        gpa: std.mem.Allocator,
        vmm: cuda.arena.Vmm,
        lazy: ?Lazy = null,
        io: ?std.Io = null, // null: no thread, every piece maps on the caller's
        d: ?*const cuda.Driver = null,
        context: cuda.abi.Context = null, // the caller's, current on the thread too
        stream: cuda.abi.Stream = null, // the thread's own
        ahead: ?Ahead = null,

        fn enter(ctx: *anyopaque) bool {
            const s: *Pieces = @ptrCast(@alignCast(ctx));
            const d = s.d.?;
            if (d.api.cuCtxSetCurrent(s.context) != cuda.abi.success) return false;
            return d.api.cuStreamCreate(&s.stream, 1) == cuda.abi.success; // CU_STREAM_NON_BLOCKING
        }
        fn zero(ctx: *anyopaque, at: u64, len: usize) anyerror!void {
            const s: *Pieces = @ptrCast(@alignCast(ctx));
            const d = s.d.?;
            try d.check(d.api.cuMemsetD8Async(at, 0, len, s.stream), "cuMemsetD8Async");
            try d.check(d.api.cuStreamSynchronize(s.stream), "cuStreamSynchronize"); // zeroed before it is marked
        }
        fn leave(ctx: *anyopaque) void {
            const s: *Pieces = @ptrCast(@alignCast(ctx));
            _ = s.d.?.api.cuStreamDestroy_v2(s.stream);
        }

        fn hookAlloc(ctx: *anyopaque, len: usize) ?u64 {
            const s: *Pieces = @ptrCast(@alignCast(ctx));
            if (len == 0) return null;
            const piece = std.mem.alignForward(usize, piece_bytes, s.vmm.granularity);
            const n = std.mem.alignForward(usize, len, s.vmm.granularity);
            const mapped = s.gpa.alloc(bool, (n + piece - 1) / piece) catch return null;
            @memset(mapped, false);
            s.lazy = .{ .base = s.vmm.va, .len = n, .piece = piece, .mapped = mapped };
            return s.vmm.va;
        }

        fn hook(s: *Pieces) cuda.memory.Hook {
            return .{ .ctx = s, .alloc = hookAlloc, .free = hookFree };
        }

        fn hookFree(ctx: *anyopaque, p: u64) bool {
            const s: *Pieces = @ptrCast(@alignCast(ctx));
            const z = if (s.lazy) |*z| z else return false;
            if (p != z.base) return false;
            z.unmapAll(Mapper.vmm(&s.vmm));
            s.gpa.free(z.mapped);
            s.lazy = null;
            return true;
        }
    };

    /// `len` bytes reserved as pieces on `device` (`ensure` maps them; with `io` a thread too); else one cuMemAlloc.
    pub fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, len: usize, device: ?cuda.abi.Device, io: ?std.Io) !Arena {
        const dev = device orelse return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
        if (len <= piece_bytes) return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
        const s: ?*Pieces = blk: {
            const vmm = cuda.arena.Vmm.open(d, dev, len) catch break :blk null;
            const s = gpa.create(Pieces) catch {
                var v = vmm;
                v.close();
                break :blk null;
            };
            s.* = .{ .gpa = gpa, .vmm = vmm, .io = io, .d = d };
            break :blk s;
        };
        const p = s orelse return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
        // the range through its own hook, the installed one untouched: the bytes are counted as every device buffer's
        const buf = cuda.DeviceBuffer.allocVia(d, len, p.hook());
        var a: Arena = .{ .buf = buf catch |err| {
            destroy(p);
            return err;
        }, .pieces = p };
        if (a.buf.owner == null) { // the hook declined, cuMemAlloc served it: no range to keep
            destroy(p);
            a.pieces = null;
        }
        return a;
    }

    /// Bytes [off, off + n) of the arena backed by memory, new pieces zeroed on `stream`; one cuMemAlloc: always.
    pub fn ensure(a: *Arena, off: u64, n: u64, stream: cuda.abi.Stream) !void {
        const s = a.pieces orelse return;
        const Zero = struct {
            d: *const cuda.Driver,
            stream: cuda.abi.Stream,
            fn piece(z: @This(), at: u64, len: usize) !void {
                try z.d.check(z.d.api.cuMemsetD8Async(at, 0, len, z.stream), "cuMemsetD8Async");
            }
        };
        const zero: Zero = .{ .d = a.buf.d, .stream = stream };
        if (s.ahead) |*x| return x.ensure(off, n, zero);
        try s.lazy.?.ensure(Mapper.vmm(&s.vmm), off, n, zero);
    }

    /// Bytes [off, off + n) mapped ahead on the arena's thread, the caller's context current on it; no io: nothing.
    pub fn post(a: *Arena, off: u64, n: u64) void {
        const s = a.pieces orelse return;
        const io = s.io orelse return;
        if (s.ahead == null) {
            if (s.d.?.api.cuCtxGetCurrent(&s.context) != cuda.abi.success or s.context == null) return;
            s.ahead = Ahead.init(s.gpa, io, &s.lazy.?, Mapper.vmm(&s.vmm), .{ .ctx = s, .enter = Pieces.enter, .zero = Pieces.zero, .leave = Pieces.leave }) catch return;
        }
        s.ahead.?.post(off, n);
    }

    /// Whether the arena maps its pieces on first use (else every byte is there from the start).
    pub fn lazy(a: *const Arena) bool {
        return a.pieces != null;
    }

    /// The thread stopped, the buffer (its pieces unmapped), then the virtual range.
    pub fn free(a: *Arena) void {
        if (a.pieces) |p| if (p.ahead) |*x| {
            x.deinit();
            p.ahead = null;
        };
        a.buf.free();
        if (a.pieces) |p| destroy(p);
        a.* = undefined;
    }

    fn destroy(p: *Pieces) void {
        p.vmm.close();
        p.gpa.destroy(p);
    }
};

const Fake = struct {
    next: u64 = 0,
    chunks: u32 = 0,
    released: u64 = 0,
    fail_at: ?u32 = null, // the chunk call that returns null
    gap_at: ?u32 = null, // the chunk call that lands one granule past its place
    zeroed: u64 = 0,
    fail_zero: bool = false,
    slow: bool = false, // a chunk call takes 50 ms
    entered: std.atomic.Value(u32) = .init(0), // chunk calls begun
    mapped_by: ?std.Thread.Id = null, // the thread of the last chunk call

    fn mapper(f: *Fake) Mapper {
        return .{ .be = .{ .ctx = f, .granularity = 2 << 20, .chunk = chunk, .release = release, .name = "fake" }, .seek = seek };
    }
    fn seek(ctx: *anyopaque, at: u64) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.next = at;
    }
    fn chunk(ctx: *anyopaque, _: usize) ?u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        _ = f.entered.fetchAdd(1, .release);
        f.mapped_by = std.Thread.getCurrentId();
        if (f.slow) std.testing.io.sleep(.fromMilliseconds(50), .awake) catch {};
        defer f.chunks += 1;
        if (f.fail_at == f.chunks) return null;
        if (f.gap_at == f.chunks) f.next += 2 << 20;
        return f.next;
    }
    fn release(ctx: *anyopaque, _: u64, len: usize) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += len;
    }
    fn piece(f: *Fake, _: u64, len: usize) !void {
        if (f.fail_zero) return error.CudaFailed;
        f.zeroed += len;
    }
    fn hooks(f: *Fake) Ahead.Hooks {
        const H = struct {
            fn enter(_: *anyopaque) bool {
                return true;
            }
            fn zero(ctx: *anyopaque, at: u64, len: usize) anyerror!void {
                const g: *Fake = @ptrCast(@alignCast(ctx));
                return g.piece(at, len);
            }
            fn leave(_: *anyopaque) void {}
        };
        return .{ .ctx = f, .enter = H.enter, .zero = H.zero, .leave = H.leave };
    }
};

test "a posted piece maps on the arena's thread, and ensure waits for the one it has in hand, never mapping it twice" {
    var f: Fake = .{ .slow = true };
    var flags: [8]bool = @splat(false);
    var z: Lazy = .{ .base = 1 << 40, .len = 2 << 30, .piece = piece_bytes, .mapped = &flags };
    var a = try Ahead.init(std.testing.allocator, std.testing.io, &z, f.mapper(), f.hooks());
    defer a.deinit();
    a.post(0, 1);
    while (f.entered.load(.acquire) == 0) std.Thread.yield() catch {}; // the thread is mapping piece 0
    try a.ensure(0, 1, &f); // needs it now: waits
    try std.testing.expect(flags[0] and !flags[1]);
    try std.testing.expectEqual(@as(u32, 1), f.chunks);
    try std.testing.expectEqual(@as(u64, piece_bytes), f.zeroed);
    try std.testing.expect(f.mapped_by.? != std.Thread.getCurrentId());
    a.post(0, piece_bytes + 1); // piece 0 mapped, piece 1 queued
    a.settle();
    try std.testing.expect(flags[1] and f.chunks == 2 and f.mapped_by.? != std.Thread.getCurrentId());
}

test "a piece the arena's thread fails to map goes back unmarked, and ensure maps it on the caller's thread" {
    var f: Fake = .{ .fail_at = 0 };
    var flags: [8]bool = @splat(false);
    var z: Lazy = .{ .base = 1 << 40, .len = 2 << 30, .piece = piece_bytes, .mapped = &flags };
    var a = try Ahead.init(std.testing.allocator, std.testing.io, &z, f.mapper(), f.hooks());
    defer a.deinit();
    a.post(0, 1);
    a.settle();
    try std.testing.expect(!flags[0] and f.chunks == 1 and f.zeroed == 0);
    try a.ensure(0, 1, &f);
    try std.testing.expect(flags[0] and f.chunks == 2);
    try std.testing.expectEqual(@as(u64, piece_bytes), f.zeroed);
    try std.testing.expectEqual(std.Thread.getCurrentId(), f.mapped_by.?);
}

test "a 6 GiB range maps only the pieces its pages use, each once, zeroed, and every mapped piece goes back" {
    var f: Fake = .{};
    var flags: [24]bool = @splat(false);
    var z: Lazy = .{ .base = 1 << 40, .len = 6 << 30, .piece = piece_bytes, .mapped = &flags };
    try z.ensure(f.mapper(), 0, 7 << 20, &f); // a page at the bottom: piece 0
    try z.ensure(f.mapper(), (6 << 30) - (360 << 20), 360 << 20, &f); // a run at the top: pieces 22 and 23
    try z.ensure(f.mapper(), (255 << 20), 2 << 20, &f); // across the edge of piece 0 and 1: piece 1 new
    try z.ensure(f.mapper(), 0, 7 << 20, &f); // again: nothing new
    try std.testing.expectEqual(@as(u32, 4), f.chunks);
    try std.testing.expectEqual(@as(u64, 4 * piece_bytes), f.zeroed);
    try std.testing.expect(flags[0] and flags[1] and flags[22] and flags[23] and !flags[2] and !flags[21]);
    z.unmapAll(f.mapper());
    try std.testing.expectEqual(@as(u64, 4 * piece_bytes), f.released);
    try std.testing.expect(std.mem.allEqual(bool, &flags, false));
    try std.testing.expectError(error.LaneMemory, z.ensure(f.mapper(), 6 << 30, 1, &f)); // past the range
}

test "a piece that fails or lands apart is not marked, and the pieces before it stay mapped" {
    var f: Fake = .{ .fail_at = 1 };
    var flags: [8]bool = @splat(false);
    var z: Lazy = .{ .base = 1 << 40, .len = 2 << 30, .piece = piece_bytes, .mapped = &flags };
    try std.testing.expectError(error.LaneMemory, z.ensure(f.mapper(), 0, 2 * piece_bytes, &f));
    try std.testing.expect(flags[0] and !flags[1]);
    var g: Fake = .{ .gap_at = 0 };
    var flags2: [8]bool = @splat(false);
    var y: Lazy = .{ .base = 1 << 40, .len = 2 << 30, .piece = piece_bytes, .mapped = &flags2 };
    try std.testing.expectError(error.LaneMemory, y.ensure(g.mapper(), 0, 1, &g));
    try std.testing.expect(!flags2[0]);
    try std.testing.expectEqual(@as(u64, piece_bytes), g.released); // the stray one went back
}

test "a piece whose zeroing fails goes back unmarked, so it is never read as zeroed nor kept mapped" {
    var f: Fake = .{ .fail_zero = true };
    var flags: [8]bool = @splat(false);
    var z: Lazy = .{ .base = 1 << 40, .len = 2 << 30, .piece = piece_bytes, .mapped = &flags };
    try std.testing.expectError(error.CudaFailed, z.ensure(f.mapper(), 0, 1, &f));
    try std.testing.expect(!flags[0]);
    try std.testing.expectEqual(@as(u64, piece_bytes), f.released);
}

test "the arena's range comes through its own hook: the installed hook is untouched, a declined one is cuMemAlloc" {
    var d = cuda.paging_test_device.Device.driver();
    const installed = cuda.memory.currentHook();
    var p: Arena.Pieces = .{ .gpa = std.testing.allocator, .vmm = undefined };
    p.vmm.granularity = 2 << 20;
    p.vmm.va = 1 << 40;
    var b = try cuda.DeviceBuffer.allocVia(&d, 3 * piece_bytes, p.hook());
    try std.testing.expectEqual(@as(u64, 1 << 40), b.ptr); // the range, not a driver allocation
    try std.testing.expect(b.owner != null and p.lazy != null);
    try std.testing.expectEqual(installed, cuda.memory.currentHook());
    b.free(); // through its owner: the flags go, nothing was mapped to unmap
    try std.testing.expect(p.lazy == null);
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 0 });
    var q: Arena.Pieces = .{ .gpa = failing.allocator(), .vmm = undefined };
    q.vmm.granularity = 2 << 20;
    q.vmm.va = 1 << 40;
    var c = try cuda.DeviceBuffer.allocVia(&d, 3 * piece_bytes, q.hook()); // no room for its flags: the hook declines
    try std.testing.expect(c.owner == null and q.lazy == null and c.ptr != 1 << 40);
    c.free();
}

test "without a device, or for one piece or less, the arena is one cuMemAlloc and every byte is there from the start" {
    var d = cuda.paging_test_device.Device.driver();
    var a = try Arena.alloc(std.testing.allocator, &d, 3 * piece_bytes, null, null);
    try std.testing.expect(!a.lazy());
    try a.ensure(0, 3 * piece_bytes, cuda.paging_test_device.Device.stream); // nothing to map
    a.free();
    var b = try Arena.alloc(std.testing.allocator, &d, piece_bytes, 0, null);
    try std.testing.expect(!b.lazy());
    b.free();
}
