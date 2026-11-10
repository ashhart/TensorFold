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
            const at = z.base + k * z.piece;
            const len = @min(z.piece, z.len - k * z.piece);
            m.seek(m.be.ctx, at);
            const got = m.be.chunk(m.be.ctx, len) orelse return error.LaneMemory;
            if (got != at) { // not at its place: the range would not hold the page
                m.be.release(m.be.ctx, got, len);
                return error.LaneMemory;
            }
            z.mapped[k] = true;
            try zero.piece(at, len);
        }
    }

    /// Every mapped piece goes back.
    pub fn unmapAll(z: *Lazy, m: Mapper) void {
        for (z.mapped, 0..) |*f, k| if (f.*) {
            m.be.release(m.be.ctx, z.base + k * z.piece, @min(z.piece, z.len - k * z.piece));
            f.* = false;
        };
    }
};

/// The pool's arena: `buf` a DeviceBuffer counted and freed as any other, a range mapped as used if `pieces` holds it.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    pieces: ?*Pieces = null,

    const Pieces = struct {
        gpa: std.mem.Allocator,
        vmm: cuda.arena.Vmm,
        want: usize = 0, // the one length the hook serves while armed (0: none)
        lazy: ?Lazy = null,

        fn hookAlloc(ctx: *anyopaque, len: usize) ?u64 {
            const s: *Pieces = @ptrCast(@alignCast(ctx));
            if (len == 0 or len != s.want) return null; // another allocation meanwhile: cuMemAlloc as before
            s.want = 0;
            const piece = std.mem.alignForward(usize, piece_bytes, s.vmm.granularity);
            const n = std.mem.alignForward(usize, len, s.vmm.granularity);
            const mapped = s.gpa.alloc(bool, (n + piece - 1) / piece) catch return null;
            @memset(mapped, false);
            s.lazy = .{ .base = s.vmm.va, .len = n, .piece = piece, .mapped = mapped };
            return s.vmm.va;
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

    /// `len` bytes reserved as pieces on `device` with virtual memory (`ensure` maps them); else one cuMemAlloc.
    pub fn alloc(gpa: std.mem.Allocator, d: *const cuda.Driver, len: usize, device: ?cuda.abi.Device) !Arena {
        const dev = device orelse return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
        if (len <= piece_bytes) return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
        const s: ?*Pieces = blk: {
            const vmm = cuda.arena.Vmm.open(d, dev, len) catch break :blk null;
            const s = gpa.create(Pieces) catch {
                var v = vmm;
                v.close();
                break :blk null;
            };
            s.* = .{ .gpa = gpa, .vmm = vmm };
            break :blk s;
        };
        const p = s orelse return .{ .buf = try cuda.DeviceBuffer.alloc(d, len) };
        // DeviceBuffer.alloc through a hook for this one call: the bytes are counted as every device buffer's are
        const before = cuda.memory.currentHook();
        cuda.memory.setHook(.{ .ctx = p, .alloc = Pieces.hookAlloc, .free = Pieces.hookFree });
        p.want = len;
        const buf = cuda.DeviceBuffer.alloc(d, len);
        p.want = 0;
        cuda.memory.setHook(before);
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
        try s.lazy.?.ensure(Mapper.vmm(&s.vmm), off, n, Zero{ .d = a.buf.d, .stream = stream });
    }

    /// Whether the arena maps its pieces on first use (else every byte is there from the start).
    pub fn lazy(a: *const Arena) bool {
        return a.pieces != null;
    }

    /// The buffer (its pieces unmapped), then the virtual range.
    pub fn free(a: *Arena) void {
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

    fn mapper(f: *Fake) Mapper {
        return .{ .be = .{ .ctx = f, .granularity = 2 << 20, .chunk = chunk, .release = release, .name = "fake" }, .seek = seek };
    }
    fn seek(ctx: *anyopaque, at: u64) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.next = at;
    }
    fn chunk(ctx: *anyopaque, _: usize) ?u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
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
        f.zeroed += len;
    }
};

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
