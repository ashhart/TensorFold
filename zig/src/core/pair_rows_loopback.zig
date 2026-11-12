//! The same seam with a host copy instead of the link: the numeric reference the real transport is checked against.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const Fence = @import("fence.zig").Fence;
const pair_layout = @import("pair_rows_layout.zig");

const FAILED = pair_layout.FAILED;
const JOBS = pair_layout.JOBS;
const Layout = pair_layout.Layout;
const PAGE = pair_layout.PAGE;
const PairRows = pair_layout.PairRows;
const Seam = pair_layout.Seam;
const elems_per_word = pair_layout.elems_per_word;
const pages = pair_layout.pages;
const stageBytes = pair_layout.stageBytes;
const pickBytes = pair_layout.pickBytes;
const seam = pair_layout.seam;
const seam_failed = pair_layout.seam_failed;
const writeHalf = pair_layout.writeHalf;
const Ref = pair_layout.Ref;

/// The same seam with a host copy instead of the link: the numeric reference; bytes arrive via `arrive`/`seed`.
pub const Loopback = struct {
    allocator: std.mem.Allocator,
    rank: u32,
    l: Layout,
    win: []u8,
    wbuf: mtl.Buffer,
    stage_pipe: mtl.Pipeline,
    gather_pipe: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,
    /// Per-exchange part bytes (stage/gather and argmax queue different sizes): the relay copies exactly this much.
    bytes: []u32,
    fence: Fence,
    /// This rank's parts: the GPU signals it per exchange, and the relay reads `bytesFor(x)` after that.
    posted: mtl.SharedEvent,
    /// The peer's parts: the relay sets it once the copy has landed, and the seam waits for it.
    landed: mtl.SharedEvent,
    x: u64 = 0,
    relay: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),

    pub fn init(device: mtl.Device, bytes: usize, rank: u32) !Loopback {
        const l: Layout = .{ .slot = pages(bytes) };
        // the window is a no-copy Metal buffer: page-aligned, page-multiple allocation, never the gpa's good luck
        const win = try std.heap.page_allocator.alloc(u8, l.window());
        errdefer std.heap.page_allocator.free(win);
        std.debug.assert(@intFromPtr(win.ptr) % PAGE == 0 and win.len % PAGE == 0);
        @memset(win, 0);
        const lib = try mtl.Library.fromSource(device, sources.core_pair_rows, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const wbuf = try device.bufferNoCopy(win.ptr, win.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer wbuf.deinit();
        const fence = try Fence.init(device);
        errdefer fence.deinit();
        const posted = try device.sharedEvent();
        errdefer posted.deinit();
        const landed = try device.sharedEvent();
        errdefer landed.deinit();
        const stage_pipe = try mtl.Pipeline.init(device, lib, "pair_stage", false);
        errdefer stage_pipe.deinit();
        const gather_pipe = try mtl.Pipeline.init(device, lib, "pair_gather", false);
        errdefer gather_pipe.deinit();
        const argmax_pipe = try mtl.Pipeline.init(device, lib, "pair_argmax", false);
        errdefer argmax_pipe.deinit();
        const pick_pipe = try mtl.Pipeline.init(device, lib, "pair_pick", false);
        errdefer pick_pipe.deinit();
        const sizes = try std.heap.page_allocator.alloc(u32, JOBS);
        errdefer std.heap.page_allocator.free(sizes);
        @memset(sizes, 0);
        return .{ .allocator = std.heap.page_allocator, .rank = rank, .l = l, .win = win, .wbuf = wbuf, .stage_pipe = stage_pipe, .gather_pipe = gather_pipe, .argmax_pipe = argmax_pipe, .pick_pipe = pick_pipe, .bytes = sizes, .fence = fence, .posted = posted, .landed = landed };
    }

    pub fn deinit(p: *Loopback) void {
        p.stopRelay();
        p.stage_pipe.deinit();
        p.gather_pipe.deinit();
        p.argmax_pipe.deinit();
        p.pick_pipe.deinit();
        p.allocator.free(p.bytes);
        p.wbuf.deinit();
        p.posted.deinit();
        p.landed.deinit();
        p.fence.deinit();
        p.allocator.free(p.win);
    }

    fn word32(p: *const Loopback, off: usize) *u32 {
        return @ptrCast(@alignCast(p.win.ptr + off));
    }

    /// Where this rank's part for exchange x goes (what the relay copies out, once `posted` passes x).
    pub fn sendOffset(p: *const Loopback, x: u32) usize {
        return p.l.send(x);
    }

    /// Where the peer's part for exchange x goes (what the relay copies in, before setting `landed` to x).
    pub fn recvOffset(p: *const Loopback, x: u32) usize {
        return p.l.recv(x);
    }

    /// Where the producer writes this rank's bf16 columns of the next exchange; the same shape as `Pair.part`.
    pub fn nextX(p: *const Loopback) u32 {
        return @truncate(p.x + 1);
    }

    pub fn part(p: *const Loopback) Seam {
        const x: u32 = @truncate(p.x + 1);
        return .{ .x = x, .ref = .{ .buf = p.wbuf, .off = p.l.send(x) } };
    }

    /// The seed pattern as a frame element: 1 + m/128, exact in bf16, so expected bytes hold either way.
    fn seedValue(r: u32, i: u32, salt: u16) f32 {
        const k = r *% 5120 +% i *% 37;
        const m: u32 = (salt & 0x1F) + ((k *% 53) & 0x1F);
        return 1.0 + @as(f32, @floatFromInt(m)) / 128.0;
    }

    /// Fill both slots with the seed and mark all arrived: arithmetic check only, timing must not use this.
    pub fn seed(p: *Loopback, rows: u32, width: u32) void {
        const bytes = stageBytes(rows, width);
        if (bytes > p.l.slot) @panic("loopback: the seeded part does not fit the window's slot");
        for (0..2) |x| {
            const recv = p.win[p.l.recv(@intCast(x))..][0..bytes];
            const send = p.win[p.l.send(@intCast(x))..][0..bytes];
            for (0..rows) |r| for (0..width) |i| {
                const at = r * width + i;
                writeHalf(recv, at, seedValue(@intCast(r), @intCast(i), 0x15));
                writeHalf(send, at, seedValue(@intCast(r), @intCast(i), 0x0A));
            };
        }
        p.satisfy();
    }

    /// Release every wait with the sentinel, as a failed link does: arithmetic check only, timing runs `startRelay`.
    pub fn satisfy(p: *Loopback) void {
        p.landed.set(seam_failed);
    }

    /// The in-process relay: wait for posted, copy in, release -- Pair's two events in order, memcpy for the wire.
    pub fn startRelay(p: *Loopback) !void {
        std.debug.assert(p.relay == null);
        p.stop.store(false, .release);
        p.relay = try std.Thread.spawn(.{}, relayLoop, .{p});
    }

    pub fn stopRelay(p: *Loopback) void {
        p.stop.store(true, .release);
        if (p.relay) |t| t.join();
        p.relay = null;
    }

    fn relayLoop(p: *Loopback) void {
        var x: u64 = 1;
        while (!p.stop.load(.acquire)) {
            if (p.posted.value() < x) continue; // spin: the host's side of the seam is part of what is measured
            const at: u32 = @truncate(x);
            const n: usize = p.bytesFor(at);
            @memcpy(p.win[p.l.recv(at)..][0..n], p.win[p.l.send(at)..][0..n]);
            p.landed.set(x);
            x += 1;
        }
    }

    /// The part's bytes for exchange `x` as the last `gather`/`argmax` queued it (the relay's copy length).
    pub fn bytesFor(p: *const Loopback, x: u32) u32 {
        return @atomicLoad(u32, &p.bytes[x % JOBS], .acquire);
    }

    /// Land a host-written peer part for exchange x (the relay's post-copy half): copy it in, then release the wait.
    pub fn arrive(p: *Loopback, x: u32, bytes: []const u8) void {
        std.debug.assert(bytes.len <= p.l.slot);
        @memcpy(p.win[p.l.recv(x)..][0..bytes.len], bytes);
        p.landed.set(x);
    }

    /// The seam `Pair.stage` encodes (before its `gather`): this rank's `width` columns of the frame row into the slot.
    pub fn stage(p: *Loopback, e: *mtl.ComputeEncoder, x: u32, src: Ref, rows: u32, width: u32, stride: u32) void {
        if (rows == 0 or width == 0 or width > stride) @panic("loopback: stage() wants rows > 0 and a width inside the source row");
        if (stageBytes(rows, width) > p.l.slot) @panic("loopback: this exchange's part does not fit the window's slot");
        e.setPipeline(p.stage_pipe);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setValue(PairRows{ .width = width, .stride = stride, .rank = p.rank, .x = x }, 1);
        e.setBuffer(src.buf, src.off, 2);
        e.setBuffer(p.wbuf, p.l.send(x), 3);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// The seam `Pair.gather` encodes: the relay, then the frame row as this rank's half and the peer's, both bf16.
    pub fn gather(p: *Loopback, e: *mtl.ComputeEncoder, x: u32, mine: Ref, out: Ref, rows: u32, width: u32, out_stride: u32) void {
        if (rows == 0 or width == 0 or width % elems_per_word != 0) @panic("loopback: gather() wants rows > 0 and a width that fills whole window words");
        if (out_stride < 2 * width) @panic("loopback: gather()'s frame row is narrower than both halves");
        if (x != @as(u32, @truncate(p.x + 1))) @panic("loopback: an exchange was encoded out of order (part() not called first)");
        if (mine.buf.id != p.wbuf.id or mine.off != p.l.send(x)) @panic("loopback: gather()'s part is not the region part() returned");
        if (stageBytes(rows, width) > p.l.slot) @panic("loopback: this exchange's part does not fit the window's slot");
        p.x += 1;
        @atomicStore(u32, &p.bytes[x % JOBS], @intCast(stageBytes(rows, width)), .release);
        seam(e, p.fence, p.posted, p.landed, x);
        e.setPipeline(p.gather_pipe);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setValue(PairRows{ .width = width, .stride = out_stride, .rank = p.rank, .x = x }, 1);
        e.setBuffer(mine.buf, mine.off, 2);
        e.setBuffer(p.wbuf, p.l.recv(x), 3);
        e.setBuffer(out.buf, out.off, 4);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// The loopback's head seam: this rank's argmax over n bf16 logits, the relay copy, then the pick over both parts.
    pub fn argmaxBf16(p: *Loopback, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        p.argmaxWidth(e, x, logits, stride, lo, n, rows, picks, 1);
    }

    /// The same seam over fp32 logits (GLM-5.3's frame): the loopback covers both widths the kernel reads.
    pub fn argmax(p: *Loopback, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        p.argmaxWidth(e, x, logits, stride, lo, n, rows, picks, 0);
    }

    fn argmaxWidth(p: *Loopback, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref, dtype: u32) void {
        if (rows == 0 or n == 0) @panic("loopback: argmax() wants rows > 0 and n > 0");
        if (x != @as(u32, @truncate(p.x + 1))) @panic("loopback: an exchange was encoded out of order (part() not called first)");
        p.x += 1;
        @atomicStore(u32, &p.bytes[x % JOBS], @intCast(pickBytes(rows)), .release); // two u32 a row: (value, id)
        e.setPipeline(p.argmax_pipe);
        e.setBuffer(logits.buf, logits.off, 0);
        e.setValue([4]u32{ stride, lo, n, dtype }, 1);
        e.setBuffer(p.wbuf, p.l.send(x), 2);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
        seam(e, p.fence, p.posted, p.landed, x); // the encoder boundary orders the argmax before the part is relayed
        e.setPipeline(p.pick_pipe);
        e.setBuffer(p.wbuf, p.l.sync(), 0);
        e.setValue([2]u32{ rows, x }, 1);
        e.setBuffer(p.wbuf, p.l.send(x), 2);
        e.setBuffer(p.wbuf, p.l.recv(x), 3);
        e.setBuffer(picks.buf, picks.off, 4);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 32 });
    }

    /// Seams the GPU refused (must be 0 for a run to be valid).
    pub fn gaveUp(p: *const Loopback) u32 {
        return @atomicLoad(u32, p.word32(p.l.sync() + FAILED * 4), .acquire);
    }

    /// There is no link to fail here, so this is `gaveUp` under the seam's name, and it reads 0.
    pub fn failed(p: *const Loopback) bool {
        return p.gaveUp() != 0;
    }
};
