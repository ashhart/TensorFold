//! GLM TP control headers and Link transport, with optional folded keeps on their own stream.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const prompt_wire = @import("cuda_prompt_wire.zig");
pub const Kind = enum(u32) { begin = 1, prefill, verify, keep, draft, end, drop, release, chunk, shared, image, block, many, redraw, idle, snapshots, pool, form, cuts, disk };

/// One step as it crosses: 64 bytes, then `n` u32 ids.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // draft: drafts to chain; chunk: where it starts; shared: its parts
    flags: u32 = 0, // begin: sampled; draft: from the prompt (1), chances (2); prefill, chunk: prompt_logprobs (1); chunk: indexer rows split (2), a capped step's rows (from bit 2); verify, shared: whole rows; snapshots: pages shared
    a: u64 = 0, // begin: the prompt cache's pages first in its table; keep: rows; prefill, chunk: the kept state it resumes (0: none); redraw: the prompt's length; form: a bit a window's rows
    seed: u64 = 0, // begin: the sampling seed; prefill, chunk: the kept state's position
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0, // begin: top_k; chunk: the prompt's length
    slot: u32 = 0, // the stream's sequence the step runs on (Engine.bind)

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

/// tf_glm_put_words' argument: a step's header (16 words) and up to `put_ids` ids.
pub const Words = extern struct { w: [96]u32 };
pub const no_keep = std.math.maxInt(u64);
const FoldHeader = extern struct {
    base: Header,
    keep: u64 = no_keep,
    keep_slot: u32 = 0,
    reserved: u32 = 0,
    comptime {
        std.debug.assert(@sizeOf(FoldHeader) == 80);
    }
};
const inline_ids = 48;
const folded_inline_ids = inline_ids - 4;
const first_bytes = 64 + inline_ids * 4;
pub const put_ids = 96 - 16;
pub const PendingKeep = struct { slot: u32, rows: u64 };

/// The most ids one step carries (a prompt at the engine's capacity).
pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer, // tensor/disk data and, with folding off, header then ids
    ids: []u32, // host copy of the last step's ids
    gpa: std.mem.Allocator,
    wait: bool, // TF_GLM_LINK_WAIT=1: every step uploaded and waited for, as before the put
    asleep: bool = false, // rank 0: the other ranks sleep on their lines (cuda.tp_watch) until its next step
    fold: bool = false, // TF_GLM_CONTROL_FOLD=1: pending keep rides the next control, never a tensor-data message
    control_buf: ?cuda.DeviceBuffer = null, // folded controls alone; disk `from` keeps `buf` and the compute stream
    control_stream: ?cuda.Stream = null, // owned only with folding on; default off uses the compute stream exactly as before
    pending_keep: ?PendingKeep = null, // leader's accepted rows on their original slot, consumed by the next control (or a standalone fallback between adjacent keeps)
    received_keep: ?PendingKeep = null, // follower's folded rows, committed on their original slot before the carried step's binding

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, asked_ids: usize) !Link {
        const max_ids = @max(asked_ids, prompt_wire.words);
        const buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        errdefer {
            var owned = buf;
            owned.free();
        }
        const wait = try @import("sys").environment.read(bool, "TF_GLM_LINK_WAIT", false);
        const fold = try @import("sys").environment.read(bool, "TF_GLM_CONTROL_FOLD", false);
        var control_stream: ?cuda.Stream = null;
        var control_buf: ?cuda.DeviceBuffer = null;
        errdefer if (control_stream) |*owned| owned.deinit();
        errdefer if (control_buf) |*owned| {
            var allocation = owned.*;
            allocation.free();
        };
        if (fold) {
            control_stream = try cuda.Stream.init(d, true);
            control_buf = try cuda.DeviceBuffer.alloc(d, 80 + max_ids * 4);
        }
        return .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa, .wait = wait, .fold = fold, .control_buf = control_buf, .control_stream = control_stream };
    }

    /// Rank 0, before it serves: whether the other ranks begin asleep; at TP its kept states' disk steps cross here.
    pub fn lead(l: *Link, asleep: bool, world: usize) void {
        l.asleep = asleep;
        if (world > 1) @import("cuda_disk_ranks.zig").link = l;
    }

    pub fn deinit(l: *Link) void {
        if (@import("cuda_disk_ranks.zig").link == l) @import("cuda_disk_ranks.zig").link = null;
        if (l.control_stream) |*owned| owned.deinit();
        if (l.control_buf) |*owned| {
            var allocation = owned.*;
            allocation.free();
        }
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn buffer(l: *Link) *const cuda.DeviceBuffer {
        return if (l.fold) &l.control_buf.? else &l.buf;
    }

    fn controlStream(l: *Link) cuda.Stream {
        return if (l.fold) l.control_stream.? else l.stream;
    }

    fn inlineIds(l: *Link) usize {
        return if (l.fold) folded_inline_ids else inline_ids;
    }

    fn bcast(l: *Link, bytes: usize) !void {
        try l.bcastAt(0, bytes);
    }
    fn bcastAt(l: *Link, offset: usize, bytes: usize) !void {
        try l.postAt(offset, bytes);
        try l.controlStream().synchronize();
    }
    fn post(l: *Link, bytes: usize) !void {
        try l.postAt(0, bytes);
    }
    fn postAt(l: *Link, offset: usize, bytes: usize) !void {
        const ptr = l.buffer().ptr + offset;
        try l.lib.check(l.lib.api.ncclBroadcast(ptr, ptr, bytes, .u8, 0, l.comm, l.controlStream().handle), "ncclBroadcast");
    }

    /// Rank 0 updates locally after this queue; a second window first sends the prior window's saved slot.
    pub fn keep(l: *Link, put: cuda.Function, slot: u32, n: u64) !void {
        if (l.fold) {
            if (l.pending_keep) |prior| {
                if (l.asleep) {
                    cuda.tp_watch.wake();
                    l.asleep = false;
                }
                try l.send(put, .{ .kind = @backingInt(Kind.keep), .slot = prior.slot, .a = prior.rows }, &.{});
            }
            l.pending_keep = .{ .slot = slot, .rows = n };
        } else {
            if (l.asleep) {
                cuda.tp_watch.wake();
                l.asleep = false;
            }
            try l.send(put, .{ .kind = @backingInt(Kind.keep), .slot = slot, .a = n }, &.{});
        }
    }

    /// Rank 0: the step and ids to every rank; up to `put_ids` ids go by `put` unwaited, longer ones upload and wait.
    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        const pending = if (l.fold and hh.kind != @backingInt(Kind.keep)) l.pending_keep else null;
        if (l.fold) l.pending_keep = null;
        const put_count = put_ids - @as(usize, if (l.fold) 4 else 0);
        if (!l.wait and ids.len <= put_count) {
            var w: Words = undefined;
            if (l.fold) {
                const fh: FoldHeader = if (pending) |p| .{ .base = hh, .keep = p.rows, .keep_slot = p.slot } else .{ .base = hh };
                @memcpy(w.w[0..20], std.mem.bytesAsSlice(u32, std.mem.asBytes(&fh)));
                @memcpy(w.w[20..][0..ids.len], ids);
            } else {
                @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
                @memcpy(w.w[16..][0..ids.len], ids);
            }
            var a: cuda.Args = .{};
            a.add(l.buffer().ptr);
            a.add(@as(u32, @intCast(@as(usize, if (l.fold) 20 else 16) + ids.len)));
            a.add(w);
            try cuda_pdl.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.controlStream(), &a);
            try l.post(first_bytes);
            if (ids.len > l.inlineIds()) try l.postAt(if (l.fold) 80 else 0, if (l.fold) ids.len * 4 else 64 + ids.len * 4);
            return;
        }
        if (l.fold) {
            const fh: FoldHeader = if (pending) |p| .{ .base = hh, .keep = p.rows, .keep_slot = p.slot } else .{ .base = hh };
            try l.buffer().uploadAsync(0, std.mem.asBytes(&fh), l.controlStream().handle);
            if (ids.len > 0 and ids.len <= l.inlineIds()) try l.buffer().uploadAsync(80, std.mem.sliceAsBytes(ids), l.controlStream().handle);
        } else {
            try l.buf.uploadAsync(0, std.mem.asBytes(&hh), l.stream.handle);
            if (ids.len > 0 and ids.len <= l.inlineIds()) try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
        }
        try l.bcast(first_bytes);
        if (ids.len > l.inlineIds()) {
            const upload_offset: usize = if (l.fold) 80 else 64;
            const broadcast_offset: usize = if (l.fold) 80 else 0;
            try l.buffer().uploadAsync(upload_offset, std.mem.sliceAsBytes(ids), l.controlStream().handle);
            try l.bcastAt(broadcast_offset, if (l.fold) ids.len * 4 else 64 + ids.len * 4);
        }
    }

    /// `bytes` at `ptr` from rank 0 into every rank's `ptr` (image rows: the follower's buffer is the same size).
    pub fn rows(l: *Link, ptr: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try l.lib.check(l.lib.api.ncclBroadcast(ptr, ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// `bytes` after the header from rank `root` into every rank's buffer (a disk step's take: cuda_disk_ranks).
    pub fn from(l: *Link, root: u32, bytes: usize) !void {
        if (bytes == 0) return;
        if (64 + bytes > l.buf.len) return error.TooManyIds;
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr + 64, l.buf.ptr + 64, bytes, .u8, @intCast(root), l.comm, l.stream.handle), "ncclBroadcast");
        try l.stream.synchronize();
    }

    /// Other ranks: the next step and its ids (valid until the next receive).
    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        l.received_keep = null;
        try l.bcast(first_bytes);
        var first: [first_bytes]u8 align(8) = undefined;
        try l.buffer().download(0, &first);
        const h = std.mem.bytesAsValue(Header, first[0..@sizeOf(Header)]).*;
        if (h.n > l.ids.len) return error.TooManyIds;
        if (l.fold) {
            const committed = std.mem.bytesAsValue(u64, first[64..72]).*;
            if (committed != no_keep) l.received_keep = .{ .slot = std.mem.bytesAsValue(u32, first[72..76]).*, .rows = committed };
        }
        if (h.n > l.inlineIds()) {
            const download_offset: usize = if (l.fold) 80 else 64;
            const broadcast_offset: usize = if (l.fold) 80 else 0;
            try l.bcastAt(broadcast_offset, if (l.fold) h.n * 4 else 64 + h.n * 4);
            try l.buffer().download(download_offset, std.mem.sliceAsBytes(l.ids[0..h.n]));
        } else if (h.n > 0) {
            const offset: usize = if (l.fold) 80 else 64;
            @memcpy(l.ids[0..h.n], std.mem.bytesAsSlice(u32, first[offset..][0 .. h.n * 4]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};
