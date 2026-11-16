//! The pool on the GPU: one device tensor a family, stacked [slots, pts] int32 page tables, the sessions' PageStore.

const std = @import("std");
const cuda = @import("cuda");
const sessions = @import("sessions");
const Pool = sessions.Pool;
const PageStore = sessions.PageStore;

/// A family as the kernels bind it (attn / kv_store / index): rows, their scales, the page table, rows-a-page shift.
pub const View = struct {
    cv: u64,
    csc: u64,
    cvs: i64,
    css: i64,
    pt: u64,
    pts: i64,
    psh: c_int,
};

const fences = 8;

pub const DevicePool = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    pool: *Pool,
    tensors: []cuda.DeviceBuffer,
    /// [slots, pts] int32: the physical tables and the split families' local tables
    table: cuda.DeviceBuffer,
    local: cuda.DeviceBuffer,
    host_table: []i32,
    host_local: []i32,
    pts: u32,
    slots: u32,
    io: cuda.Stream,
    events: [fences]cuda.Event,
    compute: cuda.Stream,
    next_fence: u64 = 0,
    staging: ?cuda.copy_source.Buffer = null,
    /// TF_CUDA_CARVEOUT=1: the display carveout the family tensors were taken from first, and the bytes it holds
    carve: ?*cuda.Carveout = null,
    carved: u64 = 0,

    /// Allocates every family's tensor (zeroed), from `carve` first when it is given, and the tables for `slots` slots.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, pool: *Pool, compute: cuda.Stream, slots: u32, max_pages: u32, carve: ?*cuda.Carveout) !DevicePool {
        const fams = pool.layout.families;
        const t = try gpa.alloc(cuda.DeviceBuffer, fams.len);
        errdefer gpa.free(t);
        var made: usize = 0;
        errdefer for (t[0..made]) |*b| drop(carve, b);
        var carved: u64 = 0;
        for (fams, 0..) |f, i| {
            const len = tensorLen(pool, f);
            t[i] = (if (carve) |c| try c.take(len) else null) orelse try cuda.DeviceBuffer.alloc(d, len);
            made += 1;
            if (t[i].borrowed) carved += len else try t[i].fill8(0, compute.handle);
        }
        const n = @as(usize, slots) * max_pages;
        var table = try cuda.DeviceBuffer.alloc(d, n * 4);
        errdefer table.free();
        try table.fill32(pool.null_page, compute.handle);
        var local = try cuda.DeviceBuffer.alloc(d, n * 4);
        errdefer local.free();
        try local.fill32(pool.local_null, compute.handle);
        const ht = try gpa.alloc(i32, n);
        errdefer gpa.free(ht);
        @memset(ht, @intCast(pool.null_page));
        const hl = try gpa.alloc(i32, n);
        errdefer gpa.free(hl);
        @memset(hl, @intCast(pool.local_null));
        var io = try cuda.Stream.init(d, true);
        errdefer io.deinit();
        var ev: [fences]cuda.Event = undefined;
        var made_events: usize = 0;
        errdefer for (ev[0..made_events]) |*e| e.deinit();
        for (&ev) |*e| {
            e.* = try cuda.Event.init(d, false);
            made_events += 1;
        }
        // fills ran on the compute stream (a legacy-stream memset races later uploads): sync before any use
        try compute.synchronize();
        // a carveout span is zeroed on the legacy stream, which the compute stream does not wait for
        if (carved > 0) try d.synchronize();
        return .{ .gpa = gpa, .d = d, .ctx = ctx, .pool = pool, .tensors = t, .table = table, .local = local, .host_table = ht, .host_local = hl, .pts = max_pages, .slots = slots, .io = io, .events = ev, .compute = compute, .carve = carve, .carved = carved };
    }

    /// A family's device tensor in bytes: its pages (familyTensorPages) of its page size.
    fn tensorLen(pool: *const Pool, f: sessions.Family) usize {
        return @as(usize, pool.familyTensorPages(f)) * f.pageBytes(pool.page);
    }

    /// Bytes of `pool` that `init` would place in `carve`: its first-fit placement, run on a copy of the spans.
    pub fn carvedBytes(pool: *const Pool, carve: ?*const cuda.Carveout) u64 {
        const c = carve orelse return 0;
        var spans = c.spans;
        var n: u64 = 0;
        for (pool.layout.families) |f| {
            const len = tensorLen(pool, f);
            if (spans.take(len) != null) n += len;
        }
        return n;
    }

    /// A family tensor back where it came from: its span to the carveout, or its allocation to the driver.
    fn drop(carve: ?*cuda.Carveout, b: *cuda.DeviceBuffer) void {
        if (b.borrowed) carve.?.give(b.*) else b.free();
    }

    pub fn close(p: *DevicePool) !void {
        try p.compute.synchronize();
        try p.io.synchronize();
        p.deinit();
    }
    pub fn deinit(p: *DevicePool) void {
        if (p.staging) |*h| h.free();
        for (p.tensors) |*b| drop(p.carve, b);
        p.gpa.free(p.tensors);
        p.table.free();
        p.local.free();
        p.gpa.free(p.host_table);
        p.gpa.free(p.host_local);
        for (&p.events) |*e| e.deinit();
        p.io.deinit();
        p.* = undefined;
    }

    /// Uploads the table entries each slot changed since the last sync, on the compute stream.
    pub fn syncTables(p: *DevicePool) !void {
        try p.compute.synchronize();
        for (p.pool.slots.items) |s| {
            if (s.id >= p.slots) return error.TooManySlots;
            const r = s.takeDirty() orelse continue;
            errdefer {
                s.dirty_lo = @min(s.dirty_lo, r.lo);
                s.dirty_hi = @max(s.dirty_hi, r.hi);
            }
            const row = @as(usize, s.id) * p.pts;
            for (r.lo..r.hi) |i| {
                p.host_table[row + i] = @intCast(s.tableAt(@intCast(i)));
                p.host_local[row + i] = @intCast(s.localTableAt(@intCast(i)));
            }
            const a = row + r.lo;
            const n = r.hi - r.lo;
            try p.table.uploadAsync(a * 4, std.mem.sliceAsBytes(p.host_table[a..][0..n]), p.compute.handle);
            try p.local.uploadAsync(a * 4, std.mem.sliceAsBytes(p.host_local[a..][0..n]), p.compute.handle);
        }
    }

    /// Family fam as the kernels take it for one slot (row mode: slot 0's row with pts, the stacked table).
    pub fn view(p: *const DevicePool, fam: u32, slot: u32) View {
        const f = p.pool.layout.families[fam];
        const split = f.split and p.pool.world > 1;
        const tab = if (split) p.local else p.table;
        const rows = f.pageRows(p.pool.page);
        const values: u64 = if (std.mem.startsWith(u8, f.name, "comp") and f.row_bytes == 584) 576 else f.row_bytes;
        return .{ .cv = p.tensors[fam].ptr, .csc = p.tensors[fam].ptr + values, .cvs = f.row_bytes, .css = f.row_bytes, .pt = tab.ptr + @as(u64, slot) * p.pts * 4, .pts = p.pts, .psh = @intCast(std.math.log2_int(u32, rows)) };
    }

    pub fn store(p: *DevicePool) PageStore {
        return .{ .ptr = p, .vtable = &vtable };
    }

    const vtable: PageStore.VTable = .{ .read = read, .write = write, .copyPage = copyPage, .fence = fence, .waitFence = waitFence, .publish = publish };

    /// Consecutive family pages as runs: one copy a run (a chunk of a long entry is mostly one or two runs).
    fn runs(pages: []const u32, i: usize) usize {
        var j = i + 1;
        while (j < pages.len and pages[j] == pages[j - 1] + 1) j += 1;
        return j;
    }

    fn stage(p: *DevicePool, len: usize) ![]u8 {
        try p.io.synchronize();
        if (p.staging) |*h| try h.wait();
        if (p.staging == null or p.staging.?.bytes.len < len) {
            if (p.staging) |*h| h.free();
            p.staging = null;
            p.staging = try cuda.copy_source.Buffer.alloc(p.d, @max(256, len));
        }
        return p.staging.?.bytes[0..len];
    }

    fn read(ptr: *anyopaque, fam: u32, pages: []const u32, dst: []u8) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        try p.ctx.makeCurrent();
        const pb: usize = @intCast(p.pool.layout.families[fam].pageBytes(p.pool.page));
        if (dst.len != pages.len * pb) return error.BadLength;
        const staged = try p.stage(dst.len);
        try p.staging.?.mark(p.io);
        var i: usize = 0;
        while (i < pages.len) {
            const j = runs(pages, i);
            try p.tensors[fam].downloadAsync(pages[i] * pb, staged[i * pb .. j * pb], p.io.handle);
            i = j;
        }
        try p.io.synchronize();
        @memcpy(dst, staged);
    }

    fn write(ptr: *anyopaque, fam: u32, pages: []const u32, src: []const u8) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        try p.ctx.makeCurrent();
        const pb: usize = @intCast(p.pool.layout.families[fam].pageBytes(p.pool.page));
        if (src.len != pages.len * pb) return error.BadLength;
        const staged = try p.stage(src.len);
        @memcpy(staged, src);
        var i: usize = 0;
        while (i < pages.len) {
            const j = runs(pages, i);
            try p.staging.?.copyTo(p.tensors[fam], pages[i] * pb, staged[i * pb .. j * pb], p.io);
            i = j;
        }
        try p.io.synchronize(); // the staging buffer goes back to the lanes after this
    }

    fn copyPage(ptr: *anyopaque, fam: u32, src: u32, dst: u32) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        const pb: usize = @intCast(p.pool.layout.families[fam].pageBytes(p.pool.page));
        const b = p.tensors[fam];
        try b.copyFrom(dst * pb, b.ptr + src * pb, pb, p.compute.handle);
    }

    fn fence(ptr: *anyopaque) anyerror!u64 {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        const t = p.next_fence;
        p.next_fence += 1;
        try p.events[t % fences].record(p.compute);
        return t;
    }

    fn waitFence(ptr: *anyopaque, token: u64) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        try p.ctx.makeCurrent();
        // the slot holds this fence's record or a later one: waiting for it waits at least as long
        try p.events[token % fences].synchronize();
    }

    fn publish(ptr: *anyopaque) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        try p.ctx.makeCurrent();
        try p.io.synchronize();
    }
};

test "the pool's families go to the display carveout first-fit, in family order, and the spans are not touched" {
    const t = std.testing;
    const fams = [_]sessions.Family{
        .{ .name = "comp.L2", .ratio = 1, .row_bytes = 584, .split = true },
        .{ .name = "index_k.L2", .ratio = 1, .row_bytes = 132, .split = true },
        .{ .name = "comp.L8", .ratio = 1, .row_bytes = 584, .split = true },
        .{ .name = "index_k.L8", .ratio = 1, .row_bytes = 132, .split = true },
    };
    var pool = try Pool.init(t.allocator, .{ .families = &fams }, 16 * 256, 2, 0);
    defer pool.deinit();
    // rank 0 of two: 8 local pages + the local null + the discard page = 10 pages a family tensor
    const comp: u64 = 10 * 256 * 584;
    const ik: u64 = 10 * 256 * 132;
    try t.expectEqual(@as(u64, 0), DevicePool.carvedBytes(&pool, null));
    var c: cuda.Carveout = .{ .d = undefined, .fd = -1, .handle = 0, .host = undefined, .dev = 0, .flags = 0, .spans = .{ .size = comp + ik + ik + 4096 } };
    // comp.L2 and index_k.L2 fit; comp.L8 does not and goes to device memory; index_k.L8 takes the room after them
    try t.expectEqual(comp + ik + ik, DevicePool.carvedBytes(&pool, &c));
    try t.expectEqual(@as(usize, 0), c.spans.n);
    c.spans = .{ .size = 2 * comp + 2 * ik };
    try t.expectEqual(2 * comp + 2 * ik, DevicePool.carvedBytes(&pool, &c));
    c.spans = .{ .size = ik - 1 };
    try t.expectEqual(@as(u64, 0), DevicePool.carvedBytes(&pool, &c));
}
