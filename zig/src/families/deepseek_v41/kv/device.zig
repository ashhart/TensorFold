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

    /// Allocates every family's tensor (zeroed) and the tables for `slots` slots of up to `max_pages` pages.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, pool: *Pool, compute: cuda.Stream, slots: u32, max_pages: u32) !DevicePool {
        const fams = pool.layout.families;
        const t = try gpa.alloc(cuda.DeviceBuffer, fams.len);
        errdefer gpa.free(t);
        var made: usize = 0;
        errdefer for (t[0..made]) |*b| b.free();
        for (fams, 0..) |f, i| {
            t[i] = try cuda.DeviceBuffer.alloc(d, @as(usize, pool.familyTensorPages(f)) * f.pageBytes(pool.page));
            made += 1;
            try t[i].fill8(0, compute.handle);
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
        for (&ev) |*e| e.* = try cuda.Event.init(d, false);
        // fills ran on the compute stream (a legacy-stream memset races later uploads): sync before any use
        try compute.synchronize();
        return .{ .gpa = gpa, .d = d, .ctx = ctx, .pool = pool, .tensors = t, .table = table, .local = local, .host_table = ht, .host_local = hl, .pts = max_pages, .slots = slots, .io = io, .events = ev, .compute = compute };
    }

    pub fn deinit(p: *DevicePool) void {
        for (p.tensors) |*b| b.free();
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
        for (p.pool.slots.items) |s| {
            if (s.id >= p.slots) return error.TooManySlots;
            const r = s.takeDirty() orelse continue;
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

    fn read(ptr: *anyopaque, fam: u32, pages: []const u32, dst: []u8) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        try p.ctx.makeCurrent();
        const pb: usize = @intCast(p.pool.layout.families[fam].pageBytes(p.pool.page));
        if (dst.len != pages.len * pb) return error.BadLength;
        var i: usize = 0;
        while (i < pages.len) {
            const j = runs(pages, i);
            try p.tensors[fam].downloadAsync(pages[i] * pb, dst[i * pb .. j * pb], p.io.handle);
            i = j;
        }
        try p.io.synchronize();
    }

    fn write(ptr: *anyopaque, fam: u32, pages: []const u32, src: []const u8) anyerror!void {
        const p: *DevicePool = @ptrCast(@alignCast(ptr));
        try p.ctx.makeCurrent();
        const pb: usize = @intCast(p.pool.layout.families[fam].pageBytes(p.pool.page));
        if (src.len != pages.len * pb) return error.BadLength;
        var i: usize = 0;
        while (i < pages.len) {
            const j = runs(pages, i);
            try p.tensors[fam].uploadAsync(pages[i] * pb, src[i * pb .. j * pb], p.io.handle);
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
