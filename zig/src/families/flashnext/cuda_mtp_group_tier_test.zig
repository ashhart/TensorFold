//! Grouped head rounds preserve real host-tier frame ownership across eviction and restore; CUDA arithmetic is stubbed.

const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const flash = @import("cuda_launch.zig");
const mtp = @import("cuda_mtp.zig");
const ops = @import("cuda_mtp_ops.zig");
const wt = @import("cuda_prompt_wt.zig");
const host_tier = @import("exl3/kv/cuda_exl3_kvhost.zig");
const kvt = @import("exl3/kv/cuda_exl3_kvtier.zig");

const B = flash.prompt.batch;
const Draft = mtp.Draft;
const page_rows = kvt.page_rows;
const head_pages = kvt.head_pages;
const none: u32 = std.math.maxInt(u32);

/// Pages a stream may grow to, and the frames its lease may pin at once: the first pages and a window of three.
const pages_n = 16;
const frames_n = head_pages + 3;
/// Bytes of one page's K/V plane in the fake store (the tier moves whole pages; their contents are not read here).
const kv_bytes = 16;
const cap = 4096;
/// Rows a verify window and a head's absorb take; the prompt ends inside page 10, past the dense keys.
const step = 8;
const prompt_rows = 10 * page_rows + 40;
const scores_ptr: u64 = 0xc000;
const page_table: flash.Pg = .{ .pt = 0x50000, .ps = 12 * 48 * 1184, .ptok = 48 };
const plan_fn = 21;

const Fake = struct {
    pub fn readoutTilesOwned(_: @This()) bool {
        return false;
    }
    pub fn fusedAttentionOwned(_: @This()) bool {
        return false;
    }
    pub fn attnPrep(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: flash.Pg) !void {}
    pub fn qsaPool(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: flash.Pg) !void {}
    pub fn qsaScores(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: flash.Pg) !void {}
    pub fn attnGate(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn attnChunks(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: flash.Pg) !void {}
    pub fn attnMerge(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn attnFused(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: flash.Pg) !void {}
    pub fn hcNormalize(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcDenseMix(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcDown(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcReduceAct(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcUpmixUp(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcUpmix(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcRinv(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
    pub fn hcMixH(_: @This(), _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) !void {}
};
const P = B.Pass(Fake);
const Layer = @typeInfo(@typeInfo(@TypeOf(P.attnBlock)).@"fn".param_types[1].?).pointer.child;

/// CUDA calls complete synchronously; assertions exercise real tier ownership, not recorded driver calls.
const Dev = struct {
    var held: [16]?[]align(16) u8 = @splat(null);

    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn copyUp(_: u64, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn copyDown(_: ?*anyopaque, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn memset(_: u64, _: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn hostDevice(out: *u64, _: ?*anyopaque, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = 0x7_0000_0000;
        return cuda.abi.success;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        const mem = std.heap.page_allocator.alignedAlloc(u8, .@"16", len) catch return cuda.abi.error_not_ready;
        for (&held) |*slot| {
            if (slot.* == null) {
                slot.* = mem;
                out.* = mem.ptr;
                return cuda.abi.success;
            }
        }
        std.heap.page_allocator.free(mem);
        return cuda.abi.error_not_ready;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) cuda.abi.Result {
        for (&held) |*slot| {
            if (slot.*) |mem| {
                if (@intFromPtr(mem.ptr) == @intFromPtr(ptr)) {
                    std.heap.page_allocator.free(mem);
                    slot.* = null;
                }
            }
        }
        return cuda.abi.success;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return cuda.abi.success;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn eventDone(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn streamSync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn ctxSync() callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuLaunchKernel = launch;
        d.api.cuMemcpyHtoDAsync_v2 = copyUp;
        d.api.cuMemcpyDtoHAsync_v2 = copyDown;
        d.api.cuMemsetD32Async = memset;
        d.api.cuMemHostGetDevicePointer_v2 = hostDevice;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventSynchronize = eventDone;
        d.api.cuEventDestroy_v2 = eventDone;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = ctxSync;
        return d;
    }
};

fn span(d: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
}

fn fun(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

/// One stream: its lease on the tier, its selection, the main model's cache views and the head's draft.
const Rig = struct {
    tl: kv.tier.Lease = .{},
    fids: std.ArrayList(u32) = .empty,
    sel: kvt.LaneSel,
    d: Draft,
    views: [wt.attn_n]cuda.DeviceBuffer,
    history: [2]i64 = .{ 0, 0 },
    stage: []align(16) u8,
    /// Rows the main model holds (the sequence's fed).
    fed: usize = 0,

    /// The main model's layers over the stream's tier selection (layer 0 of the selection's layers).
    fn mainSeq(r: *Rig) B.SeqState {
        return .{ .conv = &.{}, .state = &.{}, .kc = &r.views, .vc = &r.views, .ikc = &r.views, .pooled = &r.views, .tail = r.views[0], .history = &r.history, .cap = cap, .scores = scores_ptr, .pg = page_table, .sel = &r.sel, .sel_layer = 0 };
    }
};

const Ctx = struct {
    gpa: std.mem.Allocator,
    drv: *const cuda.Driver,
    stream: cuda.Stream,
    k: B.Kernels,
    s: B.Scratch,
    lw: Layer,
    kt: kvt.Kernels,
    host: host_tier.Host,
    backing: [2 * pages_n * kv_bytes]u8 align(16),

    fn create(gpa: std.mem.Allocator, drv: *const cuda.Driver) !*Ctx {
        const c = try gpa.create(Ctx);
        errdefer gpa.destroy(c);
        c.gpa = gpa;
        c.drv = drv;
        c.stream = .{ .d = drv, .handle = null };
        c.k = undefined;
        c.k.tiles.fact = fun(drv, 1);
        c.k.tiles.fact_fold = fun(drv, 2);
        c.k.lane = fun(drv, 3);
        c.k.select = fun(drv, 4);
        c.s = undefined;
        c.s.mixed = 0x100000000;
        c.s.xsm = 0x200000000;
        c.s.proj = 0x300000000;
        c.s.dummy = 0x400000000;
        c.s.native_part = 0;
        c.s.cap = cap;
        c.s.pos = 0x500000000;
        c.s.q = 0x600000000;
        c.s.iq = 0x700000000;
        c.s.ao = 0x800000000;
        c.s.inv = 0x900000000;
        c.s.qids = 0xa00000000;
        c.s.qnk = 0xb00000000;
        c.s.qspr = 0xc00000000;
        c.s.po = 0xd00000000;
        c.s.pm = 0xe00000000;
        c.s.pl = 0xf00000000;
        c.s.gated = 0x1100000000;
        c.s.gxs_attn = 0x1200000000;
        c.s.part = 0x1300000000;
        c.lw = undefined;
        c.lw.heads = .full;
        c.lw.proj.native = null;
        c.lw.proj.exl3 = null;
        c.lw.proj.w.ptr = 0x1000;
        c.lw.proj.s.ptr = 0x2000;
        c.lw.proj.b.ptr = 0x3000;
        c.lw.proj.k = 7680;
        c.lw.proj.n = 13952;
        c.lw.out.native = null;
        c.lw.out.exl3 = null;
        c.lw.out.w.ptr = 0x4000;
        c.lw.out.s.ptr = 0x5000;
        c.lw.out.b.ptr = 0x6000;
        c.lw.out.k = 6144;
        c.lw.out.n = 7680;
        c.lw.dense_cut = null;
        c.lw.q_scale.ptr = 0x7000;
        c.lw.k_scale.ptr = 0x8000;
        c.lw.i_scale.ptr = 0x9000;
        c.lw.ik_scale.ptr = 0xa000;
        c.kt = .{ .module = undefined, .plan = fun(drv, plan_fn), .fetch = fun(drv, 22), .remap = fun(drv, 23), .work = span(drv, 0x1400000000, 1 << 20) };
        c.host = .{ .tier = undefined, .store = .{ .d = drv, .bytes = &c.backing }, .kv_page = kv_bytes };
        try c.host.tier.init(gpa, c.host.store.bytes, 2 * frames_n, kv_bytes);
        return c;
    }

    fn destroy(c: *Ctx) void {
        c.host.tier.deinit();
        c.gpa.destroy(c);
    }

    /// A stream on the tier: its pages covered, none in a frame, its selection fresh (a new stream's).
    fn open(c: *Ctx, index: u64) !*Rig {
        const r = try c.gpa.create(Rig);
        errdefer c.gpa.destroy(r);
        r.* = .{ .sel = undefined, .d = undefined, .views = undefined, .stage = try c.gpa.alignedAlloc(u8, .@"16", 4 * (pages_n + 1)) };
        errdefer c.gpa.free(r.stage);
        try c.host.tier.reserve(&r.tl, pages_n, frames_n);
        errdefer c.host.tier.finish(&r.tl);
        try c.host.tier.cover(&r.tl, pages_n, null);
        errdefer r.fids.deinit(c.gpa);
        try r.fids.appendNTimes(c.gpa, none, pages_n);
        const base: u64 = 0x2000000000 + index * 0x1000000000;
        const table = span(c.drv, base, 4 * (pages_n + 1));
        r.sel = .{ .gpa = c.gpa, .d = c.drv, .k = &c.kt, .dev = span(c.drv, base + 0x100000000, 0), .stage = .{ .d = c.drv, .bytes = r.stage }, .itable = 0, .table = table.ptr, .tbuf = table, .host = &c.host, .arena = span(c.drv, base + 0x200000000, 2 * frames_n * kv_bytes), .tl = &r.tl, .fids = &r.fids, .kc = &.{}, .vc = &.{}, .nblocks = cap / 4, .nslots = 64, .npages = pages_n, .layers = wt.attn_n + 1 };
        for (&r.views, 0..) |*v, i| v.* = span(c.drv, base + 0x300000000 + i * 0x10000, 0);
        r.d.kc = span(c.drv, base + 0x400000000, 0);
        r.d.vc = span(c.drv, base + 0x400010000, 0);
        r.d.ikc = span(c.drv, base + 0x400020000, 0);
        r.d.pooled = span(c.drv, base + 0x400030000, 0);
        r.d.pg = page_table;
        r.d.cap = cap;
        r.d.len = 0;
        r.d.drafted = 0;
        r.d.sel = &r.sel;
        return r;
    }

    fn close(c: *Ctx, r: *Rig) void {
        c.host.tier.finish(&r.tl);
        r.fids.deinit(c.gpa);
        c.gpa.free(r.stage);
        c.gpa.destroy(r);
    }

    fn pass(c: *Ctx, seq: B.SeqState) P {
        return .{ .gpa = c.gpa, .io = undefined, .stream = c.stream, .tri = .{}, .k = &c.k, .s = &c.s, .m = undefined, .seq = seq, .hooks = undefined };
    }

    /// The main model's `n` rows from the stream's fed, each attention layer through its selection.
    fn verify(c: *Ctx, r: *Rig, n: usize) !void {
        var p = c.pass(r.mainSeq());
        p.rows = n;
        p.pos0 = r.fed;
        for (0..wt.attn_n) |ai| try p.attnBlock(&c.lw, ai);
        r.fed += n;
    }

    /// The lone head's pass (Ops.forwardHead's): the draft's own sequence view.
    fn headAlone(c: *Ctx, r: *Rig, n: usize) !void {
        var p = c.pass(r.d.headSeq(null, scores_ptr));
        p.rows = n;
        p.pos0 = r.d.len;
        try p.attnBlock(&c.lw, 0);
        r.d.len += n;
    }

    /// The grouped head's pass over `n` rows of each stream: Ops.headGroup's lanes (headLane) and its length step.
    fn headGroup(c: *Ctx, rigs: []const *Rig, n: usize) !void {
        var lanes: [2]B.Lane = undefined;
        for (rigs, 0..) |r, i| lanes[i] = ops.headLane(&r.d, n, scores_ptr);
        var p = c.pass(lanes[0].seq);
        p.lanes = lanes[0..rigs.len];
        p.rows = n * rigs.len;
        p.pos0 = 0;
        try p.attnBlock(&c.lw, 0);
        for (rigs) |r| r.d.len += n;
    }

    /// Written head rows remain in frames pinned by their stream and named by its table.
    fn expectHeadWrote(c: *Ctx, r: *Rig, n: usize) !void {
        const t = std.testing;
        try t.expectEqual(r.d.len, r.sel.done[wt.attn_n]);
        try t.expect(r.sel.head_on);
        var p = (r.d.len - n) / page_rows;
        while (p < (r.d.len + page_rows - 1) / page_rows) : (p += 1) {
            try t.expect(r.fids.items[p] != none);
            const ticket = c.host.tier.resident(r.tl.pages.ids.items[p]) orelse return error.PageNotInFrame;
            try t.expectEqual(r.fids.items[p], ticket.frame);
            try t.expect(c.host.tier.pinned[ticket.frame] > 0);
        }
        // the first pages and a window only: every page the stream and its head have left behind is out of its frames
        try t.expect(r.tl.pins.items.len <= head_pages + 2);
    }

    /// One decode round of every stream: the main window, then the grouped head absorbs it and chains one draft.
    fn round(c: *Ctx, rigs: []const *Rig) !void {
        for (rigs) |r| try c.verify(r, step);
        for (rigs) |r| {
            r.d.len -= r.d.drafted;
            r.d.drafted = 0;
        }
        try c.headGroup(rigs, step);
        for (rigs) |r| try c.expectHeadWrote(r, step);
        try c.headGroup(rigs, 1);
        for (rigs) |r| r.d.drafted += 1;
        for (rigs) |r| try c.expectHeadWrote(r, 1);
    }

    /// Restore the tier transition: release every resident frame before rewinding the stream.
    fn restore(r: *Rig, rows: usize) !void {
        try r.sel.checkRestore();
        try r.sel.restored();
        r.fed = rows;
        r.d.len = rows;
        r.d.drafted = 0;
    }
};

test "exl3: grouped head lanes release host-tier frames across rounds and tier restore" {
    var drv = Dev.driver();
    defer cuda.copy_source.synchronizeContext(&drv) catch {};
    const c = try Ctx.create(std.testing.allocator, &drv);
    defer c.destroy();
    const a = try c.open(0);
    defer c.close(a);
    const b = try c.open(1);
    defer c.close(b);
    const rigs = [_]*Rig{ a, b };

    // each prompt: the main model's rows, then the head's (the lone pass), all inside frames the lease holds
    for (rigs) |r| {
        try c.verify(r, prompt_rows);
        try c.headAlone(r, prompt_rows);
        try std.testing.expectEqual(prompt_rows, r.sel.done[wt.attn_n]);
    }

    // decode: far more pages than the lease may pin pass; the grouped head must let each go once it has written it
    for (0..120) |_| try c.round(&rigs);

    // Rewind both streams through the real tier restore transition, then reacquire frames through grouped rounds.
    const rewound = a.fed - 700;
    for (rigs) |r| {
        try Ctx.restore(r, rewound);
        try std.testing.expectEqual(@as(usize, 0), r.tl.pins.items.len);
    }
    for (0..30) |_| try c.round(&rigs);
}
