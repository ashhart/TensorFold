//! Recording CUDA calls around the production MTP pick adapter; forward arithmetic is not emulated.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const snap = @import("glm_snap_driver.zig");
const t = std.testing;
extern fn glm_mtp_pick_host([*]const u32, u32) u32;

const State = struct {
    pos: usize = 200,
    mtp_len: usize = 101,
    pub fn setMtpLen(s: *State, _: cuda.Stream, n: usize) !void {
        s.mtp_len = n;
        Fixture.active.rebases += 1;
    }
};
const Cur = struct { st: *State, in_head: usize = 0, sampling: ?u32 = null, more: []State = &.{} };
const Group = struct {
    f: *Fixture,
    pub fn allGather(g: *const Group, _: cuda.Stream, mine: u64, dst: u64, bytes: usize) !void {
        const f = g.f;
        try t.expectEqual(@as(usize, 16), bytes);
        try t.expectEqual(dst + f.e.share.rank * 16, mine);
        const words = f.words();
        try t.expectEqualSlices(u32, &words[f.e.share.rank], ptr(mine)[0..4]);
        @memcpy(ptr(dst)[0..8], @as(*const [8]u32, @ptrCast(&words)));
        f.gathers += 1;
        if (f.async_at == f.cur.in_head) f.pending_failure = true;
    }
};

const Engine = struct {
    f: *Fixture,
    d: *const cuda.Driver,
    settings: glm.ranks.Settings = .{},
    c: glm.config.Config = .{},
    local: []const u8 = &.{},
    structured: bool = false,
    share: struct { world: usize = 2, rank: usize = 0, vocab: glm.shard.Span = .{ .lo = 0, .hi = 32 } } = .{},
    pairs: cuda.DeviceBuffer,
    bm: struct { logits: u64 = 9000, fnormed: u64 = 9100, ids: u64 },
    mtp_pick: ?cuda.Function,
    m: struct { glue: struct { top1: cuda.Function, top1_sums: cuda.Function } },
    stream: cuda.Stream,
    group: ?*Group,
    lib: *const cuda.nccl.Library = undefined,
    comm: cuda.nccl.Comm = undefined,
    profiling_prompt: @FieldType(glm.engine.Engine, "profiling_prompt") = null,
    cur: *Cur,

    pub fn checkGathers(e: *Engine) !void {
        e.f.checks += 1;
        if (e.f.fail_check or e.f.proxy_failed) return error.GatherFailed;
    }
    pub fn greedyChance(e: *Engine, _: u64) !glm.mtp_chance.Pick {
        return glm.engine_pick.pick(e, e.bm.logits, true);
    }
    pub fn draw(e: *Engine, _: u64, rows: usize, pos: usize, _: u64, out: []u32) !void {
        try t.expectEqual(@as(usize, 1), rows);
        try t.expectEqual(e.cur.st.pos + 1 + e.cur.in_head - e.f.start_head, pos);
        if (e.f.async_at != null) {
            out[0] = (try glm.engine_pick.pick(e, e.bm.logits, false)).token;
            return;
        }
        e.f.syncs += 1;
        e.f.reads += 1;
        out[0] = glm.mtp_chance.pick(&e.f.words(), false).token;
    }
};

const Forward = struct {
    f: *Fixture,
    fn consume(f: *const Forward, id: u32, hidden: []const u64) !void {
        try t.expectEqualSlices(u64, &.{f.f.e.bm.fnormed}, hidden);
        const words = f.f.words();
        try t.expectEqual(glm.mtp_chance.pick(&words, false).token, id);
        f.f.previous = id;
        f.f.cache[f.f.cur.in_head] = id;
        f.f.forwards += 1;
    }
    pub fn mtp(f: *const Forward, ids: []const i32, hidden: []const u64) !void {
        f.f.uploads += 1;
        f.f.id = @intCast(ids[0]);
        try f.consume(@intCast(ids[0]), hidden);
    }
    pub fn mtpDevice(f: *const Forward, id: u64, hidden: []const u64) !void {
        try t.expectEqual(f.f.e.bm.ids, id);
        try f.consume(ptr(id)[0], hidden);
    }
};

pub const Fixture = struct {
    var active: *Fixture = undefined;
    d: cuda.Driver = undefined,
    e: Engine = undefined,
    fwd: Forward = undefined,
    group: Group = undefined,
    state: State = .{},
    cur: Cur = undefined,
    storage: [272]u32 = @splat(0xabcdef01),
    id: u32 = 0,
    previous: u32 = 17,
    syncs: usize = 0,
    reads: usize = 0,
    checks: usize = 0,
    gathers: usize = 0,
    picks: usize = 0,
    forwards: usize = 0,
    start_head: usize = 0,
    top1s: usize = 0,
    rebases: usize = 0,
    cache: [15]u32 = @splat(0xdeadbeef),
    async_at: ?usize = null,
    pending_failure: bool = false,
    proxy_failed: bool = false,
    uploads: usize = 0,
    fail_check: bool = false,
    fail_read: bool = false,
    integrated: bool = true,
    pinned: [64 << 10]u8 align(16) = undefined,

    pub fn init(f: *Fixture, rank: usize, on: bool) void {
        active = f;
        f.d = snap.Driver.init();
        f.d.api.cuLaunchKernel = launch;
        f.d.api.cuLaunchKernelEx = glm.pdl.forwardEx(launch);
        f.d.api.cuStreamSynchronize = sync;
        f.d.api.cuMemcpyDtoH_v2 = download;
        f.d.api.cuCtxGetDevice = device;
        f.d.api.cuDeviceGetAttribute = attribute;
        f.d.api.cuMemHostAlloc = hostAlloc;
        f.d.api.cuMemFreeHost = hostFree;
        f.cur = .{ .st = &f.state };
        f.fwd = .{ .f = f };
        f.group = .{ .f = f };
        f.e = .{ .f = f, .d = &f.d, .cur = &f.cur, .pairs = .{ .d = &f.d, .ptr = @intFromPtr(&f.storage[8]), .len = 256 * 4 }, .bm = .{ .ids = @intFromPtr(&f.id) }, .mtp_pick = fun(f, 3), .m = .{ .glue = .{ .top1 = fun(f, 1), .top1_sums = fun(f, 2) } }, .stream = .{ .d = &f.d, .handle = @ptrFromInt(99) }, .group = &f.group };
        f.e.share.rank = rank;
        f.e.share.vocab = .{ .lo = rank * 32768, .hi = (rank + 1) * 32768 };
        f.e.settings.mtp_chain = on;
    }

    pub fn deinit(f: *Fixture) void {
        cuda.download_stage.closeDriver(&f.d);
    }

    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, which: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        if (which != .integrated) return 1;
        out.* = @intFromBool(active.integrated);
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, bytes: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        if (bytes > active.pinned.len) return 2;
        out.* = &active.pinned;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return 0;
    }

    fn fun(f: *Fixture, n: usize) cuda.Function {
        return .{ .d = &f.d, .handle = @ptrFromInt(n) };
    }

    fn words(f: *const Fixture) [2][4]u32 {
        const j = f.cur.in_head;
        const x: f32 = @floatFromInt((j * 7 + f.previous) % 31);
        return .{ .{ @bitCast(x), @intCast(j * 43 + 2), @bitCast(@as(f32, 1.25)), 0 }, .{ @bitCast(x + (if (j % 2 == 0) @as(f32, 1) else 0)), @intCast(32768 + j * 43 + 1), @bitCast(@as(f32, 2.5)), 0 } };
    }

    pub fn run(f: *Fixture, depth: usize, out: []u32, chances: ?[]f64) !void {
        active = f;
        f.start_head = f.cur.in_head;
        var ops = glm.mtp_loop.native(&f.e, &f.fwd);
        try glm.mtp_loop.run(&ops, depth, out, chances);
        try t.expectEqualSlices(u32, &@as([8]u32, @splat(0xabcdef01)), f.storage[0..8]);
        try t.expectEqualSlices(u32, &@as([8]u32, @splat(0xabcdef01)), f.storage[264..]);
    }

    fn launch(func: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const f = active;
        if (gx != 1) return 1;
        switch (@intFromPtr(func)) {
            1, 2 => {
                f.top1s += 1;
                if (bx != 1024 or arg(i64, params, 1) != 32768 or arg(u32, params, 2) != f.e.share.vocab.lo) return 1;
                const mine = arg(u64, params, 3);
                const want = f.words()[f.e.share.rank];
                @memcpy(ptr(mine)[0..4], &want);
            },
            3 => {
                if (bx != 32 or arg(u32, params, 1) != 2) return 1;
                const pair_ptr = arg(u64, params, 0);
                ptr(arg(u64, params, 2))[0] = glm_mtp_pick_host(ptr(pair_ptr), 2);
                f.picks += 1;
            },
            else => return 1,
        }
        return 0;
    }

    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        active.syncs += 1;
        if (active.pending_failure) active.proxy_failed = true;
        active.pending_failure = false;
        return 0;
    }

    fn download(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        active.reads += 1;
        if (active.fail_read) return cuda.abi.error_not_ready;
        const bytes: [*]u8 = @ptrCast(dst.?);
        @memcpy(bytes[0..n], std.mem.sliceAsBytes(ptr(src)[0 .. n / 4]));
        return 0;
    }
};

fn ptr(p: u64) [*]u32 {
    return @ptrFromInt(p);
}

fn arg(comptime T: type, p: ?[*]?*anyopaque, i: usize) T {
    return @as(*align(1) const T, @ptrCast(p.?[i].?)).*;
}
