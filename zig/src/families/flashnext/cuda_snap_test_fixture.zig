const std = @import("std");
const cuda = @import("cuda");
const pager = @import("exl3/kv/cuda_exl3_kvpage.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const shared_fixture = @import("exl3/offload/cuda_exl3_shared_kv_test.zig");

pub const Fake = struct {
    const Copy = struct { dst: u64, src: u64, len: usize };
    var queued: [512]Copy = undefined;
    pub var pending: usize = 0;
    pub var submitted: usize = 0;
    pub var fail_at: ?usize = null;
    pub var sync_calls: usize = 0;
    pub var fail_sync_at: ?usize = null;
    pub var fail_alloc: bool = false;
    pub var fail_mapping: bool = false;
    pub var allocations: [8]?[]align(16) u8 = @splat(null);
    pub var freed_pending: bool = false;

    pub fn reset() void {
        pending = 0;
        submitted = 0;
        fail_at = null;
        sync_calls = 0;
        fail_sync_at = null;
        fail_alloc = false;
        fail_mapping = false;
        freed_pending = false;
    }

    fn copy(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        submitted += 1;
        if (fail_at == submitted) return cuda.abi.error_not_ready;
        queued[pending] = .{ .dst = dst, .src = src, .len = len };
        pending += 1;
        return 0;
    }

    fn read(dst: ?*anyopaque, src: u64, len: usize, stream: cuda.abi.Stream) callconv(.c) c_int {
        return copy(@intFromPtr(dst), src, len, stream);
    }

    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        sync_calls += 1;
        if (fail_sync_at == sync_calls) return cuda.abi.error_not_ready;
        for (queued[0..pending]) |item| {
            const dst: [*]u8 = @ptrFromInt(item.dst);
            const src: [*]const u8 = @ptrFromInt(item.src);
            @memcpy(dst[0..item.len], src[0..item.len]);
        }
        pending = 0;
        return 0;
    }

    fn contextSync() callconv(.c) c_int {
        return sync(null);
    }

    fn alloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) c_int {
        if (fail_alloc) return cuda.abi.error_not_ready;
        for (&allocations) |*slot| if (slot.* == null) {
            const bytes = std.testing.allocator.alignedAlloc(u8, .@"16", len) catch return 2;
            slot.* = bytes;
            out.* = bytes.ptr;
            return 0;
        };
        return 2;
    }

    fn device(out: *u64, address: ?*anyopaque, _: c_uint) callconv(.c) c_int {
        if (fail_mapping) return cuda.abi.error_not_ready;
        out.* = @intFromPtr(address);
        return 0;
    }

    fn free(address: ?*anyopaque) callconv(.c) c_int {
        freed_pending = freed_pending or pending != 0;
        for (&allocations) |*slot| if (slot.*) |bytes| if (@intFromPtr(bytes.ptr) == @intFromPtr(address)) {
            std.testing.allocator.free(bytes);
            slot.* = null;
            return 0;
        };
        return 1;
    }

    pub fn driver() cuda.Driver {
        var driver_ = shared_fixture.Fake.driver();
        driver_.api.cuMemcpyDtoDAsync_v2 = copy;
        driver_.api.cuMemcpyDtoHAsync_v2 = read;
        driver_.api.cuStreamSynchronize = sync;
        driver_.api.cuCtxSynchronize = contextSync;
        driver_.api.cuMemHostAlloc = alloc;
        driver_.api.cuMemHostGetDevicePointer_v2 = device;
        driver_.api.cuMemFreeHost = free;
        return driver_;
    }
};

pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    pg: pager.Paged,
    seq: prompt.Seq,

    pub fn buffer(allocator: std.mem.Allocator, driver: *const cuda.Driver, len: usize, value: u8) !cuda.DeviceBuffer {
        const bytes = try allocator.alignedAlloc(u8, .@"16", len);
        @memset(bytes, value);
        return .{ .d = driver, .ptr = @intFromPtr(bytes.ptr), .len = len, .borrowed = true };
    }

    fn buffers(allocator: std.mem.Allocator, driver: *const cuda.Driver, count: usize, len: usize) ![]cuda.DeviceBuffer {
        const out = try allocator.alloc(cuda.DeviceBuffer, count);
        for (out, 0..) |*item, index| item.* = try buffer(allocator, driver, len, @intCast(index));
        return out;
    }

    pub fn init(driver: *const cuda.Driver) !*Fixture {
        return initCap(driver, 8192);
    }

    pub fn initCap(driver: *const cuda.Driver, cap: usize) !*Fixture {
        const fixture = try std.testing.allocator.create(Fixture);
        errdefer std.testing.allocator.destroy(fixture);
        fixture.arena = .init(std.testing.allocator);
        errdefer fixture.arena.deinit();
        const allocator = fixture.arena.allocator();
        const cfg: pager.Config = .{ .window = 2052, .rows = 256, .slots = 16 };
        const layout = try pager.SnapshotLayout.init(cap, cfg, 0);
        const host = try allocator.alignedAlloc(u8, .@"16", 24 * cap * pager.row_bytes);
        @memset(host, 0x80);
        const status = try allocator.alignedAlloc(u8, .@"16", 12 * 8 * 4);
        fixture.pg = .{ .gpa = allocator, .d = driver, .k = undefined, .cfg = cfg, .cap = cap, .wr = layout.wr, .nblocks = layout.nblocks, .host = .{ .d = driver, .bytes = host }, .host_dev = @intFromPtr(host.ptr), .status_host = .{ .d = driver, .bytes = status }, .dev = try buffer(allocator, driver, layout.device, 0xFF), .tabs = try buffer(allocator, driver, layout.tables, 0), .work = undefined };
        fixture.seq = .{
            .cap = cap,
            .fed = 0,
            .history = .{ 73, 91 },
            .paged = &fixture.pg,
            .conv = try buffers(allocator, driver, wt.linear_n, (gdn.taps - 1) * gdn.conv_dim * 2),
            .state = try buffers(allocator, driver, wt.linear_n, wt.state_n * 4),
            .kc = &.{},
            .vc = &.{},
            .ikc = try buffers(allocator, driver, wt.attn_n, cap * wt.index_dim * 2),
            .pooled = try buffers(allocator, driver, wt.attn_n, prompt.pooledBytes(cap)),
            .tail = try buffer(allocator, driver, wt.ple_tail_n * wt.wide * 2, 0x7F),
            .logits = try allocator.alloc(u8, prompt.vocabulary * 2),
        };
        @memset(fixture.seq.logits, 0x81);
        return fixture;
    }

    pub fn deinit(fixture: *Fixture) void {
        if (fixture.pg.cfg.shared_lanes != 0) fixture.pg.shared_arena.?.release(fixture.pg.shared_sequence, .{ .d = fixture.pg.d, .handle = null }) catch unreachable;
        fixture.arena.deinit();
        std.testing.allocator.destroy(fixture);
    }
};

pub fn sharedFixture(driver: *const cuda.Driver, arena: *pager.shared_device.Arena) !*Fixture {
    const fixture = try Fixture.initCap(driver, arena.layout.options.cap);
    errdefer fixture.deinit();
    const sequence = try arena.register(arena.layout.options.cap);
    fixture.pg.cfg.shared_lanes = arena.layout.options.lanes;
    fixture.pg.shared_arena = arena;
    fixture.pg.shared_sequence = sequence;
    fixture.pg.dev = arena.device;
    fixture.pg.dev.borrowed = true;
    fixture.pg.tabs = arena.tables;
    fixture.pg.tabs.borrowed = true;
    fixture.pg.work = arena.work;
    fixture.pg.work.borrowed = true;
    return fixture;
}
