const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const contract = @import("exl3_shared_kv.zig");
const parts = @import("cuda_batch_state.zig");

const row_bytes = wt.kv_heads * wt.head_dim * 2;
const ratio = wt.index_ratio;
const controls_words = 8;
const work_entries = parts.att_rows * wt.index_top;
pub const max_lanes = 32;

pub const Options = struct { cap: usize, lanes: usize, window: usize, rows: usize, slots: usize };

pub const Layout = struct {
    options: Options,
    window_rows: usize,
    blocks: usize,
    lane_words: usize,
    all_words: usize,
    side_rows: usize,
    device_bytes: usize,
    marks: usize,
    controls: usize,
    owners: usize,
    stamps: usize,
    owner_generations: usize,
    lane_generations: usize,
    clock: usize,
    table_bytes: usize,
    work_bytes: usize,
    common_bytes: usize,

    pub fn init(options: Options) !Layout {
        if (options.cap == 0 or options.cap > wt.window or options.lanes == 0 or options.lanes > max_lanes or options.window < wt.dense_keys or options.rows == 0 or options.slots == 0) return error.SharedKvGeometry;
        const window_rows = try std.math.add(usize, try std.math.add(usize, options.window, options.rows), ratio);
        const blocks = (try std.math.add(usize, options.cap, ratio - 1)) / ratio + 1;
        const lane_words = try std.math.mul(usize, wt.attn_n, blocks);
        const all_words = try std.math.mul(usize, options.lanes, lane_words);
        const window_count = try std.math.mul(usize, options.lanes, wt.attn_n);
        const side_rows = try std.math.add(usize, try std.math.mul(usize, window_count, window_rows), try std.math.mul(usize, ratio, options.slots));
        if (all_words > std.math.maxInt(c_int) or side_rows > std.math.maxInt(c_int) or options.slots > std.math.maxInt(c_int)) return error.SharedKvGeometry;
        const device_bytes = try std.math.mul(usize, side_rows, 2 * row_bytes);
        const marks = try std.math.mul(usize, all_words, 4);
        const controls = try std.math.mul(usize, marks, 2);
        const owners = try std.math.add(usize, controls, try std.math.mul(usize, window_count, controls_words * 4));
        const stamps = try std.math.add(usize, owners, try std.math.mul(usize, options.slots, 4));
        const after_stamps = try std.math.add(usize, stamps, try std.math.mul(usize, options.slots, 4));
        const owner_generations = try std.math.add(usize, after_stamps, (8 - after_stamps % 8) % 8);
        const lane_generations = try std.math.add(usize, owner_generations, try std.math.mul(usize, options.slots, 8));
        const clock = try std.math.add(usize, lane_generations, try std.math.mul(usize, options.lanes, 8));
        const table_bytes = try std.math.add(usize, clock, 4);
        const work_bytes = work_entries * 4 * 4;
        return .{ .options = options, .window_rows = window_rows, .blocks = blocks, .lane_words = lane_words, .all_words = all_words, .side_rows = side_rows, .device_bytes = device_bytes, .marks = marks, .controls = controls, .owners = owners, .stamps = stamps, .owner_generations = owner_generations, .lane_generations = lane_generations, .clock = clock, .table_bytes = table_bytes, .work_bytes = work_bytes, .common_bytes = try std.math.add(usize, device_bytes, try std.math.add(usize, table_bytes, work_bytes)) };
    }

    pub fn windowOffset(layout: Layout, lane: usize, layer: usize) !usize {
        if (lane >= layout.options.lanes or layer >= wt.attn_n) return error.SharedKvKey;
        return (lane * wt.attn_n + layer) * layout.window_rows;
    }

    pub fn slotBase(layout: Layout, lane: usize, layer: usize) !usize {
        return layout.options.lanes * wt.attn_n * layout.window_rows - try layout.windowOffset(lane, layer);
    }
};

const Lane = struct { generation: u64 = 0, live: bool = false, failed: bool = false };
pub const Lease = struct { arena: u64, serial: u64, sequence: contract.Sequence };
const Active = struct {
    lease: Lease,
    owners: [max_lanes]contract.Sequence = undefined,
    count: usize,
    touched: [max_lanes]bool = @splat(false),
    stream_bound: bool = false,
    stream: cuda.abi.Stream = null,
    submitted: bool = false,
    failed: bool = false,
};
pub const View = struct { keys: u64, values: u64, inverse: u64, marks: u64, controls: u64, slot_base: usize };
pub const Kernels = struct { plan: cuda.Function, fetch: cuda.Function, remap: cuda.Function, release: cuda.Function };

pub const Arena = struct {
    allocator: std.mem.Allocator,
    driver: *const cuda.Driver,
    layout: Layout,
    id: u64,
    device: cuda.DeviceBuffer,
    tables: cuda.DeviceBuffer,
    work: cuda.DeviceBuffer,
    metadata: []align(8) u8,
    lanes: [max_lanes]Lane = @splat(.{}),
    serial: u64 = 0,
    active: ?Active = null,
    module: ?cuda.Module = null,
    kernels: ?Kernels = null,

    pub fn init(allocator: std.mem.Allocator, driver: *const cuda.Driver, options: Options) !*Arena {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const arena = try initStorage(allocator, driver, options);
        errdefer arena.destroy() catch unreachable;
        var module = try cuda.Module.load(driver, cuda.kernels.flashnext_exl3_kvpage);
        errdefer module.unload();
        arena.kernels = .{ .plan = try module.function("kvp_shared_plan"), .fetch = try module.function("kvp_fetch"), .remap = try module.function("kvp_remap"), .release = try module.function("kvp_shared_release") };
        arena.module = module;
        return arena;
    }

    pub fn initStorage(allocator: std.mem.Allocator, driver: *const cuda.Driver, options: Options) !*Arena {
        const layout = try Layout.init(options);
        const arena = try allocator.create(Arena);
        errdefer allocator.destroy(arena);
        var device = try cuda.DeviceBuffer.alloc(driver, layout.device_bytes);
        errdefer device.free();
        var tables = try cuda.DeviceBuffer.alloc(driver, layout.table_bytes);
        errdefer tables.free();
        var work = try cuda.DeviceBuffer.alloc(driver, layout.work_bytes);
        errdefer work.free();
        const metadata = try allocator.alignedAlloc(u8, .@"8", layout.table_bytes);
        errdefer allocator.free(metadata);
        @memset(metadata, 0);
        @memset(metadata[0..layout.marks], 0xff);
        @memset(metadata[layout.owners..layout.stamps], 0xff);
        errdefer cuda.completion.drain(driver, .{ .d = driver, .handle = null }, "shared KV initialization did not drain");
        try driver.check(driver.api.cuMemcpyHtoD_v2(tables.ptr, metadata.ptr, metadata.len), "cuMemcpyHtoD(shared KV tables)");
        try driver.check(driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV initialization)");
        arena.* = .{ .allocator = allocator, .driver = driver, .layout = layout, .id = try contract.identity(), .device = device, .tables = tables, .work = work, .metadata = metadata };
        return arena;
    }

    pub fn destroy(arena: *Arena) !void {
        if (arena.active != null) return error.SharedKvBusy;
        for (arena.lanes) |lane| if (lane.live) return error.SharedKvSequenceLive;
        if (arena.module) |*module| module.unload();
        arena.allocator.free(arena.metadata);
        arena.work.free();
        arena.tables.free();
        arena.device.free();
        arena.allocator.destroy(arena);
    }

    fn state(arena: *Arena, sequence: contract.Sequence) !*Lane {
        if (sequence.arena != arena.id or sequence.lane >= arena.layout.options.lanes) return error.SharedKvSequence;
        const lane = &arena.lanes[sequence.lane];
        if (!lane.live or lane.generation != sequence.generation or sequence.generation == 0) return error.SharedKvSequence;
        return lane;
    }

    pub fn register(arena: *Arena, cap: usize) !contract.Sequence {
        if (arena.active != null) return error.SharedKvBusy;
        if (cap != arena.layout.options.cap) return error.SharedKvCommonCap;
        for (arena.lanes[0..arena.layout.options.lanes], 0..) |*lane, index| {
            if (lane.live) continue;
            const next_generation = try std.math.add(u64, lane.generation, 1);
            errdefer cuda.completion.drain(arena.driver, .{ .d = arena.driver, .handle = null }, "shared KV registration did not drain");
            const lane_rows = wt.attn_n * arena.layout.window_rows;
            for (0..2) |side| {
                const start = arena.device.ptr + (side * arena.layout.side_rows + index * lane_rows) * row_bytes;
                try arena.driver.check(arena.driver.api.cuMemsetD8_v2(start, 0, lane_rows * row_bytes), "cuMemsetD8(shared KV window)");
            }
            const inverse = arena.tables.ptr + index * arena.layout.lane_words * 4;
            try arena.driver.check(arena.driver.api.cuMemsetD32_v2(inverse, 0xffffffff, arena.layout.lane_words), "cuMemsetD32(shared KV inverse)");
            try arena.driver.check(arena.driver.api.cuMemsetD32_v2(arena.tables.ptr + arena.layout.marks + index * arena.layout.lane_words * 4, 0, arena.layout.lane_words), "cuMemsetD32(shared KV marks)");
            try arena.driver.check(arena.driver.api.cuMemsetD32_v2(arena.tables.ptr + arena.layout.controls + index * wt.attn_n * controls_words * 4, 0, wt.attn_n * controls_words), "cuMemsetD32(shared KV controls)");
            try arena.driver.check(arena.driver.api.cuMemcpyHtoD_v2(arena.tables.ptr + arena.layout.lane_generations + index * 8, &next_generation, 8), "cuMemcpyHtoD(shared KV generation)");
            try arena.driver.check(arena.driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV registration)");
            lane.* = .{ .live = true, .generation = next_generation };
            return .{ .arena = arena.id, .lane = @intCast(index), .generation = next_generation };
        }
        return error.SharedKvLanes;
    }

    pub fn acquire(arena: *Arena, sequence: contract.Sequence) !Lease {
        return arena.acquireMany(&.{sequence});
    }

    pub fn acquireMany(arena: *Arena, sequences: []const contract.Sequence) !Lease {
        if (arena.active != null) return error.SharedKvBusy;
        if (sequences.len == 0 or sequences.len > arena.layout.options.lanes) return error.SharedKvLeaseOwners;
        for (sequences, 0..) |sequence, index| {
            const lane = try arena.state(sequence);
            if (lane.failed) return error.SharedKvSequenceFailed;
            for (sequences[0..index]) |previous| if (previous.lane == sequence.lane) return error.SharedKvLeaseOwners;
        }
        const serial = try std.math.add(u64, arena.serial, 1);
        const lease: Lease = .{ .arena = arena.id, .serial = serial, .sequence = sequences[0] };
        arena.serial = serial;
        arena.active = .{ .lease = lease, .count = sequences.len };
        @memcpy(arena.active.?.owners[0..sequences.len], sequences);
        return lease;
    }

    fn held(arena: *Arena, lease: Lease) !*Active {
        const active = if (arena.active) |*active| active else return error.SharedKvLease;
        if (!std.meta.eql(active.lease, lease)) return error.SharedKvLease;
        if (active.failed) return error.SharedKvPoisoned;
        return active;
    }

    pub fn submitted(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        active.submitted = true;
        for (active.owners[0..active.count]) |sequence| active.touched[sequence.lane] = true;
    }

    pub fn bindStream(arena: *Arena, lease: Lease, stream: cuda.Stream) !void {
        const active = try arena.held(lease);
        if (stream.d != arena.driver) return error.SharedKvDriver;
        if (active.stream_bound and active.stream != stream.handle) return error.SharedKvStream;
        active.stream_bound = true;
        active.stream = stream.handle;
    }

    pub fn submittedOn(arena: *Arena, lease: Lease, stream: cuda.Stream) !void {
        try arena.bindStream(lease, stream);
        try arena.submitted(lease);
    }

    fn member(arena: *Arena, lease: Lease, sequence: contract.Sequence) !*Active {
        const active = try arena.held(lease);
        _ = try arena.state(sequence);
        for (active.owners[0..active.count]) |owner| if (std.meta.eql(owner, sequence)) return active;
        return error.SharedKvLeaseOwner;
    }

    fn touch(arena: *Arena, lease: Lease, sequence: contract.Sequence) !void {
        const active = try arena.member(lease, sequence);
        active.submitted = true;
        active.touched[sequence.lane] = true;
    }

    pub fn leaseOwners(arena: *Arena, lease: Lease) ![]const contract.Sequence {
        const active = try arena.held(lease);
        return active.owners[0..active.count];
    }

    pub fn leaseFor(arena: *Arena, sequence: contract.Sequence) !Lease {
        const active = arena.active orelse return error.SharedKvLease;
        _ = try arena.member(active.lease, sequence);
        return active.lease;
    }

    pub fn clear(arena: *Arena, lease: Lease, sequence: contract.Sequence, stream: cuda.Stream) !void {
        _ = try arena.member(lease, sequence);
        try arena.bindStream(lease, stream);
        const kernels = arena.kernels orelse return error.BuiltWithoutKernels;
        var args: cuda.Args = .{};
        for ([_]u64{ arena.tables.ptr, arena.tables.ptr + arena.layout.owners, arena.tables.ptr + arena.layout.stamps, arena.tables.ptr + arena.layout.owner_generations }) |address| args.add(address);
        for ([_]usize{ arena.layout.options.slots, @as(usize, sequence.lane) * arena.layout.lane_words, (@as(usize, sequence.lane) + 1) * arena.layout.lane_words }) |value| args.add(@as(c_int, @intCast(value)));
        args.add(sequence.generation);
        try arena.touch(lease, sequence);
        errdefer arena.fail(lease) catch unreachable;
        try cuda.launch.launch(kernels.release, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &args);
        const inverse = arena.tables.ptr + @as(usize, sequence.lane) * arena.layout.lane_words * 4;
        try arena.driver.check(arena.driver.api.cuMemsetD32Async(inverse, 0xffffffff, arena.layout.lane_words, stream.handle), "cuMemsetD32Async(shared KV inverse)");
        try arena.driver.check(arena.driver.api.cuMemsetD32Async(arena.tables.ptr + arena.layout.marks + @as(usize, sequence.lane) * arena.layout.lane_words * 4, 0, arena.layout.lane_words, stream.handle), "cuMemsetD32Async(shared KV marks)");
        try arena.driver.check(arena.driver.api.cuMemsetD32Async(arena.tables.ptr + arena.layout.controls + @as(usize, sequence.lane) * wt.attn_n * controls_words * 4, 0, wt.attn_n * controls_words, stream.handle), "cuMemsetD32Async(shared KV controls)");
    }

    pub fn cancel(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        if (active.submitted) return error.SharedKvPending;
        arena.active = null;
    }

    pub fn fail(arena: *Arena, lease: Lease) !void {
        const active = try arena.held(lease);
        active.failed = true;
        for (active.touched, 0..) |touched, lane| if (touched) {
            arena.lanes[lane].failed = true;
        };
    }

    pub fn complete(arena: *Arena, lease: Lease, stream: cuda.Stream) !void {
        return arena.finish(lease, stream, true);
    }

    fn finish(arena: *Arena, lease: Lease, stream: cuda.Stream, check_status: bool) !void {
        try arena.bindStream(lease, stream);
        stream.synchronize() catch |err| {
            try arena.fail(lease);
            return err;
        };
        arena.driver.check(arena.driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV consumers)") catch |err| {
            try arena.fail(lease);
            return err;
        };
        if (check_status) {
            var controls: [wt.attn_n * controls_words]u32 = undefined;
            const active = try arena.held(lease);
            for (active.owners[0..active.count]) |owner| {
                const control_base = arena.tables.ptr + arena.layout.controls + @as(usize, owner.lane) * controls.len * 4;
                arena.driver.check(arena.driver.api.cuMemcpyDtoH_v2(&controls, control_base, controls.len * 4), "cuMemcpyDtoH(shared KV status)") catch |err| {
                    try arena.fail(lease);
                    return err;
                };
                for (0..wt.attn_n) |layer| if (controls[layer * controls_words + 2] != 0) {
                    try arena.fail(lease);
                    return error.SharedKvOverflow;
                };
            }
        }
        try arena.completeAfterConsumers(lease);
    }

    pub fn completeAfterConsumers(arena: *Arena, lease: Lease) !void {
        _ = try arena.held(lease);
        arena.active = null;
    }

    pub fn drainFailed(arena: *Arena, lease: Lease) !void {
        const active = arena.active orelse return error.SharedKvLease;
        if (!std.meta.eql(active.lease, lease) or !active.failed) return error.SharedKvLease;
        try arena.driver.check(arena.driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV failure)");
        arena.active = null;
    }

    fn binding(arena: *Arena, sequence: contract.Sequence, layer: usize) !View {
        _ = try arena.state(sequence);
        const offset = try arena.layout.windowOffset(sequence.lane, layer);
        const key_offset = @as(usize, sequence.lane) * arena.layout.lane_words + layer * arena.layout.blocks;
        return .{ .keys = arena.device.ptr + offset * row_bytes, .values = arena.device.ptr + (arena.layout.side_rows + offset) * row_bytes, .inverse = arena.tables.ptr + key_offset * 4, .marks = arena.tables.ptr + arena.layout.marks + key_offset * 4, .controls = arena.tables.ptr + arena.layout.controls + (@as(usize, sequence.lane) * wt.attn_n + layer) * controls_words * 4, .slot_base = try arena.layout.slotBase(sequence.lane, layer) };
    }

    pub fn view(arena: *Arena, lease: Lease, sequence: contract.Sequence, layer: usize) !View {
        _ = try arena.member(lease, sequence);
        return arena.binding(sequence, layer);
    }

    pub fn page(arena: *Arena, lease: Lease, stream: cuda.Stream, layer: usize, origin: usize, rows: usize, ids: u64, counts: u64, sparse: u64, host_keys: u64, host_values: u64) !void {
        return arena.pageFor(lease, lease.sequence, stream, layer, origin, rows, ids, counts, sparse, host_keys, host_values);
    }

    pub fn pageFor(arena: *Arena, lease: Lease, sequence: contract.Sequence, stream: cuda.Stream, layer: usize, origin: usize, rows: usize, ids: u64, counts: u64, sparse: u64, host_keys: u64, host_values: u64) !void {
        _ = try arena.member(lease, sequence);
        try arena.bindStream(lease, stream);
        if (rows == 0 or rows > parts.att_rows or origin % ratio != 0 or origin > arena.layout.options.cap or @min(origin / ratio, try std.math.mul(usize, rows, wt.index_top)) > arena.layout.options.slots) return error.SharedKvCapacity;
        const kernels = arena.kernels orelse return error.BuiltWithoutKernels;
        const window = try arena.view(lease, sequence, layer);
        const needed = arena.work.ptr;
        const missing = needed + work_entries * 4;
        const fetched = missing + work_entries * 4;
        var args: cuda.Args = .{};
        for ([_]u64{ ids, counts, sparse }) |address| args.add(address);
        for ([_]usize{ rows, wt.index_idw, wt.index_top, ratio, origin }) |value| args.add(@as(c_int, @intCast(value)));
        for ([_]u64{ arena.tables.ptr + arena.layout.clock, window.inverse, arena.tables.ptr }) |address| args.add(address);
        args.add(@as(c_int, @intCast(@as(usize, sequence.lane) * arena.layout.lane_words + layer * arena.layout.blocks)));
        for ([_]u64{ arena.tables.ptr + arena.layout.owners, arena.tables.ptr + arena.layout.stamps, window.marks }) |address| args.add(address);
        args.add(@as(c_int, @intCast(arena.layout.options.slots)));
        for ([_]u64{ window.controls, needed, missing, fetched, arena.tables.ptr + arena.layout.owner_generations, arena.tables.ptr + arena.layout.lane_generations }) |address| args.add(address);
        args.add(sequence.generation);
        args.add(@as(c_int, @intCast(arena.layout.lane_words)));
        args.add(@as(c_int, @intCast(arena.layout.all_words)));
        try arena.touch(lease, sequence);
        errdefer arena.fail(lease) catch unreachable;
        try cuda.launch.launch(kernels.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &args);
        var fetch_args: cuda.Args = .{};
        for ([_]u64{ host_keys, host_values, window.keys, window.values, window.controls, fetched }) |address| fetch_args.add(address);
        for ([_]usize{ window.slot_base, row_bytes / 16, ratio }) |value| fetch_args.add(@as(c_int, @intCast(value)));
        try cuda.launch.launch(kernels.fetch, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &fetch_args);
        var remap_args: cuda.Args = .{};
        for ([_]u64{ ids, counts, sparse }) |address| remap_args.add(address);
        for ([_]usize{ wt.index_idw, origin, window.slot_base }) |value| remap_args.add(@as(c_int, @intCast(value)));
        remap_args.add(window.inverse);
        remap_args.add(@as(c_int, ratio));
        remap_args.add(window.controls);
        try cuda.launch.launch(kernels.remap, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &remap_args);
    }

    pub fn release(arena: *Arena, sequence: contract.Sequence, stream: cuda.Stream) !void {
        const lane = try arena.state(sequence);
        if (arena.active != null) return error.SharedKvBusy;
        if (stream.d != arena.driver) return error.SharedKvDriver;
        const kernels = arena.kernels orelse return error.BuiltWithoutKernels;
        const failed = lane.failed;
        lane.failed = false;
        const lease = arena.acquire(sequence) catch |err| {
            lane.failed = failed;
            return err;
        };
        errdefer if (arena.active != null and !arena.active.?.failed) arena.fail(lease) catch unreachable;
        var args: cuda.Args = .{};
        for ([_]u64{ arena.tables.ptr, arena.tables.ptr + arena.layout.owners, arena.tables.ptr + arena.layout.stamps, arena.tables.ptr + arena.layout.owner_generations }) |address| args.add(address);
        for ([_]usize{ arena.layout.options.slots, @as(usize, sequence.lane) * arena.layout.lane_words, (@as(usize, sequence.lane) + 1) * arena.layout.lane_words }) |value| args.add(@as(c_int, @intCast(value)));
        args.add(sequence.generation);
        try arena.submitted(lease);
        try cuda.launch.launch(kernels.release, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &args);
        try arena.finish(lease, stream, false);
        arena.lanes[sequence.lane].live = false;
    }

    fn word(arena: *Arena, offset: usize) *u32 {
        return @ptrCast(@alignCast(arena.metadata[offset..][0..4]));
    }

    fn generation(arena: *Arena, offset: usize) *u64 {
        return @ptrCast(@alignCast(arena.metadata[offset..][0..8]));
    }

    fn downloadMetadata(arena: *Arena) !void {
        try arena.driver.check(arena.driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV metadata)");
        try arena.driver.check(arena.driver.api.cuMemcpyDtoH_v2(arena.metadata.ptr, arena.tables.ptr, arena.metadata.len), "cuMemcpyDtoH(shared KV metadata)");
    }

    fn geometry(arena: *const Arena) contract.Geometry {
        return .{ .lanes = @intCast(arena.layout.options.lanes), .layers = wt.attn_n, .blocks = @intCast(arena.layout.blocks), .slots = @intCast(arena.layout.options.slots), .row_bytes = row_bytes, .window_rows = arena.layout.window_rows, .work_words = arena.layout.work_bytes / 4 };
    }

    pub fn saveOwnedBlocks(arena: *Arena, allocator: std.mem.Allocator, lease: Lease) !contract.OwnedBlocks {
        const active = try arena.held(lease);
        if (active.count != 1) return error.SharedKvSnapshotLease;
        try arena.downloadMetadata();
        const control_base = arena.layout.controls + @as(usize, lease.sequence.lane) * wt.attn_n * controls_words * 4;
        for (0..wt.attn_n) |layer| if (arena.word(control_base + (layer * controls_words + 2) * 4).* != 0) return error.SharedKvOverflow;
        const first_key = @as(usize, lease.sequence.lane) * arena.layout.lane_words;
        var count: usize = 0;
        for (0..arena.layout.options.slots) |slot| {
            const owner = arena.word(arena.layout.owners + slot * 4).*;
            if (owner >= first_key and owner - first_key < arena.layout.lane_words and arena.generation(arena.layout.owner_generations + slot * 8).* == lease.sequence.generation) count += 1;
        }
        const entries = try allocator.alloc(contract.OwnedEntry, count);
        errdefer allocator.free(entries);
        const block_bytes = ratio * row_bytes;
        const payload = try allocator.alloc(u8, count * 2 * block_bytes);
        errdefer allocator.free(payload);
        const slot_rows = arena.layout.options.lanes * wt.attn_n * arena.layout.window_rows;
        var index: usize = 0;
        for (0..arena.layout.options.slots) |slot| {
            const owner = arena.word(arena.layout.owners + slot * 4).*;
            if (owner < first_key or owner - first_key >= arena.layout.lane_words or arena.generation(arena.layout.owner_generations + slot * 8).* != lease.sequence.generation) continue;
            if (arena.word(owner * 4).* != slot) return error.SharedKvInverse;
            const key = owner - first_key;
            entries[index] = .{ .key = .{ .layer = @intCast(key / arena.layout.blocks), .block = @intCast(key % arena.layout.blocks) }, .stamp = arena.word(arena.layout.stamps + slot * 4).*, .source_slot = @intCast(slot) };
            for (0..2) |side| {
                const address = arena.device.ptr + (side * arena.layout.side_rows + slot_rows + ratio * slot) * row_bytes;
                try arena.driver.check(arena.driver.api.cuMemcpyDtoH_v2(payload[(2 * index + side) * block_bytes ..].ptr, address, block_bytes), "cuMemcpyDtoH(shared KV owned block)");
            }
            index += 1;
        }
        var image: contract.OwnedBlocks = .{ .source = lease.sequence, .geometry = arena.geometry(), .clock = arena.word(arena.layout.clock).*, .entries = entries, .payload = payload };
        try image.validate(arena.geometry());
        return image;
    }

    fn evictMetadata(arena: *Arena, slot: usize) void {
        const owner = arena.word(arena.layout.owners + slot * 4).*;
        const owner_generation = arena.generation(arena.layout.owner_generations + slot * 8).*;
        if (owner < arena.layout.all_words and owner_generation == arena.generation(arena.layout.lane_generations + owner / arena.layout.lane_words * 8).* and arena.word(owner * 4).* == slot) {
            arena.word(owner * 4).* = 0xffffffff;
            if (arena.active) |*active| active.touched[owner / arena.layout.lane_words] = true;
        }
        arena.word(arena.layout.owners + slot * 4).* = 0xffffffff;
        arena.word(arena.layout.stamps + slot * 4).* = 0;
        arena.generation(arena.layout.owner_generations + slot * 8).* = 0;
    }

    pub fn restoreOwnedBlocks(arena: *Arena, lease: Lease, image: *const contract.OwnedBlocks) !void {
        const active = try arena.held(lease);
        if (active.count != 1) return error.SharedKvSnapshotLease;
        try arena.validateOwnedBlocks(image);
        try arena.downloadMetadata();
        const clock_base = @max(arena.word(arena.layout.clock).*, @as(u32, @intCast(image.clock)));
        const next_clock = try std.math.add(u32, try std.math.add(u32, clock_base, 1), std.math.cast(u32, try image.maxAge()) orelse return error.Overflow);
        const targets = try arena.allocator.alloc(u32, arena.layout.options.slots);
        defer arena.allocator.free(targets);
        const first_key = @as(usize, lease.sequence.lane) * arena.layout.lane_words;
        for (0..arena.layout.options.slots) |slot| {
            const owner = arena.word(arena.layout.owners + slot * 4).*;
            if (owner >= first_key and owner - first_key < arena.layout.lane_words and arena.generation(arena.layout.owner_generations + slot * 8).* == lease.sequence.generation) arena.evictMetadata(slot);
        }
        for (targets, 0..) |*target, slot| target.* = @intCast(slot);
        const Order = struct {
            fn lessThan(owner: *Arena, left: u32, right: u32) bool {
                const left_stamp = owner.word(owner.layout.stamps + @as(usize, left) * 4).*;
                const right_stamp = owner.word(owner.layout.stamps + @as(usize, right) * 4).*;
                return left_stamp < right_stamp or (left_stamp == right_stamp and left < right);
            }
        };
        std.mem.sort(u32, targets, arena, Order.lessThan);
        for (image.entries, 0..) |entry, index| {
            const slot = targets[index];
            arena.evictMetadata(slot);
            arena.word(arena.layout.owners + slot * 4).* = @intCast(first_key + @as(usize, entry.key.layer) * arena.layout.blocks + entry.key.block);
            arena.generation(arena.layout.owner_generations + slot * 8).* = lease.sequence.generation;
            arena.word(arena.layout.stamps + slot * 4).* = next_clock - @as(u32, @intCast(image.clock - entry.stamp));
        }
        arena.word(arena.layout.clock).* = next_clock;
        try arena.submitted(lease);
        errdefer arena.fail(lease) catch unreachable;
        errdefer cuda.completion.drain(arena.driver, .{ .d = arena.driver, .handle = null }, "shared KV snapshot upload did not drain");
        try arena.driver.check(arena.driver.api.cuMemcpyHtoD_v2(arena.tables.ptr, arena.metadata.ptr, arena.metadata.len), "cuMemcpyHtoD(shared KV invalidated maps)");
        const block_bytes = ratio * row_bytes;
        const slot_rows = arena.layout.options.lanes * wt.attn_n * arena.layout.window_rows;
        for (targets[0..image.entries.len], 0..) |slot, index| for (0..2) |side| {
            const address = arena.device.ptr + (side * arena.layout.side_rows + slot_rows + ratio * slot) * row_bytes;
            try arena.driver.check(arena.driver.api.cuMemcpyHtoD_v2(address, image.payload[(2 * index + side) * block_bytes ..].ptr, block_bytes), "cuMemcpyHtoD(shared KV owned block)");
        };
        try arena.driver.check(arena.driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV block payload)");
        for (targets[0..image.entries.len], image.entries) |slot, entry| arena.word((first_key + @as(usize, entry.key.layer) * arena.layout.blocks + entry.key.block) * 4).* = slot;
        try arena.driver.check(arena.driver.api.cuMemcpyHtoD_v2(arena.tables.ptr, arena.metadata.ptr, arena.layout.marks), "cuMemcpyHtoD(shared KV published maps)");
        try arena.driver.check(arena.driver.api.cuCtxSynchronize(), "cuCtxSynchronize(shared KV published maps)");
    }

    pub fn validateOwnedBlocks(arena: *const Arena, image: *const contract.OwnedBlocks) !void {
        try image.validate(arena.geometry());
        if (image.clock > std.math.maxInt(u32)) return error.Overflow;
    }

    pub fn graphKey(arena: *Arena, sequence: contract.Sequence, layer: usize, canonical: u64, origin_hash: u64) !contract.GraphKey {
        const window = try arena.binding(sequence, layer);
        return .{ .sequence = sequence, .geometry = .{ .lanes = @intCast(arena.layout.options.lanes), .layers = wt.attn_n, .blocks = @intCast(arena.layout.blocks), .slots = @intCast(arena.layout.options.slots), .row_bytes = row_bytes, .window_rows = arena.layout.window_rows, .work_words = arena.layout.work_bytes / 4 }, .bindings = .{ .keys = arena.device.ptr + arena.layout.options.lanes * wt.attn_n * arena.layout.window_rows * row_bytes, .values = arena.device.ptr + (arena.layout.side_rows + arena.layout.options.lanes * wt.attn_n * arena.layout.window_rows) * row_bytes, .owners = arena.tables.ptr + arena.layout.owners, .stamps = arena.tables.ptr + arena.layout.stamps, .clock = arena.tables.ptr + arena.layout.clock, .inverse = window.inverse, .marks = window.marks, .controls = window.controls, .work = arena.work.ptr, .window_keys = window.keys, .window_values = window.values, .canonical = canonical, .cap = arena.layout.options.cap, .rows = arena.layout.options.rows, .origin_hash = origin_hash } };
    }
};

test {
    _ = @import("cuda_exl3_shared_kv_test.zig");
}
