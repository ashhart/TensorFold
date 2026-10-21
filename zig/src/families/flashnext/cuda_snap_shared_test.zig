const std = @import("std");
const cuda = @import("cuda");
const snap = @import("cuda_snap.zig");
const pager = @import("cuda_exl3_kvpage.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const shared_fixture = @import("cuda_exl3_shared_kv_test.zig");
const fixture_helpers = @import("cuda_snap_test_fixture.zig");
const Fake = fixture_helpers.Fake;
const Fixture = fixture_helpers.Fixture;
const sharedFixture = fixture_helpers.sharedFixture;

test "exl3: native shared snapshot version 2 isolates windows canonical data and remapped entries" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 3, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    var source: ?*Fixture = try sharedFixture(&driver, arena);
    defer if (source) |fixture| fixture.deinit();
    const foreign = try sharedFixture(&driver, arena);
    defer foreign.deinit();
    source.?.seq.fed = 20;
    for (0..wt.attn_n) |layer| for ([_]bool{ false, true }) |values| {
        const target: [*]u8 = @ptrFromInt(source.?.pg.devRows(layer, values));
        @memset(target[0 .. source.?.pg.wr * pager.row_bytes], if (values) 0xa2 else 0xa1);
        const other: [*]u8 = @ptrFromInt(foreign.pg.devRows(layer, values));
        @memset(other[0 .. foreign.pg.wr * pager.row_bytes], if (values) 0xb2 else 0xb1);
    };
    shared_fixture.seed(arena, source.?.pg.shared_sequence, 0, 0, 0, 1, 0xc1);
    shared_fixture.seed(arena, source.?.pg.shared_sequence, 1, 1, 1, 2, 0xc2);
    const saved = try snap.save(std.testing.allocator, &driver, stream, &source.?.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    try std.testing.expectEqual(@as(u32, 2), saved.version);
    try std.testing.expectEqual(@as(usize, 2), saved.owned_blocks.?.entries.len);
    try std.testing.expect(try snap.pagedBytes(64, source.?.pg.cfg, 20, false) >= saved.buf.len + @sizeOf(snap.Snap) + saved.owned_blocks.?.payload.len + saved.owned_blocks.?.entries.len * @sizeOf(pager.shared_contract.OwnedEntry));
    const original_generation = source.?.pg.shared_sequence.generation;
    source.?.deinit();
    source = null;
    shared_fixture.seed(arena, foreign.pg.shared_sequence, 2, 2, 0, 3, 0xd1);
    const destination = try sharedFixture(&driver, arena);
    defer destination.deinit();
    try std.testing.expect(destination.pg.shared_sequence.generation > original_generation);
    const pending_graph: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(1) };
    try pending_graph.launchOn(stream);
    try std.testing.expect(shared_fixture.Fake.graph_pending);
    try snap.restore(stream, &destination.seq, null, saved);
    try std.testing.expect(!shared_fixture.Fake.graph_pending);
    try std.testing.expectEqual(@as(usize, 20), destination.seq.fed);
    try std.testing.expect(!destination.seq.invalidated);
    for (0..wt.attn_n) |layer| for ([_]bool{ false, true }) |values| {
        const target: [*]const u8 = @ptrFromInt(destination.pg.devRows(layer, values));
        try std.testing.expect(std.mem.allEqual(u8, target[0 .. destination.pg.wr * pager.row_bytes], if (values) 0xa2 else 0xa1));
        const other: [*]const u8 = @ptrFromInt(foreign.pg.devRows(layer, values));
        try std.testing.expect(std.mem.allEqual(u8, other[0 .. foreign.pg.wr * pager.row_bytes], if (values) 0xb2 else 0xb1));
        try std.testing.expect(std.mem.allEqual(u8, foreign.pg.hostRows(layer, values), 0x80));
        try std.testing.expect(std.mem.allEqual(u8, destination.pg.hostRows(layer, values)[0 .. 20 * pager.row_bytes], if (values) 0xa2 else 0xa1));
    };
    const foreign_lease = try arena.acquire(foreign.pg.shared_sequence);
    var foreign_blocks = try arena.saveOwnedBlocks(std.testing.allocator, foreign_lease);
    defer foreign_blocks.deinit(std.testing.allocator);
    try arena.cancel(foreign_lease);
    try std.testing.expectEqual(@as(usize, 1), foreign_blocks.entries.len);
    try std.testing.expectEqual(@as(u64, 3), foreign_blocks.entries[0].stamp);
    try std.testing.expectEqual(@as(usize, 0), foreign.seq.fed);
    try std.testing.expectEqual([2]i64{ 73, 91 }, foreign.seq.history);
    try std.testing.expect(std.mem.allEqual(u8, foreign_blocks.payload, 0xd1));
    const again = try snap.save(std.testing.allocator, &driver, stream, &destination.seq, null);
    defer snap.drop(std.testing.allocator, stream, again);
    try std.testing.expectEqualSlices(u8, saved.host.?.bytes[0..saved.buf.len], again.host.?.bytes[0..again.buf.len]);
    try std.testing.expectEqualSlices(u8, saved.owned_blocks.?.payload, again.owned_blocks.?.payload);
    for (saved.owned_blocks.?.entries, again.owned_blocks.?.entries) |original_entry, restored_entry| {
        try std.testing.expectEqual(original_entry.key, restored_entry.key);
        try std.testing.expectEqual(saved.owned_blocks.?.clock - original_entry.stamp, again.owned_blocks.?.clock - restored_entry.stamp);
    }
    try std.testing.expect(saved.owned_blocks.?.entries[0].source_slot != again.owned_blocks.?.entries[0].source_slot);
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: native shared snapshot failures poison the destination after submitted work and drain its lease" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 3, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const source = try sharedFixture(&driver, arena);
    defer source.deinit();
    const destination = try sharedFixture(&driver, arena);
    defer destination.deinit();
    const saved = try snap.save(std.testing.allocator, &driver, stream, &source.seq, null);
    defer snap.drop(std.testing.allocator, stream, saved);
    const before = Fake.submitted;
    saved.version = 1;
    try std.testing.expectError(error.SnapshotVersion, snap.restore(stream, &destination.seq, null, saved));
    try std.testing.expectEqual(before, Fake.submitted);
    try std.testing.expect(!destination.seq.invalidated);
    saved.version = 2;
    saved.owned_blocks.?.clock = @as(u64, std.math.maxInt(u32)) + 1;
    try std.testing.expectError(error.Overflow, snap.restore(stream, &destination.seq, null, saved));
    try std.testing.expectEqual(before, Fake.submitted);
    try std.testing.expect(!destination.seq.invalidated);
    saved.owned_blocks.?.clock = 0;
    Fake.fail_at = Fake.submitted + 2;
    try std.testing.expectError(error.NotReady, snap.restore(stream, &destination.seq, null, saved));
    try std.testing.expect(destination.seq.invalidated);
    try std.testing.expectEqual(@as(usize, 0), Fake.pending);
    try std.testing.expect(arena.active == null);
    try std.testing.expectError(error.SharedKvSequenceFailed, arena.acquire(destination.pg.shared_sequence));
    const available = try arena.acquire(source.pg.shared_sequence);
    try arena.cancel(available);
    try std.testing.expect(!Fake.freed_pending);
}

test "exl3: shared pager Engine sequence churn retains only the common allocation and reuses lane generations" {
    Fake.reset();
    var driver = Fake.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const baseline = cuda.usage(false).device;
    var engine: prompt.Engine = undefined;
    engine.gpa = std.testing.allocator;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.chunk = 256;
    engine.eos = 0;
    engine.page_config = .{ .window = 2052, .rows = 256, .slots = 16, .shared_lanes = 2 };
    engine.shared_arena = arena;
    engine.round_graph = .{};
    try std.testing.expectEqual(arena.layout.common_bytes, try engine.reserveSharedKv(64));
    try std.testing.expectEqual(baseline, cuda.usage(false).device);
    var previous_generation: u64 = 0;
    for (0..3) |_| {
        var sequence = try engine.newSeq(64);
        const pg = sequence.paged.?;
        try std.testing.expect(pg.dev.borrowed and pg.tabs.borrowed and pg.work.borrowed);
        try std.testing.expectEqual(arena.device.ptr, pg.dev.ptr);
        try std.testing.expectEqual(@as(usize, 0), pg.deviceBytes());
        try std.testing.expect(pg.shared_sequence.generation > previous_generation);
        previous_generation = pg.shared_sequence.generation;
        sequence.deinit(std.testing.allocator);
        try std.testing.expectEqual(baseline, cuda.usage(false).device);
    }
    try std.testing.expectError(error.SharedKvCommonCap, engine.newSeq(63));
    try std.testing.expectEqual(baseline, cuda.usage(false).device);
}

test "exl3: shared pager constructor mapping failures retire registrations without touching foreign slots" {
    Fake.reset();
    var driver = Fake.driver();
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const foreign = try arena.register(64);
    defer arena.release(foreign, stream) catch unreachable;
    shared_fixture.seed(arena, foreign, 0, 0, 0, 1, 0xb1);
    const baseline = cuda.usage(false);
    const cfg: pager.Config = .{ .window = 2052, .rows = 256, .slots = 16, .shared_lanes = 2 };
    Fake.fail_mapping = true;
    try std.testing.expectError(error.NotReady, pager.Paged.initShared(std.testing.allocator, &driver, 64, cfg, arena));
    Fake.fail_mapping = false;
    try std.testing.expectEqual(baseline.device, cuda.usage(false).device);
    try std.testing.expectEqual(baseline.host, cuda.usage(false).host);
    const available = try arena.acquire(foreign);
    var image = try arena.saveOwnedBlocks(std.testing.allocator, available);
    defer image.deinit(std.testing.allocator);
    try arena.cancel(available);
    try std.testing.expectEqual(@as(usize, 1), image.entries.len);
    try std.testing.expect(std.mem.allEqual(u8, image.payload, 0xb1));
    const pg = try pager.Paged.initShared(std.testing.allocator, &driver, 64, cfg, arena);
    pg.destroy();
    try std.testing.expectEqual(baseline.device, cuda.usage(false).device);
    try std.testing.expectEqual(baseline.host, cuda.usage(false).host);
}

test "exl3: shared snapshot mark preflight cancels its unsubmitted lease without invalidating the source" {
    Fake.reset();
    var driver = Fake.driver();
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 });
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const source = try sharedFixture(&driver, arena);
    defer source.deinit();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var draft: @import("cuda_mtp_draft.zig").Draft = undefined;
    draft.len = 1;
    const before = Fake.submitted;
    try std.testing.expectError(error.NotAtMark, snap.save(std.testing.allocator, &driver, stream, &source.seq, &draft));
    try std.testing.expectEqual(before, Fake.submitted);
    try std.testing.expect(!source.seq.invalidated);
    try std.testing.expect(arena.active == null);
    const lease = try arena.acquire(source.pg.shared_sequence);
    try arena.cancel(lease);
}

test "exl3: shared pager common charge admission and snapshot formats are independently checked" {
    const cfg: pager.Config = .{ .window = 2052, .rows = 256, .slots = 1024, .shared_lanes = 8 };
    const cap = 8192;
    const common = try pager.sharedBytesFor(cap, cfg);
    const layout = try pager.shared_device.Layout.init(.{ .cap = cap, .lanes = 8, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots });
    try std.testing.expectEqual(layout.common_bytes, common);
    try std.testing.expectEqual(@as(usize, 0), try pager.deviceBytesFor(cap, cfg));
    try std.testing.expectEqual(@as(u64, 800), try pager.admissionRoom(10000, 100, cfg));
    try std.testing.expectEqual(@as(u64, 799), try pager.admissionRoom(799, 100, cfg));
    try std.testing.expectError(error.Overflow, pager.admissionRoom(10000, std.math.maxInt(usize), cfg));
    const shared_layout = try pager.SnapshotLayout.init(cap, cfg, 64);
    var private_cfg = cfg;
    private_cfg.shared_lanes = 0;
    const private_layout = try pager.SnapshotLayout.init(cap, private_cfg, 64);
    try std.testing.expect(shared_layout.device < private_layout.device);
    try std.testing.expect(shared_layout.tables < private_layout.tables);
    try std.testing.expect(pager.configIdentity(cfg) != pager.configIdentity(private_cfg));
    try std.testing.expect(try snap.pagedBytes(cap, cfg, 64, false) > shared_layout.total + try pager.ownedSnapshotBytes(cfg));
    var bad = cfg;
    bad.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Overflow, pager.sharedBytesFor(cap, bad));
}

test "exl3: shared pager local keys bind generations and arena incarnations while TP keys remain logical" {
    Fake.reset();
    var driver = Fake.driver();
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    const options: pager.shared_device.Options = .{ .cap = 64, .lanes = 2, .window = 2052, .rows = 256, .slots = 16 };
    const arena = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, options);
    defer arena.destroy() catch unreachable;
    shared_fixture.Fake.install(arena);
    const owner = try arena.register(64);
    var pg: pager.Paged = undefined;
    pg.cfg = .{ .window = 2052, .rows = 256, .slots = 16, .shared_lanes = 2 };
    pg.cap = 64;
    pg.wr = arena.layout.window_rows;
    pg.win0 = @splat(0);
    pg.dev = arena.device;
    pg.tabs = arena.tables;
    pg.work = arena.work;
    pg.host_dev = 1234;
    pg.shared_arena = arena;
    pg.shared_sequence = owner;
    const local = pg.graphKey();
    const logical = pg.logicalGraphKey();
    try arena.release(owner, stream);
    const replacement = try arena.register(64);
    defer arena.release(replacement, stream) catch unreachable;
    pg.shared_sequence = replacement;
    try std.testing.expect(!std.meta.eql(local, pg.graphKey()));
    try std.testing.expectEqual(logical, pg.logicalGraphKey());
    const other = try pager.shared_device.Arena.initStorage(std.testing.allocator, &driver, options);
    defer other.destroy() catch unreachable;
    shared_fixture.Fake.install(other);
    const placeholder = try other.register(64);
    defer other.release(placeholder, stream) catch unreachable;
    const peer = try other.register(64);
    defer other.release(peer, stream) catch unreachable;
    pg.shared_arena = other;
    pg.shared_sequence = peer;
    try std.testing.expect(!std.meta.eql(local, pg.graphKey()));
    try std.testing.expectEqual(logical, pg.logicalGraphKey());
    pg.shared_sequence = placeholder;
    try std.testing.expect(!std.meta.eql(local, pg.graphKey()));
    try std.testing.expectEqual(logical, pg.logicalGraphKey());
}
