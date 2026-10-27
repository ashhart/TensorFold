const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const prompt = flash.prompt;
const paging = flash.exl3.kvpage;
const weights = flash.exl3.prompt_wt;
const snapshots = flash.snapshot;
const cap = 8192;

pub fn run(gpu: check.Gpu) !void {
    const previous = flash.exl3.graph.force;
    flash.exl3.graph.force = .graph;
    defer flash.exl3.graph.force = previous;
    if (!flash.exl3.recycle.enabled()) return error.RecyclingRequired;
    for ([_]bool{ false, true }) |shared| for ([_]bool{ false, true }) |host_index| try lifecycle(gpu, shared, host_index);
    check.pass("paged Engine.newSeq allocation reuse: private/shared, device/host indexer, real page fetch and snapshot round trips", .{});
}

fn zeroDevice(gpu: check.Gpu, buffer: cuda.DeviceBuffer) !void {
    const scratch = try gpu.gpa.alloc(u8, @min(buffer.len, 1 << 20));
    defer gpu.gpa.free(scratch);
    var offset: usize = 0;
    while (offset < buffer.len) {
        const size = @min(scratch.len, buffer.len - offset);
        try buffer.download(offset, scratch[0..size]);
        try check.expect(std.mem.allEqual(u8, scratch[0..size], 0), "reused allocation {x} has stale bytes at {d}", .{ buffer.ptr, offset });
        offset += size;
    }
}

fn zeroSequence(gpu: check.Gpu, stream: cuda.Stream, sequence: *prompt.Seq) !void {
    try stream.synchronize();
    for ([_][]cuda.DeviceBuffer{ sequence.conv, sequence.state, sequence.kc, sequence.vc, sequence.ikc, sequence.pooled }) |group| for (group) |buffer| try zeroDevice(gpu, buffer);
    try zeroDevice(gpu, sequence.tail);
    const pg = sequence.paged.?;
    try check.expect(sequence.fed == 0 and sequence.rank_id == 0 and !sequence.invalidated and !sequence.untaught and !pg.graph_prepared and !pg.retired_for_reuse, "new request resets ordinary and paging metadata", .{});
    try check.expect(std.meta.eql(sequence.history, [2]i64{ 73, 73 }) and std.mem.allEqual(u8, sequence.logits, 0), "new request resets history and logits", .{});
    try check.expect(std.mem.allEqual(u8, pg.host.bytes, 0) and std.mem.allEqual(u8, pg.status_host.bytes, 0), "new request clears pinned canonical rows and status", .{});
    const status = try pg.total();
    try check.expect(status.calls == 0 and status.misses == 0 and status.rebases == 0 and status.written_rows == 0 and !status.overflow, "new request clears pager counters", .{});
    for (0..weights.attn_n) |layer| {
        try check.expect(pg.win0[layer] == 0, "layer {d} window resets", .{layer});
        for (0..2) |side| try zeroDevice(gpu, .{ .d = gpu.d, .ptr = pg.devRows(layer, side == 1), .len = pg.wr * paging.row_bytes, .borrowed = true });
    }
}

fn seed(gpu: check.Gpu, stream: cuda.Stream, sequence: *prompt.Seq, at: usize, tag: u8) !void {
    const pg = sequence.paged.?;
    const origin = (at -| pg.cfg.window) / weights.index_ratio * weights.index_ratio;
    sequence.fed = at;
    sequence.history = .{ @intCast(at), tag };
    sequence.rank_id = 19;
    sequence.untaught = true;
    @memset(sequence.logits, tag);
    for ([_][]cuda.DeviceBuffer{ sequence.conv, sequence.state, sequence.kc, sequence.vc, sequence.ikc, sequence.pooled }, 0..) |group, group_index| {
        for (group, 0..) |buffer, index| try buffer.fill8(tag + @as(u8, @intCast(group_index + index)), stream.handle);
    }
    try sequence.tail.fill8(tag, stream.handle);
    try stream.synchronize();
    for (0..weights.attn_n) |layer| {
        pg.win0[layer] = origin;
        pg.rebases[layer] = layer + 3;
        pg.written[layer] = origin;
        for (0..2) |side| {
            const host = pg.hostMut(layer, side == 1);
            const words = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(host)));
            for (words, 0..) |*word, index| word.* = @truncate(index + layer * 997 + side * 32768 + tag);
            try gpu.d.check(gpu.d.api.cuMemsetD8Async(pg.devRows(layer, side == 1), tag + @as(u8, @intCast(layer + side)), pg.wr * paging.row_bytes, stream.handle), "seed pager window");
        }
    }
    try stream.synchronize();
}

fn fetch(gpu: check.Gpu, stream: cuda.Stream, sequence: *prompt.Seq) !void {
    const pg = sequence.paged.?;
    var ids: [weights.index_idw]i32 = @splat(0);
    for (0..weights.index_top) |block| for (0..weights.index_ratio) |offset| {
        ids[block * weights.index_ratio + offset] = @intCast(block * weights.index_ratio + offset);
    };
    var selected = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&ids));
    defer selected.free();
    const count = [_]i32{weights.index_top * weights.index_ratio};
    const sparse = [_]i32{1};
    var counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&count));
    defer counts.free();
    var kinds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&sparse));
    defer kinds.free();
    const lease = if (pg.shared_arena) |arena| try arena.acquire(pg.shared_sequence) else null;
    defer if (lease) |held| if (pg.shared_arena.?.active != null) pg.shared_arena.?.completeAfterConsumers(held) catch unreachable;
    errdefer gpu.d.synchronize() catch @panic("fixture paging did not drain");
    try pg.page(stream, 0, selected.ptr, counts.ptr, kinds.ptr, 1);
    if (lease) |held| try pg.shared_arena.?.complete(held, stream) else try stream.synchronize();
    const status = try pg.stats(0);
    try check.expect(status.misses > 0 and status.calls > 0 and !status.overflow, "real pager fetches canonical BF16 rows", .{});
    try selected.download(0, std.mem.asBytes(&ids));
    const slot = std.math.cast(usize, ids[0]) orelse return error.InvalidRemappedKey;
    const layout = pg.slotRows(0);
    try check.expect(slot >= layout[1] and slot + 4 <= layout[0], "real sparse keys remap into owned slots", .{});
    var actual: [4 * paging.row_bytes]u8 = undefined;
    for (0..2) |side| {
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(&actual, pg.devRows(0, side == 1) + slot * paging.row_bytes, actual.len), "read fetched block");
        try check.sameBytes("real fetched BF16 block", &actual, pg.hostRows(0, side == 1)[0..actual.len]);
    }
}

fn sameSnapshot(expected: *const snapshots.Snap, actual: *const snapshots.Snap) !void {
    try check.expect(std.meta.eql(expected.paged, actual.paged) and expected.at == actual.at and std.meta.eql(expected.history, actual.history), "restored snapshot metadata remains exact", .{});
    try check.sameBytes("restored canonical, windows, tables, GDN, indexer, PLE and logits", actual.host.?.bytes[0..actual.buf.len], expected.host.?.bytes[0..expected.buf.len]);
    if (expected.owned_blocks) |wanted| {
        const found = actual.owned_blocks orelse return error.MissingOwnedBlocks;
        try check.expect(wanted.entries.len == found.entries.len, "shared snapshot retains every owned block", .{});
        for (wanted.entries, found.entries) |before, after| try check.expect(std.meta.eql(before.key, after.key), "shared snapshot remaps logical block keys", .{});
        try check.sameBytes("restored shared BF16 blocks", found.payload, wanted.payload);
    }
}

fn identity(sequence: *const prompt.Seq) u64 {
    const pg = sequence.paged.?;
    var digest = std.hash.Wyhash.init(0);
    const addresses = [_]u64{ @intFromPtr(sequence.logits.ptr), @intFromPtr(pg), @intFromPtr(pg.status_host.bytes.ptr), sequence.tail.ptr, pg.host_dev, pg.dev.ptr, pg.tabs.ptr, pg.work.ptr, pg.devRows(0, false) };
    digest.update(std.mem.asBytes(&addresses));
    for ([_][]cuda.DeviceBuffer{ sequence.conv, sequence.state, sequence.kc, sequence.vc, sequence.ikc, sequence.pooled }) |group| for (group) |buffer| {
        const shape = [_]u64{ buffer.ptr, buffer.len };
        digest.update(std.mem.asBytes(&shape));
    };
    return digest.final();
}

fn digestSnapshot(saved: *const snapshots.Snap) u64 {
    var digest = std.hash.Wyhash.hash(0, saved.host.?.bytes[0..saved.buf.len]);
    if (saved.owned_blocks) |owned| digest ^= std.hash.Wyhash.hash(1, owned.payload);
    return digest;
}

fn lifecycle(gpu: check.Gpu, shared: bool, host_index: bool) !void {
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var engine: prompt.Engine = undefined;
    engine.gpa = gpu.gpa;
    engine.io = gpu.io;
    engine.driver = @constCast(gpu.d);
    engine.stream = &stream;
    engine.chunk = 256;
    engine.eos = 73;
    engine.round_graph = .{};
    engine.page_config = .{ .window = 2052, .rows = 256, .slots = 1024, .shared_lanes = if (shared) 3 else 0 };
    engine.shared_arena = null;
    try flash.exl3.kv.enable(&engine, false, false, host_index);
    defer flash.exl3.kv.forget(gpu.gpa, &engine);
    _ = try engine.reserveSharedKv(cap);
    defer if (engine.shared_arena) |arena| arena.destroy() catch @panic("fixture still owns shared lanes");
    defer flash.exl3.recycle.drain(&engine);
    var foreign: ?prompt.Seq = if (shared) try engine.newSeq(cap) else null;
    defer if (foreign) |*sequence| sequence.deinit(gpu.gpa);
    if (foreign) |*sequence| {
        try seed(gpu, stream, sequence, 4099, 0x12);
        try fetch(gpu, stream, sequence);
    }
    const foreign_saved = if (foreign) |*sequence| try snapshots.save(gpu.gpa, gpu.d, stream, sequence, null) else null;
    defer if (foreign_saved) |saved| snapshots.drop(gpu.gpa, stream, saved);
    var current = try engine.newSeq(cap);
    var live = true;
    defer if (live) current.deinit(gpu.gpa);
    for ([_]usize{ 4099, 6147 }) |at| {
        try seed(gpu, stream, &current, at, 0x31);
        try fetch(gpu, stream, &current);
        const saved = try snapshots.save(gpu.gpa, gpu.d, stream, &current, null);
        defer snapshots.drop(gpu.gpa, stream, saved);
        const digest = digestSnapshot(saved);
        const addresses = identity(&current);
        const pg = current.paged.?;
        const incarnation = pg.incarnation;
        const key = pg.graphKey();
        const old_owner = if (shared) pg.shared_sequence else null;
        const reused = flash.exl3.recycle.stats.seq_reused;
        live = false;
        current.deinit(gpu.gpa);
        if (old_owner) |old| {
            if (engine.shared_arena.?.acquire(old)) |held| {
                try engine.shared_arena.?.cancel(held);
                return error.StaleOwnerAccepted;
            } else |err| if (err != error.SharedKvSequence) return err;
        }
        current = try engine.newSeq(cap);
        live = true;
        try check.expect(flash.exl3.recycle.stats.seq_reused == reused + 1 and std.meta.eql(addresses, identity(&current)), "Engine.newSeq reuses every sequence/pager allocation address", .{});
        try check.expect(current.paged.?.incarnation != incarnation and !std.meta.eql(key, current.paged.?.graphKey()), "retained addresses receive a fresh strict paging identity", .{});
        if (old_owner) |old| try check.expect(current.paged.?.shared_sequence.generation > old.generation, "reattached shared owner advances generation", .{});
        try zeroSequence(gpu, stream, &current);
        try seed(gpu, stream, &current, at, 0x47);
        try fetch(gpu, stream, &current);
        try snapshots.restore(stream, &current, null, saved);
        const again = try snapshots.save(gpu.gpa, gpu.d, stream, &current, null);
        defer snapshots.drop(gpu.gpa, stream, again);
        try sameSnapshot(saved, again);
        try check.expect(digestSnapshot(saved) == digest, "source snapshot remains immutable after allocation reuse", .{});
        if (foreign) |*sequence| {
            const untouched = try snapshots.save(gpu.gpa, gpu.d, stream, sequence, null);
            defer snapshots.drop(gpu.gpa, stream, untouched);
            try sameSnapshot(foreign_saved.?, untouched);
        }
        std.debug.print("RECYCLE shared={} host_index={} at={d} tail={x} host={x} incarnation={d}->{d} snapshot={x}\n", .{ shared, host_index, at, current.tail.ptr, pg.host_dev, incarnation, pg.incarnation, digest });
    }
}
