const std = @import("std");
const sg = @import("sessions_gpu.zig");
const fwd = @import("forward.zig");
const t = std.testing;
test "Completion ownership: bounded restore rejects missing duplicate and impossible metadata before publication" {
    var k: @import("cuda_kv.zig").Kv = undefined;
    var slot: @import("kv").sessions.Slot = undefined;
    slot.id = 0;
    k.slot = &slot;
    k.opts = .{ .tokens = 256, .capacity = 64 };
    var cfg: @import("config.zig").Config = .{};
    var f: fwd.Forward = undefined;
    f.slot = .{};
    f.cfg = &cfg;
    var state: @import("forward_prefill.zig").State = .{};
    f.prefill_state = &state;
    var s: sg.Sessions = undefined;
    s.f = &f;
    s.kx = &k;
    s.pieces = &.{};
    s.ds_rings = false;
    s.ds_set = null;
    s.measure = .{ .window = cfg.window, .rings = 0, .carries = 0, .head_dim = cfg.head_dim, .hidden = cfg.hidden, .hc = cfg.hc_mult, .ds_blocks = 0, .replay = false };
    var valid = [_]u64{0};
    s.ds_valid = &valid;
    var bytes: [sg.Sessions.slot_bytes]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], 8, .little);
    var pieces = [_]sg.Piece{.{ .role = "s.L0.swa.s", .len = 8 }};
    s.pieces = &pieces;
    try t.expectError(error.BadBlob, s.uploadBlobs(&.{.{ .name = @constCast("slot"), .bytes = &bytes }}));
    try t.expectEqual(@as(u64, 0), f.slot.pos);
    s.pieces = &.{};
    const blob: @import("kv").sessions.disk.Blob = .{ .name = @constCast("slot"), .bytes = &bytes };
    try t.expectError(error.BadBlob, s.uploadBlobs(&.{ blob, blob }));
    try t.expectEqual(@as(u64, 0), f.slot.pos);
    std.mem.writeInt(u64, bytes[sg.Sessions.ds_at..][0..8], 9, .little);
    try t.expectError(error.BadBlob, s.uploadBlobs(&.{blob}));
    std.mem.writeInt(u64, bytes[sg.Sessions.ds_at..][0..8], 0, .little);
    std.mem.writeInt(u64, bytes[sg.Sessions.ced_at..][0..8], 9, .little);
    std.mem.writeInt(u64, bytes[sg.Sessions.ced_at + 8 ..][0..8], 8, .little);
    try t.expectError(error.BadBlob, s.uploadBlobs(&.{blob}));
    try t.expectEqual(@as(u64, 0), f.slot.pos);
}
test "Completion ownership: physical snapshot policy counts its complete payload and metadata" {
    var bytes: [5992448]u8 = undefined;
    var p: @import("session_snapshot.zig").Snap = undefined;
    p.bytes = &bytes;
    p.measure = 3002712;
    try t.expectEqual(@as(u64, bytes.len + @sizeOf(@TypeOf(p))), p.bounded().vtable.nbytes(&p));
}
fn historyFail(a: std.mem.Allocator) !void {
    var s: sg.Sessions = undefined;
    var histories = [_]std.ArrayList(i32){.empty};
    s.gpa = a;
    s.hist = &histories;
    defer histories[0].deinit(a);
    var ids: [128]u32 = @splat(7);
    s.commitAt(0, &ids) catch |e| {
        try t.expectEqual(@as(usize, 0), histories[0].items.len);
        return e;
    };
    try t.expectEqual(@as(usize, 128), histories[0].items.len);
}
test "Completion ownership: history allocation failure never publishes a partial token list" {
    try t.checkAllAllocationFailures(t.allocator, historyFail, .{});
}

fn allocationFails(_: *anyopaque, _: []const u32) !void {
    return error.OutOfMemory;
}
test "Completion ownership: decode and prefill reserve history before publishing accepted state" {
    var cfg: @import("config.zig").Config = .{};
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, undefined, undefined);
    defer f.arena.deinit();
    f.layers = &.{};
    f.ids = &.{ 7, 8 };
    f.sess = .{ .ptr = &f, .reserve = allocationFails, .commit = allocationFails, .save = undefined, .restore = undefined, .reset = undefined, .prompt = undefined };
    inline for (.{ false, true }) |prefill| {
        f.slot = .{ .pos = 5, .pending = .{ .start = 5, .n = 2 } };
        f.transaction.state = .{ .position = 5, .pending = 2 };
        if (prefill) try t.expectError(error.OutOfMemory, @import("forward_prefill_finish.zig").commitRoles(&f, 1, 0, "", 1)) else try t.expectError(error.OutOfMemory, f.keep(0));
        try t.expectEqual(@as(u64, 5), f.slot.pos);
        try t.expect(f.slot.pending != null);
        try t.expectEqual(@as(?u32, 2), f.transaction.state.pending);
    }
}

test "Completion ownership: disk compatibility distinguishes checkpoint layers schema and state semantics" {
    const identity = @import("session_identity.zig");
    const Kind = enum { base, checkpoint, layers, schema, semantics };
    var reference: [32]u8 = undefined;
    for (std.enums.values(Kind)) |kind| {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        identity.bind(&h, @splat(if (kind == .checkpoint) 2 else 1), if (kind == .layers) &.{ 0, 1 } else &.{0}, &.{.{ .role = "s.L0.swa.s", .len = if (kind == .schema) 16 else 8 }}, if (kind == .semantics) "replay" else "full");
        var value: [32]u8 = undefined;
        h.final(&value);
        if (kind == .base) reference = value else try t.expect(!std.mem.eql(u8, &reference, &value));
    }
}

test "Completion ownership: real Store residency totals physical snapshot allocations for several entries" {
    const sessions = @import("kv").sessions;
    var pool = try sessions.Pool.init(t.allocator, .{ .families = &.{.{ .name = "kv", .ratio = 1, .row_bytes = 8 }}, .page = 8 }, 128, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(t.allocator, &pool);
    defer host.deinit();
    var store = try sessions.Store.init(t.allocator, t.io, &pool, host.store(), null, .{ .ram_bytes = 1 << 20, .skip_covered = false });
    defer store.deinit();
    const live = try pool.newSlot(8);
    var owner: sg.Sessions = undefined;
    owner.gpa = t.allocator;
    var total: u64 = 0;
    var last: u32 = 0;
    var last_bytes: u64 = 0;
    for (0..3) |i| {
        try live.ensure(1);
        const snap = try t.allocator.create(sg.Sessions.Snap);
        snap.* = .{ .owner = &owner, .bytes = try t.allocator.alloc(u8, 32 * (i + 1)), .pos = 1, .tail = @splat(0), .tail_len = 0, .stash = .{}, .ds_rows = 0, .measure = 1 };
        const bytes = @sizeOf(sg.Sessions.Snap) + snap.bytes.len;
        const ids = [_]i32{@intCast(i + 1)};
        last = (try store.save(live, snap.bounded(), &ids, 1, 0)).?;
        _ = try live.releaseAll();
        total += bytes;
        last_bytes = bytes;
        try t.expectEqual(total, store.bounded_total);
    }
    store.policy.ram_bytes = last_bytes;
    try store.trimRam(last);
    try t.expectEqual(@as(u32, 1), store.ram_count);
    try t.expectEqual(last_bytes, store.bounded_total);
}

test "Completion ownership: config identity ignores inactive list storage and binds live semantic values" {
    var a: @import("config.zig").Config = .{};
    var b = a;
    @memset(a.kv_sources.buf[a.kv_sources.len..], 1);
    @memset(b.kv_sources.buf[b.kv_sources.len..], 2);
    @memset(&a.engram_table_dir.buf, 3);
    @memset(&b.engram_table_dir.buf, 4);
    const identity = @import("session_identity.zig");
    const first = identity.semantics(a);
    try t.expectEqualSlices(u8, &first, &identity.semantics(b));
    b.eps *= 2;
    try t.expect(!std.mem.eql(u8, &first, &identity.semantics(b)));
}
