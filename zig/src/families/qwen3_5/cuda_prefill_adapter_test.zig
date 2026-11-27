//! Actual adapter callback identity and traced native-body compile coverage without a device.
const std = @import("std");
const be = @import("lanes").backend;
const adapter = @import("cuda_prefill_trace.zig");
pub fn backend(comptime L: type, constructor: anytype) !void {
    const table: be.Backend.VTable = undefined;
    var lane: L = undefined;
    const original: be.Backend = .{ .ptr = &lane, .vtable = &table };
    const wrapped = adapter.wrapBackend(L, original);
    try std.testing.expectEqual(original.ptr, wrapped.ptr);
    try std.testing.expectEqual(original.vtable, wrapped.vtable);
    try @import("cuda_prefill_lifecycle_test.zig").run(L, constructor);
}

const lanes = @import("lanes");
const cuda = @import("cuda");
const trace = cuda.prefill_attribution;
const Fake = @import("cuda_prefill_test.zig").Fake;
var rows: [128]trace.ledger.Row = undefined;
var count: usize = 0;
fn sink(row: trace.ledger.Row) void {
    rows[count] = row;
    count += 1;
}
fn noEvents(_: *cuda.abi.Event, _: c_uint) callconv(.c) i32 {
    return 1;
}
test "CUDA prefill adapter: actual prompt chunks, fresh and successful snapshot restoration" {
    const m = @import("cuda_engine.zig").grid_rows;
    var tokens: [2 * m + 7]u32 = @splat(3);
    for ([_]bool{ false, true }) |restored| {
        var engine: Fake.Engine = .{};
        defer engine.deinit();
        var fake: Fake = .{ .e = &engine };
        defer fake.deinit();
        var saved: u8 = 0;
        var stream = try lanes.Stream.init(std.testing.allocator, .{ .id = "ledger", .prompt = &tokens, .max_new = 1, .reuse = .{ .saved = if (restored) &saved else null, .at = if (restored) m else 0 } });
        defer stream.deinit(std.testing.allocator);
        var api: cuda.abi.Api = undefined;
        api.cuEventCreate = noEvents;
        var state: trace.Trace = .{ .a = std.testing.allocator, .io = std.testing.io, .driver = undefined, .original = api, .stream = null, .sink = sink };
        defer state.entries.deinit(std.testing.allocator);
        trace.owner = &state;
        defer trace.owner = null;
        count = 0;
        try std.testing.expect(state.enter(&stream, tokens.len, m, 0));
        try @import("cuda_prefill.zig").runMode(true, &fake, &stream);
        var chunks: usize = 0;
        var saw_restore = false;
        var sum: u64 = 0;
        for (rows[0..count]) |row| {
            if (row.stage == .target_chunk) {
                try std.testing.expectEqual(sum + if (restored) @as(u64, m) else 0, row.start_token);
                try std.testing.expectEqual(@as(u64, 2), row.layers_run);
                try std.testing.expectEqual(row.rows, row.shape_bucket_rows);
                try std.testing.expectEqual(row.start_token + row.rows < tokens.len, row.unread_last_layer);
                sum += row.rows;
                chunks += 1;
            }
            if (row.stage == .restore) {
                saw_restore = true;
                try std.testing.expectEqual(if (restored) trace.ledger.Resume{ .kind = .snapshot, .tokens_restored = m, .kv_restored = true, .recurrent_restored = true } else trace.ledger.Resume{ .kind = .fresh }, row.restoration);
            }
        }
        try std.testing.expect(saw_restore);
        try std.testing.expectEqual(@as(usize, if (restored) 2 else 3), chunks);
        try std.testing.expectEqual(@as(u64, tokens.len - if (restored) @as(usize, m) else 0), sum);
        try std.testing.expectEqual(@as(usize, 0), state.depth);
    }
}

const dr = lanes.drafter;
const FakeDraft = struct {
    fns: struct { gather_taps: struct { handle: cuda.abi.Function = null } = .{}, gather_rows: struct { handle: cuda.abi.Function = null } = .{} } = .{},
    opened: usize = 0,
    absorbed: usize = 0,
    fn open(p: *anyopaque, _: *lanes.Stream) anyerror!void {
        const d: *@This() = @ptrCast(@alignCast(p));
        d.opened += 1;
    }
    fn absorb(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d: *@This() = @ptrCast(@alignCast(p));
        d.absorbed += items[0].features.rows;
    }
};
test "CUDA prefill adapter: drafter stays selected through the trailing absorb and forwards unchanged" {
    var driver: cuda.Driver = undefined;
    driver.api.cuEventCreate = noEvents;
    try trace.Trace.create(std.testing.allocator, std.testing.io, &driver, null, true);
    defer adapter.close();
    const state = trace.owner.?;
    state.sink = sink;
    count = 0;
    var tokens: [9]u32 = @splat(3);
    var stream = try lanes.Stream.init(std.testing.allocator, .{ .id = "draft-ledger", .prompt = &tokens, .max_new = 1 });
    defer stream.deinit(std.testing.allocator);
    try std.testing.expect(state.enter(&stream, tokens.len, 4096, 8192));
    state.leave();
    var d: FakeDraft = .{};
    var vtable: dr.Drafter.VTable = undefined;
    vtable.open = FakeDraft.open;
    vtable.absorb = FakeDraft.absorb;
    const original: dr.Drafter = .{ .ptr = &d, .vtable = &vtable };
    const wrapped = adapter.wrapDrafter(&d, original);
    try wrapped.open(&stream);
    try wrapped.absorb(&.{.{ .stream = &stream, .features = .{ .buffer = 1, .rows = 8, .row_bytes = 4, .space = .device, .dtype = .bf16 }, .start = 0, .follow = tokens[1..] }});
    try std.testing.expectEqual(@as(usize, 1), d.opened);
    try std.testing.expectEqual(@as(usize, 8), d.absorbed);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(.drafter_open, rows[0].stage);
    try std.testing.expectEqual(.trailing_absorb, rows[1].stage);
    try std.testing.expectEqual(@as(u64, 8), rows[1].rows);
    try std.testing.expectEqual(original.ptr, wrapped.ptr);
}
test "CUDA prefill adapter: actual DFlash2 and MTP constructors keep their off vtables" {
    var draft: @import("cuda_draft.zig").Draft = undefined;
    draft.c.kind = .dflash2;
    const first = draft.drafter();
    const second = draft.drafter();
    try std.testing.expectEqual(first.vtable, second.vtable);
    var mtp: @import("cuda_mtp.zig").Mtp = undefined;
    const one = mtp.drafter();
    const two = mtp.drafter();
    try std.testing.expectEqual(one.vtable, two.vtable);
}
