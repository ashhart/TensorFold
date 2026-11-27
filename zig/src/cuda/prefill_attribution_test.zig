//! Recorded ledger, CUDA forwarding and event-lifetime controls; no device or dynamic driver loading.
const std = @import("std");
const abi = @import("abi.zig");
const trace = @import("prefill_attribution.zig");
const ledger = trace.ledger;
const Driver = @import("driver.zig").Driver;
const t = std.testing;
var next_event: usize = 1;
var events: usize = 0;
var waits: usize = 0;
var query_ready: bool = false;
var record_count: usize = 0;
var emitted: [64]ledger.Row = undefined;
var count: usize = 0;
fn sink(row: ledger.Row) void {
    emitted[count] = row;
    count += 1;
}
fn create(out: *abi.Event, _: c_uint) callconv(.c) i32 {
    out.* = @ptrFromInt(next_event);
    next_event += 1;
    events += 1;
    return 0;
}
fn destroy(_: abi.Event) callconv(.c) i32 {
    events -= 1;
    return 0;
}
fn record(_: abi.Event, _: abi.Stream) callconv(.c) i32 {
    record_count += 1;
    return 0;
}
fn query(_: abi.Event) callconv(.c) i32 {
    return if (query_ready) 0 else 600;
}
fn elapsed(out: *f32, _: abi.Event, _: abi.Event) callconv(.c) i32 {
    out.* = 2.5;
    return 0;
}
fn sync(_: abi.Stream) callconv(.c) i32 {
    waits += 1;
    query_ready = true;
    return 0;
}
fn kernel(_: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) i32 {
    return 0;
}
fn capture(_: abi.Stream, _: abi.CaptureMode) callconv(.c) i32 {
    return 0;
}
fn endCapture(_: abi.Stream, out: *abi.Graph) callconv(.c) i32 {
    out.* = @ptrFromInt(500);
    return 0;
}
fn copy(_: ?*anyopaque, _: u64, _: usize, _: abi.Stream) callconv(.c) i32 {
    return 0;
}
fn driver() Driver {
    var d: Driver = undefined;
    d.api.cuEventCreate = create;
    d.api.cuEventDestroy_v2 = destroy;
    d.api.cuEventRecord = record;
    d.api.cuEventQuery = query;
    d.api.cuEventElapsedTime = elapsed;
    d.api.cuStreamSynchronize = sync;
    d.api.cuLaunchKernel = kernel;
    d.api.cuStreamBeginCapture_v2 = capture;
    d.api.cuStreamEndCapture = endCapture;
    d.api.cuMemcpyDtoHAsync_v2 = copy;
    return d;
}
fn setup(d: *Driver) !*trace.Trace {
    next_event = 1;
    events = 0;
    waits = 0;
    query_ready = false;
    record_count = 0;
    count = 0;
    try trace.Trace.create(t.allocator, t.io, d, @ptrFromInt(600), true);
    const owner = trace.owner.?;
    owner.sink = sink;
    try t.expect(owner.enter(d, 4096, 4096, 8192));
    return owner;
}
test "CUDA prefill ledger: recorded sequence writes independent JSON lines and exact fields" {
    const sequence = [_]ledger.Row{
        .{ .stage = .cache_allocate, .sequence = 0, .wall_ns = 11, .memory = ledger.procStatus("VmRSS: 12 kB\nVmHWM: 32 kB\n") },
        .{ .stage = .restore, .sequence = 1, .restoration = .{ .kind = .snapshot, .tokens_restored = 4096, .kv_restored = true, .recurrent_restored = true } },
        .{ .stage = .target_chunk, .sequence = 2, .rows = 4096, .layers_run = 64, .unread_last_layer = true, .shape_bucket_rows = 4096, .shape_bucket_segments = 1, .cuda_event_ns = 2500000, .event_status = .complete },
        .{ .stage = .d2h, .sequence = 3, .bytes = 8192 },
        .{ .stage = .host_sync, .sequence = 4, .operation = "cuStreamSynchronize" },
        .{ .stage = .tap_upload, .sequence = 5, .rows = 64 },
        .{ .stage = .trailing_absorb, .sequence = 6, .graph_mode = .capture_and_replay, .graph_captures = 1, .graph_replays = 1 },
        .{ .stage = .cache_publish, .sequence = 7, .scratch_bytes = 8192 },
    };
    var output: std.Io.Writer.Allocating = .init(t.allocator);
    defer output.deinit();
    for (sequence) |row| try ledger.write(&output.writer, row);
    var lines = std.mem.splitScalar(u8, output.written(), '\n');
    for (sequence) |want| {
        const line = lines.next().?;
        var got = try std.json.parseFromSlice(ledger.Row, t.allocator, line, .{});
        defer got.deinit();
        try t.expectEqualDeep(want, got.value);
    }
    try t.expectEqualStrings("", lines.next().?);
    try t.expect(lines.next() == null);
    try t.expectEqual(@as(?u64, 32768), sequence[0].memory.physical_peak_bytes);
}
test "CUDA prefill ledger: off uses no allocator, CUDA API, clock, or driver mutation" {
    var d: Driver = undefined;
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    try trace.Trace.create(failing.allocator(), undefined, &d, null, false);
    try t.expectEqual(@as(usize, 0), failing.allocations);
    try t.expect(trace.owner == null);
}
test "CUDA prefill ledger: existing wait completes chunk and copy without another wait or end marker" {
    var d = driver();
    const owner = try setup(&d);
    defer trace.Trace.close();
    const chunk = owner.begin(.{ .stage = .target_chunk, .rows = 4096 }, owner.stream, true);
    try t.expectEqual(@as(i32, 0), d.api.cuLaunchKernel(null, 1, 1, 1, 1, 1, 1, 0, owner.stream, null, null));
    try t.expectEqual(@as(i32, 0), d.api.cuMemcpyDtoHAsync_v2(null, 0, 128, owner.stream));
    try t.expectEqual(@as(usize, 0), count);
    try t.expectEqual(@as(i32, 0), d.api.cuStreamSynchronize(owner.stream));
    const before_end = record_count;
    owner.end(chunk, 0);
    try t.expectEqual(before_end, record_count);
    try t.expectEqual(@as(usize, 1), waits);
    try t.expectEqual(@as(usize, 3), count);
    try t.expectEqual(ledger.Stage.d2h, emitted[0].stage);
    var saw_chunk = false;
    var saw_copy = false;
    for (emitted[0..count]) |row| {
        if (row.stage == .target_chunk) {
            saw_chunk = true;
            try t.expectEqual(@as(?u64, 2500000), row.cuda_event_ns);
        }
        if (row.stage == .d2h) {
            saw_copy = true;
            try t.expectEqual(@as(u64, 128), row.bytes);
            try t.expectEqual(@as(?u64, 2500000), row.cuda_event_ns);
        }
    }
    try t.expect(saw_chunk and saw_copy);
    try t.expectEqual(@as(usize, 0), events);
}
test "CUDA prefill ledger: captured work gets no timing nodes; unresolved asynchronous end never waits" {
    var d = driver();
    const owner = try setup(&d);
    defer trace.Trace.close();
    const outer = owner.begin(.{ .stage = .drafter_absorb }, owner.stream, true);
    _ = d.api.cuStreamBeginCapture_v2(owner.stream, .global);
    const before = record_count;
    const inner = owner.begin(.{ .stage = .tap_upload }, owner.stream, true);
    _ = d.api.cuLaunchKernel(null, 1, 1, 1, 1, 1, 1, 0, owner.stream, null, null);
    owner.end(inner, 0);
    try t.expectEqual(before, record_count);
    var graph: abi.Graph = null;
    _ = d.api.cuStreamEndCapture(owner.stream, &graph);
    _ = d.api.cuLaunchKernel(null, 1, 1, 1, 1, 1, 1, 0, owner.stream, null, null);
    owner.end(outer, 0);
    try t.expectEqual(@as(usize, 0), waits);
    owner.collect(true);
    try t.expectEqual(@as(usize, 2), count);
    try t.expectEqual(.capture_only, emitted[0].event_status);
    try t.expectEqual(.capture, emitted[0].graph_mode);
    try t.expectEqual(.unavailable, emitted[1].event_status);
    try t.expect(!emitted[1].complete);
    try t.expectEqual(@as(usize, 0), waits);
}

test "CUDA prefill ledger: a pre-capture scope ending inside capture emits no event record or cleanup" {
    var d = driver();
    const owner = try setup(&d);
    defer trace.Trace.close();
    const outer = owner.begin(.{ .stage = .drafter_absorb }, owner.stream, true);
    _ = d.api.cuLaunchKernel(null, 1, 1, 1, 1, 1, 1, 0, owner.stream, null, null);
    _ = d.api.cuStreamBeginCapture_v2(owner.stream, .global);
    const before = record_count;
    owner.end(outer, 0);
    try t.expectEqual(before, record_count);
    try t.expectEqual(@as(usize, 0), count);
    try t.expectEqual(@as(usize, 2), events);
    var graph: abi.Graph = null;
    _ = d.api.cuStreamEndCapture(owner.stream, &graph);
    try t.expectEqual(@as(usize, 1), count);
    try t.expectEqual(.unavailable, emitted[0].event_status);
    try t.expect(!emitted[0].complete);
    try t.expectEqual(@as(usize, 0), events);
    try t.expectEqual(@as(usize, 0), waits);
}
