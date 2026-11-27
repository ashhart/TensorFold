//! Record real profiler hooks through a driver whose event calls cannot wait.
const std = @import("std");
const cuda = @import("cuda");
const ledger = @import("cuda_prefill_trace.zig");
const prof = @import("cuda_prof.zig");
const t = std.testing;
const Fake = struct {
    var creates: usize = 0;
    var live: usize = 0;
    var ready = false;
    var bad_time = false;
    var calls: std.ArrayList(u64) = .empty;
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        creates += 1;
        live += 1;
        out.* = @ptrFromInt(creates);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn query(_: cuda.abi.Event) callconv(.c) c_int {
        return if (ready) 0 else 600;
    }
    fn elapsed(out: *f32, _: cuda.abi.Event, _: cuda.abi.Event) callconv(.c) c_int {
        out.* = if (bad_time) std.math.nan(f32) else 2;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = query;
        d.api.cuEventElapsedTime = elapsed;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(launch);
        // Synchronization and wait function pointers remain undefined: any added call fails this test.
        return d;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        calls.appendSlice(t.allocator, &.{ @intFromPtr(f), gx, gy, gz, bx, by, bz, shared, @intFromPtr(stream) }) catch unreachable;
        for ("88888844", 0..) |size, i| {
            const at = args.?[i].?;
            const value: u64 = if (size == '8') @as(*const u64, @ptrCast(@alignCast(at))).* else @as(*const u32, @ptrCast(@alignCast(at))).*;
            calls.append(t.allocator, value) catch unreachable;
        }
        return 0;
    }
    fn work(stream: cuda.Stream) !void {
        const glue = @import("cuda_glue.zig");
        var kernels: glue.Kernels = undefined;
        kernels.rmsnorm_4096 = .{ .d = stream.d, .handle = @ptrFromInt(0xabc) };
        prof.context(null, 1, 2, .mixer);
        const kda = prof.span(null, stream, .kda);
        defer kda.end();
        try glue.rmsnorm(.{ .glue = &kernels }, stream, 0x100, 4096, 0x200, 0x300, 4096, 0x400, 1e-6, 8192, 4096);
    }
};
fn prefix(tmp: std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}
test "GLM prefill ledger: off hooks preserve operation bytes, on hooks add only pollable events" {
    Fake.creates = 0;
    Fake.live = 0;
    Fake.ready = false;
    Fake.bad_time = false;
    defer Fake.calls.deinit(t.allocator);
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    try Fake.work(stream);
    try t.expectEqual(@as(usize, 0), Fake.creates);
    const baseline = try t.allocator.dupe(u64, Fake.calls.items);
    defer t.allocator.free(baseline);
    Fake.calls.clearRetainingCapacity();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try prefix(tmp);
    defer t.allocator.free(dir);
    const path = try std.fmt.allocPrint(t.allocator, "{s}/cold", .{dir});
    defer t.allocator.free(path);
    const trace = try ledger.Trace.init(&d, t.io, path, 1, 2, &.{ .{ .kind = .dsa, .ffn = .dense }, .{ .kind = .kda, .ffn = .moe } });
    defer trace.deinit();
    const chunk = ledger.enter(trace, stream, 7, 0, 8192, 8192, false);
    try Fake.work(stream);
    chunk.end();
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(baseline), std.mem.sliceAsBytes(Fake.calls.items));
    try t.expectEqual(@as(usize, 8), Fake.creates);
    trace.collect(false);
    for (trace.entries.items) |e| {
        try t.expectEqual(@as(?u64, null), e.row.cuda_event_ns);
        try t.expectEqual(.pending, e.row.event_status);
    }
    try t.expect(!trace.written);
    Fake.ready = true;
    ledger.poll(trace);
    try t.expect(trace.written);
    const data = try tmp.dir.readFileAlloc(t.io, "cold.rank1.json", t.allocator, .limited(65536));
    defer t.allocator.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, data, .{});
    defer parsed.deinit();
    try t.expect(parsed.value.object.get("complete").?.bool);
    try t.expectEqual(@as(i64, 8192), parsed.value.object.get("prompt_tokens").?.integer);
    const rows = parsed.value.object.get("rows").?.array.items;
    try t.expectEqualStrings("kda", rows[2].object.get("layer_kind").?.string);
    try t.expectEqual(@as(i64, 2), rows[3].object.get("parent").?.integer);
    const count = Fake.creates;
    const second = ledger.enter(trace, stream, 8, 0, 32768, 32768, false);
    second.end();
    try Fake.work(stream);
    try t.expectEqual(count, Fake.creates);
}

test "GLM prefill ledger: no cold prompt or nonfinite timing cannot produce a complete receipt" {
    Fake.ready = true;
    Fake.bad_time = true;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try prefix(tmp);
    defer t.allocator.free(dir);
    const path = try std.fmt.allocPrint(t.allocator, "{s}/invalid", .{dir});
    defer t.allocator.free(path);
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    const trace = try ledger.Trace.init(&d, t.io, path, 0, 2, &.{});
    const resumed = ledger.enter(trace, stream, 0, 100, 100, 200, true);
    resumed.end();
    try t.expectEqual(@as(usize, 0), trace.entries.items.len);
    const cold = ledger.enter(trace, stream, 0, 0, 8, 8, false);
    cold.end();
    trace.collect(true);
    for (trace.entries.items) |e| {
        try t.expectEqual(.unavailable, e.row.event_status);
        try t.expectEqual(@as(?u64, null), e.row.cuda_event_ns);
    }
    trace.deinit();
    try t.expectEqual(@as(usize, 0), Fake.live);
    const data = try tmp.dir.readFileAlloc(t.io, "invalid.rank0.json", t.allocator, .limited(65536));
    defer t.allocator.free(data);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, data, .{});
    defer parsed.deinit();
    try t.expect(!parsed.value.object.get("complete").?.bool);
}
