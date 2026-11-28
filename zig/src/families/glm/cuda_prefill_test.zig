//! Record real profiler hooks through a driver whose event calls cannot wait.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const ledger = @import("cuda_prefill_trace.zig");
const Fake = @import("cuda_prefill_production_test.zig").Fake;
fn prefix(tmp: std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}
test "GLM prefill ledger: off hooks preserve operation bytes, on hooks add only pollable events" {
    Fake.calls = .empty;
    Fake.failure = .none;
    Fake.clock = 0;
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
    try t.expectEqualStrings("tf-glm-prefill-v2", parsed.value.object.get("schema").?.string);
    try t.expectEqual(@as(i64, 1), parsed.value.object.get("rank").?.integer);
    try t.expectEqual(@as(i64, 2), parsed.value.object.get("world").?.integer);
    try t.expectEqual(@as(i64, 7), parsed.value.object.get("slot").?.integer);
    try t.expect(parsed.value.object.get("complete").?.bool);
    try t.expectEqual(@as(i64, 8192), parsed.value.object.get("prompt_tokens").?.integer);
    const rows = parsed.value.object.get("rows").?.array.items;
    try t.expectEqualStrings("kda", rows[2].object.get("layer_kind").?.string);
    try t.expectEqual(@as(i64, 2), rows[3].object.get("parent").?.integer);
    const stages = [_][]const u8{ "request", "chunk", "kda", "norms" };
    const classes = [_][]const u8{ "aggregate", "aggregate", "kda", "norms" };
    try t.expectEqual(@as(usize, 4), rows.len);
    for (rows, stages, classes, 0..) |row, stage, class, index| {
        const object = row.object;
        try t.expectEqualStrings(stage, object.get("stage").?.string);
        try t.expectEqualStrings(class, object.get("class").?.string);
        try t.expectEqual(@as(i64, @intCast(index)), object.get("sequence").?.integer);
        try t.expectEqual(@as(i64, 0), object.get("start").?.integer);
        try t.expectEqual(@as(i64, 8192), object.get("rows").?.integer);
        try t.expectEqual(@as(i64, 0), object.get("stream").?.integer);
        try t.expectEqual(@as(i64, 2000000), object.get("cuda_event_ns").?.integer);
        try t.expectEqual(@as(i64, 0), object.get("cuda_begin_ns").?.integer);
        try t.expectEqual(@as(i64, 2000000), object.get("cuda_end_ns").?.integer);
        try t.expectEqualStrings("complete", object.get("event_status").?.string);
        if (index == 0) try t.expect(object.get("parent").? == .null) else try t.expectEqual(@as(i64, @intCast(index - 1)), object.get("parent").?.integer);
        if (index < 2) {
            try t.expect(object.get("layer").? == .null and object.get("piece").? == .null);
            try t.expectEqualStrings("outside_layer", object.get("layer_kind").?.string);
            try t.expectEqualStrings("chunk", object.get("site").?.string);
        } else {
            try t.expectEqual(@as(i64, 1), object.get("layer").?.integer);
            try t.expectEqual(@as(i64, 2), object.get("piece").?.integer);
            try t.expectEqualStrings("kda", object.get("layer_kind").?.string);
            try t.expectEqualStrings("moe", object.get("ffn_kind").?.string);
            try t.expectEqualStrings("mixer", object.get("site").?.string);
        }
    }
    const gpu = parsed.value.object.get("gpu_accounting").?.object;
    try t.expectEqual(@as(i64, 2000000), gpu.get("total_ns").?.integer);
    try t.expectEqual(gpu.get("total_ns").?.integer, gpu.get("accounted_ns").?.integer);
    const segments = gpu.get("segments").?.array.items;
    try t.expectEqual(@as(usize, 1), segments.len);
    try t.expectEqualStrings("stage", segments[0].object.get("bucket").?.string);
    try t.expectEqual(@as(i64, 3), segments[0].object.get("row").?.integer);
    try checkPartition(parsed.value.object.get("wall_accounting").?);
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

fn checkPartition(value: std.json.Value) !void {
    const object = value.object;
    const total = object.get("total_ns").?.integer;
    try t.expectEqual(total, object.get("accounted_ns").?.integer);
    var previous: i64 = 0;
    for (object.get("segments").?.array.items) |segment| {
        try t.expectEqual(previous, segment.object.get("begin_ns").?.integer);
        previous = segment.object.get("end_ns").?.integer;
    }
    try t.expectEqual(total, previous);
}
const MtpState = struct {
    mtp_len: usize = 0,
    pub fn setMtpLen(state: *MtpState, _: cuda.Stream, count: usize) !void {
        state.mtp_len = count;
        Fake.clock += 1;
    }
};
const MtpEngine = struct { mtp: bool = true, stream: cuda.Stream, cur: struct { st: MtpState } = .{ .st = .{} } };
const MtpForward = struct {
    stream: cuda.Stream,
    pub fn mtpMode(f: *MtpForward, _: []const i32, _: []const u64, _: bool) !void {
        const layer = ledger.span(f.stream, .mtp_layer);
        defer layer.end();
        const projection = ledger.span(f.stream, .mtp_projection);
        Fake.clock += 2;
        projection.end();
    }
};
test "GLM prefill ledger: real paired MTP helper emits exact A B rows and restores commit context" {
    Fake.creates = 0;
    Fake.live = 0;
    Fake.clock = 0;
    Fake.ready = false;
    Fake.bad_time = false;
    Fake.failure = .none;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try prefix(tmp);
    defer t.allocator.free(dir);
    const path = try std.fmt.allocPrint(t.allocator, "{s}/paired", .{dir});
    defer t.allocator.free(path);
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    const trace = try ledger.Trace.init(&d, t.io, path, 0, 2, &.{.{ .kind = .dsa, .ffn = .moe }});
    defer trace.deinit();
    const first = ledger.enter(trace, stream, 9, 0, 16, 28, false);
    first.end();
    const chunk = ledger.enter(trace, stream, 9, 16, 12, 28, false);
    const side: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    for ([_]usize{ 0, 8 }, [_]usize{ 8, 4 }) |offset, rows| {
        const ffn = ledger.ffnPart(stream, 0, offset, rows);
        ledger.context(0, 0, .ffn);
        const shared = ledger.span(stream, .shared_expert);
        Fake.clock += 1;
        shared.end();
        ffn.end();
    }
    const joint = ledger.jointFfn(stream, 0, 8, 4);
    const experts = ledger.span(stream, .experts);
    Fake.clock += 2;
    experts.end();
    joint.end();
    for ([_]usize{ 0, 8 }, [_]usize{ 8, 4 }) |offset, rows| {
        const ffn = ledger.ffnPart(stream, 0, offset, rows);
        ledger.context(0, 0, .ffn);
        const output = ledger.span(stream, .output_projection);
        Fake.clock += 1;
        output.end();
        const partials = ledger.span(side, .partials);
        Fake.clock += 1;
        partials.end();
        ffn.end();
    }
    var e: MtpEngine = .{ .stream = stream };
    var f: MtpForward = .{ .stream = stream };
    const ids: [8]i32 = @splat(0);
    for ([_]usize{ 0, 8 }, [_]usize{ 8, 4 }, [_]usize{ 8, 3 }) |offset, rows, mtp_rows| {
        ledger.part(offset, rows);
        ledger.context(null, null, .chunk);
        try @import("cuda_prompt_pair.zig").mtpRows(&e, &f, ids[0..mtp_rows], 0x1000);
        const commit = ledger.span(stream, .commit);
        Fake.clock += 1;
        commit.end();
    }
    chunk.end();
    Fake.ready = true;
    ledger.poll(trace);
    try t.expectEqual(@as(usize, 11), e.cur.st.mtp_len);
    const data = try tmp.dir.readFileAlloc(t.io, "paired.rank0.json", t.allocator, .limited(131072));
    defer t.allocator.free(data);
    const json = try std.json.parseFromSlice(std.json.Value, t.allocator, data, .{});
    defer json.deinit();
    try t.expect(json.value.object.get("complete").?.bool);
    var mtp_count: usize = 0;
    var commit_count: usize = 0;
    var ffn_count: usize = 0;
    var joint_count: usize = 0;
    var shared_count: usize = 0;
    var output_count: usize = 0;
    var partial_count: usize = 0;
    for (json.value.object.get("rows").?.array.items) |row| {
        const object = row.object;
        const stage = object.get("stage").?.string;
        if (std.mem.eql(u8, stage, "ffn")) {
            try t.expectEqual(@as(i64, if (ffn_count % 2 == 0) 16 else 24), object.get("start").?.integer);
            try t.expectEqual(@as(i64, if (ffn_count % 2 == 0) 8 else 4), object.get("rows").?.integer);
            try t.expectEqualStrings("aggregate", object.get("class").?.string);
            try t.expectEqualStrings("dsa", object.get("layer_kind").?.string);
            try t.expectEqualStrings("moe", object.get("ffn_kind").?.string);
            try t.expect(object.get("piece").? == .null);
            ffn_count += 1;
        } else if (std.mem.eql(u8, stage, "joint_ffn") or std.mem.eql(u8, stage, "experts")) {
            try t.expectEqual(@as(i64, 16), object.get("start").?.integer);
            try t.expectEqual(@as(i64, 12), object.get("rows").?.integer);
            joint_count += 1;
        } else if (std.mem.eql(u8, stage, "shared_expert")) {
            try t.expectEqual(@as(i64, if (shared_count == 0) 16 else 24), object.get("start").?.integer);
            try t.expectEqual(@as(i64, if (shared_count == 0) 8 else 4), object.get("rows").?.integer);
            try t.expectEqualStrings("shared_expert", object.get("class").?.string);
            shared_count += 1;
        } else if (std.mem.eql(u8, stage, "output_projection") or std.mem.eql(u8, stage, "partials")) {
            const count = if (std.mem.eql(u8, stage, "partials")) &partial_count else &output_count;
            try t.expectEqual(@as(i64, if (count.* == 0) 16 else 24), object.get("start").?.integer);
            try t.expectEqual(@as(i64, if (count.* == 0) 8 else 4), object.get("rows").?.integer);
            try t.expectEqual(@as(i64, if (count == &partial_count) 1 else 0), object.get("stream").?.integer);
            count.* += 1;
        }
        if (std.mem.eql(u8, stage, "mtp")) {
            try t.expectEqual(@as(i64, if (mtp_count == 0) 16 else 24), object.get("start").?.integer);
            try t.expectEqual(@as(i64, if (mtp_count == 0) 8 else 3), object.get("rows").?.integer);
            try t.expectEqualStrings("mtp", object.get("site").?.string);
            try t.expectEqualStrings("mtp", object.get("layer_kind").?.string);
            mtp_count += 1;
        } else if (std.mem.eql(u8, stage, "commit")) {
            try t.expectEqual(@as(i64, if (commit_count == 0) 16 else 24), object.get("start").?.integer);
            try t.expectEqual(@as(i64, if (commit_count == 0) 8 else 4), object.get("rows").?.integer);
            try t.expectEqualStrings("chunk", object.get("site").?.string);
            commit_count += 1;
        }
    }
    try t.expectEqual(@as(usize, 4), ffn_count);
    try t.expectEqual(@as(usize, 2), joint_count);
    try t.expectEqual(@as(usize, 2), shared_count);
    try t.expectEqual(@as(usize, 2), output_count);
    try t.expectEqual(@as(usize, 2), partial_count);
    try t.expectEqual(@as(usize, 2), mtp_count);
    try t.expectEqual(@as(usize, 2), commit_count);
    try checkPartition(json.value.object.get("gpu_accounting").?);
    try checkPartition(json.value.object.get("wall_accounting").?);
}

test "GLM prefill ledger: event failures and capacity exhaustion preserve real launch bytes and refuse complete receipts" {
    Fake.calls = .empty;
    defer Fake.calls.deinit(t.allocator);
    Fake.failure = .none;
    Fake.ready = true;
    Fake.bad_time = false;
    Fake.clock = 0;
    defer Fake.failure = .none;
    var d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    try Fake.work(stream);
    const expected = try t.allocator.dupe(u64, Fake.calls.items);
    defer t.allocator.free(expected);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try prefix(tmp);
    defer t.allocator.free(dir);
    for ([_]@TypeOf(Fake.failure){ .none, .create, .record, .elapsed }, 0..) |failure, index| {
        Fake.failure = failure;
        Fake.clock = 0;
        Fake.creates = 0;
        try t.expectEqual(@as(usize, 0), Fake.live);
        Fake.calls.clearRetainingCapacity();
        const path = try std.fmt.allocPrint(t.allocator, "{s}/failure{d}", .{ dir, index });
        defer t.allocator.free(path);
        const trace = try ledger.Trace.init(&d, t.io, path, 0, 2, &.{});
        if (failure == .none) trace.max_entries = 2;
        const chunk = ledger.enter(trace, stream, 1, 0, 8, 8, false);
        try Fake.work(stream);
        chunk.end();
        trace.deinit();
        try t.expectEqualSlices(u64, expected, Fake.calls.items);
        try t.expectEqual(@as(usize, 0), Fake.live);
        const name = try std.fmt.allocPrint(t.allocator, "failure{d}.rank0.json", .{index});
        defer t.allocator.free(name);
        const data = try tmp.dir.readFileAlloc(t.io, name, t.allocator, .limited(131072));
        defer t.allocator.free(data);
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, data, .{});
        defer parsed.deinit();
        try t.expect(!parsed.value.object.get("complete").?.bool);
        if (failure == .none) try t.expect(parsed.value.object.get("dropped_stages").?.integer > 0) else {
            try t.expect(parsed.value.object.get("gpu_accounting").? == .null);
            for (parsed.value.object.get("rows").?.array.items) |row| try t.expect(row.object.get("cuda_event_ns").? == .null);
        }
    }
}
