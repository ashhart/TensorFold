//! tf-drama-tower-context-trace's tracer and the per-schedule store of every stage capture, with their checks.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const Allocator = std.mem.Allocator;
const tower = drama.tower;
const Stage = tower.Stage;
const context_trace = @import("drama_tower_context_trace.zig");
const total_rows = context_trace.total_rows;
const layer_count = context_trace.layer_count;
const stages = context_trace.stages;
const stage_count = context_trace.stage_count;
const per_layer = context_trace.per_layer;
const captures_per_call = context_trace.captures_per_call;
const Errors = context_trace.Errors;
const differences = context_trace.differences;

/// Every stage of one schedule at request positions 0..187; coverage counts how often each row was written.
pub const Store = struct {
    data: [layer_count][stage_count][]u16,
    coverage: [layer_count][stage_count][]u8,

    pub fn alloc(arena: Allocator) !Store {
        var s: Store = undefined;
        for (0..layer_count) |layer| for (stages, 0..) |stage, k| {
            const count: usize = if (stage == .shared_input and layer != 0) 0 else total_rows;
            s.data[layer][k] = try arena.alloc(u16, count * tower.stageWidth(stage));
            s.coverage[layer][k] = try arena.alloc(u8, count);
            @memset(s.coverage[layer][k], 0);
        };
        return s;
    }

    pub fn rows(s: *const Store, layer: usize, k: usize, row0: usize, n: usize) []u16 {
        const w = tower.stageWidth(stages[k]);
        return s.data[layer][k][row0 * w .. (row0 + n) * w];
    }

    pub fn captured(layer: usize, k: usize) bool {
        return !(stages[k] == .shared_input and layer != 0);
    }
};

pub const Tracer = struct {
    store: *Store,
    compare: bool, // false: record into store; true: count differences against it
    staging: []align(16) u8,
    call_row0: usize = 0,
    call_rows: usize = 0,
    calls: usize = 0,
    expected: usize = 0,
    out_of_order: usize = 0,
    incomplete_calls: usize = 0,
    bad_rows: usize = 0,
    duplicate_rows: usize = 0,
    compare_differences: usize = 0,

    pub fn begin(t: *Tracer, row0: usize, rows: usize) void {
        t.call_row0 = row0;
        t.call_rows = rows;
        t.expected = 0;
    }

    pub fn end(t: *Tracer) void {
        if (t.expected != captures_per_call) t.incomplete_calls += 1;
        t.calls += 1;
    }

    pub fn capture(t: *Tracer, stream: cuda.Stream, layer: usize, stage: Stage, src: cuda.DeviceBuffer, row0: usize) !void {
        const k = @backingInt(stage);
        const index = if (stage == .shared_input) 0 else 1 + layer * per_layer + (k - 1);
        if (index != t.expected or layer >= layer_count or (stage == .shared_input and layer != 0)) t.out_of_order += 1;
        t.expected = index + 1;
        if (layer >= layer_count or !Store.captured(layer, k)) return;
        const bytes_len = t.call_rows * tower.stageWidth(stage) * 2;
        if (row0 != t.call_row0 or src.len != bytes_len or row0 + t.call_rows > total_rows) {
            t.bad_rows += 1;
            return;
        }
        if (bytes_len > t.staging.len) return error.TraceCaptureShape;
        try src.downloadAsync(0, t.staging[0..bytes_len], stream.handle);
        try stream.synchronize();
        const dst = t.store.rows(layer, k, row0, t.call_rows);
        if (t.compare) {
            t.compare_differences += differences(dst, std.mem.bytesAsSlice(u16, t.staging[0..bytes_len]));
        } else {
            @memcpy(std.mem.sliceAsBytes(dst), t.staging[0..bytes_len]);
            for (t.store.coverage[layer][k][row0 .. row0 + t.call_rows]) |*c| {
                if (c.* != 0) t.duplicate_rows += 1;
                c.* +|= 1;
            }
        }
    }
};

pub fn checkTracer(t: *const Tracer, label: []const u8, expected_calls: usize, errors: *Errors) !void {
    if (t.calls != expected_calls or t.out_of_order != 0 or t.incomplete_calls != 0 or t.bad_rows != 0 or t.duplicate_rows != 0)
        try errors.add("{s}: calls {d} (expected {d}), out of order {d}, incomplete calls {d}, bad rows {d}, duplicate rows {d}", .{
            label, t.calls, expected_calls, t.out_of_order, t.incomplete_calls, t.bad_rows, t.duplicate_rows,
        });
    if (t.compare_differences != 0) try errors.add("{s}: {d} captured values differ from the first traced run", .{ label, t.compare_differences });
}

pub fn uncovered(store: *const Store) usize {
    var count: usize = 0;
    for (0..layer_count) |layer| for (0..stage_count) |k| {
        for (store.coverage[layer][k]) |c| {
            if (c != 1) count += 1;
        }
    };
    return count;
}
