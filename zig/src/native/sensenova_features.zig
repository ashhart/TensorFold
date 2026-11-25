//! Target tap feature capture and slice lookup for external drafters (e.g. SenseNova MTP).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const be = lanes.backend;
const lb = @import("qwen27_lanes.zig");
const Metal = lb.Metal;

pub const State = struct {
    enabled: bool = false,
    taps: [q.taps.max_planes]u32 = @splat(0),
    tap_count: usize = 0,
    sink: ?be.FeatureSink = null,
    chunk_slot: u32 = 0,
    chunk_start: u64 = 0,
    chunk_rows: u32 = 0,
    chunk_valid: bool = false,
    held_start: [lb.max_streams]u64 = @splat(0),
    held_rows: [lb.max_streams]u32 = @splat(0),

    pub fn clearSlot(st: *State, slot: u32) void {
        if (slot < lb.max_streams) st.held_rows[slot] = 0;
        if (st.chunk_valid and st.chunk_slot == slot) st.chunk_valid = false;
    }

    fn matchesTaps(st: *const State, wanted: []const u32) bool {
        return st.enabled and wanted.len == st.tap_count and std.mem.eql(u32, wanted, st.taps[0..st.tap_count]);
    }
};

fn rowBytes(b: *const Metal) usize {
    return @as(usize, b.runner.taps.width) * b.features_state.tap_count * 2;
}

fn slotOffset(b: *const Metal, slot: u32, row: usize) usize {
    return (@as(usize, slot) * lb.max_rows + row) * rowBytes(b);
}

pub fn prepareFeaturesFn(ptr: *anyopaque, taps: []const u32) anyerror!void {
    const b = Metal.self(ptr);
    if (b.features_state.enabled) {
        if (b.features_state.matchesTaps(taps)) return;
        return error.TapsNotPrepared;
    }
    for (b.owners[0..b.runner.slots]) |o| if (o != null) return error.TapsAfterStreams;
    try b.runner.taps.use(b.runner.model.device, taps);
    b.runner.capture_taps = true;
    const per_row = @as(usize, b.runner.taps.width) * taps.len * 2;
    const need = lb.max_streams * lb.max_rows * per_row;
    if (b.held_taps == null) {
        b.held_taps = try b.runner.model.device.buffer(need, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    }
    @memcpy(b.features_state.taps[0..taps.len], taps);
    b.features_state.tap_count = taps.len;
    b.features_state.enabled = true;
}

pub fn prefillFeaturesFn(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, sink: be.FeatureSink) anyerror!void {
    const b = Metal.self(ptr);
    if (!b.features_state.matchesTaps(taps)) return error.TapsNotPrepared;
    b.features_state.sink = sink;
    defer b.features_state.sink = null;
    return b.backend().prefill(s);
}

pub fn recordChunk(b: *Metal, s: *lanes.Stream, slot: u32, start: usize, count: usize, last: bool) !void {
    const st = &b.features_state;
    if (!st.enabled or count == 0) return;
    try b.runner.taps.complete();
    st.chunk_slot = slot;
    st.chunk_start = start;
    st.chunk_rows = @intCast(count);
    st.chunk_valid = true;
    st.held_rows[slot] = 0;
    if (st.sink) |sink| {
        const rb = rowBytes(b);
        try sink.take(sink.ptr, s, .{
            .buffer = @intFromPtr(b.runner.taps.buffer.id),
            .offset = 0,
            .rows = @intCast(count),
            .row_bytes = @intCast(rb),
            .space = .device,
            .dtype = .bf16,
        }, start);
    }
    if (last) {
        copyRows(b, slot, 0, count - 1, 1);
        st.held_start[slot] = start + count - 1;
        st.held_rows[slot] = 1;
    }
}

pub fn recordVerify(b: *Metal, windows: []const be.Window, inputs: []const q.round_plan.Input) !void {
    const st = &b.features_state;
    if (!st.enabled) return;
    try b.runner.taps.complete();
    st.chunk_valid = false;
    for (windows, inputs, 0..) |w, in, i| {
        const rows: u32 = @intCast(w.rows());
        const first = b.round.?.firsts[i];
        copyRows(b, in.slot, 0, first, rows);
        st.held_start[in.slot] = in.start;
        st.held_rows[in.slot] = rows;
    }
}

pub fn recordKeep(b: *Metal, windows: []const be.Window, paths: []const []const u32) void {
    const st = &b.features_state;
    if (!st.enabled or b.advanced) return;
    for (windows, paths) |w, path| {
        const slot = b.slotOf(w.stream) orelse continue;
        var highest: u32 = 0;
        for (path) |row| highest = @max(highest, row);
        st.held_rows[slot] = @min(st.held_rows[slot], highest + 1);
    }
}

pub fn featuresFn(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
    const b = Metal.self(ptr);
    const st = &b.features_state;
    if (!st.matchesTaps(taps)) return error.TapsNotPrepared;
    const slot = try b.slotFor(s);
    const rb = rowBytes(b);
    if (count == 0) {
        return .{ .buffer = @intFromPtr(b.held_taps.?.id), .offset = slotOffset(b, slot, 0), .rows = 0, .row_bytes = @intCast(rb), .space = .device, .dtype = .bf16 };
    }
    if (st.held_rows[slot] > 0 and start >= st.held_start[slot] and start + count <= st.held_start[slot] + st.held_rows[slot]) {
        const rel: usize = @intCast(start - st.held_start[slot]);
        return .{ .buffer = @intFromPtr(b.held_taps.?.id), .offset = slotOffset(b, slot, rel), .rows = count, .row_bytes = @intCast(rb), .space = .device, .dtype = .bf16 };
    }
    if (st.chunk_valid and st.chunk_slot == slot and st.tap_count == 1 and start >= st.chunk_start and start + count <= st.chunk_start + st.chunk_rows) {
        const rel: usize = @intCast(start - st.chunk_start);
        return .{ .buffer = @intFromPtr(b.runner.taps.buffer.id), .offset = rel * rb, .rows = count, .row_bytes = @intCast(rb), .space = .device, .dtype = .bf16 };
    }
    return error.RowsNotHeld;
}

fn copyRows(b: *Metal, dst_slot: u32, dst_row: usize, src_row: usize, rows: usize) void {
    const t = &b.runner.taps;
    const w_bytes = @as(usize, t.width) * 2;
    const plane_bytes = @as(usize, t.capacity) * w_bytes;
    const src_base = t.buffer.contents();
    const dst_base = b.held_taps.?.contents() + slotOffset(b, dst_slot, dst_row);
    if (t.count == 1) {
        @memcpy(dst_base[0 .. rows * w_bytes], src_base[src_row * w_bytes ..][0 .. rows * w_bytes]);
        return;
    }
    for (0..rows) |r| {
        for (0..t.count) |p| {
            @memcpy(dst_base[(r * t.count + p) * w_bytes ..][0..w_bytes], src_base[p * plane_bytes + (src_row + r) * w_bytes ..][0..w_bytes]);
        }
    }
}
