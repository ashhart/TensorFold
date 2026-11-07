//! The 27B's tapped rows for a drafter in front of it (lanes.drafted): the last pass's rows by slot, on the device.
const tf = @import("tensorfold");
const lanes = tf.lanes;
const be = lanes.backend;
const Metal = @import("qwen27_lanes.zig").Metal;

/// The drafter's layers: every pass keeps their rows from now on (one layer: a row is contiguous).
pub fn prepare(ptr: *anyopaque, taps: []const u32) anyerror!void {
    const b = Metal.self(ptr);
    if (b.draft != null or taps.len != 1) return error.BadTapLayers;
    const r = b.runner;
    try r.taps.use(r.model.device, taps);
    r.capture_taps = true;
    b.external = true;
}

/// Cache rows [start, start + count) of the stream's last pass, where the taps buffer holds them.
pub fn rows(ptr: *anyopaque, s: *lanes.Stream, taps: []const u32, start: u64, count: u32) anyerror!be.Features {
    const b = Metal.self(ptr);
    const r = b.runner;
    if (!b.external or taps.len != 1 or r.taps.count != 1 or r.taps.ids[0] != taps[0]) return error.UnpreparedTaps;
    const t = r.tapped[try b.slotFor(s)];
    if (count == 0 or start < t.start or start + count > @as(u64, t.start) + t.rows) return error.TapsNotHeld;
    const width: u64 = @as(u64, r.taps.width) * 2;
    return .{ .buffer = @intFromPtr(r.taps.buffer.id), .offset = (t.first + (start - t.start)) * width, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
}
