//! Stage receipts for a single prompt; event intervals overlap their nested stages and are never additive.
const std = @import("std");
pub const Stage = enum { request, cache_allocate, cache_publish, restore, scratch, target_chunk, drafter_open, tap_upload, drafter_absorb, trailing_absorb, host_sync, d2h, h2d, d2d };
pub const Memory = struct {
    physical_source: []const u8 = "linux_proc_self_status",
    physical_peak_scope: []const u8 = "process_lifetime_including_trace",
    rss_bytes: ?u64 = null,
    physical_peak_bytes: ?u64 = null,
    device_bytes: u64 = 0,
    device_peak_bytes: u64 = 0,
    pinned_bytes: u64 = 0,
    pinned_peak_bytes: u64 = 0,
};
pub const Resume = struct {
    kind: enum { unknown, fresh, shared_prefix, snapshot } = .unknown,
    tokens_restored: u64 = 0,
    kv_restored: bool = false,
    recurrent_restored: bool = false,
    restore_failed: bool = false,
};
pub const Row = struct {
    pub fn format(row: Row, writer: *std.Io.Writer) !void {
        return write(writer, row);
    }
    schema: []const u8 = "tensorfold.cuda-prefill-stage.v1",
    request: u64 = 1,
    sequence: u64 = 0,
    parent: ?u64 = null,
    stage: Stage,
    wall_begin_ns: u64 = 0,
    wall_end_ns: u64 = 0,
    wall_ns: u64 = 0,
    cuda_event_ns: ?u64 = null,
    event_status: enum { not_launched, pending, complete, capture_only, unavailable } = .not_launched,
    complete: bool = true,
    stream: u64 = 0,
    start_token: u64 = 0,
    rows: u64 = 0,
    layers_run: u64 = 0,
    unread_last_layer: bool = false,
    skipped_last_layer_output_rows: u64 = 0,
    shape_bucket_rows: u64 = 0,
    shape_bucket_segments: u32 = 0,
    graph_captures: u64 = 0,
    graph_replays: u64 = 0,
    graph_mode: enum { eager, capture, replay, capture_and_replay } = .eager,
    launches: u64 = 0,
    bytes: u64 = 0,
    operation: []const u8 = "",
    result: i32 = 0,
    scratch_rows: u64 = 0,
    scratch_bytes: u64 = 0,
    prompt_tokens: u64 = 0,
    restoration: Resume = .{},
    memory: Memory = .{},
    dropped_stages: u64 = 0,
};
pub fn write(writer: *std.Io.Writer, row: Row) !void {
    try std.json.Stringify.value(row, .{}, writer);
    try writer.writeByte('\n');
}
pub fn procStatus(text: []const u8) Memory {
    var m: Memory = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t");
        const name = words.next() orelse continue;
        const is_rss = std.mem.eql(u8, name, "VmRSS:");
        if (!is_rss and !std.mem.eql(u8, name, "VmHWM:")) continue;
        const kb = std.fmt.parseInt(u64, words.next() orelse continue, 10) catch continue;
        if (!std.mem.eql(u8, words.next() orelse continue, "kB")) continue;
        const bytes = std.math.mul(u64, kb, 1024) catch continue;
        if (is_rss) m.rss_bytes = bytes else m.physical_peak_bytes = bytes;
    }
    return m;
}
