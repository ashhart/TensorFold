//! Prompt stage metadata; no callback adapter or scheduler ownership.
const cuda = @import("cuda");
const trace = cuda.prefill_attribution;
pub const Ticket = trace.Ticket;
pub fn open(self: anytype) !void {
    const on = try @import("sys").environment.read(bool, "TF_CUDA_PREFILL_TRACE", false);
    try trace.Trace.create(self.gpa, self.io, self.e.d, self.e.stream.handle, on);
}
pub fn begin(stage: trace.ledger.Stage, gpu: bool) Ticket {
    const t = trace.owner orelse return null;
    return t.begin(.{ .stage = stage }, t.stream, gpu);
}
pub fn end(ticket: Ticket) void {
    if (trace.owner) |t| t.end(ticket, 0);
}
pub fn chunk(e: anytype, at: usize, rows: usize, unread: bool) Ticket {
    const t = trace.owner orelse return null;
    const shape = @import("cuda_shapes.zig").Shape.init(rows, 1, false, false, 1);
    return t.begin(.{ .stage = .target_chunk, .start_token = at, .rows = rows, .layers_run = e.layers.len, .unread_last_layer = unread, .skipped_last_layer_output_rows = if (unread) rows else 0, .shape_bucket_rows = shape.total, .shape_bucket_segments = shape.nseg }, t.stream, true);
}
pub fn restored(at: usize, shared: bool, restore_failed: bool) void {
    const t = trace.owner orelse return;
    t.restoration = .{ .kind = if (at == 0) .fresh else if (shared) .shared_prefix else .snapshot, .tokens_restored = at, .kv_restored = at > 0, .recurrent_restored = at > 0, .restore_failed = restore_failed };
}

pub fn failed(ticket: Ticket) void {
    if (trace.owner) |t| if (ticket) |index| {
        t.entries.items[index].row.complete = false;
    };
}
