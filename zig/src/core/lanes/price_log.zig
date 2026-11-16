//! TF_SHARED_PRICE_LOG=1: the load-time costs once, then a line a priced shared round with what the plain rule saw.
const std = @import("std");
const Writer = std.Io.Writer;

pub const name = "TF_SHARED_PRICE_LOG";

/// Read at an engine's first priced shared round, which then logs the load-time costs; bad text fails it by name.
pub fn enabled(e: anytype) !bool {
    if (e.price_log) |on| return on;
    const on = try @import("sys").environment.read(bool, name, false);
    e.price_log = on;
    if (on) try emit(e.gpa, load, .{&e.rule});
    return on;
}

/// A timed round as the plain rule observes it: `ms` its time less extra draft steps, `head_ms` its priced head steps.
pub const Round = struct { trial: bool, sampled: bool, ms: f64, head_ms: f64, heads: []const usize, depths: []const i64 };

pub fn round(e: anytype, plans: anytype, windows: anytype, r: Round) !void {
    try emit(e.gpa, line, .{ &e.rule, plans, windows, r });
}

fn emit(gpa: std.mem.Allocator, comptime write: anytype, args: anytype) !void {
    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    try @call(.auto, write, .{&out.writer} ++ args);
    std.log.info("{s}", .{out.written()});
}

/// The shared forward's load-time ms by rows, then the head's chained, shared and extra draft steps.
pub fn load(w: *Writer, r: anytype) !void {
    try w.writeAll("shared price: load shared_ms=");
    var first = true;
    for (r.shared_costs.slots, 0..) |slot, rows| if (slot) |ms| {
        try w.print("{s}{d}:{d:.3}", .{ if (first) "" else ",", rows, ms });
        first = false;
    };
    try w.print(" step_ms={d:.3} shared_step_ms={d:.3} draft_step_ms={d:.3}", .{ r.mtp_step_ms, r.sharedStepMs(), r.draft_step_ms });
}

/// A round's trial flag, rows and time, each stream's verified drafts and next depth, the price that chose them.
pub fn line(w: *Writer, rule: anytype, plans: anytype, windows: anytype, r: Round) !void {
    var rows: usize = 0;
    for (windows) |x| rows += x.rows();
    try w.print("shared price: round streams={d} sampled={d} trial={d} rows={d} ms={d:.3} head={d:.3} forward=", .{ plans.len, @intFromBool(r.sampled), @intFromBool(r.trial), rows, r.ms, r.head_ms });
    if (rule.shared_costs.get(@intCast(rows))) |v| try w.print("{d:.3}", .{v}) else try w.writeByte('-');
    if (rule.sharedPrice(plans.len, r.sampled)) |p| try w.print(" plain={d:.3} row={d:.3}", .{ p.plain, p.row }) else try w.writeAll(" plain=- row=-");
    try w.writeAll(" drafts=");
    for (windows, 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", x.rows() - 1 });
    try w.writeAll(" depths=");
    for (0..plans.len) |i| {
        if (i > 0) try w.writeByte(',');
        const at = std.mem.indexOfScalar(usize, r.heads, i) orelse {
            try w.writeByte('-');
            continue;
        };
        try w.print("{d}", .{r.depths[at]});
    }
    // each stream's acceptance estimate by depth, `/` between depths
    try w.writeAll(" accept=");
    for (plans, 0..) |p, i| {
        if (i > 0) try w.writeByte(',');
        const st = p.stream.depth orelse {
            try w.writeByte('-');
            continue;
        };
        for (st.p, 0..) |v, j| try w.print("{s}{d:.3}", .{ if (j == 0) "" else "/", v });
    }
}

test {
    _ = @import("price_log_test.zig");
}
