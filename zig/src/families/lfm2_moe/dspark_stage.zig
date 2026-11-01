//! A lane's stage as the target's capture destination: readied before a forward, owned in place after it.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const ds = @import("dspark.zig");
const backend = @import("backend.zig");
const block = @import("dspark_block.zig");
const fwd = @import("forward.zig");

const be = lanes.backend;
const dr = lanes.drafter;

/// The stage row a forward's first row takes; a lane ahead of it rewinds, a full stage flushes.
pub const Plan = struct { slot: u32, rewind: bool = false, flush: bool = false };

/// What a capture of `rows` from `start` asks of a lane holding [end, end + staged) (null: refused).
pub fn plan(end: u64, staged: u32, start: u64, rows: usize, stage_cap: usize) ?Plan {
    if (rows > stage_cap) return null;
    if (start < end) return .{ .slot = 0, .rewind = true };
    const keep = start - end;
    if (keep > staged) return null;
    if (keep + rows <= stage_cap) return .{ .slot = @intCast(keep) };
    return .{ .slot = 0, .flush = true };
}

fn room(l: *const ds.Lane) usize {
    return @min(ds.stage_rows, l.cap);
}

fn cast(ptr: *anyopaque) *ds.Drafter {
    return @ptrCast(@alignCast(ptr));
}

pub fn source(d: *ds.Drafter) backend.StageSource {
    return .{ .ptr = d, .bind = bind, .buffer = buffer };
}

/// The stream's stage ready for a forward's rows from `start`: rows before `start` kept, the rest replaced.
fn bind(ptr: *anyopaque, s: *lanes.Stream, start: u64, rows: usize) anyerror!?backend.Stage {
    const d = cast(ptr);
    const l = d.lanes.get(s) orelse return null;
    if (start > l.end + l.staged) try d.resumeLane(s, l, start); // a prompt pass that resumed
    const p = plan(l.end, l.staged, start, rows, room(l)) orelse return null;
    if (p.rewind) {
        l.end = start;
        l.staged = 0;
    } else if (p.flush) {
        l.staged = @intCast(start - l.end);
        try d.settle();
        try d.flush(l);
    } else l.staged = p.slot;
    const wait: ?fwd.Wait = if (d.pending != null) .{ .event = d.event, .value = d.signalled } else null;
    return .{ .place = .{ .buf = l.mem, .off = l.stage_at + @as(usize, p.slot) * d.tapBytes() }, .wait = wait };
}

fn buffer(ptr: *anyopaque, s: *lanes.Stream) ?mtl.Buffer {
    const l = cast(ptr).lanes.get(s) orelse return null;
    return l.mem;
}

/// Whether `f` is rows of the lane's own stage at their place (an error when it is its stage but not there).
fn owned(l: *const ds.Lane, f: be.Features, start: u64, width: usize) !bool {
    if (f.buffer != @intFromPtr(l.mem.id)) return false;
    if (start < l.end) return error.DraftContextGap;
    const at = start - l.end;
    if (at > l.staged or at + f.rows > room(l) or f.offset != l.stage_at + at * width) return error.DraftContextGap;
    return true;
}

/// Owns taps before the target overwrites them; materializes context on hold, snapshot or overflow.
pub fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
    const d = cast(ptr);
    try d.settle();
    const width = d.tapBytes();
    // every item is accepted before any row is staged or encoded: a refused call leaves no command buffer open
    for (items) |it| {
        const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
        const f = it.features;
        if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width or it.rows != null) return error.DraftFeaturesUnsupported;
        if (it.start > l.end + l.staged) try d.resumeLane(it.stream, l, it.start); // a prompt pass that resumed
        if (it.start + f.rows + d.config.block > l.cap) return error.DraftContextGap;
        _ = try owned(l, f, it.start, width);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var run: ?@TypeOf(block.begin(d)) = null;
    errdefer if (run) |r| r.e.end(); // an encoding error closes the open encoder
    var pieces: [32]block.Piece = undefined;
    var count: usize = 0;
    var rows: usize = 0;
    for (items) |it| {
        const l = d.lanes.get(it.stream).?;
        const f = it.features;
        l.held_len = 0;
        l.slot = null;
        const src = mtl.Buffer{ .id = @ptrFromInt(f.buffer) };
        // rows from `start` replace what was staged past it; rows before the absorbed end go to the GPU now
        const keep: u32 = if (it.start >= l.end) @intCast(it.start - l.end) else 0;
        l.staged = @min(l.staged, keep);
        if (it.start >= l.end and l.staged + f.rows <= room(l)) {
            // rows captured into the stage are in place; others are copied in before the target overwrites them
            if (src.id != l.mem.id) @memcpy((l.mem.contents() + l.stage_at + l.staged * width)[0 .. f.rows * width], (src.contents() + f.offset)[0 .. f.rows * width]);
            l.staged += f.rows;
            continue;
        }
        if (run == null) run = block.begin(d);
        const e = run.?.e;
        if (ds.Drafter.stagedPiece(l)) |pc| {
            if (count == pieces.len or rows + pc.rows > ds.absorb_rows) {
                try block.absorbPass(d, e, pieces[0..count]);
                count = 0;
                rows = 0;
            }
            pieces[count] = pc;
            count += 1;
            rows += pc.rows;
        }
        var at: usize = 0;
        while (at < f.rows) {
            if (rows == ds.absorb_rows or count == pieces.len) { // the pass is full: encode it
                try block.absorbPass(d, e, pieces[0..count]);
                count = 0;
                rows = 0;
            }
            const n = @min(ds.absorb_rows - rows, f.rows - at);
            pieces[count] = .{ .lane = l, .src = .{ .buf = src, .off = f.offset + at * width }, .rows = n, .p0 = it.start + at };
            count += 1;
            rows += n;
            at += n;
        }
        l.end = it.start + f.rows;
        l.staged = 0;
    }
    const r = run orelse return;
    if (count > 0) try block.absorbPass(d, r.e, pieces[0..count]);
    run = null; // `end` closes the encoder, even when the buffer fails
    try block.end(d, r.cb, r.e);
}

test "a forward's rows take the stage row after the kept ones, and a lane ahead of them rewinds" {
    // rows [4, 6) staged on an absorbed end of 4: a forward from 6 appends, from 5 replaces the last row, from 4 all
    try std.testing.expectEqual(Plan{ .slot = 2 }, plan(4, 2, 6, 3, 16).?);
    try std.testing.expectEqual(Plan{ .slot = 1 }, plan(4, 2, 5, 3, 16).?);
    try std.testing.expectEqual(Plan{ .slot = 0 }, plan(4, 2, 4, 3, 16).?);
    // ahead of the lane: a gap is refused, a start before the absorbed end rewinds it
    try std.testing.expectEqual(@as(?Plan, null), plan(4, 2, 7, 1, 16));
    try std.testing.expectEqual(Plan{ .slot = 0, .rewind = true }, plan(4, 2, 3, 1, 16).?);
    try std.testing.expectEqual(Plan{ .slot = 0, .rewind = true }, plan(4, 0, 0, 16, 16).?);
}

test "a full stage flushes only the rows before the forward's start, and refuses more rows than it has" {
    // 14 staged of 16: a 3-row forward at the end flushes them and starts at row 0
    try std.testing.expectEqual(Plan{ .slot = 0, .flush = true }, plan(0, 14, 14, 3, 16).?);
    // a forward that reuses the anchor at 13 still fits after 13 kept rows, or flushes only those 13
    try std.testing.expectEqual(Plan{ .slot = 13 }, plan(0, 14, 13, 3, 16).?);
    try std.testing.expectEqual(Plan{ .slot = 0, .flush = true }, plan(0, 14, 13, 4, 16).?);
    try std.testing.expectEqual(@as(?Plan, null), plan(0, 0, 0, 17, 16));
    try std.testing.expectEqual(Plan{ .slot = 13 }, plan(100, 16, 113, 3, 16).?);
}
