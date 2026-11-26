//! TF_Q35_DFLASH_TRACE=<file>: one exact, versioned JSON line per verified DFlash proposal.
const std = @import("std");
const lanes = @import("lanes");
const platform = @import("sys");

const Allocator = std.mem.Allocator;
const Candidates = lanes.draft_tree.Candidates;
const lp_top = lanes.draft_tree.lp_top;
pub const lp_words = 3 + 2 * lp_top;
pub const ids_offset = 2 * 4;
pub const unary_offset = (2 + lp_top) * 4;

/// Borrowed host values: proposal capture precedes selection-base reuse; accepted rows arrive at absorption.
pub const Round = struct {
    stream: []const u8,
    position: u64,
    pending: u32,
    offered: usize,
    nodes: []const u32,
    parents: []const i32,
    own: []const f64,
    accepted_rows: []const u32,
    accepted_tokens: []const u32,
    candidates: Candidates,
};

/// Bit-preserving wire form; `own` is f64 bits, `unary` f32 bits and `hp` fixed-width BF16 hex.
pub const Row = struct { ids: []const u32 = &.{}, unary: []const u32 = &.{}, hp: []const u8 = "" };
pub const Record = struct {
    v: u8,
    stream: []const u8 = "",
    position: u64 = 0,
    pending: u32 = 0,
    offered: u32 = 0,
    rank: u32 = 0,
    topk: u32 = 0,
    nodes: []const u32 = &.{},
    parents: []const i32 = &.{},
    own: []const u64 = &.{},
    candidates: []const Row = &.{},
    accepted_rows: []const u32 = &.{},
    accepted_tokens: []const u32 = &.{},
    accepted_nodes: []const i32 = &.{},
};

pub fn parse(gpa: Allocator, text: []const u8) !std.json.Parsed(Record) {
    const got = try std.json.parseFromSlice(Record, gpa, text, .{});
    errdefer got.deinit();
    if (got.value.v != 1) return error.UnsupportedTraceVersion;
    return got;
}

fn array(w: anytype, values: anytype) !void {
    try w.writeAll("[");
    for (values, 0..) |value, i| {
        if (i != 0) try w.writeByte(',');
        try w.print("{d}", .{value});
    }
    try w.writeByte(']');
}

fn ownArray(w: anytype, values: []const f64) !void {
    try w.writeAll("[");
    for (values, 0..) |value, i| {
        if (i != 0) try w.writeByte(',');
        try w.print("{d}", .{@as(u64, @bitCast(value))});
    }
    try w.writeByte(']');
}

fn validate(r: Round) !void {
    if (r.candidates.table or r.candidates.apart != null or r.candidates.topk > lp_top or
        r.nodes.len != r.parents.len or r.nodes.len != r.own.len or r.offered > r.nodes.len or
        r.accepted_rows.len != r.accepted_tokens.len) return error.BadDraftTrace;
    const row_bytes = r.candidates.layout.words * 4;
    if (row_bytes < unary_offset + r.candidates.topk * 4 or r.candidates.words.len % row_bytes != 0 or
        r.candidates.projected.len != (r.candidates.words.len / row_bytes) * r.candidates.rank * 2) return error.BadDraftTrace;
}

/// One complete line, built before any file write so an invalid or failed record is never half a line.
pub fn writeRecord(w: anytype, r: Round) !void {
    try validate(r);
    try w.writeAll("{\"v\":1,\"stream\":");
    try std.json.Stringify.value(r.stream, .{}, w);
    try w.print(",\"position\":{d},\"pending\":{d},\"offered\":{d},\"rank\":{d},\"topk\":{d},\"nodes\":", .{ r.position, r.pending, r.offered, r.candidates.rank, r.candidates.topk });
    try array(w, r.nodes);
    try w.writeAll(",\"parents\":");
    try array(w, r.parents);
    try w.writeAll(",\"own\":");
    try ownArray(w, r.own);
    try w.writeAll(",\"candidates\":[");
    const layout = r.candidates.layout;
    const rows = r.candidates.words.len / (layout.words * 4);
    for (0..rows) |i| {
        if (i != 0) try w.writeByte(',');
        const words = std.mem.bytesAsSlice(u32, r.candidates.words[i * layout.words * 4 ..][0 .. layout.words * 4]);
        try w.writeAll("{\"ids\":");
        try array(w, words[layout.ids..][0..r.candidates.topk]);
        try w.writeAll(",\"unary\":");
        try array(w, words[layout.logits..][0..r.candidates.topk]);
        try w.writeAll(",\"hp\":\"");
        const hp = std.mem.bytesAsSlice(u16, r.candidates.projected[i * r.candidates.rank * 2 ..][0 .. r.candidates.rank * 2]);
        for (hp) |value| try w.print("{x:0>4}", .{value});
        try w.writeAll("\"}");
    }
    try w.writeAll("],\"accepted_rows\":");
    try array(w, r.accepted_rows);
    try w.writeAll(",\"accepted_tokens\":");
    try array(w, r.accepted_tokens);
    try w.writeAll(",\"accepted_nodes\":[");
    for (r.accepted_rows, 0..) |row, i| {
        if (i != 0) try w.writeByte(',');
        const node: i32 = if (row > 0 and row <= r.offered and row <= r.nodes.len) @intCast(row - 1) else -1;
        try w.print("{d}", .{node});
    }
    try w.writeAll("]}\n");
}

pub fn drop(slot: *?*Trace) void {
    if (slot.*) |t| {
        t.file.close(t.io);
        t.gpa.destroy(t);
        slot.* = null;
    }
}

pub const Trace = struct {
    gpa: Allocator,
    io: std.Io,
    file: std.Io.File,
    broken: bool = false,

    pub fn fromEnv(gpa: Allocator, io: std.Io) ?*Trace {
        const name = "TF_Q35_DFLASH_TRACE";
        const path = platform.environment.path(name, platform.environment.rawText(name)) catch {
            std.log.warn("TF_Q35_DFLASH_TRACE: invalid path; no trace", .{});
            return null;
        };
        return openPath(gpa, io, path) catch |e| {
            std.log.warn("TF_Q35_DFLASH_TRACE: cannot open the file ({t}); no trace", .{e});
            return null;
        };
    }

    pub fn line(t: *Trace, r: Round) void {
        if (t.broken) return;
        var text: std.Io.Writer.Allocating = .init(t.gpa);
        defer text.deinit();
        writeRecord(&text.writer, r) catch |e| {
            t.broken = true;
            std.log.warn("TF_Q35_DFLASH_TRACE: invalid round ({t}); the trace stops here", .{e});
            return;
        };
        const n = t.file.writeStreaming(t.io, &.{}, &.{text.written()}, 1) catch 0;
        if (n == text.written().len) return;
        t.broken = true;
        std.log.warn("TF_Q35_DFLASH_TRACE: a write gave {d} of {d} bytes; the trace stops here", .{ n, text.written().len });
    }
};

pub fn openPath(gpa: Allocator, io: std.Io, path_: ?[]const u8) !?*Trace {
    const path = path_ orelse return null;
    if (path.len == 0) return null;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    errdefer file.close(io);
    const t = try gpa.create(Trace);
    t.* = .{ .gpa = gpa, .io = io, .file = file };
    return t;
}
