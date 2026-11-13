//! Image SSE frames under OpenAI's image_generation event names, each built in its own short-lived arena.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const sse = @import("sse.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;
const log = @import("log.zig");
const openai = @import("openai.zig");

pub const Writer = struct {
    conn: *Conn,
    started: bool = false,

    pub fn start(w: *Writer) error{Closed}!void {
        if (w.started) return;
        try sse.open(w.conn);
        w.started = true;
        w.conn.response_complete = false;
    }
    fn item(a: std.mem.Allocator, index: usize, seed: u64, kind: []const u8) !*json.Object {
        const out = try json.newObject(a);
        try out.put(a, "type", .{ .string = kind });
        try out.put(a, "index", try json.intValue(a, index));
        try out.put(a, "seed", try json.intValue(a, seed));
        return out;
    }
    fn encoded(a: std.mem.Allocator, bytes: []const u8) !json.Value {
        const buffer = try a.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
        return .{ .string = std.base64.standard.Encoder.encode(buffer, bytes) };
    }
    pub fn progress(w: *Writer, index: usize, seed: u64, step: u32, total: u32) error{Closed}!void {
        var arena: std.heap.ArenaAllocator = .init(w.conn.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const out = item(a, index, seed, "image_generation.progress") catch return error.Closed;
        out.put(a, "step", json.intValue(a, step) catch return error.Closed) catch return error.Closed;
        out.put(a, "total_steps", json.intValue(a, total) catch return error.Closed) catch return error.Closed;
        try sse.event(w.conn, a, "image_generation.progress", .{ .object = out });
    }
    pub fn preview(w: *Writer, index: usize, seed: u64, step: u32, total: u32, interval: u32, picture: api.ImagePreview) error{Closed}!void {
        var arena: std.heap.ArenaAllocator = .init(w.conn.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const out = item(a, index, seed, "image_generation.partial_image") catch return error.Closed;
        const fields = .{ .{ "partial_image_index", step / interval - 1 }, .{ "step", step }, .{ "total_steps", total }, .{ "width", picture.width }, .{ "height", picture.height } };
        inline for (fields) |field| out.put(a, field[0], json.intValue(a, field[1]) catch return error.Closed) catch return error.Closed;
        out.put(a, "b64_json", encoded(a, picture.png) catch return error.Closed) catch return error.Closed;
        try sse.event(w.conn, a, "image_generation.partial_image", .{ .object = out });
    }
    pub fn completed(w: *Writer, index: usize, seed: u64, png: []const u8) error{Closed}!void {
        var arena: std.heap.ArenaAllocator = .init(w.conn.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const out = item(a, index, seed, "image_generation.completed") catch return error.Closed;
        out.put(a, "b64_json", encoded(a, png) catch return error.Closed) catch return error.Closed;
        try sse.event(w.conn, a, "image_generation.completed", .{ .object = out });
    }
    pub fn done(w: *Writer) void {
        sse.done(w.conn) catch return;
        w.conn.response_complete = true;
    }
    pub fn refused(w: *Writer, cx: *errors.Cx, id: []const u8, e: errors.Refused) void {
        if (e == error.OutOfMemory) {
            cx.kind = .server;
            cx.message = "out of memory";
        }
        var buf: [1400]u8 = undefined;
        log.line("{s}", .{log.refused(&buf, id, cx.message)});
        var arena: std.heap.ArenaAllocator = .init(w.conn.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const body = openai.wrapError(a, openai.errorBody(a, cx, null) catch return) catch return;
        sse.event(w.conn, a, "error", body) catch return;
        w.done();
    }
};
