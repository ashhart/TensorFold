//! vLLM's /tokenize and /detokenize (the ids the chat and completion routes would run), and a reply's token SHA.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const fields = @import("fields.zig");
const messages = @import("messages.zig");
const images = @import("images.zig");
const prompt = @import("prompt.zig");
const tool_specs = @import("tool_specs.zig");
const routes = @import("routes.zig");
const http_body = @import("http_body.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const paths = [_][]const u8{ "/tokenize", "/v1/tokenize", "/detokenize", "/v1/detokenize" };

fn flag(cx: *Cx, body: Value, name: []const u8, default: bool) errors.Refused!bool {
    const v = body.get(name) orelse return default;
    if (v != .bool) return cx.fail(.request, "{s} must be a boolean", .{name});
    return v.bool;
}

fn tokenize(srv: *Server, cx: *Cx, body: Value) errors.Refused!Value {
    const a = cx.a;
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    const strings = try flag(cx, body, "return_token_strs", false);
    var ids: []const u32 = undefined;
    if (body.has("messages")) {
        const generation = try flag(cx, body, "add_generation_prompt", true);
        const msgs = try messages.normalizeWith(cx, body.get("messages"), "system", srv.needs_user_after_tool, .of(srv.info.vision));
        const tools = tool_specs.active(cx, body.get("tools"), body.get("tool_choice")) catch |e| switch (e) {
            error.Refused => return cx.refuse(cx.message), // a ValueError here is a RequestError
            else => |x| return x,
        };
        const thinking = try fields.thinkingFields(cx, body, srv.effort_levels);
        const on = thinking.enable orelse srv.config.enable_thinking;
        ids = prompt.renderIds(srv, cx, msgs, tools, on, srv.effortFor(thinking.effort), generation) catch |e| switch (e) {
            error.Refused => return cx.other(cx.message),
            else => |x| return x,
        };
        // media expanded as the chat route expands them: each image's tokens, each video's frame groups
        if (try images.expand(srv.info.vision, if (srv.config.vision_urls) images.Fetch.network(srv.io) else null, srv.text, cx, msgs, ids, 0)) |x| ids = x.ids;
    } else {
        const text = body.get("prompt");
        if (text == null or text.? != .string) return cx.refuse("prompt must be a string (or send messages)");
        const special = try flag(cx, body, "add_special_tokens", true);
        ids = srv.text.encode(a, text.?.string, special) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.other("the tokenizer cannot encode this prompt"),
        };
    }
    const o = try json.newObject(a);
    try o.put(a, "count", try json.intValue(a, ids.len));
    try o.put(a, "max_model_len", if (srv.info.context_window > 0) try json.intValue(a, srv.info.context_window) else .null);
    const list = try a.alloc(Value, ids.len);
    for (ids, list) |t, *slot| slot.* = try json.intValue(a, t);
    try o.put(a, "tokens", .{ .array = list });
    if (strings) {
        const strs = try a.alloc(Value, ids.len);
        for (ids, strs) |t, *slot| slot.* = .{ .string = srv.text.tokenString(a, t) catch "" };
        try o.put(a, "token_strs", .{ .array = strs });
    }
    return .{ .object = o };
}

fn detokenize(srv: *Server, cx: *Cx, body: Value) errors.Refused!Value {
    if (body != .object) return cx.refuse("the request body must be a JSON object");
    var value = body.get("tokens") orelse Value.null;
    if (value == .array and value.array.len == 1 and value.array[0] == .array) value = value.array[0];
    if (value != .array) return cx.refuse("tokens must be a list of integer token ids (one prompt a request)");
    for (value.array) |t| if (t != .int) return cx.refuse("tokens must be a list of integer token ids (one prompt a request)");
    const vocab = srv.text.vocabSize();
    const out = try cx.a.alloc(u32, value.array.len);
    for (value.array, out) |t, *slot| {
        const n = t.int64() orelse -1;
        if (n < 0 or n >= vocab) return cx.fail(.request, "tokens token ids must be in the vocabulary's range 0 to {d}", .{@as(i64, vocab) - 1});
        slot.* = @intCast(n);
    }
    const text = srv.text.decode(cx.a, out) catch return error.OutOfMemory;
    const o = try json.newObject(cx.a);
    try o.put(cx.a, "prompt", .{ .string = text });
    return .{ .object = o };
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator, detok: bool) void {
    var cx: Cx = .{ .a = a };
    const reply = blk: {
        const body = http_body.readJson(conn, &cx) catch break :blk null;
        break :blk (if (detok) detokenize(srv, &cx, body) else tokenize(srv, &cx, body)) catch null;
    };
    if (reply) |r| return routes.sendValue(conn, a, 200, r);
    const o = json.newObject(a) catch return;
    o.put(a, "message", .{ .string = cx.message }) catch return;
    o.put(a, "type", .{ .string = "invalid_request_error" }) catch return;
    const wrapped = json.newObject(a) catch return;
    wrapped.put(a, "error", .{ .object = o }) catch return;
    routes.sendValue(conn, a, if (cx.kind == .capacity) 503 else 400, .{ .object = wrapped });
}

/// The reply's token ids hashed: drafted and ``"draft": false`` replies must match.
pub fn tokenSha(tokens: []const u32) [12]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [16]u8 = undefined;
    for (tokens, 0..) |t, i| {
        if (i > 0) h.update(",");
        h.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    var out: [12]u8 = undefined;
    const hex = std.fmt.bytesToHex(d[0..6].*, .lower);
    @memcpy(&out, &hex);
    return out;
}
