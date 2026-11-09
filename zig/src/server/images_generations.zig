//! POST /v1/images/generations: one picture as base64 PNG, written only after the engine's finished event.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const pyrepr = @import("pyrepr.zig");
const reply_text = @import("reply_text.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const ids = @import("ids.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Mailbox = @import("mailbox.zig").Mailbox;
const streaming = @import("speech_stream.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The most bytes one prompt may hold.
const max_prompt_bytes = 16 * 1024;
/// The API's own caps, which the engine's ``ImageInfo`` may only tighten.
const max_side: u16 = 1536;
const max_pixels: u32 = 1572864;
const max_steps: u32 = 200;
const default_side: u16 = 1024;
const default_steps: u32 = 50;
const default_guidance: f32 = 3;
const default_seed: u64 = 42;

/// The body fields this route answers; `quality` and every other key are refused by name.
const served_fields = [_][]const u8{ "model", "prompt", "n", "size", "response_format", "negative_prompt", "seed", "steps", "guidance" };

/// One checked request: exactly what the engine's image host needs.
const Plan = struct {
    prompt: []const u32,
    negative: []const u32,
    width: u16,
    height: u16,
    steps: u32,
    guidance: f32,
    seed: u64,
};

pub fn post(srv: *Server, c: *Conn, a: Allocator) !void {
    const cap = srv.info.image orelse {
        // dropped, so it cannot reach the next request on this connection
        _ = try http_body.read(c, a, http_body.limit);
        return routes.unknown(c, a);
    };
    var cx: Cx = .{ .a = a };
    const id = try ids.make(a, "images-", 32);
    const received = clock.nowNs(srv.io);
    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(srv.gpa);
    var result: api.ImageResult = .{};
    const raw = http_body.readJson(c, &cx) catch |e| return refused(c, a, &cx, id, e);
    const plan = prepare(srv, &cx, cap, raw) catch |e| return refused(c, a, &cx, id, e);
    generate(srv, c, &cx, plan, &result, &png) catch |e| return refused(c, a, &cx, id, e);
    reply(srv, c, &cx, received, &plan, &result, png.items) catch |e| return refused(c, a, &cx, id, e);
}

fn listed(names: []const []const u8, key: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, key)) return true;
    return false;
}

/// The refusal an error left in `cx`: out of memory is the server's 500, not an empty request error.
fn refused(c: *Conn, a: Allocator, cx: *Cx, id: []const u8, e: errors.Refused) void {
    if (e == error.OutOfMemory) {
        cx.kind = .server;
        cx.message = "out of memory";
    }
    send(c, a, cx, id);
}

/// Refuses every key this route drops: an unserved field by its own sentence, the rest by the name they sent.
fn checkFields(cx: *Cx, body: Value) errors.Refused!void {
    for (body.object.keys()) |key| {
        if (listed(&served_fields, key)) continue;
        if (std.mem.eql(u8, key, "quality") or std.mem.eql(u8, key, "style") or std.mem.eql(u8, key, "user")) return cx.fail(.request, "{s} is not supported by this model", .{key});
        return cx.fail(.request, "unknown images request field: {s}", .{key});
    }
}

/// One side of `size`: a positive multiple of 16 within `most`, or the refusal naming the value and the limit.
fn sideOf(cx: *Cx, text: []const u8, most: u16) errors.Refused!u16 {
    const n = std.fmt.parseInt(u32, text, 10) catch {
        const sent = try pyrepr.repr(cx.a, .{ .string = text });
        return cx.fail(.request, "size {s} must be a positive multiple of 16", .{sent});
    };
    if (n == 0 or n % 16 != 0) return cx.fail(.request, "size {d} must be a positive multiple of 16", .{n});
    if (n > most) return cx.fail(.request, "size {d} is past the {d} limit a side", .{ n, most });
    return @intCast(n);
}

/// The ids of valid UTF-8 text, through the server's own tokenizer.
fn encode(cx: *Cx, srv: *Server, text: []const u8) errors.Refused![]const u32 {
    return srv.text.encodePlain(cx.a, text) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.other("the tokenizer cannot encode this prompt"),
    };
}

/// Every check that can refuse the request, then its prompt ids, before anything reaches the engine.
fn prepare(srv: *Server, cx: *Cx, cap: api.ImageInfo, raw: Value) errors.Refused!Plan {
    if (raw != .object) return cx.refuse("images request body must be a JSON object");
    try checkFields(cx, raw);
    const model = raw.strField("model") orelse return cx.fail(.request, "model is required: send {s}", .{srv.config.served_name});
    if (!listed(srv.config.model_ids, model))
        return cx.fail(.request, "model {s} is not served here: send {s}", .{ try pyrepr.repr(cx.a, .{ .string = model }), srv.config.served_name });
    if (raw.get("n")) |v| {
        const count = v.int64() orelse return cx.refuse("n must be 1: this model serves one picture a request");
        if (count != 1) return cx.refuse("n must be 1: this model serves one picture a request");
    }
    var width: u16 = default_side;
    var height: u16 = default_side;
    const most = @min(max_side, cap.max_side);
    const most_pixels = @min(max_pixels, cap.max_pixels);
    if (raw.get("size")) |v| {
        const text = v.str() orelse return cx.refuse("size must be a string like 1024x1024");
        const split = std.mem.indexOfScalar(u8, text, 'x') orelse return cx.fail(.request, "size {s} must be WIDTHxHEIGHT", .{try pyrepr.repr(cx.a, v)});
        width = try sideOf(cx, text[0..split], most);
        height = try sideOf(cx, text[split + 1 ..], most);
        const pixels = @as(u32, width) * @as(u32, height);
        if (pixels > most_pixels) return cx.fail(.request, "size {d}x{d} is past the {d} pixel limit", .{ width, height, most_pixels });
    } else if (default_side > most or @as(u32, default_side) * default_side > most_pixels) {
        return cx.fail(.request, "the default size {d}x{d} does not fit this model's limits of {d} a side and {d} pixels: send a smaller size", .{ default_side, default_side, most, most_pixels });
    }
    if (raw.get("response_format")) |v| {
        const name = reply_text.pyStrip(v.str() orelse "");
        if (!std.ascii.eqlIgnoreCase(name, "b64_json")) return cx.refuse("response_format must be one of: b64_json");
    }
    const negative: []const u8 = if (raw.get("negative_prompt")) |v| switch (v) {
        .string => |s| s,
        else => return cx.fail(.request, "negative_prompt must be a string", .{}),
    } else "";
    if (!std.unicode.utf8ValidateSlice(negative)) return cx.refuse("negative_prompt must be valid UTF-8 text");
    var seed: u64 = default_seed;
    if (raw.get("seed")) |v| {
        const given = v.int64() orelse return cx.refuse("seed must be an integer");
        if (given < 0) return cx.refuse("seed must be a non-negative integer");
        seed = @intCast(given);
    }
    var steps: u32 = default_steps;
    if (raw.get("steps")) |v| {
        const given = v.int64() orelse return cx.refuse("steps must be an integer");
        const steps_most = @min(max_steps, cap.max_steps);
        if (given < 1 or given > steps_most) return cx.fail(.request, "steps must be from 1 to {d}", .{steps_most});
        steps = @intCast(given);
    }
    var guidance: f32 = default_guidance;
    if (raw.get("guidance")) |v| {
        const given = v.float64() orelse return cx.refuse("guidance must be a number");
        if (!std.math.isFinite(given) or given < 1 or given > std.math.floatMax(f32)) return cx.refuse("guidance must be a finite number from 1");
        guidance = @floatCast(given);
    }
    const asked = raw.strField("prompt") orelse return cx.refuse("prompt must be a non-empty string");
    if (!std.unicode.utf8ValidateSlice(asked)) return cx.refuse("prompt must be valid UTF-8 text");
    const prompt = reply_text.pyStrip(asked);
    if (prompt.len == 0) return cx.refuse("prompt must be a non-empty string");
    if (prompt.len > max_prompt_bytes) return cx.fail(.request, "prompt must be at most {d} bytes", .{max_prompt_bytes});
    const prompt_ids = try encode(cx, srv, prompt);
    if (prompt_ids.len > cap.max_prompt_tokens)
        return cx.fail(.request, "a prompt of {d} tokens exceeds the maximum allowed length of {d} tokens", .{ prompt_ids.len, cap.max_prompt_tokens });
    // the negative prompt is a second prompt, so it gets the prompt's limits and the prompt's sentences
    const excluded = reply_text.pyStrip(negative);
    if (excluded.len > max_prompt_bytes) return cx.fail(.request, "prompt must be at most {d} bytes", .{max_prompt_bytes});
    const negative_ids = if (excluded.len == 0) &[_]u32{} else try encode(cx, srv, excluded);
    if (negative_ids.len > cap.max_prompt_tokens)
        return cx.fail(.request, "a prompt of {d} tokens exceeds the maximum allowed length of {d} tokens", .{ negative_ids.len, cap.max_prompt_tokens });
    return .{ .prompt = prompt_ids, .negative = negative_ids, .width = width, .height = height, .steps = steps, .guidance = guidance, .seed = seed };
}

/// Submits once and waits for ``finished``; nothing has been written to the connection when it returns.
fn generate(srv: *Server, c: *Conn, cx: *Cx, plan: Plan, result: *api.ImageResult, png: *std.ArrayList(u8)) errors.Refused!void {
    const request: api.ImageRequest = .{
        .prompt = plan.prompt,
        .negative = plan.negative,
        .width = plan.width,
        .height = plan.height,
        .steps = plan.steps,
        .guidance = plan.guidance,
        .seed = plan.seed,
        .result = result,
        .png = png,
        .alloc = srv.gpa,
    };
    var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    const engine_id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submitImage(engine_id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => cx.fail(.capacity, "the engine is busy; retry shortly", .{}),
        error.Closed => cx.fail(.capacity, "the scheduler is closed", .{}),
        error.InvalidRequest => cx.fail(.request, "the engine refused this image request's prompt, size or steps", .{}),
        error.Unsupported => cx.fail(.capacity, "this model does not serve images", .{}),
    };
    // The engine holds request, png and result until finished: this thread writes nothing before that.
    _ = streaming.settle(srv, c, &box, engine_id, null) orelse return cx.fail(.capacity, "the client left before the picture was ready", .{});
    // the finished event is the last one: the reason it carried is the one the log and the refusals both read
    box.mutex.lockUncancelable(box.io);
    result.reason = box.reason;
    box.mutex.unlock(box.io);
    if (result.reason == .failed) {
        // the mailbox frees its own buffer as this function returns: the message is copied into the request's arena
        const why = if (box.message.len > 0) cx.a.dupe(u8, box.message) catch "the image generation failed" else "the image generation failed";
        return cx.fail(.server, "{s}", .{why});
    }
    if (result.reason == .cancelled) return cx.fail(.capacity, "the image generation was cancelled", .{});
}

/// The images reply: `created`, and the one picture as base64 PNG from the request's own arena.
fn reply(srv: *Server, c: *Conn, cx: *Cx, received: i96, plan: *const Plan, result: *const api.ImageResult, png: []const u8) !void {
    const a = cx.a;
    const kept = png[0..@min(result.bytes, png.len)];
    const b64 = try a.alloc(u8, std.base64.standard.Encoder.calcSize(kept.len));
    const encoded = std.base64.standard.Encoder.encode(b64, kept);
    const item = try json.newObject(a);
    try item.put(a, "b64_json", .{ .string = encoded });
    const data = try a.alloc(Value, 1);
    data[0] = .{ .object = item };
    const out = try json.newObject(a);
    try out.put(a, "created", try json.intValue(a, std.Io.Clock.real.now(srv.io).toSeconds()));
    try out.put(a, "data", .{ .array = data });
    routes.sendValue(c, a, 200, .{ .object = out });
    log.line("images prompt={d} negative={d} png={d} finish={s} seconds={d:.2}", .{ plan.prompt.len, plan.negative.len, kept.len, @tagName(result.reason), clock.seconds(clock.nowNs(srv.io) - received) });
}

/// The API's error shape at the refusal's status, logged first; a capacity refusal also carries Retry-After.
fn send(c: *Conn, a: Allocator, cx: *Cx, id: []const u8) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, cx.message)});
    const err = openai.errorBody(a, cx, null) catch return;
    const wrapped = openai.wrapError(a, err) catch return;
    if (cx.kind == .capacity) return routes.sendDraining(c, json.stringify(a, wrapped, .{}) catch return);
    routes.sendValue(c, a, cx.status(), wrapped);
}
