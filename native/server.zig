const std = @import("std");
const mx = @import("mlx.zig");
const inference = @import("session.zig");
const chat = @import("chat.zig");
const reply_text = @import("reply_text.zig");
const tool_calls = @import("tool_calls.zig");
const Request = std.http.Server.Request;
const control = @import("server_control.zig");

pub fn run(init: std.process.Init, args: []const []const u8) !void {
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var limit: usize = 0;
    var timeout_ms: i64 = 0;
    var shutdown_grace_ms: i64 = 5000;
    var name = std.fs.path.basename(args[2]);
    var defaults = try inference.Options.load(init.gpa, init.io, args[2]);
    defaults.max_tokens = 4096;
    var thinking = true;
    var vision_urls = false;
    var effort: []const u8 = "medium";
    var overrides = std.json.Value{ .object = .empty };
    defer overrides.object.deinit(init.gpa);
    var i: usize = 3;
    while (i < args.len) {
        if (std.mem.eql(u8, args[i], "--vision-urls") or std.mem.eql(u8, args[i], "--no-vision-urls")) {
            vision_urls = std.mem.eql(u8, args[i], "--vision-urls");
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--thinking") or std.mem.eql(u8, args[i], "--no-thinking")) {
            thinking = std.mem.eql(u8, args[i], "--thinking");
            i += 1;
            continue;
        }
        if (i + 1 == args.len) return error.MissingArgument;
        const flag = args[i];
        const value = args[i + 1];
        i += 2;
        if (std.mem.eql(u8, flag, "--request-timeout-seconds")) timeout_ms = try control.seconds(value) else if (std.mem.eql(u8, flag, "--shutdown-grace-seconds")) shutdown_grace_ms = try control.seconds(value) else if (std.mem.eql(u8, flag, "--host")) host = value else if (std.mem.eql(u8, flag, "--port")) port = try std.fmt.parseInt(u16, value, 10) else if (std.mem.eql(u8, flag, "--served-model-name")) name = value else if (std.mem.eql(u8, flag, "--max-requests")) limit = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, flag, "--reasoning-effort")) {
            if (!std.mem.eql(u8, value, "low") and !std.mem.eql(u8, value, "medium") and !std.mem.eql(u8, value, "xhigh")) return error.InvalidReasoningEffort;
            effort = value;
        } else {
            const fields = .{ .{ "--temperature", "temperature" }, .{ "--top-k", "top_k" }, .{ "--top-p", "top_p" }, .{ "--max-tokens", "max_tokens" }, .{ "--thinking-budget", "thinking_budget" } };
            var found = false;
            inline for (fields) |pair| if (std.mem.eql(u8, flag, pair[0])) {
                try overrides.object.put(init.gpa, pair[1], .{ .string = value });
                found = true;
            };
            if (!found) return error.UnknownArgument;
        }
    }
    defaults = try inference.Options.parseWithDefaults(init.gpa, overrides, defaults);
    var signals = control.Signals.install();
    defer signals.deinit();
    var registry = control.Registry{ .io = init.io, .timeout_ms = timeout_ms, .shutdown_grace_ms = shutdown_grace_ms };
    const monitor = try std.Thread.spawn(.{}, control.Registry.watch, .{&registry});
    defer {
        registry.finished.store(true, .release);
        monitor.join();
    }
    const address = try std.Io.net.IpAddress.parse(host, port);
    var listener = try address.listen(init.io, .{ .kernel_backlog = 128 });
    defer listener.deinit(init.io);
    const path = try std.fmt.allocPrint(init.gpa, "{s}/config.json", .{args[2]});
    defer init.gpa.free(path);
    const config = try @import("weights.zig").readFile(init.io, path);
    defer mx.allocator.free(config);
    const parsed = try std.json.parseFromSlice(std.json.Value, init.gpa, config, .{});
    defer parsed.deinit();
    if (parsed.value == .object) if (parsed.value.object.get("model_type")) |kind| if (kind == .string and std.mem.eql(u8, kind.string, "glm5_next")) try @import("glm.zig").Model.prepareRuntime();
    var jobs: [8]*Job = undefined;
    var worker = Worker{ .io = init.io, .dir = args[2], .queue = .init(&jobs), .defaults = defaults, .thinking = thinking, .effort = effort, .vision_urls = vision_urls, .control = &registry };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    defer {
        worker.queue.close(init.io);
        thread.join();
    }
    worker.ready.waitUncancelable(init.io);
    if (worker.startup_error) |err| return err;
    if (registry.stopping.load(.acquire)) return;
    var clients: std.Io.Group = .init;
    defer clients.await(init.io) catch clients.cancel(init.io);
    errdefer registry.stop();
    std.debug.print("Native inference listening at http://{s}:{d} ({s})\n", .{ host, listener.socket.address.getPort(), name });
    const Event = union(enum) { accepted: anyerror!void, stopped: anyerror!void };
    var events: [2]Event = undefined;
    var select = std.Io.Select(Event).init(init.io, &events);
    defer select.cancelDiscard();
    try select.concurrent(.accepted, acceptRequests, .{ &worker, &listener, &clients, init.gpa, name, limit });
    try select.concurrent(.stopped, control.Registry.waitStopped, .{&registry});
    switch (try select.await()) {
        inline else => |result| try result,
    }
}

fn acceptRequests(worker: *Worker, listener: *std.Io.net.Server, clients: *std.Io.Group, a: std.mem.Allocator, name: []const u8, limit: usize) anyerror!void {
    var handled: usize = 0;
    while (limit == 0 or handled < limit) : (handled += 1) {
        const stream = try listener.accept(worker.io);
        const client = worker.control.acquire(stream.socket.handle) orelse {
            stream.close(worker.io);
            if (worker.control.stopping.load(.acquire)) return;
            continue;
        };
        clients.concurrent(worker.io, connectionTask, .{ worker, a, stream, name, handled, client }) catch |err| {
            worker.control.release(client);
            stream.close(worker.io);
            return err;
        };
    }
}

const Job = struct {
    a: std.mem.Allocator,
    request: *Request,
    model: []const u8,
    sequence: usize,
    created: i64,
    client: *control.Client,
    body: std.json.Value,
    is_chat: bool,
    options: inference.Options,
    done: std.Io.Event = .unset,
    failure: ?anyerror = null,
};
const Worker = struct {
    io: std.Io,
    dir: []const u8,
    queue: std.Io.Queue(*Job),
    ready: std.Io.Event = .unset,
    startup_error: ?anyerror = null,
    control: *control.Registry,
    defaults: inference.Options,
    thinking: bool,
    effort: []const u8,
    vision_urls: bool,
    fn run(w: *Worker) void {
        w.loop() catch |err| {
            w.startup_error = err;
            w.ready.set(w.io);
        };
    }
    fn loop(w: *Worker) !void {
        try mx.init();
        defer mx.shutdown();
        var session = try inference.Session.init(w.io, w.dir);
        defer session.deinit();
        w.ready.set(w.io);
        while (w.queue.getOneUncancelable(w.io)) |job| {
            complete(&session, job.a, job.request, job.model, job.sequence, job.created, job.client, job.body, job.is_chat, job.options, w.thinking, w.effort, w.vision_urls) catch |err| {
                job.failure = err;
            };
            job.done.set(w.io);
        } else |_| {}
    }
};
fn connectionTask(w: *Worker, a: std.mem.Allocator, stream: std.Io.net.Stream, name: []const u8, sequence: usize, client: *control.Client) void {
    defer stream.close(w.io);
    defer w.control.release(client);
    var input: [65536]u8 = undefined;
    var output: [8192]u8 = undefined;
    var reader = stream.reader(w.io, &input);
    var writer = stream.writer(w.io, &output);
    var http = std.http.Server.init(&reader.interface, &writer.interface);
    var request = http.receiveHead() catch return;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    handle(w, arena.allocator(), &request, name, sequence, client) catch |err| {
        std.debug.print("HTTP request failed: {s}\n", .{@errorName(err)});
    };
}

fn json(a: std.mem.Allocator, request: *Request, status: std.http.Status, value: anytype) !void {
    const body = try std.json.Stringify.valueAlloc(a, value, .{});
    try request.respond(body, .{ .status = status, .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
}
fn failure(a: std.mem.Allocator, request: *Request, status: std.http.Status, message: []const u8) !void {
    try json(a, request, status, .{ .@"error" = .{ .message = message, .type = "invalid_request_error" } });
}
fn handle(worker: *Worker, a: std.mem.Allocator, request: *Request, model: []const u8, sequence: usize, client: *control.Client) !void {
    var route = request.head.target;
    if (std.mem.indexOfScalar(u8, route, '?')) |at| route = route[0..at];
    route = std.mem.trimEnd(u8, route, "/");
    if (request.head.method == .GET) {
        if (route.len == 0 or std.mem.eql(u8, route, "/health")) return json(a, request, .ok, .{ .status = "ok", .model = model, .warming = false });
        if (std.mem.eql(u8, route, "/v1/models") or std.mem.eql(u8, route, "/models")) return json(a, request, .ok, .{ .object = "list", .data = &.{.{ .id = model, .object = "model", .created = std.Io.Clock.real.now(worker.io).toSeconds(), .owned_by = "tensorfold" }} });
        return failure(a, request, .not_found, "Unknown route");
    }
    const is_chat = std.mem.eql(u8, route, "/v1/chat/completions") or std.mem.eql(u8, route, "/chat/completions");
    if (request.head.method != .POST or (!is_chat and !std.mem.eql(u8, route, "/v1/completions") and !std.mem.eql(u8, route, "/completions"))) return failure(a, request, .not_found, "Unknown route");
    var body_buffer: [8192]u8 = undefined;
    const body_reader = try request.readerExpectContinue(&body_buffer);
    const bytes = body_reader.allocRemaining(a, .limited(32 * 1024 * 1024)) catch {
        client.cancellation().check() catch |err| return requestFailure(a, request, err);
        return failure(a, request, .bad_request, "Invalid or oversized request body");
    };
    const body = std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always }) catch return failure(a, request, .bad_request, "Invalid JSON");
    const options = inference.Options.parseWithDefaults(a, body.value, worker.defaults) catch |err| return failure(a, request, .bad_request, @errorName(err));
    if (body.value.object.get("model")) |value| if (value != .string or !std.mem.eql(u8, value.string, model)) return failure(a, request, .not_found, "Unknown model");
    client.cancellation().check() catch |err| return requestFailure(a, request, err);
    var job = Job{ .a = a, .request = request, .model = model, .sequence = sequence, .created = std.Io.Clock.real.now(worker.io).toSeconds(), .client = client, .body = body.value, .is_chat = is_chat, .options = options };
    if (try worker.queue.putUncancelable(worker.io, &.{&job}, 0) == 0) return failure(a, request, .service_unavailable, "Inference queue is full");
    job.done.waitUncancelable(worker.io);
    if (job.failure) |err| return err;
}

fn requestFailure(a: std.mem.Allocator, request: *Request, err: anyerror) !void {
    return failure(a, request, switch (err) {
        error.RequestTimedOut => .request_timeout,
        error.ServerStopping => .service_unavailable,
        else => .bad_request,
    }, @errorName(err));
}

fn complete(session: *inference.Session, a: std.mem.Allocator, request: *Request, model: []const u8, sequence: usize, created: i64, client: *control.Client, body: std.json.Value, is_chat: bool, requested: inference.Options, default_thinking: bool, default_effort: []const u8, vision_urls: bool) !void {
    const cancellation = client.cancellation();
    cancellation.check() catch |err| return requestFailure(a, request, err);
    var options = requested;
    var ids: std.ArrayList(i32) = .empty;
    var raw_images = body.object.get("images") orelse .null;
    var thinking = false;
    var tools: std.json.Value = .null;
    var max_calls: ?usize = null;
    var gate: ?@import("call_gate.zig").Gate = null;
    if (is_chat) {
        tools = chat.activeTools(a, body) catch |err| return failure(a, request, .bad_request, @errorName(err));
        if (body.object.get("parallel_tool_calls")) |parallel| if (parallel != .null) {
            if (parallel != .bool) return failure(a, request, .bad_request, "parallel_tool_calls must be a boolean");
            if (!parallel.bool) max_calls = 1;
        };
        const rendered = session.renderChat(a, body, default_thinking, default_effort) catch |err| return failure(a, request, .bad_request, @errorName(err));
        try ids.appendSlice(a, try chat.encode(a, &session.tokenizer, rendered));
        raw_images = rendered.images;
        thinking = rendered.thinking;
        if (chat.requiresCall(body)) {
            const form = (try session.chat_template.?.callForm(a, &session.tokenizer)) orelse return failure(a, request, .bad_request, "This template cannot mark required tool calls");
            var names: std.ArrayList([]const u8) = .empty;
            if (tools == .array) for (tools.array.items) |spec| try names.append(a, try chat.toolName(spec));
            gate = @import("call_gate.zig").Gate.init(a, &session.tokenizer, ids.items, form, names.items, session.backend == .gemma) catch |err| return failure(a, request, .bad_request, @errorName(err));
        }
    } else switch (body.object.get("prompt") orelse return failure(a, request, .bad_request, "Missing prompt")) {
        .string => |value| {
            for (try session.tokenizer.encode(a, value)) |id| try ids.append(a, @intCast(id));
        },
        .array => |values| for (values.items) |value| {
            if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(i32)) return failure(a, request, .bad_request, "Invalid prompt token");
            try ids.append(a, @intCast(value.integer));
        },
        else => return failure(a, request, .bad_request, "Prompt must be text or token IDs"),
    }
    if (!thinking) options.thinking_budget = 0;
    session.validate(ids.items, options) catch |err| return failure(a, request, .bad_request, @errorName(err));
    if (raw_images == .array and raw_images.array.items.len > 0 and session.backend != .qwen) return failure(a, request, .bad_request, "This model does not support image inputs");
    const images = @import("image_source.zig").loadWithCancellation(a, session.io, raw_images, vision_urls, cancellation) catch |err| return requestFailure(a, request, err);
    const id = try std.fmt.allocPrint(a, "{s}cmpl-{d}-{d}", .{ if (is_chat) "chat" else "", created, sequence });
    const markers: reply_text.Markers = if (session.backend == .gemma) reply_text.gemma_markers else .{};
    cancellation.check() catch |err| return requestFailure(a, request, err);
    if (options.stream) {
        var buffer: [8192]u8 = undefined;
        var response = try request.respondStreaming(&buffer, .{ .respond_options = .{ .keep_alive = false, .extra_headers = &.{ .{ .name = "content-type", .value = "text/event-stream" }, .{ .name = "cache-control", .value = "no-cache" } } } });
        var state = Stream{ .a = a, .writer = &response.writer, .transport = request.server.out, .id = id, .model = model, .created = created, .is_chat = is_chat, .thinking = thinking, .markers = markers, .tools = tools, .max_calls = max_calls, .cancellation = cancellation };
        if (is_chat) try state.chatChunk(.{ .role = "assistant", .content = "" }, null);
        var reply = session.generateImages(mx.allocator, ids.items, options, .{ .context = &state, .emit = Stream.emit, .cancellation = cancellation, .gate = if (gate) |*g| g else null }, images) catch |err| {
            const error_body = try std.json.Stringify.valueAlloc(a, .{ .@"error" = .{ .message = @errorName(err) } }, .{});
            try response.writer.print("data: {s}\n\ndata: [DONE]\n\n", .{error_body});
            try response.end();
            return;
        };
        defer reply.deinit(mx.allocator);
        if (is_chat) {
            try state.chatText(true);
            try state.chatChunk(std.json.Value{ .object = .empty }, if (state.calls_sent > 0) "tool_calls" else @tagName(reply.finish_reason));
        } else try state.chunk("", @tagName(reply.finish_reason));
        try response.writer.writeAll("data: [DONE]\n\n");
        try response.end();
    } else {
        var reply = session.generateImages(mx.allocator, ids.items, options, .{ .cancellation = cancellation, .gate = if (gate) |*g| g else null }, images) catch |err| return requestFailure(a, request, err);
        defer reply.deinit(mx.allocator);
        const usage = .{ .prompt_tokens = reply.prompt_tokens, .completion_tokens = reply.tokens.items.len, .total_tokens = reply.prompt_tokens + reply.tokens.items.len };
        if (is_chat) {
            const parts = if (thinking) reply_text.splitThinking(reply.content, true, markers) else reply_text.Parts{ .content = reply.content };
            var parsed = try tool_calls.parse(a, parts.content, tools, max_calls, id);
            if (max_calls != null and tools == .array and tools.array.items.len > 0) parsed.content = try tool_calls.singleContent(a, parsed.content);
            try json(a, request, .ok, .{ .id = id, .object = "chat.completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .message = .{ .role = "assistant", .content = parsed.content, .reasoning_content = parts.reasoning, .tool_calls = parsed.calls }, .finish_reason = if (parsed.calls.len > 0) "tool_calls" else @tagName(reply.finish_reason) }}, .usage = usage });
        } else try json(a, request, .ok, .{ .id = id, .object = "text_completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .text = reply.content, .finish_reason = @tagName(reply.finish_reason), .logprobs = @as(?u8, null) }}, .usage = usage });
    }
}

const Stream = struct {
    a: std.mem.Allocator,
    writer: *std.Io.Writer,
    transport: *std.Io.Writer,
    id: []const u8,
    model: []const u8,
    created: i64,
    cancellation: @import("cancellation.zig").Cancellation,
    is_chat: bool = false,
    thinking: bool = false,
    markers: reply_text.Markers = .{},
    accumulated: std.ArrayList(u8) = .empty,
    sent_content: usize = 0,
    sent_reasoning: usize = 0,
    tools: std.json.Value = .null,
    max_calls: ?usize = null,
    calls_sent: usize = 0,
    tool_stream: @import("tool_stream.zig").Streamer = .{},
    fn chunk(s: *Stream, value: []const u8, finish: ?[]const u8) !void {
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "text_completion", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .text = value, .finish_reason = finish, .logprobs = @as(?u8, null) }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn emit(context: ?*anyopaque, value: []const u8) !void {
        if (value.len == 0) return;
        const s: *Stream = @ptrCast(@alignCast(context.?));
        try s.cancellation.check();
        if (!s.is_chat) return s.chunk(value, null);
        try s.accumulated.appendSlice(s.a, value);
        try s.chatText(false);
    }
    fn chatChunk(s: *Stream, delta: anytype, finish: ?[]const u8) !void {
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "chat.completion.chunk", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .delta = delta, .finish_reason = finish }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn chatText(s: *Stream, finished: bool) !void {
        const parts = if (s.thinking) reply_text.splitThinking(s.accumulated.items, finished, s.markers) else reply_text.Parts{ .content = s.accumulated.items };
        if (parts.reasoning.len > s.sent_reasoning) {
            try s.chatChunk(.{ .reasoning_content = parts.reasoning[s.sent_reasoning..] }, null);
            s.sent_reasoning = parts.reasoning.len;
        }
        const has_tools = s.tools == .array and s.tools.array.items.len > 0;
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const a = scratch.allocator();
        if (has_tools and s.max_calls == null) try s.tool_stream.feed(a, parts.content, s.tools, s, toolDelta);
        var parsed = if (finished) try tool_calls.parse(a, parts.content, s.tools, s.max_calls, s.id) else tool_calls.Result{ .content = if (has_tools) try tool_calls.preview(a, parts.content, s.tools, s.max_calls) else parts.content };
        if (s.max_calls != null and finished and has_tools) parsed.content = try tool_calls.singleContent(a, parsed.content);
        if (parsed.content.len > s.sent_content) {
            try s.chatChunk(.{ .content = parsed.content[s.sent_content..] }, null);
            s.sent_content = parsed.content.len;
        }
        for (parsed.calls, 0..) |call, index| {
            if (index < s.tool_stream.count) continue;
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = index, .id = call.id, .type = "function", .function = .{ .name = call.function.name, .arguments = "" } }} }, null);
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = index, .function = .{ .arguments = call.function.arguments } }} }, null);
            s.calls_sent += 1;
        }
    }
    fn toolDelta(context: ?*anyopaque, delta: @import("tool_stream.zig").Delta) !void {
        const s: *Stream = @ptrCast(@alignCast(context.?));
        if (delta.name) |name| {
            const id = try std.fmt.allocPrint(s.a, "call_{s}_{d}", .{ s.id, delta.index });
            try s.chatChunk(.{ .tool_calls = &.{.{ .index = delta.index, .id = id, .type = "function", .function = .{ .name = name, .arguments = "" } }} }, null);
            s.calls_sent += 1;
        } else try s.chatChunk(.{ .tool_calls = &.{.{ .index = delta.index, .function = .{ .arguments = delta.arguments } }} }, null);
    }
};
