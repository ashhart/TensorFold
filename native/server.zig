const std = @import("std");
const mx = @import("mlx.zig");
const inference = @import("session.zig");
const Request = std.http.Server.Request;

pub fn run(init: std.process.Init, args: []const []const u8) !void {
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var limit: usize = 0;
    var name = std.fs.path.basename(args[2]);
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        if (i + 1 == args.len) return error.MissingArgument;
        if (std.mem.eql(u8, args[i], "--host")) host = args[i + 1] else if (std.mem.eql(u8, args[i], "--port")) port = try std.fmt.parseInt(u16, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--served-model-name")) name = args[i + 1] else if (std.mem.eql(u8, args[i], "--max-requests")) limit = try std.fmt.parseInt(usize, args[i + 1], 10) else return error.UnknownArgument;
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
    var worker = Worker{ .io = init.io, .dir = args[2], .queue = .init(&jobs) };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    defer {
        worker.queue.close(init.io);
        thread.join();
    }
    worker.ready.waitUncancelable(init.io);
    if (worker.startup_error) |err| return err;
    var clients: std.Io.Group = .init;
    defer clients.await(init.io) catch clients.cancel(init.io);
    std.debug.print("Native inference listening at http://{s}:{d} ({s})\n", .{ host, port, name });
    var handled: usize = 0;
    while (limit == 0 or handled < limit) : (handled += 1) {
        const stream = try listener.accept(init.io);
        if (worker.connections.fetchAdd(1, .acq_rel) >= 32) {
            _ = worker.connections.fetchSub(1, .acq_rel);
            stream.close(init.io);
            continue;
        }
        clients.concurrent(init.io, connectionTask, .{ &worker, init.gpa, stream, name, handled }) catch |err| {
            _ = worker.connections.fetchSub(1, .acq_rel);
            stream.close(init.io);
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
    socket: std.posix.fd_t,
    prompt: std.json.Value,
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
    connections: std.atomic.Value(usize) = .init(0),
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
            complete(&session, job.a, job.request, job.model, job.sequence, job.created, job.socket, job.prompt, job.options) catch |err| {
                job.failure = err;
            };
            job.done.set(w.io);
        } else |_| {}
    }
};
fn connectionTask(w: *Worker, a: std.mem.Allocator, stream: std.Io.net.Stream, name: []const u8, sequence: usize) void {
    defer _ = w.connections.fetchSub(1, .acq_rel);
    defer stream.close(w.io);
    var input: [65536]u8 = undefined;
    var output: [8192]u8 = undefined;
    var reader = stream.reader(w.io, &input);
    var writer = stream.writer(w.io, &output);
    var http = std.http.Server.init(&reader.interface, &writer.interface);
    var request = http.receiveHead() catch return;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    handle(w, arena.allocator(), &request, name, sequence, stream.socket.handle) catch |err| {
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
fn handle(worker: *Worker, a: std.mem.Allocator, request: *Request, model: []const u8, sequence: usize, socket: std.posix.fd_t) !void {
    var route = request.head.target;
    if (std.mem.indexOfScalar(u8, route, '?')) |at| route = route[0..at];
    route = std.mem.trimEnd(u8, route, "/");
    if (request.head.method == .GET) {
        if (route.len == 0 or std.mem.eql(u8, route, "/health")) return json(a, request, .ok, .{ .status = "ok", .model = model, .warming = false });
        if (std.mem.eql(u8, route, "/v1/models") or std.mem.eql(u8, route, "/models")) return json(a, request, .ok, .{ .object = "list", .data = &.{.{ .id = model, .object = "model", .created = std.Io.Clock.real.now(worker.io).toSeconds(), .owned_by = "tensorfold" }} });
        return failure(a, request, .not_found, "Unknown route");
    }
    if (request.head.method != .POST or (!std.mem.eql(u8, route, "/v1/completions") and !std.mem.eql(u8, route, "/completions"))) return failure(a, request, .not_found, "Unknown route");
    var body_buffer: [8192]u8 = undefined;
    const body_reader = try request.readerExpectContinue(&body_buffer);
    const bytes = body_reader.allocRemaining(a, .limited(32 * 1024 * 1024)) catch return failure(a, request, .bad_request, "Invalid or oversized request body");
    const body = std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always }) catch return failure(a, request, .bad_request, "Invalid JSON");
    const options = inference.Options.parse(a, body.value) catch |err| return failure(a, request, .bad_request, @errorName(err));
    if (body.value.object.get("model")) |value| if (value != .string or !std.mem.eql(u8, value.string, model)) return failure(a, request, .not_found, "Unknown model");
    const prompt = body.value.object.get("prompt") orelse return failure(a, request, .bad_request, "Missing prompt");
    var job = Job{ .a = a, .request = request, .model = model, .sequence = sequence, .created = std.Io.Clock.real.now(worker.io).toSeconds(), .socket = socket, .prompt = prompt, .options = options };
    if (try worker.queue.putUncancelable(worker.io, &.{&job}, 0) == 0) return failure(a, request, .service_unavailable, "Inference queue is full");
    job.done.waitUncancelable(worker.io);
    if (job.failure) |err| return err;
}

fn complete(session: *inference.Session, a: std.mem.Allocator, request: *Request, model: []const u8, sequence: usize, created: i64, socket: std.posix.fd_t, prompt: std.json.Value, options: inference.Options) !void {
    var ids: std.ArrayList(i32) = .empty;
    switch (prompt) {
        .string => |value| {
            for (try session.tokenizer.encode(a, value)) |id| try ids.append(a, @intCast(id));
        },
        .array => |values| for (values.items) |value| {
            if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(i32)) return failure(a, request, .bad_request, "Invalid prompt token");
            try ids.append(a, @intCast(value.integer));
        },
        else => return failure(a, request, .bad_request, "Prompt must be text or token IDs"),
    }
    session.validate(ids.items, options) catch |err| return failure(a, request, .bad_request, @errorName(err));
    const id = try std.fmt.allocPrint(a, "cmpl-{d}-{d}", .{ created, sequence });
    var connection = Connection{ .socket = socket };
    if (options.stream) {
        var buffer: [8192]u8 = undefined;
        var response = try request.respondStreaming(&buffer, .{ .respond_options = .{ .keep_alive = false, .extra_headers = &.{ .{ .name = "content-type", .value = "text/event-stream" }, .{ .name = "cache-control", .value = "no-cache" } } } });
        var state = Stream{ .a = a, .writer = &response.writer, .transport = request.server.out, .id = id, .model = model, .created = created, .connection = connection };
        var reply = session.generate(mx.allocator, ids.items, options, .{ .context = &state, .emit = Stream.emit, .cancelled = Stream.cancelled }) catch |err| {
            const error_body = try std.json.Stringify.valueAlloc(a, .{ .@"error" = .{ .message = @errorName(err) } }, .{});
            try response.writer.print("data: {s}\n\ndata: [DONE]\n\n", .{error_body});
            try response.end();
            return;
        };
        defer reply.deinit(mx.allocator);
        try state.chunk("", @tagName(reply.finish_reason));
        try response.writer.writeAll("data: [DONE]\n\n");
        try response.end();
    } else {
        var reply = session.generate(mx.allocator, ids.items, options, .{ .context = &connection, .cancelled = Connection.cancelled }) catch |err| return failure(a, request, .bad_request, @errorName(err));
        defer reply.deinit(mx.allocator);
        try json(a, request, .ok, .{ .id = id, .object = "text_completion", .created = created, .model = model, .choices = &.{.{ .index = @as(usize, 0), .text = reply.content, .finish_reason = @tagName(reply.finish_reason), .logprobs = @as(?u8, null) }}, .usage = .{ .prompt_tokens = ids.items.len, .completion_tokens = reply.tokens.items.len, .total_tokens = ids.items.len + reply.tokens.items.len } });
    }
}

const Stream = struct {
    a: std.mem.Allocator,
    writer: *std.Io.Writer,
    transport: *std.Io.Writer,
    id: []const u8,
    model: []const u8,
    created: i64,
    connection: Connection,
    fn cancelled(context: ?*anyopaque) bool {
        const s: *Stream = @ptrCast(@alignCast(context.?));
        return Connection.cancelled(&s.connection);
    }
    fn chunk(s: *Stream, value: []const u8, finish: ?[]const u8) !void {
        const body = try std.json.Stringify.valueAlloc(s.a, .{ .id = s.id, .object = "text_completion", .created = s.created, .model = s.model, .choices = &.{.{ .index = @as(usize, 0), .text = value, .finish_reason = finish, .logprobs = @as(?u8, null) }} }, .{});
        try s.writer.print("data: {s}\n\n", .{body});
        try s.writer.flush();
        try s.transport.flush();
    }
    fn emit(context: ?*anyopaque, value: []const u8) !void {
        if (value.len == 0) return;
        const s: *Stream = @ptrCast(@alignCast(context.?));
        return s.chunk(value, null);
    }
};

const Connection = struct {
    socket: std.posix.fd_t,
    fn cancelled(context: ?*anyopaque) bool {
        const s: *Connection = @ptrCast(@alignCast(context.?));
        var byte: [1]u8 = undefined;
        const result = std.c.recv(s.socket, &byte, 1, std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT);
        if (result == 0) return true;
        if (result > 0) return false;
        return switch (std.posix.errno(result)) {
            .AGAIN, .INTR => false,
            else => true,
        };
    }
};
