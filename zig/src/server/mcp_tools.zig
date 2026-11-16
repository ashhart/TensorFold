//! The tools `tensorfold mcp` offers: names, schemas and hints, the argument checks, and each call's answer.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const rpc = @import("mcp_rpc.zig");
const mcp = @import("mcp.zig");
const cache = @import("mcp_cache.zig");
const serve = @import("mcp_serve.zig");
const host = @import("benchmark_host.zig");
const server = @import("benchmark_server.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const none: Value = .null;

pub const Tool = enum {
    machine,
    models,
    info,
    pull,
    serve,
    stop,
    status,

    /// pull and serve can take minutes, so each runs on its own thread while the session keeps reading.
    pub fn long(t: Tool) bool {
        return t == .pull or t == .serve;
    }
};

/// How long a 2026-07-28 client may keep the tool list: it changes only with the binary.
pub const ttl_ms: u64 = 3_600_000;

const Kind = enum { string, integer };
const Param = struct { name: []const u8, kind: Kind = .string, required: bool = false, about: []const u8 };
const Hints = struct { read_only: bool = false, destructive: bool = false, idempotent: bool = true, open_world: bool = false };
const Spec = struct { tool: Tool, title: []const u8, about: []const u8, params: []const Param = &.{}, hints: Hints };

const model_about = "A cached Hugging Face repo id (owner/name), or the absolute path of an existing checkpoint directory";

/// The tools in the order tools/list gives them, which never varies.
const specs = [_]Spec{
    .{ .tool = .machine, .title = "Describe this machine", .about = "This machine: OS, CPU, memory and GPUs, and the TensorFold version, backend and model types that serve here.", .hints = .{ .read_only = true } },
    .{ .tool = .models, .title = "List cached models", .about = "The checkpoints in the local Hugging Face cache that a TensorFold family serves, with their revisions and sizes on disk.", .hints = .{ .read_only = true } },
    .{ .tool = .info, .title = "Describe a model", .about = "One checkpoint's family, format, context, weights and the memory it needs to start, as tensorfold info prints them.", .params = &.{.{ .name = "model", .required = true, .about = model_about }}, .hints = .{ .read_only = true } },
    .{ .tool = .pull, .title = "Download a model", .about = "Downloads a Hugging Face repo into the local cache as tensorfold pull does: checked against the Hub, resumable, and refused when no TensorFold family serves it. Reports progress when the request asks for it.", .params = &.{
        .{ .name = "repo", .required = true, .about = "The Hugging Face repo id, owner/name" },
        .{ .name = "revision", .about = "A branch, tag or 40-hex commit; main when left out" },
    }, .hints = .{ .open_world = true } },
    .{ .tool = .serve, .title = "Start a local server", .about = "Starts tensorfold serve for a cached model on a free 127.0.0.1 port with a fresh API key, waits until it answers, and returns its base URL, served name and key. Serving the same model and settings again returns the server already running.", .params = &.{
        .{ .name = "model", .required = true, .about = model_about },
        .{ .name = "context", .kind = .integer, .about = "The context length in tokens; the model family's default when left out" },
        .{ .name = "drafter", .about = "A drafter checkpoint for faster decoding, named as model is" },
    }, .hints = .{} },
    .{ .tool = .stop, .title = "Stop a local server", .about = "Stops a server this session started, by the id serve gave.", .params = &.{.{ .name = "id", .required = true, .about = "The server's id from serve or status" }}, .hints = .{ .destructive = true } },
    .{ .tool = .status, .title = "List local servers", .about = "The servers this session started: id, model, served name, base URL and whether each still runs.", .hints = .{ .read_only = true } },
};

fn specOf(t: Tool) Spec {
    for (specs) |s| if (s.tool == t) return s;
    unreachable;
}

/// tools/list's result: every tool with its input schema and hints, cacheable for an hour under 2026-07-28.
pub fn list(a: Allocator, v: rpc.Version) !*json.Object {
    const items = try a.alloc(Value, specs.len);
    for (specs, items) |s, *item| item.* = try describe(a, s);
    const body = try json.newObject(a);
    try body.put(a, "tools", .{ .array = items });
    if (v.stateless()) {
        try body.put(a, "ttlMs", try json.intValue(a, ttl_ms));
        try body.put(a, "cacheScope", .{ .string = "public" });
    }
    return body;
}

fn describe(a: Allocator, s: Spec) !Value {
    const props = try json.newObject(a);
    var required: std.ArrayList(Value) = .empty;
    for (s.params) |p| {
        const prop = try json.newObject(a);
        try prop.put(a, "type", .{ .string = @tagName(p.kind) });
        if (p.kind == .integer) try prop.put(a, "minimum", try json.intValue(a, 1));
        try prop.put(a, "description", .{ .string = p.about });
        try props.put(a, p.name, .{ .object = prop });
        if (p.required) try required.append(a, .{ .string = p.name });
    }
    const schema = try json.newObject(a);
    try schema.put(a, "type", .{ .string = "object" });
    try schema.put(a, "properties", .{ .object = props });
    if (required.items.len > 0) try schema.put(a, "required", .{ .array = required.items });
    try schema.put(a, "additionalProperties", .{ .bool = false });
    const hints = try json.newObject(a);
    try hints.put(a, "title", .{ .string = s.title });
    try hints.put(a, "readOnlyHint", .{ .bool = s.hints.read_only });
    try hints.put(a, "destructiveHint", .{ .bool = s.hints.destructive });
    try hints.put(a, "idempotentHint", .{ .bool = s.hints.idempotent });
    try hints.put(a, "openWorldHint", .{ .bool = s.hints.open_world });
    const o = try json.newObject(a);
    try o.put(a, "name", .{ .string = @tagName(s.tool) });
    try o.put(a, "title", .{ .string = s.title });
    try o.put(a, "description", .{ .string = s.about });
    try o.put(a, "inputSchema", .{ .object = schema });
    try o.put(a, "annotations", .{ .object = hints });
    return .{ .object = o };
}

/// A call checked against its tool's schema, then run; an argument problem is a tool error the model can correct.
pub fn call(c: *mcp.Call) !*json.Object {
    if (try check(c.a, specOf(c.tool), c.args)) |why| return failed(c.a, "{s}", .{why});
    const args = c.args;
    return switch (c.tool) {
        .machine => machine(c),
        .models => cache.models(c),
        .info => cache.info(c, args.strField("model").?),
        .pull => cache.pull(c, args.strField("repo").?, args.strField("revision")),
        .serve => serve.start(c, .{ .model = args.strField("model").?, .context = if (args.get("context")) |n| n.int64() else null, .drafter = args.strField("drafter") }),
        .stop => serve.stop(c, args.strField("id").?),
        .status => serve.status(c),
    };
}

/// The first way ``args`` breaks the tool's schema: an unknown or empty argument, a wrong type, a missing one.
fn check(a: Allocator, s: Spec, args: Value) !?[]const u8 {
    const tool = @tagName(s.tool);
    if (args == .object) for (args.object.keys(), args.object.values()) |key, v| {
        const p = for (s.params) |p| {
            if (std.mem.eql(u8, p.name, key)) break p;
        } else return try std.fmt.allocPrint(a, "{s} takes no argument named {s}{s}", .{ tool, key, try takes(a, s) });
        if (v == .null and !p.required) continue; // an optional argument sent as null is left out
        switch (p.kind) {
            .string => if (v != .string or v.string.len == 0) return try std.fmt.allocPrint(a, "{s}: {s} must be a non-empty string", .{ tool, key }),
            .integer => if ((v.int64() orelse 0) < 1 or (v.int64() orelse 0) > std.math.maxInt(u32))
                return try std.fmt.allocPrint(a, "{s}: {s} must be a whole number from 1 to {d}", .{ tool, key, std.math.maxInt(u32) }),
        }
    };
    for (s.params) |p| if (p.required and (args.get(p.name) orelse none) == .null)
        return try std.fmt.allocPrint(a, "{s} needs {s}: {s}", .{ tool, p.name, p.about });
    return null;
}

fn takes(a: Allocator, s: Spec) ![]const u8 {
    if (s.params.len == 0) return "; it takes no arguments";
    var names: std.ArrayList(u8) = .empty;
    for (s.params, 0..) |p, i| try names.print(a, "{s}{s}", .{ if (i == 0) "" else ", ", p.name });
    return std.fmt.allocPrint(a, "; it takes {s}", .{names.items});
}

/// A tool's answer: ``text`` for the model, and the same facts as structured content.
pub fn answered(a: Allocator, text: []const u8, structured: *json.Object) !*json.Object {
    const body = try json.newObject(a);
    try body.put(a, "content", try textContent(a, text));
    try body.put(a, "structuredContent", .{ .object = structured });
    try body.put(a, "isError", .{ .bool = false });
    return body;
}

/// A tool error: the reason as text with ``isError`` set, so the model sees it and can correct the call.
pub fn failed(a: Allocator, comptime fmt: []const u8, args: anytype) !*json.Object {
    const body = try json.newObject(a);
    try body.put(a, "content", try textContent(a, try std.fmt.allocPrint(a, fmt, args)));
    try body.put(a, "isError", .{ .bool = true });
    return body;
}

fn textContent(a: Allocator, text: []const u8) !Value {
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "text" });
    try part.put(a, "text", .{ .string = text });
    const items = try a.alloc(Value, 1);
    items[0] = .{ .object = part };
    return .{ .array = items };
}

pub fn optString(s: ?[]const u8) Value {
    return if (s) |x| .{ .string = x } else .null;
}

pub fn optInt(a: Allocator, n: ?u64) !Value {
    return if (n) |x| json.intValue(a, x) else .null;
}

pub fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// The machine tool: the benchmark's allowlisted host facts (no hostname or serials), with this build's version.
fn machine(c: *mcp.Call) !*json.Object {
    const a = c.a;
    const ctx = c.ctx();
    const engines = ctx.engines(a);
    const backend = server.buildBackend();
    const probe = host.detect(a, ctx.io, ctx.environ, backend);
    const h = probe.hardware;
    var text: std.Io.Writer.Allocating = .init(a);
    const w = &text.writer;
    try w.print("TensorFold {s} ({s} backend) on {t} {t}\n", .{ ctx.version, backend, builtin.os.tag, builtin.cpu.arch });
    try w.print("CPU: {s}\n", .{h.cpu_model orelse "unknown"});
    if (h.system_memory_bytes) |m| try w.print("memory: {d:.1} GiB ({s})\n", .{ gib(m), h.memory_type }) else try w.writeAll("memory: unknown\n");
    const gpus = try a.alloc(Value, h.gpus.len);
    for (h.gpus, gpus) |g, *slot| {
        try w.print("GPU: {s}", .{g.name});
        if (g.memory_bytes) |m| try w.print(", {d:.1} GiB", .{gib(m)});
        if (g.cores) |n| try w.print(", {d} cores", .{n});
        try w.writeAll("\n");
        const o = try json.newObject(a);
        try o.put(a, "name", .{ .string = g.name });
        try o.put(a, "memory_bytes", try optInt(a, g.memory_bytes));
        try o.put(a, "cores", try optInt(a, g.cores));
        slot.* = .{ .object = o };
    }
    if (h.gpus.len == 0) try w.writeAll("GPU: none found\n");
    const types = try a.alloc(Value, engines.families.len);
    for (engines.families, types) |f, *slot| slot.* = .{ .string = f.model_type };
    try w.print("model types served: {d}\n", .{types.len});
    const o = try json.newObject(a);
    try o.put(a, "os", .{ .string = @tagName(builtin.os.tag) });
    try o.put(a, "arch", .{ .string = @tagName(builtin.cpu.arch) });
    try o.put(a, "cpu", optString(h.cpu_model));
    try o.put(a, "memory_bytes", try optInt(a, h.system_memory_bytes));
    try o.put(a, "memory_type", .{ .string = h.memory_type });
    try o.put(a, "gpus", .{ .array = gpus });
    try o.put(a, "driver", optString(probe.driver));
    try o.put(a, "tensorfold_version", .{ .string = ctx.version });
    try o.put(a, "backend", .{ .string = backend });
    try o.put(a, "model_types", .{ .array = types });
    return answered(a, text.written(), o);
}
