//! The cache tools: models and info read the Hugging Face cache, and pull fills it by `tensorfold pull`'s own code.
const std = @import("std");
const json = @import("json");
const checkpoint = @import("checkpoint_cli");
const mcp = @import("mcp.zig");
const tools = @import("mcp_tools.zig");
const rpc = @import("mcp_rpc.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const hub = checkpoint.hub;

/// A model argument as a checkpoint directory: a cached repo id, or the absolute path of an existing checkpoint.
pub fn resolve(a: Allocator, io: std.Io, env: *const std.process.Environ.Map, model: []const u8, problem: *[]const u8) !?[]const u8 {
    if (std.fs.path.isAbsolute(model)) {
        var parts = std.mem.tokenizeAny(u8, model, "/\\");
        while (parts.next()) |part| if (std.mem.eql(u8, part, "..")) {
            problem.* = try std.fmt.allocPrint(a, "{s}: a path with .. is refused; name the checkpoint directory itself", .{model});
            return null;
        };
        if (!hub.isDir(io, model) or @import("checkpoint_config").optional(a, io, model) == null) {
            problem.* = try std.fmt.allocPrint(a, "{s} is not a checkpoint directory (no config.json there)", .{model});
            return null;
        }
        return model;
    }
    if (!repoId(model)) {
        problem.* = try std.fmt.allocPrint(a, "{s} is neither a Hugging Face repo id (owner/name) nor the absolute path of a checkpoint directory; relative paths are refused", .{model});
        return null;
    }
    if (try hub.cachedSnapshot(a, io, try hub.cacheDir(a, env, null), model)) |snapshot| return snapshot;
    problem.* = try std.fmt.allocPrint(a, "{s} is not in the Hugging Face cache; download it with the pull tool first", .{model});
    return null;
}

/// A repo id as the Hub names them: owner/name, no ``--`` or ``..``, neither part starting or ending in - or .
pub fn repoId(text: []const u8) bool {
    if (!hub.isRepoIdLike(text) or std.mem.indexOf(u8, text, "..") != null or std.mem.indexOf(u8, text, "--") != null) return false;
    var parts = std.mem.splitScalar(u8, text, '/');
    while (parts.next()) |part| for ([_]u8{ part[0], part[part.len - 1] }) |end| {
        if (end == '-' or end == '.') return false;
    };
    return true;
}

/// The models tool: the cached checkpoints `tensorfold models` lists, with each one's revision and size on disk.
pub fn models(c: *mcp.Call) !*json.Object {
    const a = c.a;
    const io = c.ctx().io;
    const root = try hub.cacheDir(a, c.ctx().environ, null);
    const repos = try hub.cachedRepos(a, io, c.ctx().environ);
    var text: std.Io.Writer.Allocating = .init(a);
    var items: std.ArrayList(Value) = .empty;
    for (repos) |repo| {
        const snapshot = try hub.cachedSnapshot(a, io, root, repo) orelse continue;
        const family = hub.family(hub.modelType(a, io, snapshot)) orelse continue;
        const bytes = try hub.sizeOf(a, io, snapshot);
        const revision = std.fs.path.basename(snapshot);
        // the line `tensorfold models` prints, with the size and revision after it
        try text.writer.print("{s}: {s} ({s}), {d:.2} GiB, revision {s}\n", .{ repo, family.title, family.model_type, tools.gib(bytes), revision });
        const o = try json.newObject(a);
        try o.put(a, "repo", .{ .string = repo });
        try o.put(a, "family", .{ .string = family.title });
        try o.put(a, "model_type", .{ .string = family.model_type });
        try o.put(a, "revision", .{ .string = revision });
        try o.put(a, "size_bytes", try json.intValue(a, bytes));
        try items.append(a, .{ .object = o });
    }
    if (items.items.len == 0) try text.writer.writeAll("no checkpoints in the Hugging Face cache that a TensorFold family serves; download one with the pull tool\n");
    const o = try json.newObject(a);
    try o.put(a, "models", .{ .array = items.items });
    return tools.answered(a, text.written(), o);
}

/// The info tool: the lines `tensorfold info` prints for one checkpoint, and its facts as numbers.
pub fn info(c: *mcp.Call, model: []const u8) !*json.Object {
    const a = c.a;
    const io = c.ctx().io;
    var problem: []const u8 = "";
    const dir = try resolve(a, io, c.ctx().environ, model, &problem) orelse return tools.failed(a, "{s}", .{problem});
    var out: std.Io.Writer.Allocating = .init(a);
    const code = try checkpoint.info.run(a, io, &out.writer, c.ctx().environ, null, dir);
    const text = try std.mem.replaceOwned(u8, a, out.written(), dir, model); // run names the directory it was given
    if (code != 0) return tools.failed(a, "{s}", .{std.mem.trim(u8, text, " \n")});
    const family = hub.family(hub.modelType(a, io, dir)) orelse return tools.failed(a, "{s}", .{std.mem.trim(u8, text, " \n")});
    const details = hub.configDetails(a, io, dir);
    const o = try json.newObject(a);
    try o.put(a, "model", .{ .string = model });
    try o.put(a, "family", .{ .string = if (details.is_bolt) "Nemotron 3.5 Bolt" else family.title }); // as info titles it
    try o.put(a, "model_type", .{ .string = family.model_type });
    if (hub.quantization(a, io, dir)) |q| {
        const quantization = try json.newObject(a);
        try quantization.put(a, "bits", try json.intValue(a, q.bits));
        try quantization.put(a, "group_size", try json.intValue(a, q.group));
        try o.put(a, "quantization", .{ .object = quantization });
    } else try o.put(a, "quantization", .null);
    try o.put(a, "modelopt_formats", if (details.modelopt_formats) |formats| try rpc.strings(a, formats) else .null);
    try o.put(a, "context", try tools.optInt(a, hub.configInt(a, io, dir, "max_position_embeddings")));
    const memory = checkpoint.memory(a, io, dir);
    try o.put(a, "weights_bytes", try tools.optInt(a, if (memory) |m| m.weights else null));
    try o.put(a, "memory_floor_bytes", try tools.optInt(a, if (memory) |m| m.floor else null));
    return tools.answered(a, text, o);
}

/// The repos being pulled now: a second pull of one waits for none and is refused, as both would write its blobs.
pub const Pulls = struct {
    mutex: std.Io.Mutex = .init,
    repos: std.ArrayList([]const u8) = .empty,

    fn claim(p: *Pulls, gpa: Allocator, io: std.Io, repo: []const u8) !bool {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        for (p.repos.items) |r| if (std.ascii.eqlIgnoreCase(r, repo)) return false;
        try p.repos.append(gpa, try gpa.dupe(u8, repo));
        return true;
    }

    fn release(p: *Pulls, gpa: Allocator, io: std.Io, repo: []const u8) void {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        for (p.repos.items, 0..) |r, i| if (std.ascii.eqlIgnoreCase(r, repo)) {
            gpa.free(p.repos.swapRemove(i));
            return;
        };
    }

    pub fn deinit(p: *Pulls, gpa: Allocator) void {
        for (p.repos.items) |r| gpa.free(r);
        p.repos.deinit(gpa);
    }
};

/// Pull's terminal lines as they come: each kept for the answer and sent as a progress notification.
const Lines = struct {
    call: *mcp.Call,
    text: std.ArrayList(u8) = .empty,
    sent: usize = 0,
    writer: std.Io.Writer = .{ .vtable = &.{ .drain = drain }, .buffer = &.{} },

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const l: *Lines = @alignCast(@fieldParentPtr("writer", w));
        var n: usize = 0;
        for (data[0 .. data.len - 1]) |d| {
            l.text.appendSlice(l.call.a, d) catch return error.WriteFailed;
            n += d.len;
        }
        for (0..splat) |_| {
            l.text.appendSlice(l.call.a, data[data.len - 1]) catch return error.WriteFailed;
            n += data[data.len - 1].len;
        }
        while (std.mem.indexOfScalarPos(u8, l.text.items, l.sent, '\n')) |end| {
            const line = std.mem.trim(u8, l.text.items[l.sent..end], " \t\r");
            l.sent = end + 1;
            if (line.len > 0) l.call.report(line);
        }
        return n;
    }
};

/// The pull tool: ``tensorfold pull REPO[@REVISION]`` into the same cache, its lines as progress notifications.
pub fn pull(c: *mcp.Call, repo: []const u8, revision: ?[]const u8) !*json.Object {
    const a = c.a;
    const ctx = c.ctx();
    if (!repoId(repo)) return tools.failed(a, "{s} is not a Hugging Face repo id (owner/name); pull takes a repo id, never a path", .{repo});
    const spec = if (revision) |r| try std.fmt.allocPrint(a, "{s}@{s}", .{ repo, r }) else repo;
    const pulls = &c.session.pulls;
    if (!try pulls.claim(ctx.gpa, ctx.io, repo)) return tools.failed(a, "{s} is already being pulled by this session; wait for that pull to finish", .{repo});
    defer pulls.release(ctx.gpa, ctx.io, repo);
    var out: Lines = .{ .call = c };
    var err: std.Io.Writer.Allocating = .init(a);
    const code = try checkpoint.pull.run(a, ctx.io, &out.writer, &err.writer, ctx.environ, null, spec);
    if (code != 0) {
        const why = std.mem.trim(u8, err.written(), " \n");
        if (why.len > 0) return tools.failed(a, "pull {s} failed: {s}", .{ spec, why });
        return tools.failed(a, "pull {s} failed: the hub answered with an error or no files, as for a repo or revision that is missing, private or gated", .{spec});
    }
    const snapshot = try hub.cachedSnapshot(a, ctx.io, try hub.cacheDir(a, ctx.environ, null), repo) orelse
        return tools.failed(a, "pull {s} finished, but the cache holds no snapshot of it", .{spec});
    const model_type = try a.dupe(u8, hub.modelType(a, ctx.io, snapshot));
    const family = hub.family(model_type);
    const sha = std.fs.path.basename(snapshot);
    const text = if (family) |f|
        try std.fmt.allocPrint(a, "Pulled {s} at revision {s}: {s} ({s}); serve it with the serve tool.\n", .{ repo, sha, f.title, f.model_type })
    else
        try std.fmt.allocPrint(a, "Pulled {s} at revision {s} (model_type {s}), a drafter for serve's drafter argument.\n", .{ repo, sha, model_type });
    const o = try json.newObject(a);
    try o.put(a, "repo", .{ .string = repo });
    try o.put(a, "revision", .{ .string = sha });
    try o.put(a, "family", tools.optString(if (family) |f| f.title else null));
    try o.put(a, "model_type", .{ .string = model_type });
    return tools.answered(a, text, o);
}
