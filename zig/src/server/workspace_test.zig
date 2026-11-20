//! Workspace completions through the real HTTP preparation, lane host and exact prompt cache.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const json = @import("json");
const chat = @import("chat.zig");
const cli = @import("cli.zig");
const model_text = @import("model_text.zig");
const server_mod = @import("server.zig");
const workspace = @import("workspace.zig");
const errors = @import("errors.zig");
const pc = api.prompt_cache;
const t = std.testing;
const Allocator = std.mem.Allocator;

fn byteIds(comptime bytes: []const u8) [bytes.len]u32 {
    var out: [bytes.len]u32 = undefined;
    for (bytes, &out) |byte, *id| id.* = byte;
    return out;
}

const ByteText = struct {
    fn text(self: *@This()) model_text.Text {
        return .{ .ctx = self, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

const Files = struct {
    tmp: std.testing.TmpDir,
    root: []u8,
    state: []u8,

    fn init() !Files {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.createDirPath(t.io, "repo");
        const base = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
        defer t.allocator.free(base);
        return .{
            .tmp = tmp,
            .root = try std.fs.path.join(t.allocator, &.{ base, "repo" }),
            .state = try std.fs.path.join(t.allocator, &.{ base, "state" }),
        };
    }
    fn deinit(f: *Files) void {
        t.allocator.free(f.root);
        t.allocator.free(f.state);
        f.tmp.cleanup();
    }
    fn write(f: *Files, name: []const u8, bytes: []const u8) !void {
        const path = try std.fs.path.join(t.allocator, &.{ "repo", name });
        defer t.allocator.free(path);
        try f.tmp.dir.writeFile(t.io, .{ .sub_path = path, .data = bytes });
    }
};

const Gone = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

const Harness = struct {
    gpa: Allocator,
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    store: pc.Store = undefined,
    host: api.LaneHost = undefined,
    srv: *server_mod.Server = undefined,
    text: ByteText = .{},
    prompt: []u32 = &.{},
    cuts: []u32 = &.{},
    submits: usize = 0,

    fn init(h: *Harness, gpa: Allocator, files: *const Files) !void {
        h.* = .{ .gpa = gpa, .cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7), .target = .{ .gpa = gpa, .prefill_chunks = 3 } };
        errdefer h.cfg.deinit(gpa);
        errdefer h.target.deinit();
        h.core = lanes.Engine.init(gpa, &h.cfg, h.target.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.store = pc.Store.init(gpa, .{ .ptr = &h.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 1, .min_prompt = 0 }, 1 << 20);
        errdefer h.store.deinit();
        h.host = api.LaneHost.init(gpa, t.io, &h.core, .{ .lanes = 1, .prompt_cache = true });
        h.host.cache = &h.store;
        try h.host.start();
        errdefer h.host.stop();
        h.srv = try server_mod.Server.init(gpa, t.io, h.engine(), h.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false }, null);
        errdefer h.srv.deinit();
        h.srv.workspace = try workspace.Service.init(gpa, t.io, files.root, files.state, @splat(7), h.srv.text, h.srv.info, .{});
    }

    fn deinit(h: *Harness) void {
        h.srv.deinit();
        h.host.stop();
        h.store.deinit();
        h.core.deinit();
        h.target.deinit();
        h.cfg.deinit(h.gpa);
        h.gpa.free(h.prompt);
        h.gpa.free(h.cuts);
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return fake.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        fake.drop(saved);
    }

    fn engine(h: *Harness) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        return h.host.engine().info();
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.submits += 1;
        h.gpa.free(h.prompt);
        h.gpa.free(h.cuts);
        h.prompt = h.gpa.dupe(u32, request.prompt) catch return error.InvalidRequest;
        h.cuts = h.gpa.dupe(u32, request.shared_prefixes) catch return error.InvalidRequest;
        return h.host.engine().submit(id, request, sink);
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.host.engine().cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, tokens: []u32) void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.host.engine().status(out, tokens);
    }
    fn memory(ctx: *anyopaque, reset: bool) ?api.Memory {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        return h.host.engine().memory(reset);
    }

    fn run(h: *Harness, a: Allocator, call: workspace.Request, prompt: []const u8) !chat.Reply {
        var cx: errors.Cx = .{ .a = a };
        return chat.run(h.srv, &cx, .{ .prompt = .{ .text = prompt }, .workspace = call, .max_tokens = 1, .fields = .{ .object = try json.newObject(a) } }, null, Gone{});
    }
};

test "workspace completions reuse exact source cuts cold warm and edited" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files = try Files.init();
    defer files.deinit();
    try files.write("b.txt", "bravo");
    try files.write("a.txt", "alpha");
    var h: Harness = undefined;
    try h.init(t.allocator, &files);
    defer h.deinit();
    const sources = [_][]const u8{ "b.txt", "a.txt" };
    const cold = try h.run(a, .{ .op = .cold, .sources = &sources }, "R");
    try t.expectEqual(@as(usize, 0), cold.cached_tokens);
    try t.expectEqualSlices(u32, &byteIds("Ralphabravo"), h.prompt);
    try t.expectEqualSlices(u32, &.{ 1, 6, 11 }, h.cuts);
    try t.expectEqual(@as(i64, 0), cold.runtime.get("workspace").?.get("manifest_reused_tokens").?.int64().?);
    const hits = h.store.counts.hits;
    const warm = try h.run(a, .{ .op = .warm }, "");
    try t.expectEqualSlices(u32, &byteIds("Ralphabravo"), h.prompt);
    try t.expect(warm.cached_tokens > 0);
    try t.expect(h.store.counts.hits > hits);
    const detail = warm.runtime.get("workspace").?;
    try t.expectEqual(@as(i64, 11), detail.get("manifest_reused_tokens").?.int64().?);
    try files.write("b.txt", "BRAVO");
    const edited = try h.run(a, .{ .op = .edit, .sources = &sources }, "R");
    try t.expectEqualSlices(u32, &byteIds("RalphaBRAVO"), h.prompt);
    try t.expectEqual(@as(usize, 6), edited.cached_tokens);
    const edit_detail = edited.runtime.get("workspace").?;
    try t.expectEqual(@as(i64, 6), edit_detail.get("manifest_reused_tokens").?.int64().?);
    try t.expectEqual(@as(i64, 1), edit_detail.get("changed_files").?.int64().?);
}

test "workspace warm restart reuses IDs but an empty cache reports no KV hit" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files = try Files.init();
    defer files.deinit();
    try files.write("a.txt", "alpha");
    var expected: []u32 = undefined;
    {
        var cold_engine: Harness = undefined;
        try cold_engine.init(t.allocator, &files);
        defer cold_engine.deinit();
        const cold = try cold_engine.run(a, .{ .op = .cold, .sources = &.{"a.txt"} }, "R");
        expected = try t.allocator.dupe(u32, cold_engine.prompt);
        try t.expectEqual(@as(usize, 0), cold.cached_tokens);
    }
    defer t.allocator.free(expected);
    var fresh: Harness = undefined;
    try fresh.init(t.allocator, &files);
    defer fresh.deinit();
    const warm = try fresh.run(a, .{ .op = .warm }, "");
    try t.expectEqualSlices(u32, expected, fresh.prompt);
    try t.expectEqual(@as(usize, 0), fresh.store.counts.hits);
    try t.expectEqual(@as(usize, 0), warm.cached_tokens);
    try t.expectEqual(@as(i64, 6), warm.runtime.get("workspace").?.get("manifest_reused_tokens").?.int64().?);
    var cx: errors.Cx = .{ .a = a };
    try t.expectError(error.Refused, workspace.parse(&cx, (try json.parse(a, "{\"prompt\":\"R\",\"workspace\":{\"op\":\"warm\"}}")).ok, false, true));
}

test "workspace source and namespace refusals do not reach the engine" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files = try Files.init();
    defer files.deinit();
    try files.write("a.txt", "alpha");
    var h: Harness = undefined;
    try h.init(t.allocator, &files);
    defer h.deinit();
    _ = try h.run(a, .{ .op = .cold, .sources = &.{"a.txt"} }, "R");
    const submissions = h.submits;
    const before = h.prompt;
    try files.write("a.txt", "Alpha");
    try t.expectError(error.Refused, h.run(a, .{ .op = .warm }, ""));
    try t.expectEqualSlices(u32, before, h.prompt);
    try t.expectEqual(submissions, h.submits);
    var cx: errors.Cx = .{ .a = a };
    const duplicate = (try json.parse(a, "{\"prompt\":\"R\",\"workspace\":{\"op\":\"edit\",\"sources\":[\"a.txt\",\"a.txt\"]}}")).ok;
    try t.expectError(error.Refused, workspace.parse(&cx, duplicate, false, true));
}

test "workspace CLI is opt in and requires both paths" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    const args = try cli.parse(a, &.{ "m", "--workspace", "/repo", "--workspace-state=/state" }, &usage);
    try t.expectEqualStrings("/repo", args.workspace.?);
    try t.expectEqualStrings("/state", args.workspace_state.?);
    try t.expect((try cli.parse(a, &.{"m"}, &usage)).workspace == null);
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--workspace", "/repo" }, &usage));
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--workspace-state", "/state" }, &usage));
}

test "workspace tokenizer bytes bind preprocessing identities" {
    var files = try Files.init();
    defer files.deinit();
    const checkpoint: [32]u8 = @splat(3);
    const tokenizer: [32]u8 = @splat(4);
    const missing = try workspace.modelIdentity(t.allocator, t.io, files.root, checkpoint, tokenizer);
    try files.write("tokenizer.json", "{\"post_processor\":{\"id\":\"first\"}}");
    const first = try workspace.modelIdentity(t.allocator, t.io, files.root, checkpoint, tokenizer);
    try files.write("tokenizer.json", "{\"post_processor\":{\"id\":\"other\"}}");
    const other = try workspace.modelIdentity(t.allocator, t.io, files.root, checkpoint, tokenizer);
    try t.expect(!std.mem.eql(u8, &first, &other));
    try t.expect(!std.mem.eql(u8, &missing, &first));
}

test "workspace compaction refuses before manifest publication or submission" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files = try Files.init();
    defer files.deinit();
    try files.write("a.txt", "alpha");
    var h: Harness = undefined;
    try h.init(t.allocator, &files);
    defer h.deinit();
    h.srv.config.compact_at = .auto;
    try t.expectError(error.Refused, h.run(a, .{ .op = .cold, .sources = &.{"a.txt"} }, "R"));
    try t.expectEqual(@as(usize, 0), h.submits);
    h.srv.config.compact_at = null;
    try t.expectError(error.Refused, h.run(a, .{ .op = .warm }, ""));
    try t.expectEqual(@as(usize, 0), h.submits);
}
