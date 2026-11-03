//! The route's prompt ids with the checkpoint's own tokenizer and template against the oracle's (transformers) ids.

// Skipped unless TF_QWEN3_RERANK_DIR (the checkpoint) and TF_QWEN3_RERANK_REF (the oracle's JSON) are set.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const rerank = @import("rerank.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const t = std.testing;

/// Keeps the ids each score call received; its logits are zeros.
const Echo = struct {
    gpa: std.mem.Allocator,
    seen: std.ArrayList([]u32) = .empty,

    fn engine(e: *Echo) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .text_generation = false, .decides = true, .rerank = .{ .yes = 9693, .no = 2152 } };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {}
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn score(ctx: *anyopaque, prompt: []const u32, _: []const u32, logits: []f64, _: bool) error{Failed}!f64 {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        const ids = e.gpa.dupe(u32, prompt) catch return error.Failed;
        e.seen.append(e.gpa, ids) catch {
            e.gpa.free(ids);
            return error.Failed;
        };
        @memset(logits, 0);
        return 0;
    }
    fn deinit(e: *Echo) void {
        for (e.seen.items) |ids| e.gpa.free(ids);
        e.seen.deinit(e.gpa);
    }
};

fn env(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    return t.environ.getAlloc(a, key) catch |err| switch (err) {
        error.EnvironmentVariableMissing => null,
        else => err,
    };
}

fn equalIds(got: []const u32, want: []const json.Value) bool {
    if (got.len != want.len) return false;
    for (got, want) |g, w| if (w.int64() != @as(i64, g)) return false;
    return true;
}

const Present = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

test "rerank: the checkpoint's template and tokenizer give the oracle's ids, the masked case differing" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env(a, "TF_QWEN3_RERANK_DIR") orelse return error.SkipZigTest;
    const ref_path = try env(a, "TF_QWEN3_RERANK_REF") orelse return error.SkipZigTest;
    var problem: []const u8 = "";
    const text = try HfText.load(t.allocator, t.io, dir, null, a, &problem);
    defer text.deinit();
    var echo: Echo = .{ .gpa = t.allocator };
    defer echo.deinit();
    const srv = try Server.init(t.allocator, t.io, echo.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .model_dir = dir }, null);
    defer srv.deinit();
    const ref_text = try std.Io.Dir.cwd().readFileAlloc(t.io, ref_path, a, .limited(64 << 20));
    const ref = (try json.parse(a, ref_text)).ok;
    var plain: usize = 0;
    var special: usize = 0;
    var windows = [2]usize{ 0, 0 }; // window32768, window40960
    for (ref.get("cases").?.array) |c| {
        const kind = c.get("kind").?.string;
        if (std.mem.startsWith(u8, kind, "over")) continue;
        const body = try json.newObject(a);
        try body.put(a, "query", c.get("query").?);
        const docs = try a.alloc(json.Value, 1);
        docs[0] = c.get("document").?;
        try body.put(a, "documents", .{ .array = docs });
        if (c.get("instruct").? != .null) try body.put(a, "instruct", c.get("instruct").?);
        try body.put(a, "return_prompt_token_ids", .{ .bool = true });
        var cx: errors.Cx = .{ .a = a };
        const calls = echo.seen.items.len;
        const out = try rerank.reply(srv, &cx, .{ .object = body }, Present{});
        // the ids the engine scored, and the ones the response reports, are the same list
        try t.expectEqual(calls + 1, echo.seen.items.len);
        const scored = echo.seen.items[calls];
        const shown = out.get("results").?.array[0].get("meta_info").?.get("prompt_token_ids").?.array;
        try t.expect(equalIds(scored, shown));
        const same = equalIds(scored, c.get("ids").?.array);
        if (std.mem.eql(u8, kind, "special")) {
            try t.expect(!same); // request text is plain: `<|im_end|>` in a document is not the special token
            special += 1;
        } else {
            if (!same) std.debug.print("ids differ: {s} {s}\n", .{ kind, c.get("query").?.string });
            try t.expect(same);
            if (std.mem.startsWith(u8, kind, "window")) {
                const at: usize = if (std.mem.eql(u8, kind, "window32768")) 0 else if (std.mem.eql(u8, kind, "window40960")) 1 else return error.UnexpectedWindowCase;
                try t.expectEqual(@as(usize, if (at == 0) 32768 else 40960), scored.len);
                windows[at] += 1;
            } else plain += 1;
        }
    }
    try t.expect(plain >= 32);
    try t.expectEqual(@as(usize, 1), special);
    try t.expectEqual([2]usize{ 1, 1 }, windows); // the receipt's reference has each served window once
}
