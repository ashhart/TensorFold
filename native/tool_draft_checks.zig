const std = @import("std");
const draft = @import("tool_draft.zig");

pub fn check(io: std.Io, model: []const u8, path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 * 1024 * 1024));
    const cases = try std.json.parseFromSlice(struct {
        structures: []const struct { tools: std.json.Value, text: []const u8, open_at_start: bool, end_text: bool, expected: ?[]const u8 },
        streams: []const struct { tools: std.json.Value, prompt_len: usize, open_at_start: bool, fallback: bool, events: []const struct { context: []const i32, max_draft: usize, tree: bool, accepted: usize, tokens: []const i32, parents: []const i32, confident: bool, match: usize, telemetry: std.json.Value } },
        copies: []const []const struct { context: []const i32, max_draft: usize, accepted: usize, tokens: []const i32, confident: bool, match: usize, silent_for: usize },
    }, a, bytes, .{});
    const directory = try std.Io.Dir.cwd().realPathFileAlloc(io, model, a);
    var tokenizer = try @import("vendor/tokenizer.zig").loadTokenizer(io, a, directory);
    defer tokenizer.deinit();
    for (cases.value.structures, 0..) |case, i| {
        var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer scratch.deinit();
        const allocator = scratch.allocator();
        const actual = try draft.structure(allocator, try draft.schema(allocator, case.tools), case.text, case.open_at_start, case.end_text);
        errdefer std.debug.print("Structure case {d}: {f}\n", .{ i, std.json.fmt(case, .{}) });
        if (case.expected) |expected| try std.testing.expectEqualStrings(expected, actual orelse return error.MissingToolDraft) else if (actual != null) return error.UnexpectedToolDraft;
    }
    var count: usize = 0;
    for (cases.value.streams, 0..) |stream, i| {
        var proposer = try draft.Proposer.init(std.heap.page_allocator, &tokenizer, stream.tools, stream.prompt_len);
        defer proposer.deinit();
        proposer.open_at_start = stream.open_at_start;
        proposer.fallback_enabled = stream.fallback;
        for (stream.events, 0..) |event, j| {
            errdefer std.debug.print("Tool draft stream {d}, event {d}\n", .{ i, j });
            const proposal = if (event.tree) try proposer.proposeTree(a, event.context, event.max_draft) else try proposer.propose(a, event.context, event.max_draft);
            try std.testing.expectEqualSlices(i32, event.tokens, proposal.tokens[0..proposal.len]);
            try std.testing.expectEqualSlices(i32, event.parents, proposal.parents[0..proposal.len]);
            try std.testing.expectEqual(event.confident, proposer.confident());
            try std.testing.expectEqual(event.match, proposer.matchLength());
            proposer.observe(proposal.len, event.accepted);
            const telemetry = proposer.telemetry();
            inline for (comptime std.meta.fieldNames(@TypeOf(telemetry))) |field| if (event.telemetry.object.get(field)) |value| try std.testing.expectEqual(@as(usize, @intCast(value.integer)), @field(telemetry, field));
            count += 1;
        }
    }
    for (cases.value.copies) |events| {
        var copy = @import("suffix_lookup.zig").Lookup{ .allocator = std.heap.page_allocator };
        defer copy.deinit();
        for (events) |event| {
            const proposal = try copy.propose(event.context, event.max_draft);
            try std.testing.expectEqualSlices(i32, event.tokens, proposal.tokens[0..proposal.len]);
            try std.testing.expectEqual(event.confident, copy.last_confident);
            try std.testing.expectEqual(event.match, copy.last_match);
            copy.observe(proposal.len, event.accepted);
            try std.testing.expectEqual(event.silent_for, copy.silent_for);
            count += 1;
        }
    }
    std.debug.print("PASS: {d} upstream tool structure cases and {d} proposal/copy events for {s}\n", .{ cases.value.structures.len, count, model });
}
