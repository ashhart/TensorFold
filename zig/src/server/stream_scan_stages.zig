//! The stream path's whole-text steps alone, each timed on the same reply text at several reply lengths.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const tool_minimax = @import("tool_minimax.zig");
const tool_lfm2 = @import("tool_lfm2.zig");
const kimi_reply = @import("kimi_reply.zig");
const model_text = @import("model_text.zig");
const scan = @import("stream_scan_text.zig");
const Allocator = std.mem.Allocator;

const io = std.testing.io;
const full = builtin.mode != .debug;
const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"parameters\":{\"type\":\"object\"}}}]";

/// One step's arguments at one reply length: the text and tokens a chunk's commit hands it.
const At = struct {
    text: []const u8, // the visible text so far
    ids: []const u32,
    close: usize, // the close token's byte offset, in the content phase
    answer: []const u8, // the text after the close, in the content phase
    model: model_text.Text,
    kimi: model_text.Text,
    kimi_ids: []const u32,
    tools: []const json.Value,
};

const Step = struct { name: []const u8, site: []const u8, run: *const fn (Allocator, *const At) anyerror!void };

fn closePending(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway(try reply_text.closeWhere(a, reply_text.decodeText, at.model, at.ids, scan.think_end));
}
fn minimaxDsml(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway((try tool_minimax.dsml(a, at.text, at.tools)).len);
}
fn splitPending(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway((try reply_text.splitThinkingAt(a, at.text, false, reply_text.think_markers, .pending)).reasoning.len);
}
fn stopVisible(_: Allocator, at: *const At) !void {
    const stops: reply_text.Stops = .{ .strings = &.{"END OF ANSWER"} };
    std.mem.doNotOptimizeAway(stops.visible(at.text, true).len);
}
fn kimiParse(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway((try kimi_reply.parse(a, at.kimi, at.kimi_ids, true, at.tools)).reasoning.len);
}
fn closeRewrite(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway(try tool_minimax.closeAfterRewrite(a, at.text, at.tools, .{ .token = at.close }));
}
fn splitToken(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway((try reply_text.splitThinkingAt(a, at.text, false, reply_text.think_markers, .{ .token = at.close })).answer.len);
}
fn answerSteps(a: Allocator, at: *const At) !void {
    var hidden: std.ArrayList(u8) = .empty;
    const answer = try tool_lfm2.dsml(a, at.answer, at.tools, false);
    const shown = try reply_text.hideInto(a, &hidden, answer, false, true, reply_text.openersOf(reply_text.think_markers));
    std.mem.doNotOptimizeAway(reply_text.heldBack(reply_text.streamingVisible(shown), true).len);
}
fn itemCopy(a: Allocator, at: *const At) !void {
    std.mem.doNotOptimizeAway((try std.mem.concat(a, u8, &.{ at.text, " delta" })).len);
}

/// Reasoning-phase steps (the close token not yet written), then content-phase steps (written, prose after it).
const reasoning_steps = [_]Step{
    .{ .name = "close token scan", .site = "chat_generation.zig:114 -> reply_text.zig:100", .run = closePending },
    .{ .name = "MiniMax block search", .site = "chat_generation.zig:111 -> tool_minimax.zig:17", .run = minimaxDsml },
    .{ .name = "think split, openers", .site = "chat_generation.zig:116 -> reply_text.zig:176", .run = splitPending },
    .{ .name = "stop strings (one)", .site = "chat_generation.zig:110 -> reply_text.zig:387", .run = stopVisible },
    .{ .name = "Kimi re-parse", .site = "chat_generation.zig:109 -> chat_kimi.zig:20 -> kimi_reply.zig:179", .run = kimiParse },
    .{ .name = "Responses item copy", .site = "responses.zig:355", .run = itemCopy },
};
const content_steps = [_]Step{
    .{ .name = "MiniMax block search", .site = "chat_generation.zig:111 -> tool_minimax.zig:17", .run = minimaxDsml },
    .{ .name = "MiniMax close rewrite", .site = "chat_generation.zig:115 -> tool_minimax.zig:34", .run = closeRewrite },
    .{ .name = "think split at token", .site = "chat_generation.zig:116 -> reply_text.zig:165", .run = splitToken },
    .{ .name = "answer: lfm2, hide, held", .site = "chat_generation.zig:124-126", .run = answerSteps },
};

/// The median of ``reps`` runs of ``step``, in microseconds; each run allocates in a fresh arena.
fn median(gpa: Allocator, step: Step, at: *const At, reps: usize) !f64 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var times: [301]i96 = undefined;
    for (times[0..reps]) |*t| {
        _ = arena.reset(.retain_capacity);
        const start = std.Io.Clock.awake.now(io).toNanoseconds();
        try step.run(arena.allocator(), at);
        t.* = std.Io.Clock.awake.now(io).toNanoseconds() - start;
    }
    std.mem.sort(i96, times[0..reps], {}, std.sort.asc(i96));
    return @as(f64, @floatFromInt(times[reps / 2])) / 1e3;
}

test "stream scan bench: each whole-text step's time at 1k to 30k reply tokens" {
    const gpa = std.heap.smp_allocator;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const lengths = if (full) [_]usize{ 1_000, 5_000, 10_000, 20_000, 30_000 } else [_]usize{ 100, 200, 300, 300, 300 };
    const top = lengths[lengths.len - 1];
    var plain: scan.Text = .{};
    const k = try scan.kimi(gpa);
    defer k.deinit();
    const r = try scan.reply(a, .qwen, .{ .reasoning = top, .content = 500, .value = 10 }, null);
    const kr = try scan.reply(a, .kimi, .{ .reasoning = top, .content = 500, .value = 10 }, k.text());
    const tools = (try json.parse(a, tools_json)).ok.array;
    const reps: usize = if (full) 301 else 3;
    if (full) std.debug.print("\nwhole-text steps, median of {d} runs, us (reasoning phase at N reply tokens; content phase 250 prose tokens after N)\n", .{reps});
    for (reasoning_steps ++ content_steps, 0..) |step, i| {
        const content = i >= reasoning_steps.len;
        if (full) std.debug.print("  {s:<26}", .{step.name});
        for (lengths) |n| {
            const end = if (content) r.content_from + 250 else n;
            const ids = if (content) try std.mem.concat(a, u32, &.{ r.ids[0..n], r.ids[top..end] }) else r.ids[0..n];
            const text = try plain.text().decode(a, ids);
            const at: At = .{
                .text = text,
                .ids = ids,
                .close = r.reasoning_ends[n - 1],
                .answer = if (content) std.mem.trimStart(u8, text[r.reasoning_ends[n - 1] + "</think>".len ..], "\n") else "",
                .model = plain.text(),
                .kimi = k.text(),
                .kimi_ids = kr.ids[0..n],
                .tools = tools,
            };
            const us = try median(gpa, step, &at, reps);
            if (full) std.debug.print(" {d:>8.2}", .{us});
        }
        if (full) std.debug.print("   {s}\n", .{step.site});
    }
}
