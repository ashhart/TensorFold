//! MOSS-Image's out-of-service log, out of process (a test cannot log at err); a bad LoRA fails its request alone.
const std = @import("std");
const api = @import("engine_api");
const host = @import("moss_image_host.zig");

pub const std_options: std.Options = .{ .log_level = .err, .logFn = capture };
var lines: [1][256]u8 = undefined;
var lengths: [1]usize = undefined;
var count: usize = 0;

fn capture(comptime _: std.log.Level, comptime _: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (count == lines.len) return;
    const line = std.fmt.bufPrint(&lines[count], format, args) catch return;
    lengths[count] = line.len;
    count += 1;
}

/// Counts the one `finished` event a picture's pass sends.
fn counted(ctx: *anyopaque, _: api.Id, event: *const api.Event) void {
    const n: *usize = @ptrCast(@alignCast(ctx));
    if (event.* == .finished) n.* += 1;
}

/// The last `finished` event's reason and message.
const Last = struct { reason: ?api.Reason = null, message: []const u8 = "" };

fn last(ctx: *anyopaque, _: api.Id, event: *const api.Event) void {
    const l: *Last = @ptrCast(@alignCast(ctx));
    if (event.* == .finished) l.* = .{ .reason = event.finished.reason, .message = event.finished.message };
}

pub fn main() !void {
    const a = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var result: api.ImageResult = .{};
    var png = std.ArrayList(u8).empty;
    var events: usize = 0;
    const sink: api.Sink = .{ .ctx = &events, .event = &counted };
    const request: api.ImageRequest = .{ .prompt = &.{}, .width = 16, .height = 16, .steps = 1, .guidance = 1, .seed = 7, .result = &result, .png = &png, .alloc = a };
    var h: host.Host = .{
        .gpa = a,
        .io = io,
        .driver = undefined,
        .ctx = undefined,
        .encoder = undefined,
        .model = undefined,
        .decoder = undefined,
        .tokenizer = undefined,
        .before = &.{},
        .after = &.{},
        .empty = &.{},
        .scheduler = undefined,
    };
    // each refusal core.runtime_lora or the folder gives fails that request alone: no err line, the engine still serves
    const refusals = @typeInfo(@import("moss_image").lora.Refusal).error_set.error_names.?;
    var seen: Last = .{};
    const told: api.Sink = .{ .ctx = &seen, .event = &last };
    inline for (refusals, 2..) |r, id| {
        const at = h.queue.admit(id) orelse return error.NoSlot;
        h.jobs[at] = .{ .id = id, .request = &request, .sink = told };
        seen = .{};
        host.Host.fail(&h, at, @field(anyerror, r));
        if (h.broken) return error.StoppedOnALora;
        if (seen.reason != .failed or seen.message.len == 0) return error.LoraReasonMissing;
        if (std.mem.indexOf(u8, seen.message, "LoRA") == null) return error.LoraReasonUnnamed;
        if (h.queue.held() != 0) return error.RefusedSlotHeld;
    }
    if (count != 0) return error.LoraRefusalLogged;
    std.debug.print("moss lora refusals: {d} kinds, each one failed event naming the LoRA, the engine still serving\n", .{refusals.len});

    const slot = h.queue.admit(1) orelse return error.NoSlot;
    h.jobs[slot] = .{ .id = 1, .request = &request, .sink = sink };
    host.Host.fail(&h, slot, error.LatentRows);
    if (!h.broken) return error.NotStopped;
    if (events != 1) return error.WrongEventCount;
    if (count != 1) return error.WrongLogCount;
    const expected = "moss image: a picture's pass failed (LatentRows); this engine serves no more";
    if (!std.mem.eql(u8, expected, lines[0][0..lengths[0]])) return error.WrongLogText;
    if (h.queue.held() != 0) return error.StoppedSlotHeld;
    std.debug.print("moss out of service: one err line with the reason, one failed event, the engine stopped\n", .{});
}
