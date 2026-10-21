//! Experimental Qwen3.5 BF16 scoring runner with full-vocabulary-normalized selected-token logprobs.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;
const Request = struct { input_ids: []const u32, labels: []const u32 = &.{} };
const Batch = struct { requests: []const Request };
fn writeAll(bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.c.write(1, bytes[at..].ptr, bytes.len - at);
        if (n <= 0) return error.Write;
        at += @intCast(n);
    }
}
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.Usage;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, try std.fmt.parseInt(c_int, args[2], 10));
    defer context.deinit();
    const e = try E.Engine.init(gpa, io, &context, args[1], false);
    defer e.deinit();
    var work: [4]E.Cache = undefined;
    var made: usize = 0;
    defer for (work[0..made]) |*cache| cache.deinit(gpa);
    for (&work) |*cache| {
        cache.* = try E.Cache.init(gpa, e, 65536);
        made += 1;
    }
    try writeAll("{\"ready\":true,\"engine\":\"TensorFold-qwen35-bf16-experimental\"}\n");
    var line_: std.ArrayList(u8) = .empty;
    defer line_.deinit(gpa);
    var readbuf: [65536]u8 = undefined;
    var readat: usize = 0;
    var readlen: usize = 0;
    while (true) {
        line_.clearRetainingCapacity();
        while (true) {
            if (readat == readlen) {
                const n = std.c.read(0, &readbuf, readbuf.len);
                if (n == 0) return;
                if (n < 0) return error.Read;
                readat = 0;
                readlen = @intCast(n);
            }
            const remaining = readbuf[readat..readlen];
            const newline = std.mem.indexOfScalar(u8, remaining, '\n');
            const count = newline orelse remaining.len;
            if (line_.items.len + count > 4 * 1024 * 1024) return error.TooLarge;
            try line_.appendSlice(gpa, remaining[0..count]);
            readat += count;
            if (newline != null) {
                readat += 1;
                break;
            }
        }
        const parsed = try std.json.parseFromSlice(Batch, gpa, line_.items, .{});
        defer parsed.deinit();
        const requests = parsed.value.requests;
        if (requests.len == 0 or requests.len > 4) return error.InvalidBatch;
        const start = std.Io.Clock.awake.now(io).toNanoseconds();
        var offsets: [4]usize = undefined;
        // Fresh states keep this development tool independent of the native shared prompt-cache owner.
        for (requests, 0..) |r, i| {
            if (r.input_ids.len == 0 or r.input_ids.len > 65530) return error.InvalidPrompt;
            for (r.input_ids) |id| if (id >= e.g.vocab) return error.InvalidToken;
            if (r.labels.len > 16) return error.InvalidLabelCount;
            work[i].score_labels[0] = @intCast(r.labels.len);
            for (r.labels, 0..) |id, j| {
                if (id >= e.g.vocab) return error.InvalidLabel;
                work[i].score_labels[j + 1] = id;
            }
            offsets[i] = 0;
            try work[i].reset(e);
        }
        // Process independent prompts together while retaining each final row and state.
        while (true) {
            var parts: [4]E.Part = undefined;
            var ids: [128]u32 = undefined;
            var n: usize = 0;
            var total: usize = 0;
            for (requests, 0..) |r, i| {
                if (offsets[i] >= r.input_ids.len) continue;
                const rows = @min(128 / requests.len, r.input_ids.len - offsets[i]);
                parts[n] = .{ .cache = &work[i], .rows = rows };
                n += 1;
                @memcpy(ids[total..][0..rows], r.input_ids[offsets[i]..][0..rows]);
                offsets[i] += rows;
                total += rows;
            }
            if (n == 0) break;
            try e.forward(parts[0..n], ids[0..total], .scores);
        }
        var result: std.Io.Writer.Allocating = .init(gpa);
        defer result.deinit();
        const w = &result.writer;
        try w.writeAll("{\"results\":[");
        for (requests, 0..) |r, i| {
            if (i > 0) try w.writeByte(',');
            const logits = work[i].scores;
            const best = work[i].score_token;
            const lse: f64 = work[i].score_lse;
            try w.print("{{\"token\":{d},\"logprobs\":[", .{best});
            for (r.labels, 0..) |_, j| {
                if (j > 0) try w.writeByte(',');
                try w.print("{d}", .{@as(f64, logits[j]) - lse});
            }
            try w.print("],\"cached_tokens\":{d}}}", .{@as(usize, 0)});
        }
        const ns = std.Io.Clock.awake.now(io).toNanoseconds() - start;
        try w.print("],\"engine_ms\":{d}}}\n", .{@as(f64, @floatFromInt(ns)) / 1e6});
        try writeAll(result.written());
    }
}
