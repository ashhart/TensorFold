//! One attempt's SSE lines: events, usage, telemetry, status and timings.
const std = @import("std");
const json = @import("json.zig");
const suite = @import("benchmark_suite.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const max_event = 512 * 1024;
pub const max_stream = 8 * 1024 * 1024;

pub const Failure = error{ InvalidStream, ServerError, InvalidUsage, TruncatedStream, UnexpectedFinish, MissingUsage, EmptyOutput, TokenCountMismatch, OutOfMemory };

pub fn code(f: Failure) suite.ErrorCode {
    return switch (f) {
        error.InvalidStream, error.OutOfMemory => .invalid_stream,
        error.ServerError => .server_error,
        error.InvalidUsage => .invalid_usage,
        error.TruncatedStream => .truncated_stream,
        error.UnexpectedFinish => .unexpected_finish,
        error.MissingUsage => .missing_usage,
        error.EmptyOutput => .empty_output,
        error.TokenCountMismatch => .token_count_mismatch,
    };
}

/// One attempt's stream state; ``arrived`` stamps are monotonic nanoseconds.
pub const Stream = struct {
    a: Allocator,
    data: std.ArrayList(u8) = .empty,
    data_lines: usize = 0,
    event_bytes: usize = 0,
    total: usize = 0,
    first: ?i96 = null,
    last: ?i96 = null,
    deliveries: u32 = 0,
    usage: ?Value = null,
    runtime: *json.Object,
    done: bool = false,
    finished: bool = false,
    finish_reason: ?Value = null,

    pub fn init(a: Allocator) Allocator.Error!Stream {
        return .{ .a = a, .runtime = try json.newObject(a) };
    }

    /// One line with its newline, read at ``arrived``; true once ``[DONE]`` is seen.
    pub fn line(s: *Stream, with_newline: []const u8, arrived: i96) Failure!bool {
        s.total += with_newline.len;
        if (s.total > max_stream) return error.InvalidStream;
        const raw = std.mem.trimEnd(u8, with_newline, "\n");
        if (raw.len > max_event) return error.InvalidStream;
        const text = std.mem.trimEnd(u8, raw, "\r");
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidStream;
        if (text.len == 0) {
            defer {
                s.data.clearRetainingCapacity();
                s.data_lines = 0;
                s.event_bytes = 0;
            }
            if (s.data_lines > 0) return s.event(s.data.items, arrived);
            return false;
        }
        if (!std.mem.startsWith(u8, text, "data:")) return false;
        var value = text[5..];
        if (value.len > 0 and value[0] == ' ') value = value[1..];
        if (s.data_lines > 0) try s.data.append(s.a, '\n');
        try s.data.appendSlice(s.a, value);
        s.data_lines += 1;
        s.event_bytes += raw.len;
        if (s.event_bytes > max_event) return error.InvalidStream;
        return false;
    }

    fn event(s: *Stream, text: []const u8, arrived: i96) Failure!bool {
        if (std.mem.eql(u8, std.mem.trim(u8, text, " \t\n\r\x0b\x0c"), "[DONE]")) {
            s.done = true;
            return true;
        }
        const chunk = switch (try json.parseText(s.a, try s.a.dupe(u8, text))) {
            .ok => |v| v,
            .err => return error.InvalidStream,
        };
        if (chunk != .object) return error.InvalidStream;
        if (chunk.field("error") != null) return error.ServerError;
        if (chunk.field("usage")) |u| {
            if (u != .object) return error.InvalidUsage;
            s.usage = u;
        }
        const reported = chunk.get("tensorfold") orelse chunk.get("runtime");
        if (reported) |r| if (r == .object) for (r.object.keys(), r.object.values()) |k, v| try s.runtime.put(s.a, k, v);
        const choices = chunk.get("choices") orelse Value{ .array = &.{} };
        if (choices != .array or choices.array.len > 1) return error.InvalidStream;
        var has_text = false;
        for (choices.array) |choice| {
            if (choice != .object) return error.InvalidStream;
            if (choice.get("index")) |i| if ((i.float64() orelse return error.InvalidStream) != 0) return error.InvalidStream;
            const delta = choice.get("delta") orelse Value{ .null = {} };
            if (choice.has("delta") and delta != .object) return error.InvalidStream;
            const pieces = [_]?Value{ choice.get("text"), delta.get("content"), delta.get("reasoning_content") };
            for (pieces) |p| if (p) |x| if (x != .null and x != .string) return error.InvalidStream;
            has_text = for (pieces) |p| {
                if (p) |x| if (x == .string and x.string.len > 0) break true;
            } else false;
            if (s.finished and has_text) return error.InvalidStream;
            if (choice.field("finish_reason")) |reason| {
                if (reason != .string or s.finished) return error.InvalidStream;
                s.finished = true;
                s.finish_reason = reason;
            }
        }
        if (has_text) {
            if (s.first == null) s.first = arrived;
            s.last = arrived;
            s.deliveries += 1;
        }
        return false;
    }

    /// The checks after the stream ends, in a fixed order; counts land in ``sample`` as they are read.
    pub fn finish(s: *Stream, tokens: u64, sample: *suite.Sample) Failure!void {
        if (!s.done) return error.TruncatedStream;
        const reason = if (s.finish_reason) |r| r.string else return error.UnexpectedFinish;
        if (!std.mem.eql(u8, reason, "stop") and !std.mem.eql(u8, reason, "length")) return error.UnexpectedFinish;
        const usage = s.usage orelse return error.MissingUsage;
        const prompt = count(usage.get("prompt_tokens")) orelse return error.InvalidUsage;
        const completion = count(usage.get("completion_tokens")) orelse return error.InvalidUsage;
        sample.prompt_tokens = prompt;
        sample.completion_tokens = completion;
        const details = usage.get("prompt_tokens_details");
        var cached = if (details != null and details.? == .object) count(details.?.get("cached_tokens")) else null;
        if (cached == null) cached = count(s.runtime.get("cached_tokens") orelse s.runtime.get("cached"));
        if (cached != null and cached.? > prompt) return error.InvalidUsage;
        sample.cached_tokens = cached;
        if (s.first == null or completion == 0) return error.EmptyOutput;
        if (completion > tokens) return error.TokenCountMismatch;
        if (completion < tokens) {
            sample.status = .early_eos;
        } else if (s.deliveries < 2 or s.last.? <= s.first.?) {
            sample.status = .unmeasured;
            sample.error_code = .unmeasured_delivery;
        } else sample.status = .ok;
    }

    /// The times and telemetry every attempt keeps, failed ones included.
    pub fn timings(s: *const Stream, start: i96, end: i96, sample: *suite.Sample) void {
        sample.end_to_end_seconds = @max(0, seconds(end - start));
        sample.server_decode_tps = runtimeNumber(s.runtime, "tokens_per_second", "decode_tps");
        sample.server_decode_seconds = runtimeNumber(s.runtime, "decode_seconds", "decode_s");
        sample.prefill_seconds = runtimeNumber(s.runtime, "prefill_seconds", "prefill_s");
        sample.token_sha = tokenSha(s.a, s.runtime.get("token_sha"));
        if (s.first) |f| sample.ttft_seconds = @max(0, seconds(f - start));
        if (s.deliveries >= 2 and s.last.? > s.first.?) {
            const span = seconds(s.last.? - s.first.?);
            sample.delivery_seconds = span;
            if (sample.completion_tokens) |c| if (c > 1) {
                sample.delivery_tps = @as(f64, @floatFromInt(c - 1)) / span;
            };
        }
    }
};

pub fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(@as(i64, @intCast(ns)))) / 1e9;
}

/// A count: a non-negative JSON integer (bools are not).
fn count(v: ?Value) ?u64 {
    const x = v orelse return null;
    if (x != .int or x.int.len == 0 or x.int[0] == '-') return null;
    return std.fmt.parseInt(u64, x.int, 10) catch null;
}

/// ``key``, else ``fallback``, from the telemetry block: a finite non-negative number, else null.
fn runtimeNumber(runtime: *const json.Object, key: []const u8, fallback: []const u8) ?f64 {
    const v = runtime.get(key) orelse runtime.get(fallback) orelse return null;
    if (v != .int and v != .float) return null;
    const f = v.float64() orelse return null;
    return if (std.math.isFinite(f) and f >= 0) f else null;
}

fn tokenSha(a: Allocator, v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    if (x != .string or (x.string.len != 12 and x.string.len != 64)) return null;
    for (x.string) |ch| if (!std.ascii.isHex(ch)) return null;
    return std.ascii.allocLowerString(a, x.string) catch null;
}
