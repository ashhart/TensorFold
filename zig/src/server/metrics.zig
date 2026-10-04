//! Prometheus text for GET /metrics, family for family as the Python server writes it.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");

const prefix = "tensorfold:";
/// Upper edges shared by the request and time-to-first-token histograms; +Inf is added when rendered.
pub const buckets = [_]f64{ 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0 };

pub const Histogram = struct {
    counts: [buckets.len + 1]u64 = @splat(0),
    total: f64 = 0,
    n: u64 = 0,

    pub fn observe(h: *Histogram, raw: f64) void {
        const value = @max(0, raw);
        h.n += 1;
        h.total += value;
        for (buckets, 0..) |edge, i| if (value <= edge) {
            h.counts[i] += 1;
            return;
        };
        h.counts[buckets.len] += 1;
    }
};

/// Counters of finished requests; gauges are read from the engine at scrape time.
pub const Metrics = struct {
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    prompt: u64 = 0,
    generation: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    disconnects: u64 = 0,
    latency: Histogram = .{},
    ttft: Histogram = .{},
    decode: Histogram = .{},
    requests: std.ArrayList(struct { key: []const u8, status: u16, count: u64 }) = .empty,

    pub fn note(m: *Metrics, io: std.Io, prompt: usize, generation: usize, drafted: u64, accepted: u64, latency: f64, ttft: ?f64, decode: ?f64) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.prompt += prompt;
        m.generation += generation;
        m.drafted += drafted;
        m.accepted += accepted;
        m.latency.observe(latency);
        if (ttft) |t| m.ttft.observe(t);
        if (decode) |d| m.decode.observe(d);
    }

    /// One HTTP reply under its key label, counted once per request.
    pub fn httpRequest(m: *Metrics, io: std.Io, key: []const u8, status: u16) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        for (m.requests.items) |*r| if (r.status == status and std.mem.eql(u8, r.key, key)) {
            r.count += 1;
            return;
        };
        const owned = m.gpa.dupe(u8, key) catch return;
        m.requests.append(m.gpa, .{ .key = owned, .status = status, .count = 1 }) catch {};
    }

    pub fn disconnected(m: *Metrics, io: std.Io) void {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        m.disconnects += 1;
    }

    /// The scrape body, ending in a newline.
    pub fn render(m: *Metrics, io: std.Io, w: *std.Io.Writer, engine: api.Engine, window: u32) !void {
        var status: api.Status = .{};
        var streams: [512]u32 = undefined;
        engine.status(&status, &streams);
        m.mutex.lockUncancelable(io);
        const snap = m.*;
        const requests = try m.gpa.dupe(@TypeOf(m.requests.items[0]), m.requests.items);
        defer m.gpa.free(requests);
        m.mutex.unlock(io);
        std.mem.sort(@TypeOf(requests[0]), requests, {}, struct {
            fn less(_: void, x: @TypeOf(requests[0]), y: @TypeOf(requests[0])) bool {
                const o = std.mem.order(u8, x.key, y.key);
                return o == .lt or (o == .eq and x.status < y.status);
            }
        }.less);
        if (requests.len > 0) {
            try family(w, "requests_total", "counter", "HTTP replies by API key label and status.");
            for (requests) |r| try w.print("{s}requests_total{{key=\"{s}\",status=\"{d}\"}} {d}\n", .{ prefix, r.key, r.status, r.count });
        }
        try gauge(w, "requests_running", "gauge", "Requests in prefill or decode.", status.running);
        try gauge(w, "requests_waiting", "gauge", "Requests queued or held until a lane is free.", status.waiting);
        try gauge(w, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.", snap.prompt);
        try gauge(w, "generation_tokens_total", "counter", "Generated tokens of finished requests.", snap.generation);
        const live = streams[0..@min(status.streams, streams.len)];
        try family(w, "kv_cache_usage_ratio", "gauge", "Tokens in a stream cache divided by that stream's context window.");
        try pools(w, "kv_cache_usage_ratio", "pool", live, window);
        try gauge(w, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.", snap.drafted);
        try gauge(w, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try histogram(w, "request_latency_seconds", "Seconds from arrival to the reply leaving.", snap.latency);
        try histogram(w, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.", snap.ttft);
        try histogram(w, "request_decode_seconds", "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.", snap.decode);
        if (footprint()) |bytes| try gauge(w, "process_footprint_bytes", "gauge", "This process's physical footprint as the OS counts it, Metal buffers included; only where the platform reports one (macOS).", bytes);
        try gauge(w, "num_requests_running", "gauge", "Requests in prefill or decode. A mirror of tensorfold:requests_running.", status.running);
        try gauge(w, "num_requests_waiting", "gauge", "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.", status.waiting);
        try family(w, "kv_cache_usage_perc", "gauge", "A stream's cache occupancy under vLLM's name; same streams and ratios as tensorfold:kv_cache_usage_ratio.");
        try pools(w, "kv_cache_usage_perc", "stream", live, window);
        try gauge(w, "spec_decode_num_draft_tokens_total", "counter", "Draft tokens verified on finished requests, this server's single draft counter.", snap.drafted);
        try gauge(w, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.", snap.accepted);
        try histogram(w, "e2e_request_latency_seconds", "Seconds from arrival to the reply leaving, under vLLM's name.", snap.latency);
        try histogram(w, "request_decode_time_seconds", "Seconds a finished request spent decoding, under vLLM's name.", snap.decode);
        try gauge(w, "client_disconnections_total", "counter", "Requests a client left before the reply left the server.", snap.disconnects);
        if (status.preemptions) |p| try gauge(w, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.", p);
    }
};

fn family(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} {s}\n", .{ prefix, name, help, prefix, name, kind });
}

fn gauge(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8, value: u64) !void {
    try family(w, name, kind, help);
    try w.print("{s}{s} {d}\n", .{ prefix, name, value });
}

fn pools(w: *std.Io.Writer, name: []const u8, label: []const u8, lengths: []const u32, window: u32) !void {
    if (lengths.len == 0) return w.print("{s}{s}{{{s}=\"0\"}} 0\n", .{ prefix, name, label });
    for (lengths, 0..) |n, i| {
        const ratio: f64 = if (window == 0) 0 else @min(1.0, @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(window)));
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}{{{s}=\"{d}\"}} {s}\n", .{ prefix, name, label, i, num(&buf, ratio) });
    }
}

fn histogram(w: *std.Io.Writer, name: []const u8, help: []const u8, h: Histogram) !void {
    try w.print("# HELP {s}{s} {s}\n# TYPE {s}{s} histogram\n", .{ prefix, name, help, prefix, name });
    var cumulative: u64 = 0;
    for (buckets, 0..) |edge, i| {
        cumulative += h.counts[i];
        var buf: [40]u8 = undefined;
        try w.print("{s}{s}_bucket{{le=\"{s}\"}} {d}\n", .{ prefix, name, trimmed(&buf, edge, 4), cumulative });
    }
    try w.print("{s}{s}_bucket{{le=\"+Inf\"}} {d}\n", .{ prefix, name, cumulative + h.counts[buckets.len] });
    var buf: [40]u8 = undefined;
    try w.print("{s}{s}_sum {s}\n{s}{s}_count {d}\n", .{ prefix, name, num(&buf, h.total), prefix, name, h.n });
}

/// ``_num``: an integral value as an int, else six decimals without trailing zeros.
fn num(buf: []u8, value: f64) []const u8 {
    if (value == @trunc(value) and @abs(value) < 1e18) return std.fmt.bufPrint(buf, "{d}", .{@as(i64, @intFromFloat(value))}) catch "0";
    return trimmed(buf, value, 6);
}

fn trimmed(buf: []u8, value: f64, comptime places: u8) []const u8 {
    const text = std.fmt.bufPrint(buf, "{d:." ++ std.fmt.comptimePrint("{d}", .{places}) ++ "}", .{value}) catch return "0";
    return std.mem.trimEnd(u8, std.mem.trimEnd(u8, text, "0"), ".");
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

/// The process's physical footprint where macOS reports one (rusage_info_v4).
fn footprint() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var info: [43]u64 = undefined;
    if (proc_pid_rusage(getpid(), 4, &info) != 0) return null;
    return info[9];
}

test "edges and numbers" {
    var buf: [40]u8 = undefined;
    try std.testing.expectEqualStrings("0.01", trimmed(&buf, 0.01, 4));
    try std.testing.expectEqualStrings("120", trimmed(&buf, 120.0, 4));
    try std.testing.expectEqualStrings("0.333333", num(&buf, 1.0 / 3.0));
    try std.testing.expectEqualStrings("2", num(&buf, 2.0));
}
