//! The scrape with the ledger's families: parked, live and stale bodies, each family once, label values escaped.
const std = @import("std");
const api = @import("engine_api");
const ledger = api.load_ledger;
const load_metrics = @import("load_metrics.zig");
const Metrics = @import("metrics.zig").Metrics;

const id: load_metrics.Identity = .{ .model = "glm \"flash\"", .backend = "cuda", .version = "1.2.3" };

fn snapshot(parked: bool) ledger.Snapshot {
    return .{ .instance = @splat('a'), .start_s = 1793200000, .parked = parked, .lanes = 4, .http = @splat(0) };
}

/// Fails when a `# TYPE` line names a family twice.
fn once(body: []const u8) !void {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(std.testing.allocator);
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "# TYPE ")) continue;
        const name = line["# TYPE ".len..std.mem.lastIndexOfScalar(u8, line, ' ').?];
        const had = try seen.fetchPut(std.testing.allocator, name, {});
        if (had != null) {
            std.debug.print("family {s} twice\n", .{name});
            return error.TestFamilyTwice;
        }
    }
}

fn has(body: []const u8, text: []const u8) bool {
    return std.mem.indexOf(u8, body, text) != null;
}

test "a parked scrape shows the server values and no engine value" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try m.renderParked(std.testing.io, &out.writer, .{ .parked = true });
    try load_metrics.render(&out.writer, snapshot(true), id);
    const body = out.written();
    try once(body);
    try std.testing.expect(has(body, "tensorfold:engine_parked 1\n"));
    try std.testing.expect(has(body, "tensorfold:lanes 4\n"));
    try std.testing.expect(has(body, "tensorfold:http_requests_active{route=\"chat_completions\"} 0\n"));
    try std.testing.expect(has(body, "tensorfold:http_requests_active{route=\"other\"} 0\n"));
    for ([_][]const u8{ "requests_running", "requests_waiting", "num_requests_running", "generation_tokens_running", "waiting_prompt_tokens", "scheduler_tokens_total", "kv_capacity_mode" }) |name|
        try std.testing.expect(!has(body, name));
}

test "a live scrape shows each group once, beside the request counts metrics.zig renders" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var s = snapshot(false);
    s.http[@backingInt(ledger.RouteGroup.responses)] = 2;
    s.requests = .{ .running = 3, .waiting = 2, .waiting_prompt_tokens = 9000, .filling_left = 512, .started = 40, .requeued = 1, .yields = 2 };
    s.work = .{ .prefill_compute = 1000, .prefill_cache = 400, .decode = 300, .prefill_busy_ns = 1_500_000_000 };
    s.capacity = .{ .mode = .paged, .lanes_free = 1 };
    s.pool = .{ .capacity = 4096, .available = 1024, .reclaimable = 512, .promised = 256 };
    const status: api.Status = .{ .running = 3, .waiting = 2, .preemptions = 3 };
    try m.render(std.testing.io, &out.writer, status, &.{}, .{}, null, 4096);
    try load_metrics.render(&out.writer, s, id);
    const body = out.written();
    try once(body);
    for ([_][]const u8{
        "tensorfold:engine_parked 0\n",
        "tensorfold:http_requests_active{route=\"responses\"} 2\n",
        "tensorfold:waiting_prompt_tokens 9000\n",
        "tensorfold:filling_prompt_tokens_left 512\n",
        "tensorfold:requests_started_total 40\n",
        "tensorfold:scheduler_tokens_total{mode=\"prefill_compute\"} 1000\n",
        "tensorfold:scheduler_tokens_total{mode=\"decode\"} 300\n",
        "tensorfold:prefill_busy_seconds_total 1.5\n",
        "tensorfold:kv_capacity_mode{mode=\"paged\"} 1\n",
        "tensorfold:lanes_free 1\n",
        "tensorfold:kv_pool_available_tokens{pool=\"main\"} 1024\n",
        "tensorfold:preemptions_total 3\n",
    }) |line| try std.testing.expect(has(body, line));
    try std.testing.expect(!has(body, "kv_pool_run_available"));
}

test "an unknown fill leaves its gauge out; a stale resident engine shows the ledger's counts" {
    const gpa = std.testing.allocator;
    var m: Metrics = .{ .gpa = gpa };
    defer m.deinit();
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var s = snapshot(false);
    s.requests = .{ .running = 2, .waiting = 1, .filling_left = null, .fill_unknown = true };
    try m.renderParked(std.testing.io, &out.writer, .{ .running = 2, .waiting = 1 });
    try load_metrics.render(&out.writer, s, id);
    const body = out.written();
    try once(body);
    try std.testing.expect(has(body, "tensorfold:requests_running 2\n"));
    try std.testing.expect(has(body, "tensorfold:num_requests_waiting 1\n"));
    try std.testing.expect(!has(body, "filling_prompt_tokens_left"));
}

test "server_info escapes its label values" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try load_metrics.render(&out.writer, snapshot(false), .{ .model = "a\\b\"c\nd", .backend = "", .version = "" });
    try std.testing.expect(has(out.written(), "server_info{model=\"a\\\\b\\\"c\\nd\",backend=\"\",version=\"\",instance=\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"} 1\n"));
}
