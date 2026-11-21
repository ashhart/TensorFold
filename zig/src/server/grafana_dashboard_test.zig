//! packaging/grafana/tensorfold-engine.json queries only the tensorfold: families that metrics.zig writes.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Names = std.StringHashMapUnmanaged(void);

const dashboard_path = "packaging/grafana/tensorfold-engine.json";
const metrics_path = "zig/src/server/metrics.zig";

/// PromQL functions, aggregations and keywords the dashboard may use; any other word must be a metric.
const words = [_][]const u8{
    "histogram_quantile", "sum",     "rate",   "irate",    "increase",   "clamp_min",   "clamp_max", "max",
    "min",                "avg",     "count",  "abs",      "vector",     "absent",      "topk",      "bottomk",
    "by",                 "without", "on",     "ignoring", "group_left", "group_right", "bool",      "offset",
    "and",                "or",      "unless",
};

/// Every family metrics.zig declares, with a histogram's _bucket, _sum and _count series.
fn emitted(a: Allocator, source: []const u8) !Names {
    var names: Names = .empty;
    for ([_][]const u8{ "gauge(w, \"", "family(w, \"", "histogram(w, \"" }) |call| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, source, at, call)) |hit| {
            const start = hit + call.len;
            const end = std.mem.indexOfScalarPos(u8, source, start, '"') orelse return error.BadSource;
            const suffixes: []const []const u8 = if (call[0] == 'h') &.{ "_bucket", "_sum", "_count" } else &.{""};
            for (suffixes) |suffix| try names.put(a, try std.fmt.allocPrint(a, "tensorfold:{s}{s}", .{ source[start..end], suffix }), {});
            at = end;
        }
    }
    return names;
}

fn identChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == ':';
}

fn known(word: []const u8) bool {
    for (words) |w| if (std.mem.eql(u8, w, word)) return true;
    return false;
}

/// Checks each metric an expression names and returns how many it named; labels, ranges and strings are skipped.
fn check(expr: []const u8, names: *const Names) !usize {
    if (std.mem.startsWith(u8, expr, "label_values(")) {
        const comma = std.mem.lastIndexOfScalar(u8, expr, ',') orelse return error.BadQuery;
        return check(expr["label_values(".len..comma], names);
    }
    var found: usize = 0;
    var depth: usize = 0;
    var i: usize = 0;
    while (i < expr.len) {
        const c = expr[i];
        if (c == '"' or c == '\'') {
            i += 1;
            while (i < expr.len and expr[i] != c) i += if (expr[i] == '\\') 2 else 1;
            i += 1;
        } else if (c == '{' or c == '[') {
            depth += 1;
            i += 1;
        } else if (c == '}' or c == ']') {
            depth -|= 1;
            i += 1;
        } else if (depth > 0) {
            i += 1;
        } else if (c == '$') { // a Grafana variable
            i += 1;
            while (i < expr.len and identChar(expr[i])) i += 1;
        } else if (std.ascii.isDigit(c) or (c == '.' and i + 1 < expr.len and std.ascii.isDigit(expr[i + 1]))) {
            i += 1; // a number or duration: 0.95, 1e-9, 5m
            while (i < expr.len and (std.ascii.isAlphanumeric(expr[i]) or expr[i] == '.' or
                ((expr[i] == '-' or expr[i] == '+') and (expr[i - 1] == 'e' or expr[i - 1] == 'E')))) i += 1;
        } else if (std.ascii.isAlphabetic(c) or c == '_' or c == ':') {
            const start = i;
            while (i < expr.len and identChar(expr[i])) i += 1;
            const word = expr[start..i];
            if (known(word)) {
                const grouping = std.mem.eql(u8, word, "by") or std.mem.eql(u8, word, "without") or std.mem.eql(u8, word, "on") or
                    std.mem.eql(u8, word, "ignoring") or std.mem.eql(u8, word, "group_left") or std.mem.eql(u8, word, "group_right");
                var j = i;
                while (j < expr.len and expr[j] == ' ') j += 1;
                if (grouping and j < expr.len and expr[j] == '(') i = (std.mem.indexOfScalarPos(u8, expr, j, ')') orelse return error.BadQuery) + 1;
                continue;
            }
            if (!names.contains(word)) return error.UnknownMetric;
            found += 1;
        } else i += 1;
    }
    return found;
}

/// Every panel query and template variable query; each target must name at least one written metric.
fn checkDashboard(a: Allocator, bytes: []const u8, names: *const Names) !usize {
    const root = switch (try json.parse(a, bytes)) {
        .ok => |v| v,
        else => return error.BadDashboard,
    };
    const panels = (root.get("panels") orelse return error.BadDashboard).items() orelse return error.BadDashboard;
    var queries: usize = 0;
    for (panels) |panel| {
        const targets = (panel.get("targets") orelse return error.BadDashboard).items() orelse return error.BadDashboard;
        if (targets.len == 0) return error.BadDashboard;
        for (targets) |target| {
            if (try check(target.strField("expr") orelse return error.BadDashboard, names) == 0) return error.NoMetric;
            queries += 1;
        }
    }
    const variables = ((root.get("templating") orelse return error.BadDashboard).get("list") orelse return error.BadDashboard).items() orelse return error.BadDashboard;
    for (variables) |variable| {
        const query = variable.get("query") orelse continue;
        if (query != .object) continue;
        if (try check(query.strField("query") orelse return error.BadDashboard, names) == 0) return error.NoMetric;
        queries += 1;
    }
    return queries;
}

test "grafana dashboard: every query names a tensorfold: metric that metrics.zig writes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const source = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, metrics_path, a, .limited(1 << 20));
    const dashboard = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, dashboard_path, a, .limited(1 << 20));
    const names = try emitted(a, source);
    try std.testing.expect(try checkDashboard(a, dashboard, &names) >= 20);
    // a metric renamed in the engine, or a dashboard query renamed, is caught
    const renamed_source = try std.mem.replaceOwned(u8, a, source, "\"requests_waiting\"", "\"requests_queued\"");
    const stale = try emitted(a, renamed_source);
    try std.testing.expectError(error.UnknownMetric, checkDashboard(a, dashboard, &stale));
    const renamed_query = try std.mem.replaceOwned(u8, a, dashboard, "tensorfold:kv_cache_usage_ratio", "tensorfold:kv_cache_usage");
    try std.testing.expectError(error.UnknownMetric, checkDashboard(a, renamed_query, &names));
}

test "grafana dashboard: numbers, variables, label lists and unprefixed names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = try emitted(a, "try gauge(w, \"lanes\", \"gauge\", \"x\", 1);\ntry histogram(w, \"ttft\", \"x\", h);\n");
    try std.testing.expectEqual(@as(usize, 2), try check("histogram_quantile(0.95, sum by (le) (rate(tensorfold:ttft_bucket{a=~\"$b\"}[$__rate_interval]))) / clamp_min(tensorfold:lanes, 1e-9)", &names));
    try std.testing.expectEqual(@as(usize, 1), try check("label_values(tensorfold:lanes, instance)", &names));
    try std.testing.expectError(error.UnknownMetric, check("rate(process_resident_bytes[5m])", &names));
    try std.testing.expectError(error.UnknownMetric, check("tensorfold:ttft", &names));
}
