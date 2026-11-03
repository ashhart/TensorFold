//! `tensorfold slide audit`: explicitly request a read-only pocket audit from a running --slide server.
const std = @import("std");
const json = @import("json");
const stream = @import("benchmark_stream.zig");
const A = std.mem.Allocator;
pub const usage = "usage: tensorfold slide audit [--server URL] [--threshold RATE] [--key-stdin] [--json]\n";

pub const Options = struct { server: []const u8 = "http://127.0.0.1:8080", threshold: f64 = 0.01, key_stdin: bool = false, json: bool = false };

pub fn parse(args: []const []const u8) !Options {
    var o: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--json")) {
            o.json = true;
        } else if (std.mem.eql(u8, args[i], "--key-stdin")) {
            o.key_stdin = true;
        } else if (std.mem.eql(u8, args[i], "--server") or std.mem.eql(u8, args[i], "--threshold")) {
            const name = args[i];
            i += 1;
            if (i == args.len) return error.AuditUsage;
            if (std.mem.eql(u8, name, "--server")) o.server = args[i] else o.threshold = std.fmt.parseFloat(f64, args[i]) catch return error.AuditUsage;
        } else return error.AuditUsage;
    }
    if (stream.splitUrl(o.server) == null or !std.math.isFinite(o.threshold) or o.threshold < 0 or o.threshold > 1)
        return error.AuditUsage;
    return o;
}

pub fn main(init: std.process.Init, args: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    if (args.len == 1 and (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h"))) {
        try std.Io.File.stdout().writeStreamingAll(init.io, usage);
        return 0;
    }
    const argv = try a.alloc([]const u8, args.len);
    for (argv, args) |*v, x| v.* = x;
    const o = parse(argv) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    var token: ?[]const u8 = null;
    var input_buf: [4096]u8 = undefined;
    if (o.key_stdin) {
        var input = std.Io.File.stdin().readerStreaming(init.io, &input_buf);
        const bytes = input.interface.allocRemaining(a, .limited(4096)) catch return 2;
        const key = std.mem.trim(u8, bytes, " \t\r\n");
        if (key.len == 0 or key.len > 256 or std.mem.indexOfAny(u8, key, "\r\n") != null) return 2;
        token = key;
    }
    const base = std.mem.trimEnd(u8, o.server, "/");
    const suffix = if (std.mem.endsWith(u8, base, "/v1")) "/slide/graph" else "/v1/slide/graph";
    const url = try std.fmt.allocPrint(a, "{s}{s}?audit=1&threshold={d}", .{ base, suffix, o.threshold });
    var client: std.http.Client = .{ .allocator = a, .io = init.io };
    defer client.deinit();
    const r = stream.fetch(a, init.io, &client, .{ .url = url, .token = token, .limit = 16 << 20, .timeout_s = 3600 });
    if (r.failed != .none or r.status != 200) {
        std.debug.print("tensorfold: pocket audit failed (HTTP {d}, {t}); server must support pockets and --slide\n", .{ r.status, r.failed });
        return 1;
    }
    const root = switch (try json.parse(a, r.body)) {
        .ok => |v| v,
        .err => return 1,
    };
    const audit = root.field("audit") orelse return 1;
    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(init.io, &buf);
    if (o.json) {
        try out.interface.print("{s}\n", .{try json.stringify(a, audit, .{})});
    } else try display(&out.interface, audit);
    try out.interface.flush();
    return 0;
}

pub fn display(w: *std.Io.Writer, audit: json.Value) !void {
    const facts = (audit.field("facts") orelse return error.BadAudit).items() orelse return error.BadAudit;
    try w.print("Pocket audit ({s}), read-only; off-subject threshold {d:.4}, activation floor {d:.6}\n", .{
        audit.strField("version") orelse "",                                               (audit.field("off_subject_threshold") orelse return error.BadAudit).float64() orelse 0,
        (audit.field("activation_floor") orelse return error.BadAudit).float64() orelse 0,
    });
    for (facts) |f| {
        try w.print("Fact {d}, subject: {s}\n", .{ intField(f, "id") orelse 0, f.strField("subject") orelse "" });
        for ((f.field("neurons") orelse return error.BadAudit).items() orelse return error.BadAudit) |n| {
            const rate = n.field("off_subject") orelse return error.BadAudit;
            const flag = n.field("flagged") orelse return error.BadAudit;
            try w.print("  L{d} neuron {d} ({s}): {d}/{d} off-subject ({d:.4}) {s}\n", .{
                intField(n, "layer") orelse 0,                   intField(n, "neuron") orelse 0,     n.strField("key") orelse "",
                intField(rate, "fired") orelse 0,                intField(rate, "prompts") orelse 0, (rate.field("rate") orelse @as(json.Value, .null)).float64() orelse 0,
                if (flag == .bool and flag.bool) "FLAG" else "",
            });
            const cohorts = n.field("cohorts") orelse return error.BadAudit;
            inline for (.{ "own", "subject", "other", "general" }) |name| {
                const group = cohorts.field(name) orelse return error.BadAudit;
                const count = intField(group, "prompts") orelse 0;
                try w.print("    {s}: {d}/{d}{s}\n", .{ name, intField(group, "fired") orelse 0, count, if (count == 0) " (unchecked)" else "" });
            }
            for ((n.field("strongest_off_subject") orelse return error.BadAudit).items() orelse return error.BadAudit) |p|
                try w.print("    {s}: {s} (activation {d:.6}, row {d})\n", .{
                    p.strField("cohort") orelse "",                                           p.strField("prompt") orelse "",
                    (p.field("activation") orelse @as(json.Value, .null)).float64() orelse 0, intField(p, "row") orelse 0,
                });
        }
    }
}

test "audit CLI: bounded threshold, authenticated input option and credential-free URL" {
    const o = try parse(&.{ "--server", "http://127.0.0.1:8080", "--threshold", "0.2", "--json", "--key-stdin" });
    try std.testing.expect(o.json and o.key_stdin and o.threshold == 0.2);
    for ([_][]const []const u8{ &.{"--server"}, &.{ "--server", "http://user:password@localhost" }, &.{ "--threshold", "nan" }, &.{ "--threshold", "-0.1" }, &.{"--bogus"} }) |argv|
        try std.testing.expectError(error.AuditUsage, parse(argv));
}

fn intField(v: json.Value, key: []const u8) ?i64 {
    return (v.field(key) orelse return null).int64();
}
