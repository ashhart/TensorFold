//! The command's flags, parsed as argparse does, and the checks on them in a fixed order with fixed words.
const std = @import("std");
const fields = @import("fields.zig");
const upload = @import("benchmark_upload.zig");
const Allocator = std.mem.Allocator;

pub const Action = enum { run, publish, inspect, login, logout, suites, help };
pub const Backend = enum { auto, mlx, cuda, rocm };

pub const Args = struct {
    action: Action = .run,
    target: []const u8 = "",
    file: ?[]const u8 = null,
    backend: Backend = .auto,
    server: ?[]const u8 = null,
    model_id: ?[]const u8 = null,
    tokens: i64 = 256,
    reps: i64 = 5,
    temperatures: []const f64 = &.{ 1.0, 0.0 },
    serial: bool = false,
    context: ?i64 = null,
    drafter: []const u8 = "auto",
    timeout: f64 = 600,
    startup_timeout: f64 = 600,
    output: ?[]const u8 = null,
    publish: bool = false,
    yes: bool = false,
    upload_url: []const u8 = upload.default_api,
    token_stdin: bool = false,
};

pub const usage_line = "usage: tensorfold benchmark [-h] [--backend {auto,mlx,cuda,rocm}] [--server SERVER] [--model-id MODEL_ID] [--tokens TOKENS] [--reps REPS] [--temperatures TEMPERATURES] [--serial] [--context CONTEXT] [--drafter DRAFTER] [--timeout TIMEOUT] [--startup-timeout STARTUP_TIMEOUT] [--output OUTPUT] [--publish] [--yes] [--upload-url UPLOAD_URL] [--token-stdin] [target] [file]";

const Flag = struct { name: []const u8, value: bool = true, help: []const u8 };

pub const flags = [_]Flag{
    .{ .name = "--backend", .help = "auto, mlx, cuda or rocm (default: auto, this build's backend)" },
    .{ .name = "--server", .help = "attach to an existing server; result stays unranked without a server manifest" },
    .{ .name = "--model-id", .help = "model alias to request from an attached server" },
    .{ .name = "--tokens", .help = "output tokens per request (default: 256)" },
    .{ .name = "--reps", .help = "measured repetitions per workload and temperature (default: 5)" },
    .{ .name = "--temperatures", .help = "comma-separated sampling temperatures (default: 1.0,0)" },
    .{ .name = "--serial", .value = false, .help = "use the same engine with drafts disabled" },
    .{ .name = "--context", .help = "explicit prompt plus reply capacity" },
    .{ .name = "--drafter", .help = "a drafter checkpoint directory or cached repo ID for the managed server; auto (default) and none pass no drafter, so the engine keeps its own default" },
    .{ .name = "--download", .value = false, .help = "not supported natively; run tensorfold pull MODEL first" },
    .{ .name = "--timeout", .help = "request timeout in seconds (default: 600)" },
    .{ .name = "--startup-timeout", .help = "managed server startup timeout in seconds (default: 600)" },
    .{ .name = "--output", .help = "receipt path (default: ~/.tensorfold/benchmarks/RUN_ID.json)" },
    .{ .name = "--publish", .value = false, .help = "review and upload the saved receipt" },
    .{ .name = "--yes", .value = false, .help = "explicitly approve public upload without the interactive prompt" },
    .{ .name = "--upload-url", .help = "benchmark API base URL (default: " ++ upload.default_api ++ ")" },
    .{ .name = "--token-stdin", .value = false, .help = "for login, read a contributor token from stdin" },
    .{ .name = "--help", .value = false, .help = "show this help message and exit" },
};

fn usage(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{ Usage, OutOfMemory } {
    problem.* = try std.fmt.allocPrint(a, fmt, args);
    return error.Usage;
}

fn invalid(problem: *[]const u8, message: []const u8) error{Invalid} {
    problem.* = message;
    return error.Invalid;
}

/// The flag a long option names, exactly or by a unique prefix as argparse allows.
fn find(a: Allocator, name: []const u8, problem: *[]const u8) error{ Usage, OutOfMemory }!?Flag {
    var hits: std.ArrayList([]const u8) = .empty;
    var hit: ?Flag = null;
    for (flags) |f| {
        if (std.mem.eql(u8, f.name, name)) return f;
        if (std.mem.startsWith(u8, f.name, name) and name.len > 2) {
            try hits.append(a, f.name);
            hit = f;
        }
    }
    if (hits.items.len > 1) return usage(a, problem, "ambiguous option: {s} could match {s}", .{ name, try std.mem.join(a, ", ", hits.items) });
    return hit;
}

/// argparse's pass, then the checks; ``Usage`` exits 2 with the usage line, ``Invalid`` exits 1.
pub fn parse(a: Allocator, argv: []const []const u8, problem: *[]const u8) error{ Usage, Invalid, OutOfMemory }!Args {
    var out: Args = .{};
    var positional: std.ArrayList([]const u8) = .empty;
    var temperatures: []const u8 = "1.0,0";
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (std.mem.eql(u8, token, "-h")) return .{ .action = .help };
        if (!std.mem.startsWith(u8, token, "--") or token.len == 2) {
            if (positional.items.len == 2) return usage(a, problem, "unrecognized arguments: {s}", .{token});
            try positional.append(a, token);
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const flag = try find(a, token[0 .. eq orelse token.len], problem) orelse return usage(a, problem, "unrecognized arguments: {s}", .{token});
        const name = flag.name;
        var value: []const u8 = "";
        if (flag.value) {
            if (eq) |e| value = token[e + 1 ..] else {
                i += 1;
                if (i >= argv.len or std.mem.startsWith(u8, argv[i], "--")) return usage(a, problem, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
        } else if (eq) |e| return usage(a, problem, "argument {s}: ignored explicit argument '{s}'", .{ name, token[e + 1 ..] });
        const is = struct {
            fn f(x: []const u8, y: []const u8) bool {
                return std.mem.eql(u8, x, y);
            }
        }.f;
        if (is(name, "--help")) return .{ .action = .help };
        if (is(name, "--download")) return usage(a, problem, "--download is not supported natively; run tensorfold pull MODEL first", .{});
        if (is(name, "--backend")) out.backend = std.meta.stringToEnum(Backend, value) orelse return usage(a, problem, "argument --backend: invalid choice: '{s}' (choose from auto, mlx, cuda, rocm)", .{value});
        if (is(name, "--server")) out.server = value;
        if (is(name, "--model-id")) out.model_id = value;
        if (is(name, "--tokens")) out.tokens = try int(a, problem, name, value);
        if (is(name, "--reps")) out.reps = try int(a, problem, name, value);
        if (is(name, "--temperatures")) temperatures = value;
        if (is(name, "--serial")) out.serial = true;
        if (is(name, "--context")) out.context = try int(a, problem, name, value);
        if (is(name, "--drafter")) out.drafter = value;
        if (is(name, "--timeout")) out.timeout = try float(a, problem, name, value);
        if (is(name, "--startup-timeout")) out.startup_timeout = try float(a, problem, name, value);
        if (is(name, "--output")) out.output = value;
        if (is(name, "--publish")) out.publish = true;
        if (is(name, "--yes")) out.yes = true;
        if (is(name, "--upload-url")) out.upload_url = value;
        if (is(name, "--token-stdin")) out.token_stdin = true;
    }
    if (positional.items.len > 0) out.target = positional.items[0];
    if (positional.items.len > 1) out.file = positional.items[1];
    try checks(a, &out, temperatures, problem);
    return out;
}

fn int(a: Allocator, problem: *[]const u8, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!i64 {
    const text = try fields.pyIntText(a, v) orelse return usage(a, problem, "argument {s}: invalid int value: '{s}'", .{ name, v });
    return std.fmt.parseInt(i64, text, 10) catch if (text[0] == '-') std.math.minInt(i64) else std.math.maxInt(i64);
}

fn float(a: Allocator, problem: *[]const u8, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    return fields.pyFloat(v) orelse usage(a, problem, "argument {s}: invalid float value: '{s}'", .{ name, v });
}

/// The checks after parsing; the first that fails names the problem.
fn checks(a: Allocator, out: *Args, temperatures: []const u8, problem: *[]const u8) error{ Invalid, OutOfMemory }!void {
    if (out.target.len == 0) return invalid(problem, "Choose a model ID, or benchmark publish/inspect/login/logout/suites");
    for ([_]Action{ .publish, .inspect, .login, .logout, .suites }) |act| if (std.mem.eql(u8, out.target, @tagName(act))) {
        out.action = act;
    };
    if (out.action == .publish or out.action == .inspect) {
        if (out.file == null) return invalid(problem, "This action needs a receipt JSON file");
        return;
    }
    if (out.action != .run) {
        if (out.file != null) return invalid(problem, "This action does not take a receipt file");
        return;
    }
    if (out.file != null or out.token_stdin) return invalid(problem, "Unexpected benchmark argument");
    var temps: std.ArrayList(f64) = .empty;
    var it = std.mem.splitScalar(u8, temperatures, ',');
    while (it.next()) |item| try temps.append(a, fields.pyFloat(item) orelse return invalid(problem, "Temperatures must be comma-separated finite numbers"));
    const t = temps.items;
    var distinct = true;
    for (t, 0..) |x, i| for (t[0..i]) |y| {
        if (x == y) distinct = false;
    };
    if (t.len < 1 or t.len > 4 or !distinct) return invalid(problem, "Choose one to four distinct temperatures");
    for (t) |x| if (!std.math.isFinite(x) or x < 0 or x > 2) return invalid(problem, "Temperatures must be finite numbers between 0 and 2");
    out.temperatures = t;
    if (out.tokens < 2 or out.tokens > 8192 or out.reps < 1 or out.reps > 20) return invalid(problem, "Tokens must be 2..8192 and repetitions 1..20");
    if (out.context) |c| if (c < 1 or c > 10_000_000) return invalid(problem, "Context must be a positive token count");
    for ([_]f64{ out.timeout, out.startup_timeout }) |s| if (!std.math.isFinite(s) or s <= 0) return invalid(problem, "Timeouts must be finite positive seconds");
}

/// ``-h``: the usage line and one line per flag.
pub fn help(w: *std.Io.Writer) !void {
    try w.print("{s}\n\nRun the community-v1 benchmark and optionally publish its receipt.\n\npositional arguments:\n  target  model ID or checkpoint directory, or publish/inspect/login/logout/suites\n  file    receipt JSON for publish or inspect\n\noptions:\n", .{usage_line});
    for (flags) |f| try w.print("  {s:<18} {s}\n", .{ f.name, f.help });
}

test "the checks run in a fixed order with fixed words; parse errors are usage errors" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const invalid_cases = [_]struct { []const []const u8, []const u8 }{
        .{ &.{}, "Choose a model ID, or benchmark publish/inspect/login/logout/suites" },
        .{ &.{"inspect"}, "This action needs a receipt JSON file" },
        .{ &.{ "logout", "run.json" }, "This action does not take a receipt file" },
        .{ &.{ "example/model", "run.json" }, "Unexpected benchmark argument" },
        .{ &.{ "example/model", "--token-stdin" }, "Unexpected benchmark argument" },
        .{ &.{ "example/model", "--temperatures", "1,x" }, "Temperatures must be comma-separated finite numbers" },
        .{ &.{ "example/model", "--temperatures", "1,1.0" }, "Choose one to four distinct temperatures" },
        .{ &.{ "example/model", "--temperatures", "0,0.5,1,1.5,2" }, "Choose one to four distinct temperatures" },
        .{ &.{ "example/model", "--temperatures", "inf" }, "Temperatures must be finite numbers between 0 and 2" },
        .{ &.{ "example/model", "--temperatures", "-1" }, "Temperatures must be finite numbers between 0 and 2" },
        .{ &.{ "example/model", "--tokens", "1" }, "Tokens must be 2..8192 and repetitions 1..20" },
        .{ &.{ "example/model", "--reps", "21" }, "Tokens must be 2..8192 and repetitions 1..20" },
        .{ &.{ "example/model", "--context", "0" }, "Context must be a positive token count" },
        .{ &.{ "example/model", "--timeout", "nan" }, "Timeouts must be finite positive seconds" },
        .{ &.{ "example/model", "--startup-timeout", "0" }, "Timeouts must be finite positive seconds" },
    };
    for (invalid_cases) |c| {
        try std.testing.expectError(error.Invalid, parse(a, c[0], &problem));
        try std.testing.expectEqualStrings(c[1], problem);
    }
    for ([_][]const []const u8{ &.{ "m", "--backend", "metal" }, &.{ "m", "--tokens", "x" }, &.{ "m", "--timeout" }, &.{ "m", "--bogus" }, &.{ "m", "--download" }, &.{ "a", "b", "c" }, &.{ "m", "--t", "1" }, &.{ "m", "--serial=1" } }) |argv|
        try std.testing.expectError(error.Usage, parse(a, argv, &problem));
    const ok = try parse(a, &.{ "example/model", "--temp=0", "--tokens", " 1_0 ", "--serial", "--server", "http://127.0.0.1:1" }, &problem);
    try std.testing.expect(ok.action == .run and ok.tokens == 10 and ok.serial and ok.temperatures.len == 1 and ok.temperatures[0] == 0);
    try std.testing.expectEqual(Action.publish, (try parse(a, &.{ "publish", "run.json", "--yes" }, &problem)).action);
    try std.testing.expectEqual(Action.help, (try parse(a, &.{ "suites", "-h" }, &problem)).action);
}

test "--backend takes rocm, and the usage line, help and invalid choice name it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expectEqual(Backend.rocm, (try parse(a, &.{ "m", "--backend", "rocm" }, &problem)).backend);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--backend", "hip" }, &problem));
    try std.testing.expectEqualStrings("argument --backend: invalid choice: 'hip' (choose from auto, mlx, cuda, rocm)", problem);
    var out: std.Io.Writer.Allocating = .init(a);
    try help(&out.writer);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "[--backend {auto,mlx,cuda,rocm}]") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "auto, mlx, cuda or rocm (default: auto") != null);
}

test "the --drafter help says how to name a drafter and that auto passes none" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try help(&out.writer);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "--drafter          a drafter checkpoint directory or cached repo ID") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "auto (default) and none pass no drafter, so the engine keeps its own default") != null);
}
