//! ``tensorfold serve`` flags: one table for parsing them and for ``capabilities --json``.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    value, // --flag VALUE or --flag=VALUE
    store_true, // --flag
    append, // repeatable --flag VALUE
};

pub const Flag = struct {
    name: []const u8,
    kind: Kind = .value,
    /// argparse's choices; empty: any value of the flag's type.
    choices: []const []const u8 = &.{},
    /// This binary honours the flag (and lists it in ``capabilities``); the rest are refused as unsupported.
    native: bool = false,
    /// The values it honours when only some of the choices.
    native_values: ?[]const []const u8 = null,
};

const backend_values: []const []const u8 = if (builtin.os.tag == .macos) &.{ "auto", "mlx" } else &.{ "auto", "cuda" };

/// Every Python serve flag (``cli_args.build_parser``), the ones this binary serves marked native.
pub const flags = [_]Flag{
    .{ .name = "--host", .native = true },
    .{ .name = "--port", .native = true },
    .{ .name = "--name", .native = true },
    .{ .name = "--alias", .kind = .append, .native = true },
    .{ .name = "--api-key", .kind = .append, .native = true },
    .{ .name = "--api-key-file", .native = true },
    .{ .name = "--metrics-open", .kind = .store_true, .native = true },
    .{ .name = "--dashboard", .kind = .store_true, .native = true },
    .{ .name = "--vision", .kind = .store_true },
    .{ .name = "--vision-urls", .kind = .store_true },
    .{ .name = "--vision-offload", .kind = .store_true },
    .{ .name = "--vision-max-images" },
    .{ .name = "--vision-image-tokens" },
    .{ .name = "--context", .native = true },
    .{ .name = "--speed-up", .native = true },
    .{ .name = "--max-tokens", .native = true },
    .{ .name = "--temperature", .native = true },
    .{ .name = "--top-p", .native = true },
    .{ .name = "--top-k", .native = true },
    .{ .name = "--min-p", .native = true },
    .{ .name = "--thinking", .kind = .store_true, .native = true },
    .{ .name = "--no-thinking", .kind = .store_true, .native = true },
    .{ .name = "--reasoning-effort", .choices = &.{ "low", "medium", "high", "xhigh" }, .native = true },
    .{ .name = "--thinking-budget", .native = true },
    .{ .name = "--loop-guard", .kind = .store_true, .native = true },
    .{ .name = "--no-drafts", .kind = .store_true, .native = true },
    .{ .name = "--drafter" },
    .{ .name = "--drafter-bits" },
    .{ .name = "--mtp-drafts" },
    .{ .name = "--mtp-confidence" },
    .{ .name = "--lane-kernels", .choices = &.{ "auto", "on", "off" } },
    .{ .name = "--prompt-cache-gib", .native = true },
    .{ .name = "--prompt-cache-over-cap", .kind = .store_true, .native = true },
    .{ .name = "--checkpoint-slots" },
    .{ .name = "--spill-gib" },
    .{ .name = "--snapshot-dir", .native = true, .native_values = &.{"none"} },
    .{ .name = "--max-snapshots", .native = true, .native_values = &.{"0"} },
    .{ .name = "--parallel", .native = true },
    .{ .name = "--decode-share" },
    .{ .name = "--prefill-pass" },
    .{ .name = "--pass-cache-gib" },
    .{ .name = "--mlx-cache-gib" },
    .{ .name = "--ssd-experts" },
    .{ .name = "--ple-on-ssd", .kind = .store_true },
    .{ .name = "--no-update-check", .kind = .store_true, .native = true },
    .{ .name = "--backend", .choices = &.{ "auto", "mlx", "cuda" }, .native = true, .native_values = backend_values },
    .{ .name = "--tp", .choices = &.{ "1", "2" } },
    .{ .name = "--rank", .choices = &.{ "0", "1" } },
    .{ .name = "--master" },
    .{ .name = "--master-port" },
    .{ .name = "--kv-dtype", .choices = &.{ "bf16", "int8", "int4" } },
    .{ .name = "--prefill-fp8", .kind = .store_true },
    .{ .name = "--no-prefill-fp8", .kind = .store_true },
    .{ .name = "--precision", .choices = &.{ "checkpoint", "full" } },
};

/// The variables this binary honours as the Python engine does.
pub const env = [_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_NO_LIVE", "TENSORFOLD_SEED_SALT", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_NO_UPDATE_CHECK", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE" };

pub const Args = struct {
    model: []const u8 = "",
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    name: []const u8 = "",
    alias: []const []const u8 = &.{},
    api_key: []const []const u8 = &.{},
    api_key_file: ?[]const u8 = null,
    metrics_open: bool = false,
    dashboard: bool = false,
    context: ?i64 = null,
    speed_up: ?[]const u8 = null, // speed-up mode: this Mac's settings for the two-Mac link (Flash Next)
    prompt_cache_gib: ?f64 = null, // kept prompt states' budget (0: none; null: what 70% of RAM leaves past the loaded server)
    prompt_cache_over_cap: bool = false, // a --prompt-cache-gib past that is kept, not refused
    max_tokens: i64 = 4096,
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
    thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    loop_guard: bool = false,
    no_drafts: bool = false,
    parallel: []const u8 = "auto",
    backend: []const u8 = "auto",
};

/// A usage error's message (argparse's ``error:`` line); the caller exits 2.
pub const Usage = struct { message: []const u8 = "" };

fn find(name: []const u8) ?Flag {
    for (flags) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

fn fail(u: *Usage, a: Allocator, comptime fmt: []const u8, args: anytype) error{ Usage, OutOfMemory } {
    u.message = try std.fmt.allocPrint(a, fmt, args);
    return error.Usage;
}

/// ``serve`` arguments as the switch passes them: the model and full flag names.
pub fn parse(a: Allocator, argv: []const []const u8, u: *Usage) error{ Usage, OutOfMemory }!Args {
    var out: Args = .{};
    var alias: std.ArrayList([]const u8) = .empty;
    var keys: std.ArrayList([]const u8) = .empty;
    var model: ?[]const u8 = null;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (!std.mem.startsWith(u8, token, "--")) {
            if (model != null) return fail(u, a, "unrecognized arguments: {s}", .{token});
            model = token;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = find(name) orelse return fail(u, a, "unrecognized arguments: {s}", .{token});
        var value: ?[]const u8 = null;
        if (flag.kind != .store_true) {
            if (eq) |e| value = token[e + 1 ..] else {
                i += 1;
                if (i >= argv.len) return fail(u, a, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
            if (flag.choices.len > 0) for (flag.choices) |c| {
                if (std.mem.eql(u8, c, value.?)) break;
            } else return fail(u, a, "argument {s}: invalid choice: '{s}'", .{ name, value.? });
        } else if (eq != null) return fail(u, a, "argument {s}: ignored explicit argument '{s}'", .{ name, token[eq.? + 1 ..] });
        if (!flag.native) return fail(u, a, "{s} is not served by the native engine yet; serve this command with --engine python", .{name});
        if (flag.native_values) |allowed| if (value) |v| for (allowed) |x| {
            if (std.mem.eql(u8, x, v)) break;
        } else return fail(u, a, "{s} {s} is not served by the native engine; serve this command with --engine python", .{ name, v });
        try apply(a, &out, name, value, u, &alias, &keys);
    }
    out.model = model orelse return fail(u, a, "the following arguments are required: model", .{});
    out.alias = alias.items;
    out.api_key = keys.items;
    return out;
}

fn int(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!i64 {
    return std.fmt.parseInt(i64, std.mem.trim(u8, v, " "), 10) catch fail(u, a, "argument {s}: invalid int value: '{s}'", .{ name, v });
}

fn float(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    return @import("fields.zig").pyFloat(v) orelse fail(u, a, "argument {s}: invalid float value: '{s}'", .{ name, v });
}

/// A finite GiB count from 0 (off) to below 2^34, whose bytes fit a u64.
fn gib(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    const g = try float(u, a, name, v);
    if (!std.math.isFinite(g) or g < 0 or g >= 1 << 34) return fail(u, a, "argument {s}: expected GiB from 0 (off) to below 2^34: '{s}'", .{ name, v });
    return g;
}

fn apply(a: Allocator, out: *Args, name: []const u8, value: ?[]const u8, u: *Usage, alias: *std.ArrayList([]const u8), keys: *std.ArrayList([]const u8)) error{ Usage, OutOfMemory }!void {
    const v = value orelse "";
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--host")) out.host = v else if (is(name, "--port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --port: invalid port: '{s}'", .{v});
        out.port = @intCast(p);
    } else if (is(name, "--name")) out.name = v else if (is(name, "--alias")) try alias.append(a, v) else if (is(name, "--api-key")) try keys.append(a, v) else if (is(name, "--api-key-file")) out.api_key_file = v else if (is(name, "--metrics-open")) out.metrics_open = true else if (is(name, "--dashboard")) out.dashboard = true else if (is(name, "--context")) out.context = try int(u, a, name, v) else if (is(name, "--speed-up")) out.speed_up = v else if (is(name, "--prompt-cache-gib")) out.prompt_cache_gib = try gib(u, a, name, v) else if (is(name, "--prompt-cache-over-cap")) out.prompt_cache_over_cap = true else if (is(name, "--max-tokens")) out.max_tokens = try int(u, a, name, v) else if (is(name, "--temperature")) out.temperature = try float(u, a, name, v) else if (is(name, "--top-p")) out.top_p = try float(u, a, name, v) else if (is(name, "--top-k")) out.top_k = try int(u, a, name, v) else if (is(name, "--min-p")) out.min_p = try float(u, a, name, v) else if (is(name, "--thinking")) out.thinking = true else if (is(name, "--no-thinking")) out.thinking = false else if (is(name, "--reasoning-effort")) out.reasoning_effort = v else if (is(name, "--thinking-budget")) out.thinking_budget = try int(u, a, name, v) else if (is(name, "--loop-guard")) out.loop_guard = true else if (is(name, "--no-drafts")) out.no_drafts = true else if (is(name, "--parallel")) out.parallel = v else if (is(name, "--backend")) out.backend = v;
}

/// ``--parallel``: "auto" is up to 8 requests at once; a number caps it.
pub fn parallel(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    if (std.ascii.eqlIgnoreCase(t, "auto")) return 8;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    return @intCast(@max(1, @min(n, 4096)));
}

/// What ``capabilities --json`` reports about the engine side: its version, chip, backends and families.
pub const Engines = struct {
    version: []const u8,
    chip: ?[]const u8 = null,
    backends: []const []const u8 = &.{},
    /// model_type to the weight formats it reads, as gate entries name them.
    families: []const api.Family = &.{},
};

/// The native capabilities document: the table's native flags and the honoured variables.
pub fn capabilities(w: *std.Io.Writer, e: Engines) !void {
    try w.print("{{\"schema\": 1, \"engine\": \"zig\", \"version\": \"{s}\", \"chip\": ", .{e.version});
    if (e.chip) |c| try w.print("\"{s}\"", .{c}) else try w.writeAll("null");
    try w.writeAll(", \"backends\": [");
    for (e.backends, 0..) |b, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", b });
    try w.writeAll("], \"families\": {");
    for (e.families, 0..) |f, i| {
        try w.print("{s}\"{s}\": [", .{ if (i > 0) ", " else "", f.model_type });
        for (f.formats, 0..) |fmt, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", fmt });
        try w.writeAll("]");
    }
    try w.writeAll("}, \"serve\": {\"flags\": {");
    var first = true;
    for (flags) |f| {
        if (!f.native) continue;
        try w.print("{s}\"{s}\": {{", .{ if (first) "" else ", ", f.name });
        first = false;
        const values = f.native_values orelse f.choices;
        if (values.len > 0) {
            try w.writeAll("\"values\": [");
            for (values, 0..) |v, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", v });
            try w.writeAll("]");
        }
        try w.writeAll("}");
    }
    try w.writeAll("}}, \"env\": [");
    for (env, 0..) |name, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", name });
    try w.writeAll("]}\n");
}

test "parse and capabilities share the table" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const args = try parse(a, &.{ "/models/x", "--port", "9000", "--api-key=k1", "--api-key", "k2", "--no-thinking", "--reasoning-effort", "low" }, &u);
    try std.testing.expectEqual(@as(u16, 9000), args.port);
    try std.testing.expectEqual(@as(usize, 2), args.api_key.len);
    try std.testing.expect(!args.thinking);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--drafter", "x" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--reasoning-effort", "max" }, &u));
    try std.testing.expectEqual(@as(?f64, 12.5), (try parse(a, &.{ "m", "--prompt-cache-gib", "12.5" }, &u)).prompt_cache_gib);
    try std.testing.expectEqual(@as(?f64, 0), (try parse(a, &.{ "m", "--prompt-cache-gib", "0" }, &u)).prompt_cache_gib);
    try std.testing.expect(!(try parse(a, &.{ "m", "--prompt-cache-gib", "16" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--prompt-cache-gib", "16", "--prompt-cache-over-cap" }, &u)).prompt_cache_over_cap);
    for ([_][]const u8{ "1e300", "inf", "nan", "-1", "17179869184" }) |bad| try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--prompt-cache-gib", bad }, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0.6.5" });
    const doc = out.written();
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--no-thinking\": {}") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--drafter\"") == null);
}
