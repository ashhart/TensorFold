//! ``tensorfold benchmark``: the actions, a run from start to receipt, the result table and exit codes.
const std = @import("std");
const builtin = @import("builtin");
const json = @import("json");
const cli = @import("cli.zig");
const hub = @import("hub.zig");
const args_mod = @import("benchmark_args.zig");
const suite = @import("benchmark_suite.zig");
const receipt = @import("benchmark_receipt.zig");
const stream = @import("benchmark_stream.zig");
const server = @import("benchmark_server.zig");
const model = @import("benchmark_model.zig");
const host = @import("benchmark_host.zig");
const manifest = @import("benchmark_manifest.zig");
const upload = @import("benchmark_upload.zig");
const check = @import("benchmark_check.zig");
const stop = @import("benchmark_stop.zig");
const Allocator = std.mem.Allocator;
const Args = args_mod.Args;

/// Publication effects can be replaced by inert services when exercising command entry points.
pub const Publication = struct {
    credential: *const fn (Allocator, std.Io, upload.Options, *[]const u8) anyerror!?[]const u8 = upload.credential,
    verify_public: *const fn (Allocator, std.Io, upload.Options, json.Value, *[]const u8) anyerror!bool = upload.verifyPublic,
    upload: *const fn (Allocator, std.Io, upload.Options, json.Value, *[]const u8) anyerror!?upload.Ack = upload.upload,
};

pub const Input = struct { tty: bool, reader: *std.Io.Reader, fd: ?std.posix.fd_t };

/// Everything the command touches, so tests can run it with fakes, captured writers and a temporary home.
pub const Context = struct {
    a: Allocator,
    io: std.Io,
    environ: *const std.process.Environ.Map,
    home: []const u8,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    input: Input,
    engines: cli.Engines,
    self_exe: ?[]const u8 = null,
    hf_base: []const u8 = upload.hf_models,
    signals: bool = false,
    publication: Publication = .{},
};

pub fn main(init: std.process.Init, argv: []const []const u8, engines: cli.Engines) !u8 {
    const io = init.io;
    var out_buf: [16 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var in_buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &out_buf);
    var err = std.Io.File.stderr().writerStreaming(io, &err_buf);
    var in = std.Io.File.stdin().readerStreaming(io, &in_buf);
    defer out.interface.flush() catch {};
    defer err.interface.flush() catch {};
    const ctx: Context = .{
        .a = init.arena.allocator(),
        .io = io,
        .environ = init.environ_map,
        .home = init.environ_map.get("HOME") orelse ".",
        .out = &out.interface,
        .err = &err.interface,
        .input = .{ .tty = std.Io.File.stdin().isTty(io) catch false, .reader = &in.interface, .fd = std.posix.STDIN_FILENO },
        .engines = engines,
        .signals = true,
    };
    return command(&ctx, argv);
}

fn say(ctx: *const Context, comptime fmt: []const u8, args: anytype) !void {
    try ctx.out.print(fmt ++ "\n", args);
    try ctx.out.flush();
}

fn fail(ctx: *const Context, message: []const u8) !u8 {
    try ctx.err.print("tensorfold benchmark: {s}\n", .{message});
    try ctx.err.flush();
    return 1;
}

/// A run or prompt ended by a stop signal: Ctrl-C says so; every signal maps to its shell code.
fn cancelled(ctx: *const Context) !u8 {
    const sig = stop.requested() orelse .INT;
    if (sig == .INT) {
        try ctx.err.writeAll("Benchmark cancelled.\n");
        try ctx.err.flush();
    }
    return stop.exitCode(sig);
}

fn nowMs(ctx: *const Context) i64 {
    return std.Io.Clock.real.now(ctx.io).toMilliseconds();
}

fn expand(ctx: *const Context, path: []const u8) ![]const u8 {
    return if (std.mem.startsWith(u8, path, "~/")) std.fs.path.join(ctx.a, &.{ ctx.home, path[2..] }) else path;
}

pub fn command(ctx: *const Context, argv: []const []const u8) !u8 {
    var problem: []const u8 = "";
    const args = args_mod.parse(ctx.a, argv, &problem) catch |e| switch (e) {
        error.Usage => {
            try ctx.err.print("{s}\ntensorfold benchmark: error: {s}\n", .{ args_mod.usage_line, problem });
            try ctx.err.flush();
            return 2;
        },
        error.Invalid => return fail(ctx, problem),
        error.OutOfMemory => |x| return x,
    };
    if (ctx.signals and (args.action == .run or args.action == .publish or args.action == .login)) stop.install();
    defer if (ctx.signals) stop.restore();
    switch (args.action) {
        .help => {
            try args_mod.help(ctx.out);
            try ctx.out.flush();
            return 0;
        },
        .suites => {
            try ctx.out.writeAll(suite.suites_text);
            try ctx.out.flush();
            return 0;
        },
        .logout => {
            upload.logout(ctx.a, ctx.io, ctx.home) catch return fail(ctx, "Could not remove the local upload credentials");
            try say(ctx, "Local benchmark upload credentials removed.", .{});
            return 0;
        },
        .login => return login(ctx, args),
        .inspect, .publish => {
            const v = try receipt.load(ctx.a, ctx.io, try expand(ctx, args.file.?), nowMs(ctx), &problem) orelse return fail(ctx, problem);
            try show(ctx, v);
            return if (args.action == .publish) publish(ctx, args, v) else 0;
        },
        .run => return run(ctx, args),
    }
}

/// One line from the input, read as its own task so Ctrl-C ends the wait; null when stopped.
fn readLine(ctx: *const Context) error{Unavailable}!?[]const u8 {
    const Task = struct {
        r: *std.Io.Reader,
        line: []const u8 = "",
        done: std.atomic.Value(bool) = .init(false),
        fn go(t: *@This()) void {
            t.line = t.r.takeDelimiterInclusive('\n') catch |e| switch (e) {
                error.EndOfStream => blk: {
                    const rest = t.r.buffered();
                    t.r.toss(rest.len);
                    break :blk rest;
                },
                else => "",
            };
        }
    };
    var task: Task = .{ .r = ctx.input.reader };
    return switch (stream.bounded(ctx.io, null, Task.go, &task)) {
        .finished => std.mem.trim(u8, task.line, " \t\r\n"),
        .stopped, .deadline => null,
        .unavailable => error.Unavailable,
    };
}

/// A hidden-echo read on a terminal, restored before returning and by a second Ctrl-C.
fn readSecret(ctx: *const Context) error{Unavailable}!?[]const u8 {
    const fd = ctx.input.fd orelse return readLine(ctx);
    const saved = std.posix.tcgetattr(fd) catch return readLine(ctx);
    var quiet = saved;
    quiet.lflag.ECHO = false;
    stop.guardTerminal(fd, saved);
    std.posix.tcsetattr(fd, .FLUSH, quiet) catch {};
    defer {
        std.posix.tcsetattr(fd, .FLUSH, saved) catch {};
        stop.releaseTerminal();
    }
    return readLine(ctx);
}

fn login(ctx: *const Context, args: Args) !u8 {
    var problem: []const u8 = "";
    const token = if (args.token_stdin) readLine(ctx) else if (ctx.input.tty) blk: {
        try say(ctx, "Use a contributor upload token issued by the TensorFold benchmark administrator.", .{});
        try ctx.err.writeAll("Upload token: ");
        try ctx.err.flush();
        const t = readSecret(ctx);
        try ctx.err.writeAll("\n");
        break :blk t;
    } else return fail(ctx, "Use login --token-stdin to provide a contributor token without exposing it in arguments");
    const t = (token catch return fail(ctx, "Could not read the upload token")) orelse return cancelled(ctx);
    if (!upload.tokenOk(t)) return fail(ctx, "Invalid benchmark upload token");
    const api = upload.apiUrl(args.upload_url, &problem) orelse return fail(ctx, problem);
    const name = try upload.login(ctx.a, ctx.io, options(ctx, api), t, &problem) orelse return if (problem.len == 0) cancelled(ctx) else fail(ctx, problem);
    try say(ctx, "Benchmark uploads enabled for {s}.", .{name});
    return 0;
}

fn options(ctx: *const Context, api: []const u8) upload.Options {
    return .{ .api = api, .hf_base = ctx.hf_base, .environ = ctx.environ, .home = ctx.home, .now_ms = nowMs(ctx) };
}

/// Publishing: the board's rules and the upload token, the public check, preview, consent, then the upload.
fn publish(ctx: *const Context, args: Args, v: json.Value) !u8 {
    var problem: []const u8 = "";
    const api = upload.apiUrl(args.upload_url, &problem) orelse return fail(ctx, problem);
    const o = options(ctx, api);
    if (!check.receipt(v, .publish, o.now_ms, &problem)) return fail(ctx, problem);
    if (!upload.publicationAllowed(v, &problem)) return fail(ctx, problem);
    _ = try ctx.publication.credential(ctx.a, ctx.io, o, &problem) orelse return fail(ctx, problem);
    if (!try ctx.publication.verify_public(ctx.a, ctx.io, o, v, &problem)) return if (problem.len == 0) cancelled(ctx) else fail(ctx, problem);
    try ctx.out.writeAll("Public upload preview:\n");
    try receipt.preview(ctx.out, v);
    try say(ctx, "New submissions are community-reported and pending moderation.", .{});
    if (!args.yes) {
        if (!ctx.input.tty) return fail(ctx, "Publishing needs explicit consent; inspect the receipt and pass --yes");
        try ctx.out.writeAll("Upload these public fields to the benchmark receiver? [y/N] ");
        try ctx.out.flush();
        const answer = (readLine(ctx) catch return fail(ctx, "Could not read the answer; the receipt remains local")) orelse return cancelled(ctx);
        if (!std.ascii.eqlIgnoreCase(answer, "y") and !std.ascii.eqlIgnoreCase(answer, "yes")) {
            try say(ctx, "Upload cancelled; the receipt remains local.", .{});
            return 0;
        }
    }
    const ack = try ctx.publication.upload(ctx.a, ctx.io, o, v, &problem) orelse return if (problem.len == 0) cancelled(ctx) else fail(ctx, problem);
    try say(ctx, "Submission {s}: {s}", .{ ack.id, ack.status });
    if (ack.results_url) |u| try say(ctx, "{s}", .{u});
    return 0;
}

/// The run line, one row per cell with medians, and the notes.
fn show(ctx: *const Context, v: json.Value) !void {
    const repo = json.strOr(v.get("model").?.get("repo_id"));
    try say(ctx, "Run {s} | {s} | {s}", .{ json.strOr(v.get("run_id")), json.strOr(v.get("runtime").?.get("backend")), if (repo.len > 0) repo else "local model" });
    try say(ctx, "Protocol {s}; compare only within this protocol and manifest digest.", .{json.strOr(v.get("suite").?.get("id"))});
    try say(ctx, "Workload       Temp    Stream delivery tok/s    First text, s    Status", .{});
    const settings = v.get("settings").?;
    for (try suite.receiptCells(ctx.a, v)) |c| {
        var t: [32]u8 = undefined;
        const rate = if (c.delivery) |d| try std.fmt.allocPrint(ctx.a, "{d:.2}", .{d}) else "not measured";
        const ttft = if (c.ttft) |d| try std.fmt.allocPrint(ctx.a, "{d:.3}", .{d}) else "not measured";
        try say(ctx, "{s:<14} {s:<6} {s:<25} {s:<16} {s}", .{ c.fixture_id, suite.fmtG(&t, c.temperature), rate, ttft, if (c.protocol_complete) "complete" else "custom/incomplete" });
    }
    try say(ctx, "Stream delivery includes transport buffering; it is not an engine token clock.", .{});
    if (settings.get("managed").? != .bool or !settings.get("managed").?.bool) {
        const source = json.strOr(v.get("hardware").?.get("source"));
        const version = json.strOr(v.get("runtime").?.get("tensorfold_version"));
        const pinned = settings.get("rank_count").? != .null and v.get("model").?.get("revision").? != .null;
        if (std.mem.eql(u8, source, "unavailable") or std.mem.eql(u8, version, "unknown"))
            try say(ctx, "Attached-server hardware and checkpoint revisions are unknown; this result is unranked.", .{})
        else if (!pinned)
            try say(ctx, "The attached server's manifest does not attest a rank count and checkpoint revision; this result is unranked.", .{});
    }
    if (std.mem.eql(u8, json.strOr(v.get("runtime").?.get("backend")), "rocm"))
        try say(ctx, "This rocm receipt is saved locally; it cannot be published until the board accepts rocm.", .{});
}

/// What an engine that cannot sample answers a request with a temperature (server.zig's refusal).
pub const greedy_only = "decodes greedily only";

const Printer = struct {
    ctx: *const Context,
    reps: i64,
    heard: [3][]const u8 = undefined,
    heard_n: usize = 0,
    greedy: bool = false,

    /// A refused attempt's reason, once per distinct reason and at most three; a greedy-only engine is named once.
    fn said(p: *anyopaque, s: *const suite.Sample, text: []const u8) void {
        const self: *Printer = @ptrCast(@alignCast(p));
        for (self.heard[0..self.heard_n]) |h| if (std.mem.eql(u8, h, text)) return;
        if (self.heard_n == self.heard.len) return;
        self.heard[self.heard_n] = text;
        self.heard_n += 1;
        say(self.ctx, "Server answered {s}", .{text}) catch {};
        if (self.greedy or s.temperature <= 0 or std.mem.indexOf(u8, text, greedy_only) == null) return;
        self.greedy = true;
        say(self.ctx, "This engine decodes greedily only, so the cells above temperature 0 fail and the run cannot be ranked; for a local run, use --temperatures 0.", .{}) catch {};
    }

    fn call(p: *anyopaque, s: *const suite.Sample) void {
        const self: *Printer = @ptrCast(@alignCast(p));
        var t: [32]u8 = undefined;
        var phase: [32]u8 = undefined;
        var rate: [64]u8 = undefined;
        const ph = if (s.repeat < 0) "warm-up" else std.fmt.bufPrint(&phase, "repeat {d}/{d}", .{ s.repeat + 1, self.reps }) catch "";
        const measured = if (s.delivery_tps) |r| std.fmt.bufPrint(&rate, " {d:.2} delivery tok/s", .{r}) catch "" else "";
        say(self.ctx, "{s} T={s} {s}: {s}{s}", .{ s.fixture_id, suite.fmtG(&t, s.temperature), ph, @tagName(s.status), measured }) catch {};
    }
};

const hip_drafter = "the HIP engine runs no DFlash2 drafter yet";
/// Why a ROCm run refuses a named ``--drafter`` before it starts.
pub const rocm_drafter = "--drafter: " ++ hip_drafter ++ ", so a rocm run takes none; --drafter none runs without it";

/// The drafter ``--drafter auto`` names: the family's own; none under --serial or on rocm (HIP runs no DFlash2 yet).
pub fn autoDrafter(config: json.Value, backend: []const u8, serial: bool) ?[]const u8 {
    if (serial or std.mem.eql(u8, backend, "rocm")) return null;
    return model.knownDrafter(config);
}

/// A run from checks to receipt: attached or with its own serve child, then the table and the exit code.
fn run(ctx: *const Context, args: Args) !u8 {
    const a = ctx.a;
    var problem: []const u8 = "";
    const backend = if (args.backend == .auto) server.buildBackend() else @tagName(args.backend);
    const rocm = std.mem.eql(u8, backend, "rocm");
    if (!std.mem.eql(u8, backend, "mlx") and !std.mem.eql(u8, backend, "cuda") and !rocm)
        return fail(ctx, try std.fmt.allocPrint(a, "community-v2 runs cover mlx, cuda and rocm; this build serves {s}", .{backend}));
    if (rocm and args.publish) return fail(ctx, check.rocm_local);
    if (args.publish and !upload.v2_submissions_enabled) return fail(ctx, upload.v2_disabled);
    if (args.publish) {
        const api = upload.apiUrl(args.upload_url, &problem) orelse return fail(ctx, problem);
        _ = try ctx.publication.credential(a, ctx.io, options(ctx, api), &problem) orelse return fail(ctx, problem);
    }
    const run_id = receipt.runId(ctx.io);
    const bench_dir = try std.fs.path.join(a, &.{ ctx.home, ".tensorfold", "benchmarks" });
    const settings: receipt.Settings = .{ .tokens = @intCast(args.tokens), .repetitions = @intCast(args.reps), .temperatures = args.temperatures, .serial = args.serial, .context_tokens = args.context, .managed = args.server == null };
    var facts: struct { model: receipt.Model, hardware: receipt.Hardware, runtime: receipt.Runtime } = undefined;
    var rank_count: ?u8 = if (args.server == null) 1 else null;
    var target: stream.Target = undefined;
    var chat_only = args.chat_only;
    var plan: ?server.Plan = null;
    var note: ?[]const u8 = null;
    var wait: ?[]const u8 = null;
    var startup_s: f64 = args.startup_timeout orelse 600;
    if (args.server) |url| {
        if (args.backend == .auto) return fail(ctx, "--server needs an explicit --backend mlx, cuda or rocm; client hardware does not identify a remote server");
        if (stream.splitUrl(url) == null) return fail(ctx, "Server URL must be HTTP(S) without credentials, query or fragment");
        var b = std.mem.trimEnd(u8, url, "/");
        if (std.mem.endsWith(u8, b, "/v1")) b = b[0 .. b.len - 3];
        target = .{ .base = b, .served = args.model_id orelse args.target };
        facts = .{
            .model = model.attached(args.target, hub.isDir(ctx.io, try expand(ctx, args.target))),
            .hardware = host.unavailable,
            .runtime = .{ .tensorfold_version = "unknown", .backend = backend, .python_version = "unknown", .platform = if (std.mem.eql(u8, backend, "mlx")) "macos" else "linux" },
        };
    } else {
        if (args.model_id != null) return fail(ctx, "--model-id applies only to an attached server");
        if (args.startup_timeout) |s| if (s > 3600) return fail(ctx, "Startup timeout must be greater than 0 and at most 3600 seconds");
        const dir = try hub.resolve(a, ctx.io, ctx.environ, args.target, &problem) orelse return fail(ctx, problem);
        const config = (model.readConfig(a, ctx.io, dir, &problem) catch |e| {
            if (e == error.OutOfMemory) return e;
            return fail(ctx, problem);
        }) orelse return fail(ctx, "Checkpoint is incomplete; run tensorfold pull MODEL");
        if (config.value != .object) return fail(ctx, "Checkpoint is incomplete; run tensorfold pull MODEL");
        const kind = model.modelType(config.value) orelse "unknown";
        for (ctx.engines.families) |f| {
            if (std.mem.eql(u8, f.model_type, kind)) break;
        } else return fail(ctx, try std.fmt.allocPrint(a, "This build does not serve model_type {s}; tensorfold-native capabilities --json lists what it serves", .{kind}));
        const auto = std.mem.eql(u8, args.drafter, "auto");
        const given = !auto and !std.mem.eql(u8, args.drafter, "none") and args.drafter.len > 0;
        if (rocm and given) return fail(ctx, rocm_drafter);
        // auto passes the family's own drafter only when serve's cache lookup finds it; nothing is downloaded.
        const known = if (auto) autoDrafter(config.value, backend, args.serial) else null;
        const drafter = if (args.serial) null else if (given) args.drafter else known;
        var uncached: []const u8 = "";
        var drafter_dir: ?[]const u8 = null;
        if (drafter) |d| {
            drafter_dir = try hub.resolve(a, ctx.io, ctx.environ, d, &uncached);
            if (drafter_dir == null and given) return fail(ctx, uncached);
        }
        const drafter_label = if (drafter_dir != null and !hub.isDir(ctx.io, try expand(ctx, drafter.?)))
            drafter
        else
            null;
        chat_only = model.chatOnly(config, ctx.engines.families);
        const probe = host.detect(a, ctx.io, ctx.environ, backend);
        facts = .{
            .model = try model.describeConfig(a, ctx.io, args.target, dir, hub.isDir(ctx.io, try expand(ctx, args.target)), ctx.engines.families, drafter_label, config),
            .hardware = probe.hardware,
            .runtime = .{ .tensorfold_version = ctx.engines.version, .backend = backend, .python_version = "none", .platform = if (builtin.os.tag == .macos) "macos" else "linux", .driver = probe.driver },
        };
        if (args.startup_timeout == null) {
            const bytes = model.weightBytes(ctx.io, dir);
            startup_s = server.startupTimeout(bytes);
            if (startup_s > 600) wait = try std.fmt.allocPrint(a, "The server gets up to {d} s to load {d:.0} GB of weights; --startup-timeout sets another limit.", .{ startup_s, @as(f64, @floatFromInt(bytes)) / 1e9 });
        }
        plan = .{ .exe = ctx.self_exe orelse try std.process.executablePathAlloc(ctx.io, a), .dir = dir, .backend = backend, .tokens = @intCast(args.tokens), .context = args.context, .drafter_dir = drafter_dir, .serial = args.serial };
        var u: cli.Usage = .{};
        _ = cli.parse(a, (try server.argv(a, plan.?))[2..], &u) catch |e| switch (e) {
            error.Usage => return fail(ctx, try std.fmt.allocPrint(a, "the managed server refuses its flags: {s}", .{u.message})),
            else => |x| return x,
        };
        if (args.serial and given) note = try std.fmt.allocPrint(a, "--serial passes no drafter, so --drafter {s} is not used.", .{args.drafter});
        if (rocm and auto and !args.serial and model.knownDrafter(config.value) != null) note = "This run passes no drafter: " ++ hip_drafter ++ ".";
        if (known) |k| note = if (drafter_dir != null)
            try std.fmt.allocPrint(a, "This run passes the cached drafter {s}; --drafter none runs without it.", .{k})
        else
            try std.fmt.allocPrint(a, "This run passes no drafter: {s}.", .{uncached});
    }
    const dest = if (args.output) |o| try expand(ctx, o) else blk: {
        _ = std.Io.Dir.cwd().createDirPathStatus(ctx.io, bench_dir, .fromMode(0o700)) catch {};
        break :blk try std.fmt.allocPrint(a, "{s}/{s}.json", .{ bench_dir, &run_id });
    };
    if (!try receipt.writable(a, ctx.io, dest, &problem)) return fail(ctx, try std.fmt.allocPrint(a, "{s}; choose another --output", .{problem}));
    try say(ctx, "{s}: code and chat, {d} temperatures, {d} repeats after warm-up.", .{ suite.id, args.temperatures.len, args.reps });
    try say(ctx, "This generates model replies; use Ctrl-C to cancel. Nothing is uploaded unless you publish.", .{});
    if (note) |n| try say(ctx, "{s}", .{n});
    if (wait) |w| try say(ctx, "{s}", .{w});
    // the attached server's own facts, read once nothing more can refuse the run
    if (args.server != null) if (try manifest.fetch(a, ctx.io, target.base, backend)) |got| {
        facts.hardware = got.hardware;
        facts.runtime = got.runtime;
        facts.model.revision = got.revision;
        facts.model.config_sha256 = got.config_sha256;
        facts.model.tokenizer_sha256 = got.tokenizer_sha256;
        facts.model.family = got.family;
        facts.model.quantization = got.quantization;
        rank_count = got.rank_count;
    };
    var managed: ?server.Managed = null;
    if (plan) |p| {
        var tail: []const []const u8 = &.{};
        managed = try server.Managed.start(a, ctx.io, ctx.environ, p, .{ .startup_timeout_s = startup_s, .bench_dir = bench_dir, .tail = &tail }, &problem) orelse {
            if (stop.requested() != null) return cancelled(ctx);
            const code = try fail(ctx, problem);
            if (tail.len == 1) try ctx.err.print("Last server log line: {s}\n", .{tail[0]});
            if (tail.len > 1) try ctx.err.writeAll("Last server log lines:\n");
            if (tail.len > 1) for (tail) |line| try ctx.err.print("  {s}\n", .{line});
            try ctx.err.flush();
            return code;
        };
        target = .{ .base = managed.?.base(), .served = server.served_name };
    }
    var printer: Printer = .{ .ctx = ctx, .reps = args.reps };
    const samples = stream.run(a, ctx.io, target, .{ .tokens = @intCast(args.tokens), .reps = @intCast(args.reps), .temperatures = args.temperatures, .serial = args.serial, .timeout_s = args.timeout, .chat_only = chat_only }, .{ .ctx = &printer, .call = Printer.call, .said = Printer.said }) catch |e| {
        if (managed) |*m| _ = m.stop();
        return e;
    };
    const cleanup = if (managed) |*m| m.stop() else null;
    if (stop.requested()) |sig| if (sig != .INT) return stop.exitCode(sig);
    var stamp: [32]u8 = undefined;
    const created = receipt.createdAt(std.Io.Clock.real.now(ctx.io).toNanoseconds(), &stamp);
    var recorded = settings;
    recorded.rank_count = rank_count;
    const v = try receipt.value(a, &run_id, created, facts.model, facts.runtime, facts.hardware, recorded, samples);
    try show(ctx, v);
    if (!try receipt.save(a, ctx.io, v, dest, nowMs(ctx), &problem)) return fail(ctx, problem);
    try say(ctx, "Receipt saved: {s}", .{dest});
    if (cleanup) |c| return fail(ctx, c);
    const was_cancelled = stop.requested() != null or for (samples) |s| {
        if (s.error_code == .cancelled) break true;
    } else false;
    if (was_cancelled) return stop.exitCode(.INT);
    if (args.publish) {
        const code = try publish(ctx, args, v);
        if (code != 0) return code;
    }
    for (samples) |s| if (s.status != .ok) {
        try say(ctx, "Incomplete attempts remain in the saved receipt (exit 2). " ++
            "A cell has aggregates only when every repetition succeeds; early EOS is excluded from full-length comparisons.", .{});
        return 2;
    };
    return 0;
}
