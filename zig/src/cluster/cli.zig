//! `tensorfold cluster init|check|status`, `tensorfold serve [MODEL] --cluster FILE` and `tensorfold node`.
const std = @import("std");
const config = @import("config.zig");
const launch = @import("launch.zig");
const budget = @import("budget.zig");
const status = @import("status.zig");
const topology = @import("topology.zig");
const bringup = @import("bringup.zig");
const cli_model = @import("cli_model.zig");

const Allocator = std.mem.Allocator;
const W = std.Io.Writer;

pub const Env = struct {
    io: std.Io,
    a: Allocator,
    out: *W,
    err: *W,
    runner: launch.Runner,
};

pub const usage =
    \\usage: tensorfold cluster init --node NAME=ADDRESS[@VIA]... [--cuda NAME]... [--ssh-user U] [--ssh-key FILE] [--name CLUSTER] [--probes DIR]
    \\       tensorfold cluster check --cluster FILE [--model ID] [--model-config FILE [--model-files FILE]] [--probe | --probes DIR]
    \\       tensorfold cluster status --cluster FILE [--model ID] [--probe | --probes DIR]
    \\       tensorfold serve [MODEL] --cluster FILE [--leader NAME] [--dry-run] [--model-config FILE [--model-files FILE]] [--probes DIR]
    \\       tensorfold node --name NAME --cluster FILE [--leader NAME] [--model ID] [--probes DIR]
    \\--probes DIR reads each node's probe output from DIR/NAME/ and records launches in DIR/launches.log (fake hosts).
    \\
;

/// The commands this module serves: cluster, node, and serve with --cluster.
pub fn wants(args: []const []const u8) bool {
    if (args.len == 0) return false;
    if (std.mem.eql(u8, args[0], "cluster") or std.mem.eql(u8, args[0], "node")) return true;
    if (!std.mem.eql(u8, args[0], "serve")) return false;
    for (args[1..]) |x| if (std.mem.eql(u8, x, "--cluster") or std.mem.startsWith(u8, x, "--cluster=")) return true;
    return false;
}

/// The exit code: 0 ok, 1 refused or invalid, 2 usage, 3 the MCDMA endpoints are not linked into this build.
pub fn run(env_in: Env, args: []const []const u8) !u8 {
    var env = env_in;
    var saved: launch.Saved = undefined;
    if (flag(args, "--probes")) |dir| {
        saved = .{ .io = env.io, .dir = dir };
        env.runner = saved.runner();
    }
    if (args.len < 1) return bad(env);
    if (std.mem.eql(u8, args[0], "cluster") and args.len >= 2) {
        if (std.mem.eql(u8, args[1], "init")) return initCmd(env, args[2..]);
        if (std.mem.eql(u8, args[1], "check")) return checkCmd(env, args[2..], false);
        if (std.mem.eql(u8, args[1], "status")) return checkCmd(env, args[2..], true);
    }
    if (std.mem.eql(u8, args[0], "serve")) return serveCmd(env, args[1..]);
    if (std.mem.eql(u8, args[0], "node")) return nodeCmd(env, args[1..]);
    return bad(env);
}

/// The entry both binaries share: argv after the program name, stdout and stderr, a real runner.
pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    var out_buf: [64 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &out_buf);
    var err = std.Io.File.stderr().writer(init.io, &err_buf);
    var sys: System = .{ .io = init.io };
    const args = try a.alloc([]const u8, argv.len);
    for (args, argv) |*x, y| x.* = y;
    const code = try run(.{ .io = init.io, .a = a, .out = &out.interface, .err = &err.interface, .runner = sys.runner() }, args);
    try out.interface.flush();
    try err.interface.flush();
    return code;
}

fn bad(env: Env) !u8 {
    try env.err.writeAll(usage);
    return 2;
}

fn flag(args: []const []const u8, name: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], name) and i + 1 < args.len) return args[i + 1];
        if (std.mem.startsWith(u8, args[i], name) and args[i].len > name.len and args[i][name.len] == '=') return args[i][name.len + 1 ..];
    }
    return null;
}

fn has(args: []const []const u8, name: []const u8) bool {
    for (args) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

fn source(env: Env, args: []const []const u8) cli_model.Source {
    return .{ .io = env.io, .a = env.a, .runner = env.runner, .live = has(args, "--probe") or flag(args, "--probes") != null, .shape_path = flag(args, "--model-config"), .files_path = flag(args, "--model-files") };
}

fn initCmd(env: Env, args: []const []const u8) !u8 {
    var nodes: std.ArrayList(config.Node) = .empty;
    var i: usize = 0;
    while (i + 1 < args.len) : (i += 1) {
        if (!std.mem.eql(u8, args[i], "--node")) continue;
        const spec = args[i + 1];
        const eq = std.mem.indexOfScalar(u8, spec, '=') orelse return bad(env);
        const at = std.mem.indexOfScalar(u8, spec[eq + 1 ..], '@');
        const address = if (at) |k| spec[eq + 1 ..][0..k] else spec[eq + 1 ..];
        try nodes.append(env.a, .{ .name = spec[0..eq], .address = address, .via = if (at) |k| spec[eq + 1 + k + 1 ..] else null, .backend = .metal });
    }
    i = 0;
    while (i + 1 < args.len) : (i += 1) if (std.mem.eql(u8, args[i], "--cuda")) {
        for (nodes.items) |*n| if (std.mem.eql(u8, n.name, args[i + 1])) {
            n.backend = .cuda;
        };
    };
    if (nodes.items.len == 0) return bad(env);
    const c: config.Config = .{ .name = flag(args, "--name") orelse "cluster", .ssh = .{ .user = flag(args, "--ssh-user"), .key = flag(args, "--ssh-key") }, .nodes = nodes.items, .models = &.{} };
    var probs: config.Problems = .{ .a = env.a };
    config.validate(c, &probs);
    if (!probs.ok()) return problems(env, probs);
    const found = try launch.discover(env.a, env.runner, c);
    try launch.writeJson(env.out, found.config);
    return 0;
}

fn problems(env: Env, probs: config.Problems) !u8 {
    for (probs.list.items) |p| try env.err.print("{s}\n", .{p});
    return 1;
}

const Loaded = struct { config: config.Config, json: []const u8 };

fn load(env: Env, args: []const []const u8) !?Loaded {
    const path = flag(args, "--cluster") orelse return null;
    const json = std.Io.Dir.cwd().readFileAlloc(env.io, path, env.a, .limited(16 << 20)) catch |err| {
        try env.err.print("{s}: {s}\n", .{ path, @errorName(err) });
        return null;
    };
    var probs: config.Problems = .{ .a = env.a };
    const c = config.parse(env.a, json, &probs) catch {
        _ = try problems(env, probs);
        return null;
    };
    return .{ .config = c, .json = json };
}

fn checkCmd(env: Env, args: []const []const u8, show_nodes: bool) !u8 {
    const l = (try load(env, args)) orelse return 1;
    const c = l.config;
    const src = source(env, args);
    if (show_nodes) {
        const names = try env.a.alloc([]const u8, c.nodes.len);
        for (c.nodes, names) |n, *x| x.* = n.name;
        const invs = cli_model.inventories(src, c, names) catch |err| return reportErr(env, err);
        try status.nodes(env.out, invs);
        if (src.live) try status.links(env.out, invs, try topology.edges(env.a, invs));
    }
    var code: u8 = 0;
    for (c.models) |m| {
        if (flag(args, "--model")) |want| if (!std.mem.eql(u8, want, m.id)) continue;
        try env.out.print("\nmodel {s}: {s}, {d} nodes\n", .{ m.id, @tagName(m.mode), m.nodes.len });
        if (m.mode == .disaggregated) {
            try env.out.print("prefill on {d} nodes, decode on {d}: KV handoff over MCDMA; outside the exactness contract (a prefill computed elsewhere is not this node's own)\n", .{ m.prefill.len, m.decode.len });
            continue;
        }
        const invs = cli_model.inventories(src, c, m.nodes) catch |err| return reportErr(env, err);
        const pl = cli_model.planModel(src, m, invs) catch |err| {
            try env.out.print("not planned: {s}\n", .{@errorName(err)});
            continue;
        };
        try status.placement(env.out, &pl.p, pl.needs, pl.fits);
        try cli_model.summary(env.out, &pl, m);
        try status.costs(env.out, &pl.p, pl.needs, &pl.shape, m.context);
        if (!budget.allFit(pl.fits)) code = 1;
    }
    if (show_nodes) try env.out.writeAll("\nload progress: no cluster is running from this host\n");
    return code;
}

fn reportErr(env: Env, err: anyerror) !u8 {
    try env.err.print("{s}: give memory_gb for every node, or pass --probe or --probes DIR\n", .{@errorName(err)});
    return 1;
}

fn serveCmd(env: Env, args: []const []const u8) !u8 {
    const l = (try load(env, args)) orelse return bad(env);
    const c = l.config;
    const id = if (args.len > 0 and !std.mem.startsWith(u8, args[0], "--")) args[0] else if (c.models.len == 1) c.models[0].id else return bad(env);
    const m = c.model(id) orelse {
        try env.err.print("no model \"{s}\" in the cluster file\n", .{id});
        return 1;
    };
    const leader = flag(args, "--leader") orelse m.nodes[0];
    const code = try checkCmd(env, args, false);
    if (code != 0) return code;
    const fake = flag(args, "--probes") != null;
    const started = try launch.launch(env.a, env.runner, c, l.json, m, leader, has(args, "--dry-run") or !fake);
    try env.out.print("\nleader {s}; the control step starts tensorfold node over key-only ssh on:", .{leader});
    for (started) |s| try env.out.print(" {s}{s}", .{ s.name, if (s.ok) "" else " (FAILED)" });
    try env.out.writeAll("\n");
    for (started) |s| if (!s.ok) return 1;
    if (!fake) {
        try env.err.writeAll("this build does not link the MCDMA endpoints: no node was started, and nothing runs over TCP\n");
        return if (has(args, "--dry-run")) 0 else 3;
    }
    const src = source(env, args);
    const invs = try cli_model.inventories(src, c, m.nodes);
    const pl = try cli_model.planModel(src, m, invs);
    const r = bringup.simulate(env.a, invs, try topology.edges(env.a, invs), pl.p.digest) catch |err| {
        try env.err.print("the fake cluster did not come up: {s}\n", .{@errorName(err)});
        return 1;
    };
    try env.out.print("fake cluster up: {d} nodes agree on plan {x:0>16} (leader {x}, epoch {d}) after {d} ms simulated; rounds need the real endpoints and kernels\n", .{ r.nodes, r.plan, r.leader, r.epoch, r.agreed_ms });
    return 0;
}

fn nodeCmd(env: Env, args: []const []const u8) !u8 {
    const l = (try load(env, args)) orelse return bad(env);
    const name = flag(args, "--name") orelse return bad(env);
    const i = l.config.find(name) orelse {
        try env.err.print("no node \"{s}\" in the cluster file\n", .{name});
        return 1;
    };
    const inv = try launch.inventory(env.a, env.runner, l.config, l.config.nodes[i], true);
    try status.nodes(env.out, &.{inv});
    try env.err.writeAll("node: joining needs the MCDMA endpoints, which this build does not link\n");
    return 3;
}

/// Runs commands with std.process (ssh for remote hosts, /bin/sh for this one).
pub const System = struct {
    io: std.Io,

    pub fn runner(s: *System) launch.Runner {
        return .{ .ptr = s, .run = runFn };
    }

    fn runFn(ptr: *anyopaque, a: Allocator, _: []const u8, argv: []const []const u8) anyerror!launch.Output {
        const s: *System = @ptrCast(@alignCast(ptr));
        const res = try std.process.run(a, s.io, .{ .argv = argv, .stdout_limit = .limited(64 << 20), .stderr_limit = .limited(1 << 20) });
        const code: u8 = switch (res.term) {
            .exited => |c| c,
            else => 255,
        };
        return .{ .code = code, .stdout = res.stdout };
    }
};

test {
    _ = @import("cli_test.zig");
}
