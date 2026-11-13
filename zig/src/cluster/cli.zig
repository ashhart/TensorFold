//! `tensorfold cluster init|check|status`, `tensorfold serve [MODEL] --cluster FILE` and `tensorfold node`.
const std = @import("std");
const config = @import("config.zig");
const launch = @import("launch.zig");
const budget = @import("budget.zig");
const status = @import("status.zig");
const topology = @import("topology.zig");
const bringup = @import("bringup.zig");
const cli_model = @import("cli_model.zig");
const mcdma_link = @import("mcdma_link.zig");
const node = @import("node.zig");

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
    \\usage: tensorfold config validate FILE
    \\       tensorfold cluster init --node NAME=ADDRESS[@VIA]... [--cuda NAME]... [--ssh-user U] [--ssh-key FILE] [--name CLUSTER] [--probes DIR]
    \\       tensorfold cluster check --cluster FILE [--model ID] [--model-config FILE [--model-files FILE]] [--probe | --probes DIR]
    \\       tensorfold cluster status --cluster FILE [--model ID] [--probe | --probes DIR]
    \\       tensorfold serve [MODEL] --cluster FILE [--leader NAME] [--dry-run] [--hold SECONDS] [--model-config FILE [--model-files FILE]] [--probe | --probes DIR]
    \\       tensorfold node --name NAME --cluster FILE [--leader NAME] [--model ID] [--probes DIR]
    \\--probes DIR reads each node's probe output from DIR/NAME/ and records launches in DIR/launches.log (fake hosts).
    \\On real hosts a disaggregated model's nodes open their MCDMA links and agree on its plan; serve holds the cluster
    \\--hold SECONDS (default 0) and says goodbye, node holds until a node says goodbye. A converged model is refused.
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

/// Exit: 0 ok, 1 refused, not planned, invalid or failed, 2 usage, 3 no MCDMA library (undeclared, absent, not ABI 1).
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
        if (std.mem.eql(u8, args[1], "check")) return checkCmd(env, args[2..], false, flag(args, "--model"));
        if (std.mem.eql(u8, args[1], "status")) return checkCmd(env, args[2..], true, flag(args, "--model"));
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
    var out = std.Io.File.stdout().writerStreaming(init.io, &out_buf); // streaming: positional writes from both would overwrite each other in one redirected file
    var err = std.Io.File.stderr().writerStreaming(init.io, &err_buf);
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

/// The flags a command takes: with a value (`--x V` or `--x=V`), and switches.
const Flags = struct { values: []const []const u8, switches: []const []const u8 = &.{}, positionals: usize = 0 };

const serve_flags: Flags = .{ .values = &.{ "--cluster", "--leader", "--hold", "--model-config", "--model-files", "--probes" }, .switches = &.{ "--dry-run", "--probe" }, .positionals = 1 };
const node_flags: Flags = .{ .values = &.{ "--name", "--cluster", "--leader", "--model", "--probes" } };

/// Refuse by name a flag `cmd` does not take, a missing value or a stray argument, before anything is read or opened.
fn known(env: Env, cmd: []const u8, args: []const []const u8, f: Flags) !bool {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const x = args[i];
        if (!std.mem.startsWith(u8, x, "--")) {
            if (i < f.positionals) continue;
            try env.err.print("{s}: unexpected argument {s}\n", .{ cmd, x });
            return false;
        }
        const eq = std.mem.indexOfScalar(u8, x, '=');
        const name = x[0 .. eq orelse x.len];
        if (contains(f.values, name)) {
            if (eq != null) continue;
            if (i + 1 >= args.len) {
                try env.err.print("{s}: {s} needs a value\n", .{ cmd, name });
                return false;
            }
            i += 1;
        } else if (contains(f.switches, name)) {
            if (eq == null) continue;
            try env.err.print("{s}: {s} takes no value\n", .{ cmd, name });
            return false;
        } else {
            try env.err.print("{s}: unknown flag {s}\n", .{ cmd, name });
            return false;
        }
    }
    return true;
}

fn contains(list: []const []const u8, name: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
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
    const found = launch.discover(env.a, env.runner, c) catch |err| {
        if (err == error.GpuLimitMissing) return 1;
        return err;
    };
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

/// Every model in the file, or only `only`.
fn checkCmd(env: Env, args: []const []const u8, show_nodes: bool, only: ?[]const u8) !u8 {
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
        if (only) |want| if (!std.mem.eql(u8, want, m.id)) continue;
        try env.out.print("\nmodel {s}: {s}, {d} nodes\n", .{ m.id, @tagName(m.mode), m.nodes.len });
        if (m.mode == .disaggregated) {
            try env.out.print("prefill on {d} nodes, decode on {d}: KV handoff over MCDMA; outside the exactness contract (a prefill computed elsewhere is not this node's own)\n", .{ m.prefill.len, m.decode.len });
            continue;
        }
        const invs = cli_model.inventories(src, c, m.nodes) catch |err| return reportErr(env, err);
        const pl = cli_model.planModel(src, m, invs) catch |err| {
            try env.out.print("not planned: {s}\n", .{@errorName(err)});
            code = 1;
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
    if (err == error.GpuLimitMissing) return 1;
    try env.err.print("{s}: give memory_gb for every node, or pass --probe or --probes DIR\n", .{@errorName(err)});
    return 1;
}

/// Converged rounds on real hosts need a real Shard on every rank; only the toy shard exists.
fn refuseConverged(env: Env, m: config.Model) !void {
    try env.err.print("model {s} is converged: its rounds on real hosts need a real backend.Shard (cluster/backend.zig) on every node, and only the toy shard exists, so it is refused and no node was started; a disaggregated model brings its nodes up over MCDMA\n", .{m.id});
}

fn serveCmd(env: Env, args: []const []const u8) !u8 {
    if (!try known(env, "serve", args, serve_flags)) return bad(env);
    const l = (try load(env, args)) orelse return bad(env);
    const c = l.config;
    const id = if (args.len > 0 and !std.mem.startsWith(u8, args[0], "--")) args[0] else if (c.models.len == 1) c.models[0].id else return bad(env);
    const m = c.model(id) orelse {
        try env.err.print("no model \"{s}\" in the cluster file\n", .{id});
        return 1;
    };
    const hold = seconds(args, "--hold") orelse return bad(env);
    // a disaggregated model's nodes are its prefill and decode lists, and no tensor-parallel group spans the two
    const disaggregated = m.mode == .disaggregated;
    const members = try mcdma_link.ranks(env.a, m);
    const leader = flag(args, "--leader") orelse if (mcdma_link.rankOf(members, m.nodes[0]) != null) m.nodes[0] else members[0];
    if (disaggregated and mcdma_link.rankOf(members, leader) == null) {
        try env.err.print("--leader {s} is not one of model {s}'s prefill or decode nodes\n", .{ leader, m.id });
        return 1;
    }
    const code = try checkCmd(env, args, false, m.id);
    const fake = flag(args, "--probes") != null;
    const dry = has(args, "--dry-run");
    if (!fake and !disaggregated) {
        try refuseConverged(env, m);
        return if (dry) code else 1;
    }
    if (code != 0) return code;
    // real hosts refuse before any node starts: every node's links, library and window, then this node's library file
    if (!fake) {
        var why: []const u8 = "";
        mcdma_link.specAll(env.a, c, m, .{}, &why) catch |err| {
            try env.err.print("{s}\n", .{why});
            return if (mcdma_link.libraryError(err)) 3 else 1;
        };
    }
    const here: ?Prepared = if (fake) null else switch (try prepare(env, c, m, leader)) {
        .ok => |p| p,
        .code => |x| return x,
    };
    var started_model = m;
    started_model.nodes = members;
    const started = try launch.launch(env.a, env.runner, c, l.json, started_model, leader, dry);
    try env.out.print("\nleader {s}; the control step starts tensorfold node over key-only ssh on:", .{leader});
    for (started) |s| try env.out.print(" {s}{s}", .{ s.name, if (s.ok) "" else " (FAILED)" });
    try env.out.writeAll("\n");
    for (started) |s| if (!s.ok) return 1;
    if (here) |p| {
        try describe(env, c, p);
        if (dry) {
            try env.out.writeAll("dry run: no node was started and no link opened\n");
            return 0;
        }
        const inv = launch.inventory(env.a, env.runner, c, c.nodes[c.find(leader).?], true) catch |err| {
            if (err == error.GpuLimitMissing) return 1;
            return err;
        };
        return hostRun(env, p, inv, hold);
    }
    const src = source(env, args);
    if (disaggregated) {
        const invs = cli_model.inventories(src, c, members) catch |err| {
            if (err == error.GpuLimitMissing) return 1;
            return err;
        };
        const plan = mcdma_link.planDigest(c, m, members, .{});
        // cabled as the real nodes are, by the declared links (a CUDA host probes no Thunderbolt peers), else as probed
        const es = if (c.links.len > 0) try mcdma_link.cables(env.a, c, members) else try topology.edges(env.a, invs);
        const r = bringup.simulate(env.a, invs, es, plan) catch |err| {
            try env.err.print("the fake cluster did not come up: {s}\n", .{@errorName(err)});
            return 1;
        };
        try env.out.print("fake cluster up: {d} nodes agree on disaggregated plan {x:0>16} (prefill on {d}, decode on {d}; leader {x}, epoch {d}) after {d} ms simulated; the KV hand-off needs the real endpoints and engines\n", .{ r.nodes, r.plan, m.prefill.len, m.decode.len, r.leader, r.epoch, r.agreed_ms });
        return 0;
    }
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
    if (!try known(env, "node", args, node_flags)) return bad(env);
    const l = (try load(env, args)) orelse return bad(env);
    const c = l.config;
    const name = flag(args, "--name") orelse return bad(env);
    const i = c.find(name) orelse {
        try env.err.print("no node \"{s}\" in the cluster file\n", .{name});
        return 1;
    };
    if (flag(args, "--probes") != null) {
        const inv = try launch.inventory(env.a, env.runner, c, c.nodes[i], true);
        try status.nodes(env.out, &.{inv});
        try env.err.writeAll("node: fake hosts (--probes) have no MCDMA links to join; serve --cluster --probes brings every node up in its own process\n");
        return 1;
    }
    const id = flag(args, "--model") orelse if (c.models.len == 1) c.models[0].id else return bad(env);
    const m = c.model(id) orelse {
        try env.err.print("no model \"{s}\" in the cluster file\n", .{id});
        return 1;
    };
    if (m.mode != .disaggregated) {
        try refuseConverged(env, m);
        return 1;
    }
    const p = switch (try prepare(env, c, m, name)) {
        .ok => |x| x,
        .code => |x| return x,
    };
    try describe(env, c, p);
    const inv = launch.inventory(env.a, env.runner, c, c.nodes[i], true) catch |err| {
        if (err == error.GpuLimitMissing) return 1;
        return err;
    };
    try status.nodes(env.out, &.{inv});
    return hostRun(env, p, inv, null);
}

const Prepared = struct { spec: mcdma_link.Spec, plan: u64 };

/// This node's rank, links, window and plan from the file, and its library found; else the exit code (3: the library).
fn prepare(env: Env, c: config.Config, m: config.Model, name: []const u8) !union(enum) { ok: Prepared, code: u8 } {
    var why: []const u8 = "";
    const s = mcdma_link.spec(env.a, c, m, name, .{}, &why) catch |err| {
        try env.err.print("{s}\n", .{why});
        return .{ .code = if (mcdma_link.libraryError(err)) 3 else 1 };
    };
    mcdma_link.findLibrary(env.io, s.library) catch |err| {
        try env.err.print("{s}: no MCDMA library at {s} ({s}): build libmcdma-fabric (tools/zig/mcdma_pair/build_mcdma.sh) and give its absolute path as the node's \"library\"\n", .{ name, s.library, @errorName(err) });
        return .{ .code = 3 };
    };
    return .{ .ok = .{ .spec = s, .plan = mcdma_link.planDigest(c, m, s.names, .{}) } };
}

/// What this node will open: its rank, each link's ports, GID, path MTU and kind, the window, the plan.
fn describe(env: Env, c: config.Config, p: Prepared) !void {
    const s = p.spec;
    try env.out.print("{s}: rank {d} of {d} (", .{ s.names[s.rank], s.rank, s.names.len });
    for (s.names, 0..) |n, r| try env.out.print("{s}{d} {s} {s}", .{ if (r == 0) "" else ", ", r, n, @tagName(c.nodes[c.find(n).?].backend) });
    try env.out.print("); plan {x:0>16}; window {d} bytes, staging {d} a rank; library {s}\n", .{ p.plan, s.config.window_bytes, s.config.staging_bytes, s.library });
    for (s.links) |x| try env.out.print("  link \"{s}\" to {s}: {s} via {s}, port {d} to {d}, gid {d}, path MTU {d}, kind {s}\n", .{ x.name, s.names[x.peer], x.device, x.via, x.port, x.peer_port, x.gid, x.mtu, if (x.kind) |k| @tagName(k) else "any" });
}

/// Bring this node up over MCDMA and hold: `hold_ns`, then goodbye (serve), or null until a node says goodbye (node).
fn hostRun(env: Env, p: Prepared, probed: node.Inventory, hold_ns: ?u64) !u8 {
    // membership knows a node by its unique name in the file (its id is the name's), not its probed hardware id
    var inv = probed;
    inv.name = .of(p.spec.names[p.spec.rank]);
    try env.out.flush();
    var diag: bringup.Diag = .{};
    const got = bringup.onHost(std.heap.c_allocator, p.spec, inv, p.plan, .{ .hold_ns = hold_ns }, env.out, &diag) catch |err| {
        const who = diag.culprit.name.str();
        switch (err) {
            error.PlanMismatch => try env.err.print("{s} computed plan {x:0>16} and this node {x:0>16}: every node must read the same cluster file (its links, path MTUs and node lists)\n", .{ who, diag.culprit.plan, p.plan }),
            error.WrongPeer => try env.err.print("{s} answered on the link the cluster file gives {s}: check each link's via\n", .{ who, diag.culprit.expected }),
            error.MemberLost => try env.err.print("{s} is {s}{s}{s}\n", .{ who, if (diag.culprit.state) |st| @tagName(st) else "gone", if (diag.reason.len > 0) "; MCDMA: " else "", diag.reason }),
            error.NeverSettled, error.NeverAgreed => try env.err.print("{s}: not every node of the plan joined and agreed within the settle time\n", .{@errorName(err)}),
            else => try env.err.print("mcdma: {s}\n", .{@errorName(err)}),
        }
        return if (mcdma_link.libraryError(err)) 3 else 1;
    };
    if (got.ended_by) |x| {
        try env.out.print("{s} said goodbye; this node said goodbye and closed its links\n", .{x.name.str()});
        if (hold_ns != null) {
            try env.err.print("{s} left before the hold ended\n", .{x.name.str()});
            return 1;
        }
    } else try env.out.print("held the cluster {d} s, said goodbye and closed the links; rounds need the KV hand-off (#1594 step 2b)\n", .{(hold_ns orelse 0) / std.time.ns_per_s});
    return 0;
}

/// Whole seconds given as `name`, in nanoseconds: 0 when absent, null when not a number.
fn seconds(args: []const []const u8, name: []const u8) ?u64 {
    const text = flag(args, name) orelse return 0;
    const s = std.fmt.parseInt(u32, text, 10) catch return null;
    return @as(u64, s) * std.time.ns_per_s;
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
