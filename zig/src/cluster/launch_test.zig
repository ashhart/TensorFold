//! The control step against fake hosts: key-only ssh argv, `cluster init` discovering a mesh, and node launches.
const std = @import("std");
const fabric = @import("fabric");
const node = @import("node.zig");
const config = @import("config.zig");
const launch = @import("launch.zig");
const fixtures = @import("probe_fixtures.zig");

const Fake = struct {
    cables: []const fixtures.Cable,
    calls: std.ArrayList([]const []const u8) = .empty,
    fail: ?[]const u8 = null,

    fn runner(f: *Fake) launch.Runner {
        return .{ .ptr = f, .run = run };
    }

    fn run(ptr: *anyopaque, a: std.mem.Allocator, name: []const u8, argv: []const []const u8) anyerror!launch.Output {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        try f.calls.append(a, argv);
        const host = argv[argv.len - 2];
        const cmd = argv[argv.len - 1];
        const at = std.mem.lastIndexOfScalar(u8, host, '@') orelse return error.NoUser;
        const n: u8 = host[at + 1 + "host".len] - '0';
        if (name[1] - '0' != n) return error.WrongNode;
        if (std.mem.indexOf(u8, cmd, "tensorfold node") != null) {
            if (f.fail) |bad| if (std.mem.indexOf(u8, cmd, bad) != null) return .{ .code = 255, .stdout = "" };
            return .{ .code = 0, .stdout = "started 4242\n" };
        }
        const tb = try fixtures.render(a, n, 4, f.cables);
        const text = if (std.mem.startsWith(u8, cmd, "sysctl")) fixtures.sysctl_a else if (std.mem.indexOf(u8, cmd, "SPDisplays") != null) fixtures.displays else if (std.mem.indexOf(u8, cmd, "SPThunderbolt") != null) tb[0] else if (std.mem.startsWith(u8, cmd, "ibv_devinfo")) tb[1] else fixtures.df;
        return .{ .code = 0, .stdout = text };
    }
};

const four =
    \\{ "schema": "tensorfold-cluster/1", "cluster": "lab", "ssh": { "user": "tf", "key": "/keys/cluster" },
    \\  "nodes": [ { "name": "n0", "address": "host0", "backend": "metal" },
    \\             { "name": "n1", "address": "host1", "backend": "metal", "via": "n0" },
    \\             { "name": "n2", "address": "host2", "backend": "metal", "via": "n0" },
    \\             { "name": "n3", "address": "host3", "backend": "metal", "via": "n0" } ],
    \\  "models": { "m": { "path": "/models/m", "parallel": { "tensor": 4, "expert": 4 } } } }
;

const mesh = [_]fixtures.Cable{ .{ .a = 0, .a_bus = 1, .b = 3, .b_bus = 1 }, .{ .a = 0, .a_bus = 2, .b = 2, .b_bus = 2 }, .{ .a = 0, .a_bus = 3, .b = 1, .b_bus = 3 }, .{ .a = 1, .a_bus = 1, .b = 2, .b_bus = 1 }, .{ .a = 1, .a_bus = 2, .b = 3, .b_bus = 2 }, .{ .a = 2, .a_bus = 3, .b = 3, .b_bus = 3 } };

test "a cuda node is probed with Linux tools and keeps a declared gpu limit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Rec = struct {
        cmds: std.ArrayList([]const u8) = .empty,
        fn runner(r: *@This()) launch.Runner {
            return .{ .ptr = r, .run = run };
        }
        fn run(ptr: *anyopaque, alloc: std.mem.Allocator, _: []const u8, argv: []const []const u8) anyerror!launch.Output {
            const r: *@This() = @ptrCast(@alignCast(ptr));
            const cmd = argv[argv.len - 1];
            try r.cmds.append(alloc, cmd);
            if (std.mem.startsWith(u8, cmd, "cat /proc/meminfo")) return .{ .code = 0, .stdout = "MemTotal:       131072000 kB\nMemAvailable:    80000000 kB\n" };
            if (std.mem.startsWith(u8, cmd, "nvidia-smi")) return .{ .code = 1, .stdout = "" };
            if (std.mem.startsWith(u8, cmd, "ibv_devinfo")) return .{ .code = 127, .stdout = "" };
            if (std.mem.startsWith(u8, cmd, "df")) return .{ .code = 0, .stdout = "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/nvme0n1p2 1 1 10 1% /\n" };
            if (std.mem.startsWith(u8, cmd, "cat /etc/machine-id")) return .{ .code = 0, .stdout = "abc123\n" };
            return .{ .code = 127, .stdout = "" };
        }
    };
    var rec: Rec = .{};
    var nodes = [_]config.Node{.{ .name = "gb10", .address = "192.0.2.30", .backend = .cuda, .gpu_limit = 100 * node.gib }};
    const c: config.Config = .{ .name = "lab", .nodes = &nodes, .models = &.{} };
    const inv = try launch.inventory(a, rec.runner(), c, nodes[0], true);
    try std.testing.expectEqual(@as(usize, launch.linux_probes.len), rec.cmds.items.len);
    try std.testing.expect(std.mem.startsWith(u8, rec.cmds.items[0], "cat /proc/meminfo"));
    try std.testing.expect(std.mem.indexOf(u8, rec.cmds.items[1], "nvidia-smi") != null);
    try std.testing.expectEqual(node.Backend.cuda, inv.backend);
    try std.testing.expectEqual(@as(u64, 100 * node.gib), inv.gpu_limit);
    try std.testing.expectEqual(@as(u64, 131_072_000 * 1024), inv.memory);
    try std.testing.expectEqualStrings("CUDA", inv.chip.str());
}

test "a GB10 with no memory figure is refused unless the cluster file sets gpu_limit_gb" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Rec = struct {
        fn runner() launch.Runner {
            return .{ .ptr = undefined, .run = run };
        }
        fn run(_: *anyopaque, _: std.mem.Allocator, _: []const u8, argv: []const []const u8) anyerror!launch.Output {
            const cmd = argv[argv.len - 1];
            if (std.mem.startsWith(u8, cmd, "cat /proc/meminfo")) return .{ .code = 0, .stdout = "MemTotal:       131072000 kB\nMemAvailable:    80000000 kB\n" };
            if (std.mem.startsWith(u8, cmd, "nvidia-smi")) return .{ .code = 0, .stdout = "NVIDIA GB10, [N/A], [N/A]\n" };
            if (std.mem.startsWith(u8, cmd, "ibv_devinfo")) return .{ .code = 0, .stdout = "" };
            if (std.mem.startsWith(u8, cmd, "df")) return .{ .code = 0, .stdout = "Filesystem 1K-blocks Used Available Use% Mounted on\n/dev/nvme0n1p2 1 1 10 1% /\n" };
            if (std.mem.startsWith(u8, cmd, "cat /etc/machine-id")) return .{ .code = 0, .stdout = "abc123\n" };
            return .{ .code = 127, .stdout = "" };
        }
    };
    var open = [_]config.Node{.{ .name = "gb10", .address = "192.0.2.30", .backend = .cuda }};
    const bare: config.Config = .{ .name = "lab", .nodes = &open, .models = &.{} };
    try std.testing.expectError(error.GpuLimitMissing, launch.inventory(a, Rec.runner(), bare, open[0], true));
    var capped = [_]config.Node{.{ .name = "gb10", .address = "192.0.2.30", .backend = .cuda, .gpu_limit = 100 * node.gib }};
    const set: config.Config = .{ .name = "lab", .nodes = &capped, .models = &.{} };
    const inv = try launch.inventory(a, Rec.runner(), set, capped[0], true);
    try std.testing.expect(inv.unified);
    try std.testing.expectEqual(@as(u64, 100 * node.gib), inv.gpu_limit);
    try std.testing.expectEqualStrings("NVIDIA GB10", inv.chip.str());
}

test "ssh for a node behind a jump host is key-only and goes through it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, four, &probs);
    const argv = try launch.sshArgv(a, c, c.nodes[2], "true");
    const joined = try std.mem.join(a, " ", argv);
    try std.testing.expect(std.mem.indexOf(u8, joined, "-i /keys/cluster -o IdentitiesOnly=yes -o BatchMode=yes") != null);
    try std.testing.expect(std.mem.indexOf(u8, joined, "ProxyCommand=ssh -i /keys/cluster -o IdentitiesOnly=yes -o BatchMode=yes -W %h:%p tf@host0") != null);
    try std.testing.expectEqualStrings("tf@host2", argv[argv.len - 2]);
    try std.testing.expect(std.mem.indexOf(u8, joined, "assword") == null);
}

test "cluster init discovers four Macs, their cables and memory, and writes a file that parses back" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, four, &probs);
    var fake: Fake = .{ .cables = &mesh };
    const found = try launch.discover(a, fake.runner(), c);
    for (found.config.nodes) |n| {
        try std.testing.expectEqual(@as(usize, 3), n.ports.?.len);
        try std.testing.expectEqual(@as(?u64, 512 * node.gib), n.memory);
        try std.testing.expectEqualStrings("Apple M3 Ultra", n.chip.?);
    }
    try std.testing.expectEqualStrings("n3", found.config.nodes[0].ports.?[0].peer);
    try std.testing.expectEqual(@as(usize, 4 * launch.mac_probes.len), fake.calls.items.len);
    var out: std.Io.Writer.Allocating = .init(a);
    try launch.writeJson(&out.writer, found.config);
    var again: config.Problems = .{ .a = a };
    const back = try config.parse(a, out.written(), &again);
    try std.testing.expectEqual(@as(usize, 4), back.nodes.len);
    try std.testing.expectEqualStrings("n0", back.nodes[1].via.?);
    try std.testing.expectEqual(@as(usize, 3), back.nodes[2].ports.?.len);
}

test "serve starts tensorfold node on every other model node, reports a node that fails, and a dry run starts none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, four, &probs);
    var fake: Fake = .{ .cables = &mesh, .fail = "--name n2" };
    const started = try launch.launch(a, fake.runner(), c, four, c.model("m").?, "n0", false);
    try std.testing.expectEqual(@as(usize, 3), started.len);
    try std.testing.expect(started[0].ok and !started[1].ok and started[2].ok);
    const cmd = started[0].argv[started[0].argv.len - 1];
    try std.testing.expect(std.mem.indexOf(u8, cmd, "nohup tensorfold node --name n1 --leader n0 --model m") != null);
    try std.testing.expect(std.mem.indexOf(u8, cmd, "base64 --decode > ~/.config/tensorfold/cluster.json") != null);
    const calls = fake.calls.items.len;
    _ = try launch.launch(a, fake.runner(), c, four, c.model("m").?, "n0", true);
    try std.testing.expectEqual(calls, fake.calls.items.len);
}

test "a node's \"bin\" is the tensorfold its launch runs over ssh; without one, tensorfold from the login PATH" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.mem.replaceOwned(u8, a, four, "\"name\": \"n1\", \"address\": \"host1\", \"backend\": \"metal\",", "\"name\": \"n1\", \"address\": \"host1\", \"backend\": \"metal\", \"bin\": \"/opt/tf/bin/tensorfold\",");
    var probs: config.Problems = .{ .a = a };
    const c = try config.parse(a, text, &probs);
    var fake: Fake = .{ .cables = &mesh };
    const started = try launch.launch(a, fake.runner(), c, text, c.model("m").?, "n0", true);
    try std.testing.expectEqualStrings("tf@host1", started[0].argv[started[0].argv.len - 2]);
    try std.testing.expect(std.mem.indexOf(u8, started[0].argv[started[0].argv.len - 1], "nohup /opt/tf/bin/tensorfold node --name n1 --leader n0 --model m --cluster") != null);
    try std.testing.expect(std.mem.indexOf(u8, started[1].argv[started[1].argv.len - 1], "nohup tensorfold node --name n2 --leader n0 --model m --cluster") != null);
}

test "the node launch backgrounds only the node: the exact remote command, braces around nohup and echo" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const cmd = try launch.nodeCommand(arena.allocator(), "{}", "/opt/tf/bin/tensorfold", "n1", "n0", "m");
    try std.testing.expectEqualStrings("mkdir -p ~/.config/tensorfold ~/.cache/tensorfold && printf %s e30= | base64 --decode > ~/.config/tensorfold/cluster.json && { nohup /opt/tf/bin/tensorfold node --name n1 --leader n0 --model m --cluster ~/.config/tensorfold/cluster.json >> ~/.cache/tensorfold/node.log 2>&1 < /dev/null & echo started $!; }", cmd);
    // the braces hold the background job to the node alone, and the node holds none of ssh's stdin, stdout or stderr
    const open = std.mem.indexOf(u8, cmd, "&& { nohup ").?;
    const amp = std.mem.lastIndexOf(u8, cmd, " & echo started $!; }").?;
    try std.testing.expect(std.mem.endsWith(u8, cmd, " & echo started $!; }") and open < amp);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, cmd, " & "));
    const job = cmd[open..amp];
    for ([_][]const u8{ ">> ~/.cache/tensorfold/node.log", "2>&1", "< /dev/null" }) |redirect| try std.testing.expect(std.mem.indexOf(u8, job, redirect) != null);
}

test "the launch's remote command returns at once, before the node it started exits, so ssh does not wait on it" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const home = try tmp.dir.realPathFileAlloc(io, ".", a);
    // a stand-in node that runs 4 s; ssh, like std.process.run, reads the command's stdout until it closes
    try tmp.dir.writeFile(io, .{ .sub_path = "node.sh", .data = "sleep 4\n" });
    const bin = try std.fmt.allocPrint(a, "/bin/sh {s}/node.sh", .{home});
    const cmd = try launch.nodeCommand(a, "{\"cluster\": \"x\"}", bin, "n1", "n0", "m");
    const t0 = fabric.words.nowNs();
    const res = try std.process.run(a, io, .{ .argv = &.{ "/bin/sh", "-c", try std.fmt.allocPrint(a, "HOME={s}; export HOME; {s}", .{ home, cmd }) } });
    const took = fabric.words.nowNs() - t0;
    try std.testing.expect(res.term == .exited and res.term.exited == 0);
    try std.testing.expect(std.mem.startsWith(u8, res.stdout, "started "));
    try std.testing.expect(took < 2 * std.time.ns_per_s);
    const written = try tmp.dir.readFileAlloc(io, ".config/tensorfold/cluster.json", a, .limited(1 << 10));
    try std.testing.expectEqualStrings("{\"cluster\": \"x\"}", written);
}
