//! The control step against fake hosts: key-only ssh argv, `cluster init` discovering a mesh, and node launches.
const std = @import("std");
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
