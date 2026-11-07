//! Control step over key-only ssh: read-only probes, `cluster init`, starting `tensorfold node`; never the data plane.
const std = @import("std");
const node = @import("node.zig");
const config = @import("config.zig");
const probe = @import("probe.zig");
const topology = @import("topology.zig");

const Allocator = std.mem.Allocator;

pub const Output = struct { code: u8, stdout: []const u8 };

/// Runs a command for node `name` (over ssh, or locally) and returns its exit code and stdout; tests swap in a fake.
pub const Runner = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque, a: Allocator, name: []const u8, argv: []const []const u8) anyerror!Output,
};

/// Fake hosts: each node's probe output read from DIR/NAME/<tool>.txt, and node launches appended to DIR/launches.log.
pub const Saved = struct {
    io: std.Io,
    dir: []const u8,

    pub fn runner(s: *Saved) Runner {
        return .{ .ptr = s, .run = run };
    }

    const files = [_]struct { []const u8, []const u8 }{
        .{ "sysctl", "sysctl.txt" },
        .{ "system_profiler SPDisplaysDataType", "displays.txt" },
        .{ "system_profiler SPThunderboltDataType", "thunderbolt.txt" },
        .{ "ibv_devinfo", "ibv_devinfo.txt" },
        .{ "df -k /System/Volumes/Data", "df.txt" },
        .{ "cat /proc/meminfo", "meminfo.txt" },
        .{ "nvidia-smi", "nvidia.txt" },
        .{ "df -k /", "df-root.txt" },
        .{ "cat /etc/machine-id", "machine-id.txt" },
    };

    fn run(ptr: *anyopaque, a: Allocator, name: []const u8, argv: []const []const u8) anyerror!Output {
        const s: *Saved = @ptrCast(@alignCast(ptr));
        const cmd = argv[argv.len - 1];
        const cwd = std.Io.Dir.cwd();
        if (std.mem.indexOf(u8, cmd, " node --name ") != null) {
            const path = try std.fs.path.join(a, &.{ s.dir, "launches.log" });
            const log = try cwd.createFile(s.io, path, .{ .truncate = false });
            defer log.close(s.io);
            const line = try std.fmt.allocPrint(a, "{s}: {s}\n", .{ name, try std.mem.join(a, " ", argv) });
            try log.writePositionalAll(s.io, line, try log.length(s.io));
            return .{ .code = 0, .stdout = "started 0\n" };
        }
        for (files) |f| if (std.mem.startsWith(u8, cmd, f[0])) {
            const path = try std.fs.path.join(a, &.{ s.dir, name, f[1] });
            const text = cwd.readFileAlloc(s.io, path, a, .limited(64 << 20)) catch return .{ .code = 1, .stdout = "" };
            return .{ .code = 0, .stdout = text };
        };
        return .{ .code = 127, .stdout = "" };
    }
};

/// ssh with only the key: no passwords, no prompts; nodes with `via` go through their jump host.
pub fn sshArgv(a: Allocator, c: config.Config, n: config.Node, remote: []const u8) ![]const []const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(a, "ssh");
    if (c.ssh.key) |k| try argv.appendSlice(a, &.{ "-i", k });
    try argv.appendSlice(a, &.{ "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15" });
    if (n.via) |via| {
        const j = c.nodes[c.find(via).?];
        const key = if (c.ssh.key) |k| try std.fmt.allocPrint(a, "-i {s} ", .{k}) else "";
        try argv.appendSlice(a, &.{ "-o", try std.fmt.allocPrint(a, "ProxyCommand=ssh {s}-o IdentitiesOnly=yes -o BatchMode=yes -W %h:%p {s}", .{ key, try target(a, c, j) }) });
    }
    try argv.append(a, try target(a, c, n));
    try argv.append(a, remote);
    return argv.items;
}

fn target(a: Allocator, c: config.Config, n: config.Node) ![]const u8 {
    return if (c.ssh.user) |u| std.fmt.allocPrint(a, "{s}@{s}", .{ u, n.address }) else n.address;
}

/// The read-only tools each probe runs: nothing installed, written or started.
pub const mac_probes = [_][]const u8{
    "sysctl hw.memsize hw.pagesize iogpu.wired_limit_mb vm.page_free_count vm.page_speculative_count machdep.cpu.brand_string kern.osproductversion",
    "system_profiler SPDisplaysDataType",
    "system_profiler SPThunderboltDataType",
    "ibv_devinfo -v",
    "df -k /System/Volumes/Data",
};

/// Linux tools for a CUDA host. nvidia-smi, ibv_devinfo and machine-id may be missing; meminfo and df may not.
pub const linux_probes = [_][]const u8{
    "cat /proc/meminfo",
    "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits",
    "ibv_devinfo -v",
    "df -k /",
    "cat /etc/machine-id",
};

fn optionalProbe(cmd: []const u8) bool {
    return std.mem.startsWith(u8, cmd, "ibv_devinfo") or std.mem.startsWith(u8, cmd, "nvidia-smi") or std.mem.startsWith(u8, cmd, "cat /etc/machine-id");
}

fn probeOut(a: Allocator, r: Runner, c: config.Config, n: config.Node, local: bool, cmd: []const u8) ![]const u8 {
    const argv: []const []const u8 = if (local) &.{ "/bin/sh", "-c", cmd } else try sshArgv(a, c, n, cmd);
    const out = try r.run(r.ptr, a, n.name, argv);
    if (out.code != 0) return if (optionalProbe(cmd)) "" else error.ProbeFailed;
    return out.stdout;
}

/// One node's inventory from its tools over ssh (or locally when `local`); a CUDA node with Linux tools, not macOS's.
pub fn inventory(a: Allocator, r: Runner, c: config.Config, n: config.Node, local: bool) !node.Inventory {
    if (n.backend == .cuda) {
        var inv = probe.inventoryLinux(.{
            .name = n.name,
            .meminfo = try probeOut(a, r, c, n, local, linux_probes[0]),
            .nvidia = try probeOut(a, r, c, n, local, linux_probes[1]),
            .devinfo = try probeOut(a, r, c, n, local, linux_probes[2]),
            .df = try probeOut(a, r, c, n, local, linux_probes[3]),
            .machine_id = try probeOut(a, r, c, n, local, linux_probes[4]),
        });
        inv.backend = n.backend;
        if (n.memory) |m| inv.memory = m;
        if (n.gpu_limit) |g| inv.gpu_limit = g;
        if (std.mem.indexOf(u8, inv.chip.str(), "GB10") != null and inv.gpu_limit == 0) {
            std.log.warn("refused: {s} is a GB10 and nvidia-smi printed no memory. Set gpu_limit_gb for {s} in the cluster file.", .{ n.name, n.name });
            return error.GpuLimitMissing;
        }
        return inv;
    }
    var texts: [mac_probes.len][]const u8 = undefined;
    for (mac_probes, &texts) |cmd, *t| t.* = try probeOut(a, r, c, n, local, cmd);
    var inv = probe.inventory(.{ .name = n.name, .sysctl = texts[0], .displays = texts[1], .thunderbolt = texts[2], .devinfo = texts[3], .df = texts[4] });
    inv.backend = n.backend;
    if (n.memory) |m| inv.memory = m;
    if (n.gpu_limit) |g| inv.gpu_limit = g;
    return inv;
}

/// `cluster init`: every node's inventory and its cables resolved to node names, ready to write and edit.
pub fn discover(a: Allocator, r: Runner, c: config.Config) !struct { config: config.Config, inventories: []node.Inventory } {
    const invs = try a.alloc(node.Inventory, c.nodes.len);
    for (c.nodes, invs) |n, *inv| inv.* = try inventory(a, r, c, n, false);
    const es = try topology.edges(a, invs);
    var out = c;
    out.nodes = try a.dupe(config.Node, c.nodes);
    for (out.nodes, invs, 0..) |*n, inv, i| {
        n.chip = try a.dupe(u8, inv.chip.str());
        n.memory = inv.memory;
        n.gpu_limit = inv.gpu_limit;
        var ports: std.ArrayList(config.Port) = .empty;
        for (es) |e| {
            if (e.a == i) try ports.append(a, .{ .device = try a.dupe(u8, inv.ports()[e.a_port].device.str()), .peer = c.nodes[e.b].name, .gbps = e.gbps });
            if (e.b == i) try ports.append(a, .{ .device = try a.dupe(u8, inv.ports()[e.b_port].device.str()), .peer = c.nodes[e.a].name, .gbps = e.gbps });
        }
        n.ports = ports.items;
    }
    return .{ .config = out, .inventories = invs };
}

/// The command that writes cluster.json and starts `bin` node detached, its stdio off ssh, so ssh returns at once.
pub fn nodeCommand(a: Allocator, json: []const u8, bin: []const u8, name: []const u8, leader: []const u8, model: []const u8) ![]const u8 {
    const b64 = try a.alloc(u8, std.base64.standard.Encoder.calcSize(json.len));
    _ = std.base64.standard.Encoder.encode(b64, json);
    return std.fmt.allocPrint(a, "mkdir -p ~/.config/tensorfold ~/.cache/tensorfold && printf %s {s} | base64 --decode > ~/.config/tensorfold/cluster.json && {{ nohup {s} node --name {s} --leader {s} --model {s} --cluster ~/.config/tensorfold/cluster.json >> ~/.cache/tensorfold/node.log 2>&1 < /dev/null & echo started $!; }}", .{ b64, bin, name, leader, model });
}

pub const Started = struct { name: []const u8, argv: []const []const u8, ok: bool };

/// `serve --cluster`: start `tensorfold node` on every model node except the leader (this host).
pub fn launch(a: Allocator, r: Runner, c: config.Config, json: []const u8, model: config.Model, leader: []const u8, dry: bool) ![]Started {
    var out: std.ArrayList(Started) = .empty;
    for (model.nodes) |name| {
        if (std.mem.eql(u8, name, leader)) continue;
        const n = c.nodes[c.find(name) orelse return error.UnknownNode];
        const argv = try sshArgv(a, c, n, try nodeCommand(a, json, n.bin orelse "tensorfold", name, leader, model.id));
        var ok = true;
        if (!dry) {
            const res = try r.run(r.ptr, a, name, argv);
            ok = res.code == 0 and std.mem.startsWith(u8, res.stdout, "started ");
        }
        try out.append(a, .{ .name = name, .argv = argv, .ok = ok });
    }
    return out.items;
}

/// Write `c` back out as cluster.json (what `cluster init` prints for the owner to edit).
pub fn writeJson(w: *std.Io.Writer, c: config.Config) !void {
    try w.print("{{\n  \"schema\": \"{s}\",\n  \"cluster\": \"{s}\",\n  \"transport\": \"mcdma\",\n", .{ config.schema, c.name });
    if (c.ssh.user != null or c.ssh.key != null) {
        try w.writeAll("  \"ssh\": {");
        if (c.ssh.user) |u| try w.print(" \"user\": \"{s}\"{s}", .{ u, if (c.ssh.key != null) "," else "" });
        if (c.ssh.key) |k| try w.print(" \"key\": \"{s}\"", .{k});
        try w.writeAll(" },\n");
    }
    try w.writeAll("  \"nodes\": [\n");
    for (c.nodes, 0..) |n, i| {
        try w.print("    {{ \"name\": \"{s}\", \"address\": \"{s}\", \"backend\": \"{s}\"", .{ n.name, n.address, @tagName(n.backend) });
        if (n.via) |v| try w.print(", \"via\": \"{s}\"", .{v});
        if (n.chip) |ch| try w.print(", \"chip\": \"{s}\"", .{ch});
        if (n.memory) |m| try w.print(", \"memory_gb\": {d:.0}", .{@as(f64, @floatFromInt(m)) / @as(f64, @floatFromInt(node.gib))});
        if (n.gpu_limit) |g| try w.print(", \"gpu_limit_gb\": {d:.1}", .{@as(f64, @floatFromInt(g)) / @as(f64, @floatFromInt(node.gib))});
        if (n.library) |l| try w.print(", \"library\": \"{s}\"", .{l});
        if (n.bin) |b| try w.print(", \"bin\": \"{s}\"", .{b});
        if (n.ports) |ps| {
            try w.writeAll(",\n      \"ports\": [");
            for (ps, 0..) |p, k| try w.print("{s}{{ \"device\": \"{s}\", \"peer\": \"{s}\", \"gbps\": {d} }}", .{ if (k == 0) "" else ", ", p.device, p.peer, p.gbps });
            try w.writeAll("]");
        } else try w.writeAll(", \"ports\": \"auto\"");
        try w.print(" }}{s}\n", .{if (i + 1 < c.nodes.len) "," else ""});
    }
    try w.writeAll("  ],\n");
    if (c.links.len > 0) {
        try w.writeAll("  \"links\": [\n");
        for (c.links, 0..) |l, i| {
            try w.print("    {{ \"name\": \"{s}\"", .{l.name});
            if (l.kind) |k| try w.print(", \"kind\": \"{s}\"", .{@tagName(k)});
            try w.print(", \"mtu\": {d},\n      \"ends\": [", .{l.mtu});
            for (l.ends, 0..) |e, k| {
                try w.print("{s}{{ \"node\": \"{s}\", \"device\": \"{s}\", \"via\": \"{s}\", \"port\": {d}", .{ if (k == 0) "" else ",\n                ", e.node, e.device, e.via, e.port });
                if (e.gid) |g| try w.print(", \"gid\": {d}", .{g});
                try w.writeAll(" }");
            }
            try w.print("] }}{s}\n", .{if (i + 1 < c.links.len) "," else ""});
        }
        try w.writeAll("  ],\n");
    }
    try w.writeAll("  \"models\": {}\n}\n");
}

test {
    _ = @import("launch_test.zig");
}
