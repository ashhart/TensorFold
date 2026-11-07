//! cluster.json (tensorfold-cluster/1): nodes, how to reach them, their MCDMA links, per model its layout and budgets.
const std = @import("std");
const fabric = @import("fabric");
const node = @import("node.zig");
const plan = @import("plan.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const LinkKind = fabric.rdma.LinkKind;

pub const schema = "tensorfold-cluster/1";

pub const Ssh = struct { user: ?[]const u8 = null, key: ?[]const u8 = null };

/// One declared port: its RDMA device and the node its cable reaches.
pub const Port = struct { device: []const u8, peer: []const u8, gbps: u32 = 0 };

pub const Node = struct {
    name: []const u8,
    /// Host or IP for the control step (ssh launch); never the data plane.
    address: []const u8,
    /// A node this one is reached through (its ssh jump host).
    via: ?[]const u8 = null,
    backend: node.Backend,
    chip: ?[]const u8 = null,
    memory: ?u64 = null,
    gpu_limit: ?u64 = null,
    /// null means discover the ports from the hardware.
    ports: ?[]Port = null,
    root: ?[]const u8 = null,
    /// libmcdma-fabric on this node: an absolute path (MCDMA's library is loaded from no other place).
    library: ?[]const u8 = null,
    /// This node's tensorfold by absolute path (ssh's PATH has only system directories); null: `tensorfold` from PATH.
    bin: ?[]const u8 = null,
};

/// One end of an MCDMA link: its node, RDMA device, meeting interface and peer address, exchange port, GID index.
pub const End = struct {
    node: []const u8,
    /// One device, or Thunderbolt members joined with "+".
    device: []const u8,
    /// The interface, a slash, then the other end's address on the same segment ("en7/192.0.2.10").
    via: []const u8,
    /// This end's exchange port; the other end's `peer_port`.
    port: u16,
    /// null: MCDMA's default (1, or -1 to scan for a bond of more than two devices); a ConnectX end always gives it.
    gid: ?c_int = null,

    /// The GID index MCDMA opens this end with.
    pub fn gidIndex(e: End) c_int {
        return e.gid orelse if (std.mem.count(u8, e.device, "+") + 1 > 2) -1 else 1;
    }
};

/// An MCDMA link, declared once: both ends read one name, kind and path MTU; each end's peer port is the other's port.
pub const Link = struct {
    name: []const u8,
    /// null takes whichever kind connects; .tb5 or .cx5 refuses the other.
    kind: ?LinkKind = null,
    /// The RC path MTU, the same on both ends and at most each port's active MTU.
    mtu: c_int = 4096,
    ends: [2]End,

    /// Which end is `name`'s, if either.
    pub fn endOf(l: Link, name: []const u8) ?u1 {
        for (l.ends, 0..) |e, i| if (std.mem.eql(u8, e.node, name)) return @intCast(i);
        return null;
    }

    /// Whether the link joins nodes `a` and `b`, in either order.
    pub fn joins(l: Link, a: []const u8, b: []const u8) bool {
        const i = l.endOf(a) orelse return false;
        return std.mem.eql(u8, l.ends[1 - i].node, b);
    }
};

pub const Drafter = struct { path: []const u8, bytes: u64 = 0, head: bool = true };

pub const Model = struct {
    id: []const u8,
    /// One path for every node, or per node name.
    path: ?[]const u8 = null,
    paths: []const [2][]const u8 = &.{},
    nodes: []const []const u8 = &.{},
    layout: plan.Layout = .{},
    mode: plan.Mode = .converged,
    prefill: []const []const u8 = &.{},
    decode: []const []const u8 = &.{},
    drafter: ?Drafter = null,
    context: u64 = 131072,
    streams: u32 = 1,
    rows: u32 = 16,
    /// Prompt rows a round takes while streams decode; 0 lets admission size it.
    chunk: u32 = 0,
    /// MLA attention by heads or by streams; null lets admission choose.
    mla: ?plan.MlaSplit = null,
    vision: bool = false,
    margin: u64 = 4 * node.gib,

    pub fn pathFor(m: Model, name: []const u8) ?[]const u8 {
        for (m.paths) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return m.path;
    }

    /// The planner's options for this model.
    pub fn options(m: Model) plan.Options {
        var o: plan.Options = .{ .mode = m.mode, .layout = m.layout, .streams = m.streams, .context = m.context, .rows = m.rows, .vision = m.vision, .margin = m.margin, .mla = m.mla orelse .heads };
        if (m.chunk > 0) o.chunk = m.chunk;
        if (m.drafter) |d| {
            o.drafter_bytes = d.bytes;
            o.drafter_head = d.head;
        } else o.drafter_head = false;
        return o;
    }
};

pub const Config = struct {
    name: []const u8,
    ssh: Ssh = .{},
    nodes: []Node,
    /// The MCDMA links of the data plane, each declared once; none for fake hosts and planning.
    links: []Link = &.{},
    models: []Model,

    pub fn find(c: Config, name: []const u8) ?usize {
        for (c.nodes, 0..) |n, i| if (std.mem.eql(u8, n.name, name)) return i;
        return null;
    }

    /// The link between nodes `a` and `b`, if one is declared.
    pub fn linkBetween(c: Config, a: []const u8, b: []const u8) ?Link {
        for (c.links) |l| if (l.joins(a, b)) return l;
        return null;
    }

    pub fn model(c: Config, id: []const u8) ?Model {
        for (c.models) |m| if (std.mem.eql(u8, m.id, id)) return m;
        return null;
    }
};

/// Every problem found, each as "path: message", so `cluster check` can print them all at once.
pub const Problems = struct {
    a: Allocator,
    list: std.ArrayList([]const u8) = .empty,

    pub fn add(p: *Problems, comptime fmt: []const u8, args: anytype) void {
        const msg = std.fmt.allocPrint(p.a, fmt, args) catch return;
        p.list.append(p.a, msg) catch {};
    }

    pub fn ok(p: Problems) bool {
        return p.list.items.len == 0;
    }
};

/// Parse and validate; on any problem returns error.Invalid with every problem in `probs`.
pub fn parse(a: Allocator, text: []const u8, probs: *Problems) !Config {
    const root = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch |err| {
        probs.add("cluster.json: not JSON ({s})", .{@errorName(err)});
        return error.Invalid;
    };
    if (root != .object) {
        probs.add("cluster.json: the top level must be an object", .{});
        return error.Invalid;
    }
    const o = root.object;
    if (!std.mem.eql(u8, str(o.get("schema")) orelse "", schema)) probs.add("schema: must be \"{s}\"", .{schema});
    const transport = str(o.get("transport")) orelse "mcdma";
    if (!std.mem.eql(u8, transport, "mcdma")) probs.add("transport: \"{s}\" is not supported; the data plane is MCDMA direct memory access only (no TCP)", .{transport});
    var c: Config = .{ .name = str(o.get("cluster")) orelse "", .nodes = &.{}, .models = &.{} };
    if (c.name.len == 0) probs.add("cluster: a name is required", .{});
    if (o.get("ssh")) |s| if (s == .object) {
        c.ssh = .{ .user = str(s.object.get("user")), .key = str(s.object.get("key")) };
    };
    c.nodes = try nodes(a, o.get("nodes"), probs);
    c.links = try links(a, o.get("links"), probs);
    c.models = try models(a, o.get("models"), probs);
    const every = try a.alloc([]const u8, c.nodes.len);
    for (c.nodes, every) |n, *e| e.* = n.name;
    for (c.models) |*m| if (m.nodes.len == 0) {
        m.nodes = every;
    };
    validate(c, probs);
    if (!probs.ok()) return error.Invalid;
    return c;
}

fn nodes(a: Allocator, v: ?Value, probs: *Problems) ![]Node {
    const arr = v orelse {
        probs.add("nodes: at least one node is required", .{});
        return &.{};
    };
    if (arr != .array) {
        probs.add("nodes: must be an array", .{});
        return &.{};
    }
    var out: std.ArrayList(Node) = .empty;
    for (arr.array.items, 0..) |item, i| {
        if (item != .object) {
            probs.add("nodes[{d}]: must be an object", .{i});
            continue;
        }
        const x = item.object;
        const name = str(x.get("name")) orelse "";
        const backend_text = str(x.get("backend")) orelse "";
        const backend: ?node.Backend = if (std.mem.eql(u8, backend_text, "metal")) .metal else if (std.mem.eql(u8, backend_text, "cuda")) .cuda else null;
        if (backend == null) probs.add("nodes[{d}] ({s}): backend must be \"metal\" or \"cuda\"", .{ i, name });
        var n: Node = .{ .name = name, .address = str(x.get("address")) orelse "", .via = str(x.get("via")), .backend = backend orelse .metal, .chip = str(x.get("chip")) orelse str(x.get("gpu")), .root = str(x.get("model_root")) };
        if (x.get("library")) |l| n.library = str(l) orelse "";
        if (x.get("bin")) |b| n.bin = str(b) orelse "";
        if (num(x.get("memory_gb"))) |g| n.memory = @intFromFloat(g * @as(f64, @floatFromInt(node.gib)));
        if (num(x.get("gpu_limit_gb"))) |g| n.gpu_limit = @intFromFloat(g * @as(f64, @floatFromInt(node.gib)));
        if (x.get("ports")) |p| switch (p) {
            .string => |s| if (!std.mem.eql(u8, s, "auto")) probs.add("nodes[{d}] ({s}).ports: \"auto\" or a list", .{ i, name }),
            .array => |list| {
                var ports: std.ArrayList(Port) = .empty;
                for (list.items) |pi| {
                    if (pi != .object) continue;
                    try ports.append(a, .{ .device = str(pi.object.get("device")) orelse "", .peer = str(pi.object.get("peer")) orelse "", .gbps = @intFromFloat(num(pi.object.get("gbps")) orelse 0) });
                }
                n.ports = ports.items;
            },
            else => probs.add("nodes[{d}] ({s}).ports: \"auto\" or a list", .{ i, name }),
        };
        try out.append(a, n);
    }
    return out.items;
}

/// The links as written; what they mean (nodes, kinds, GIDs, pairs) is `validate`'s.
fn links(a: Allocator, v: ?Value, probs: *Problems) ![]Link {
    const arr = v orelse return &.{};
    if (arr != .array) {
        probs.add("links: must be an array, one entry per MCDMA link", .{});
        return &.{};
    }
    var out: std.ArrayList(Link) = .empty;
    for (arr.array.items, 0..) |item, i| {
        if (item != .object) {
            probs.add("links[{d}]: must be an object", .{i});
            continue;
        }
        const x = item.object;
        const name = str(x.get("name")) orelse "";
        var l: Link = .{ .name = name, .ends = undefined };
        if (x.get("kind")) |k| {
            const kind: ?LinkKind = if (str(k)) |s| std.meta.stringToEnum(LinkKind, s) else null;
            if (kind == .tb5 or kind == .cx5) l.kind = kind else probs.add("links[{d}] ({s}).kind: \"tb5\" or \"cx5\"", .{ i, name });
        }
        if (x.get("mtu")) |m| {
            if (m == .integer and std.math.cast(c_int, m.integer) != null) l.mtu = @intCast(m.integer) else l.mtu = 0;
        }
        const ends = x.get("ends") orelse Value{ .null = {} };
        if (ends != .array or ends.array.items.len != 2 or ends.array.items[0] != .object or ends.array.items[1] != .object) {
            probs.add("links[{d}] ({s}).ends: two objects, one per node the link joins", .{ i, name });
            continue;
        }
        var whole = true;
        for (&l.ends, ends.array.items, 0..) |*e, ev, k| {
            const y = ev.object;
            e.* = .{ .node = str(y.get("node")) orelse "", .device = str(y.get("device")) orelse "", .via = str(y.get("via")) orelse "", .port = 0 };
            if (y.get("port")) |p| {
                if (p == .integer and p.integer >= 1 and p.integer <= 65535) e.port = @intCast(p.integer) else {
                    probs.add("links[{d}] ({s}).ends[{d}].port: an exchange port from 1 to 65535", .{ i, name, k });
                    whole = false;
                }
            }
            if (y.get("gid")) |g| {
                if (g == .integer and g.integer >= -1 and g.integer <= 255) e.gid = @intCast(g.integer) else {
                    probs.add("links[{d}] ({s}).ends[{d}].gid: a GID index from 0 to 255, or -1 to scan (Thunderbolt only)", .{ i, name, k });
                    whole = false;
                }
            }
        }
        if (whole) try out.append(a, l);
    }
    return out.items;
}

fn names(a: Allocator, v: ?Value) ![]const []const u8 {
    const arr = v orelse return &.{};
    if (arr != .array) return &.{};
    var out: std.ArrayList([]const u8) = .empty;
    for (arr.array.items) |x| if (x == .string) try out.append(a, x.string);
    return out.items;
}

fn degree(v: ?Value, path: []const u8, probs: *Problems) u32 {
    const x = v orelse return 0;
    switch (x) {
        .string => |s| if (std.mem.eql(u8, s, "auto")) return 0,
        .integer => |i| if (i >= 1 and i <= 256) return @intCast(i),
        else => {},
    }
    probs.add("{s}: a degree from 1 to 256, or \"auto\"", .{path});
    return 0;
}

fn models(a: Allocator, v: ?Value, probs: *Problems) ![]Model {
    const obj = v orelse return &.{};
    if (obj != .object) {
        probs.add("models: must be an object keyed by model id", .{});
        return &.{};
    }
    var out: std.ArrayList(Model) = .empty;
    var it = obj.object.iterator();
    while (it.next()) |kv| {
        const id = kv.key_ptr.*;
        const x = kv.value_ptr.*;
        if (x != .object) {
            probs.add("models.{s}: must be an object", .{id});
            continue;
        }
        const o = x.object;
        var m: Model = .{ .id = id, .nodes = try names(a, o.get("nodes")), .prefill = try names(a, o.get("prefill")), .decode = try names(a, o.get("decode")) };
        if (o.get("path")) |p| switch (p) {
            .string => |s| m.path = s,
            .object => |per| {
                var list: std.ArrayList([2][]const u8) = .empty;
                var pit = per.iterator();
                while (pit.next()) |e| if (e.value_ptr.* == .string) try list.append(a, .{ e.key_ptr.*, e.value_ptr.string });
                m.paths = list.items;
            },
            else => probs.add("models.{s}.path: a path, or an object of node name to path", .{id}),
        } else probs.add("models.{s}.path: required", .{id});
        if (o.get("parallel")) |p| if (p == .object) {
            var buf: [3][96]u8 = undefined;
            m.layout = .{
                .tensor = degree(p.object.get("tensor"), std.fmt.bufPrint(&buf[0], "models.{s}.parallel.tensor", .{id}) catch "parallel.tensor", probs),
                .pipeline = degree(p.object.get("pipeline"), std.fmt.bufPrint(&buf[1], "models.{s}.parallel.pipeline", .{id}) catch "parallel.pipeline", probs),
                .expert = degree(p.object.get("expert"), std.fmt.bufPrint(&buf[2], "models.{s}.parallel.expert", .{id}) catch "parallel.expert", probs),
            };
        };
        const mode = str(o.get("mode")) orelse "converged";
        if (std.mem.eql(u8, mode, "disaggregated")) m.mode = .disaggregated else if (!std.mem.eql(u8, mode, "converged")) probs.add("models.{s}.mode: \"converged\" or \"disaggregated\"", .{id});
        if (o.get("drafter")) |d| if (d == .object) {
            m.drafter = .{ .path = str(d.object.get("path")) orelse "", .bytes = @intFromFloat((num(d.object.get("size_gb")) orelse 0) * @as(f64, @floatFromInt(node.gib))), .head = if (d.object.get("uses_target_head")) |h| h == .bool and h.bool else true };
        };
        if (num(o.get("context"))) |n| m.context = @intFromFloat(n);
        if (num(o.get("streams"))) |n| m.streams = @intFromFloat(n);
        if (num(o.get("rows"))) |n| m.rows = @intFromFloat(n);
        if (num(o.get("chunk"))) |n| m.chunk = @intFromFloat(n);
        if (str(o.get("mla"))) |split| {
            if (std.mem.eql(u8, split, "heads")) m.mla = .heads else if (std.mem.eql(u8, split, "streams")) m.mla = .streams else if (!std.mem.eql(u8, split, "auto")) probs.add("models.{s}.mla: \"heads\", \"streams\" or \"auto\"", .{id});
        }
        if (o.get("vision")) |b| m.vision = b == .bool and b.bool;
        if (o.get("memory")) |mem| if (mem == .object) if (num(mem.object.get("margin_gb"))) |g| {
            m.margin = @intFromFloat(g * @as(f64, @floatFromInt(node.gib)));
        };
        try out.append(a, m);
    }
    return out.items;
}

/// Names, links, layouts and backend mixes that cannot work, each with what to change.
pub fn validate(c: Config, probs: *Problems) void {
    for (c.nodes, 0..) |n, i| {
        if (!validName(n.name)) probs.add("nodes[{d}].name: 1-20 characters of [A-Za-z0-9_-] (it names the node's MCDMA links)", .{i});
        if (n.address.len == 0) probs.add("nodes[{d}] ({s}).address: required for the launch step", .{ i, n.name });
        for (c.nodes[0..i]) |m| if (std.mem.eql(u8, m.name, n.name)) probs.add("nodes[{d}].name: \"{s}\" is used twice", .{ i, n.name });
        if (n.via) |via| {
            if (std.mem.eql(u8, via, n.name)) probs.add("nodes[{d}] ({s}).via: a node cannot be reached through itself", .{ i, n.name });
            if (c.find(via)) |j| {
                if (c.nodes[j].via != null) probs.add("nodes[{d}] ({s}).via: {s} is itself reached through another node; use one hop", .{ i, n.name, via });
            } else probs.add("nodes[{d}] ({s}).via: no node is named \"{s}\"", .{ i, n.name, via });
        }
        if (n.ports) |ports| for (ports) |p| {
            if (c.find(p.peer) == null) probs.add("nodes[{d}] ({s}).ports: {s} reaches \"{s}\", which is not a node", .{ i, n.name, p.device, p.peer });
        };
        if (n.library) |lib| if (!std.fs.path.isAbsolute(lib) or std.mem.indexOfScalar(u8, lib, 0) != null) {
            probs.add("nodes[{d}] ({s}).library: libmcdma-fabric's absolute path (MCDMA's library is loaded from no other place)", .{ i, n.name });
        };
        if (n.bin) |bin| if (!binOk(bin)) {
            probs.add("nodes[{d}] ({s}).bin: the absolute path of that node's tensorfold, in [A-Za-z0-9_./+-] (it goes into the launch's remote command)", .{ i, n.name });
        };
    }
    for (c.links, 0..) |l, i| checkLink(c, l, i, probs);
    for (c.models) |m| checkModel(c, m, probs);
}

/// A link as MCDMA's endpoint checks it (mcdma.zig's validate, memberCount, tp_settings), failing here by name first.
fn checkLink(c: Config, l: Link, i: usize, probs: *Problems) void {
    const name = l.name;
    if (!validName(name)) probs.add("links[{d}].name: 1-20 characters of [A-Za-z0-9_-] (both ends give MCDMA this name)", .{i});
    for (c.links[0..i]) |old| if (std.mem.eql(u8, old.name, name)) probs.add("links[{d}].name: \"{s}\" is used twice", .{ i, name });
    if (std.mem.indexOfScalar(c_int, &fabric.mcdma.path_mtus, l.mtu) == null) probs.add("links[{d}] ({s}).mtu: 1024, 2048 or 4096, the same on both ends and at most each port's active MTU (MCDMA runs any other value at 1024)", .{ i, name });
    if (std.mem.eql(u8, l.ends[0].node, l.ends[1].node)) probs.add("links[{d}] ({s}).ends: a link joins two different nodes", .{ i, name });
    // mcdma.Endpoint takes one link per peer and one link per device
    for (c.links[0..i]) |old| if (old.joins(l.ends[0].node, l.ends[1].node)) {
        probs.add("links[{d}] ({s}): {s} and {s} are already joined by link \"{s}\"; MCDMA takes one link per pair of nodes", .{ i, name, l.ends[0].node, l.ends[1].node, old.name });
        break;
    };
    for (l.ends, 0..) |e, k| {
        const at = c.find(e.node) orelse {
            probs.add("links[{d}] ({s}).ends[{d}].node: no node is named \"{s}\"", .{ i, name, k, e.node });
            continue;
        };
        const backend = c.nodes[at].backend;
        var members: usize = 1;
        if (fabric.verbs.deviceParts(probs.a, e.device)) |parts| {
            members = parts.len;
            probs.a.free(parts);
        } else |_| probs.add("links[{d}] ({s}).ends[{d}].device: an RDMA device, or up to 8 distinct Thunderbolt devices joined with \"+\"", .{ i, name, k });
        if (!viaOk(e.via, members)) probs.add("links[{d}] ({s}).ends[{d}].via: the interface, a slash and the other end's address (\"en7/192.0.2.10\"), one per device or one for all", .{ i, name, k });
        if (e.port == 0) probs.add("links[{d}] ({s}).ends[{d}].port: required, the exchange port this end listens on", .{ i, name, k }) else if (@as(usize, e.port) + members > 65536) probs.add("links[{d}] ({s}).ends[{d}].port: {d} devices take ports {d} and up, past 65535", .{ i, name, k, members, e.port });
        if (l.kind == .cx5) {
            if (members > 1) probs.add("links[{d}] ({s}).ends[{d}].device: a ConnectX link takes one device, no bond", .{ i, name, k });
            if (e.gid == null or e.gid.? < 0) probs.add("links[{d}] ({s}).ends[{d}].gid: a ConnectX link needs its RoCE v2 GID index (-1 scans Thunderbolt only)", .{ i, name, k }) else if (backend == .metal and e.gid.? != 0) probs.add("links[{d}] ({s}).ends[{d}].gid: {d} on a Mac; the MCDMA kext has GID index 0 only", .{ i, name, k, e.gid.? });
        }
        if (l.kind == .tb5 and backend == .cuda) probs.add("links[{d}] ({s}): {s} is a CUDA host; Thunderbolt links join Macs, a CUDA host takes a ConnectX link (\"kind\": \"cx5\")", .{ i, name, e.node });
        for (c.links[0..i]) |old| if (old.endOf(e.node)) |j| if (std.mem.eql(u8, old.ends[j].device, e.device)) {
            probs.add("links[{d}] ({s}).ends[{d}].device: {s}'s {s} already carries link \"{s}\"; one link per device", .{ i, name, k, e.node, e.device, old.name });
        };
    }
}

/// An absolute path the launch can put in its remote shell command as it is: no spaces, quotes or other shell syntax.
fn binOk(path: []const u8) bool {
    if (path.len < 2 or path[0] != '/') return false;
    for (path) |ch| if (!std.ascii.isAlphanumeric(ch) and std.mem.indexOfScalar(u8, "_./+-", ch) == null) return false;
    return true;
}

/// MCDMA's via: one interface/address for every member, or one per member, none empty or with spaces.
fn viaOk(via: []const u8, members: usize) bool {
    if (via.len == 0) return false;
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, via, '+');
    while (it.next()) |part| {
        if (part.len == 0) return false;
        for (part) |ch| if (ch <= ' ' or ch == 127) return false;
        count += 1;
    }
    return count == 1 or count == members;
}

fn checkModel(c: Config, m: Model, probs: *Problems) void {
    const members = m.nodes;
    for (members) |n| if (c.find(n) == null) probs.add("models.{s}.nodes: \"{s}\" is not a node", .{ m.id, n });
    for (m.paths) |p| if (c.find(p[0]) == null) probs.add("models.{s}.path: \"{s}\" is not a node", .{ m.id, p[0] });
    // the nodes that load the model: a disaggregated model's prefill and decode ("nodes" may default to every node)
    const loading: []const []const []const u8 = if (m.mode == .disaggregated) &.{ m.prefill, m.decode } else &.{members};
    if (m.path == null and m.paths.len > 0) for (loading) |list| for (list) |n| {
        if (m.pathFor(n) == null) probs.add("models.{s}.path: no path for node {s}", .{ m.id, n });
    };
    const n: u32 = @intCast(members.len);
    const l = m.layout;
    if (m.mode == .disaggregated) {
        if (m.prefill.len == 0 or m.decode.len == 0) probs.add("models.{s}: disaggregated mode needs \"prefill\" and \"decode\" node lists", .{m.id});
        for (m.prefill) |p| for (m.decode) |d| if (std.mem.eql(u8, p, d)) probs.add("models.{s}: {s} is both a prefill and a decode node", .{ m.id, p });
        for (m.prefill) |p| if (c.find(p) == null) probs.add("models.{s}.prefill: \"{s}\" is not a node", .{ m.id, p });
        for (m.decode) |p| if (c.find(p) == null) probs.add("models.{s}.decode: \"{s}\" is not a node", .{ m.id, p });
        return;
    }
    if (m.prefill.len > 0 or m.decode.len > 0) probs.add("models.{s}: prefill and decode lists need \"mode\": \"disaggregated\"", .{m.id});
    if (l.tensor > 0 and l.pipeline > 0 and l.tensor * l.pipeline != n) probs.add("models.{s}.parallel: tensor {d} x pipeline {d} must equal its {d} nodes", .{ m.id, l.tensor, l.pipeline, n });
    if (l.tensor > 0 and n % l.tensor != 0) probs.add("models.{s}.parallel.tensor: {d} does not divide its {d} nodes", .{ m.id, l.tensor, n });
    if (l.pipeline > 0 and n % l.pipeline != 0) probs.add("models.{s}.parallel.pipeline: {d} does not divide its {d} nodes", .{ m.id, l.pipeline, n });
    if (l.expert > 0 and l.tensor > 0 and l.tensor % l.expert != 0) probs.add("models.{s}.parallel.expert: {d} must divide tensor {d}", .{ m.id, l.expert, l.tensor });
    const t = if (l.tensor > 0) l.tensor else if (l.pipeline > 0) n / @max(l.pipeline, 1) else n;
    if (t == 0) return;
    var stage: u32 = 0;
    while (stage * t < n) : (stage += 1) {
        const group = members[stage * t .. @min(n, stage * t + t)];
        for (group[1..]) |x| {
            const a_idx = c.find(group[0]) orelse continue;
            const b_idx = c.find(x) orelse continue;
            if (c.nodes[a_idx].backend == c.nodes[b_idx].backend) continue;
            probs.add("models.{s}: {s} ({s}) and {s} ({s}) share a tensor-parallel group; Metal and CUDA kernels give different bits, so put each backend in its own pipeline stage or use \"mode\": \"disaggregated\"", .{ m.id, group[0], @tagName(c.nodes[a_idx].backend), x, @tagName(c.nodes[b_idx].backend) });
        }
    }
}

/// MCDMA's link-name rule: 1-20 characters of [A-Za-z0-9_-].
pub fn validName(name: []const u8) bool {
    if (name.len < 1 or name.len > 20) return false;
    for (name) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '_' and ch != '-') return false;
    return true;
}

fn str(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn num(v: ?Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

test {
    _ = @import("config_test.zig");
}
