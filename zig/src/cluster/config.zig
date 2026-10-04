//! cluster.json (schema tensorfold-cluster/1): the nodes, how to reach them, and per model its nodes, layout and budgets.
const std = @import("std");
const node = @import("node.zig");
const plan = @import("plan.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

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
    models: []Model,

    pub fn find(c: Config, name: []const u8) ?usize {
        for (c.nodes, 0..) |n, i| if (std.mem.eql(u8, n.name, name)) return i;
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
    }
    for (c.models) |m| checkModel(c, m, probs);
}

fn checkModel(c: Config, m: Model, probs: *Problems) void {
    const members = m.nodes;
    for (members) |n| if (c.find(n) == null) probs.add("models.{s}.nodes: \"{s}\" is not a node", .{ m.id, n });
    for (m.paths) |p| if (c.find(p[0]) == null) probs.add("models.{s}.path: \"{s}\" is not a node", .{ m.id, p[0] });
    if (m.path == null and m.paths.len > 0) for (members) |n| {
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
