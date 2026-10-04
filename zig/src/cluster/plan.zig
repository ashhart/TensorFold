//! The placement planner: every tensor to ranks under one layout (pipeline x tensor, experts spread or split), deterministically.
const std = @import("std");
const node = @import("node.zig");
const checkpoint = @import("checkpoint.zig");
const model = @import("model.zig");
const roles = @import("roles.zig");
const canon = @import("canon.zig");

const Allocator = std.mem.Allocator;
const Range = canon.Range;

/// Converged: one lane round spans every node, prompt rows and decode rows together. Disaggregated: prefill and decode sets.
pub const Mode = enum { converged, disaggregated };

/// Degrees; 0 means the planner chooses. nodes = pipeline x tensor; expert divides tensor (1: every expert split by TP).
pub const Layout = struct {
    tensor: u32 = 0,
    pipeline: u32 = 0,
    expert: u32 = 0,
};

pub const LatentIn = enum { replicate, column };

/// MLA attention by heads (the latent cache replicated on every rank) or by streams (weights replicated, caches split).
pub const MlaSplit = enum(u8) { heads, streams };

pub const Options = struct {
    mode: Mode = .converged,
    layout: Layout = .{},
    /// Canonical reduction slices for every tensor-parallel sum (a power of two dividing the heads).
    slices: u32 = 8,
    /// Canonical expert groups for the routed sum (a power of two dividing the experts).
    groups: u32 = 8,
    /// Column split costs one more step a MoE layer and saves every rank three quarters of the latent input reads.
    latent_in: LatentIn = .column,
    mla: MlaSplit = .heads,
    streams: u32 = 1,
    context: u64 = 131072,
    rows: u32 = 16,
    chunk: u32 = 512,
    drafter_bytes: u64 = 0,
    /// The drafter reads the target's whole LM head on the leader.
    drafter_head: bool = true,
    vision: bool = false,
    margin: u64 = 4 * node.gib,
    os_floor: u64 = 16 * node.gib,
};

pub const Kind = enum(u3) { replicate, column, row, expert, single, skip };

/// One tensor's place: its stage, and for experts the EP block that holds it, for singles the rank.
pub const Place = packed struct(u32) { kind: Kind, stage: u8, owner: u12, _: u9 = 0 };

pub const Stage = struct {
    layers: Range,
    /// Ranks [first, first + tensor).
    first: u32,
    tensor: u32,
    expert: u32,
    slices: []Range,
    /// Expert groups per EP block.
    groups: []Range,
};

pub const Plan = struct {
    opts: Options,
    nodes: []const node.Inventory,
    layout: Layout,
    stages: []Stage,
    stage_of: []u8,
    place: []Place,
    digest: u64,

    pub fn ranks(p: *const Plan) u32 {
        return @intCast(p.nodes.len);
    }

    /// The rank that runs a stream's MLA attention when it is split by streams (the leader picks slots to balance).
    pub fn streamOwner(p: *const Plan, stage: u32, slot: u32) u32 {
        const st = p.stages[stage];
        return st.first + slot % st.tensor;
    }

    pub fn stageOfRank(p: *const Plan, r: u32) u32 {
        for (p.stages, 0..) |s, i| if (r >= s.first and r < s.first + s.tensor) return @intCast(i);
        unreachable;
    }

    /// The bytes of tensor `t` that rank `r` holds.
    pub fn bytesOn(p: *const Plan, t: checkpoint.Tensor, i: usize, r: u32) u64 {
        const pl = p.place[i];
        const st = p.stages[pl.stage];
        if (pl.kind == .skip) return 0;
        if (pl.kind == .single) return if (pl.owner == r) t.bytes else 0;
        if (r < st.first or r >= st.first + st.tensor) return 0;
        const local = r - st.first;
        return switch (pl.kind) {
            .replicate => t.bytes,
            .column, .row => share(t.bytes, st.slices[local], p.opts.slices),
            .expert => blk: {
                const per = st.tensor / st.expert;
                if (local / per != pl.owner) break :blk 0;
                break :blk if (per == 1) t.bytes else t.bytes / per;
            },
            else => unreachable,
        };
    }
};

pub fn share(bytes: u64, r: Range, units: u32) u64 {
    return @intCast(@as(u128, bytes) * r.len() / units);
}

pub const Error = error{ NoNodes, BadLayout, BadSlices, BadGroups, TooManyStages };

/// The layout the planner chooses when a degree is 0: one stage with tensor parallelism over every node, experts spread over all.
pub fn resolve(s: *const model.Shape, n: u32, want: Layout) Error!Layout {
    if (n == 0) return error.NoNodes;
    var l = want;
    if (l.pipeline == 0 and l.tensor == 0) l.pipeline = 1;
    if (l.pipeline == 0) l.pipeline = n / l.tensor;
    if (l.tensor == 0) l.tensor = n / l.pipeline;
    if (l.pipeline * l.tensor != n) return error.BadLayout;
    if (l.expert == 0) l.expert = if (s.experts > 0) l.tensor else 1;
    if (l.tensor % l.expert != 0 or l.pipeline > s.layers or l.pipeline > 255) return error.BadLayout;
    return l;
}

/// Plan `c` over `nodes` (in id order; rank = index, rank 0 leads).
pub fn plan(a: Allocator, c: checkpoint.Checkpoint, s: *const model.Shape, nodes: []const node.Inventory, opts: Options) !Plan {
    if (!std.math.isPowerOfTwo(opts.slices) or (s.heads > 0 and s.heads % opts.slices != 0)) return error.BadSlices;
    if (s.experts > 0 and (!std.math.isPowerOfTwo(opts.groups) or s.experts % opts.groups != 0)) return error.BadGroups;
    const n: u32 = @intCast(nodes.len);
    const layout = try resolve(s, n, opts.layout);
    var p: Plan = .{ .opts = opts, .nodes = nodes, .layout = layout, .stages = try a.alloc(Stage, layout.pipeline), .stage_of = try a.alloc(u8, s.layers), .place = try a.alloc(Place, c.tensors.len), .digest = 0 };
    try stages(a, &p, c, s);
    for (c.tensors, p.place) |t, *pl| pl.* = placeOf(&p, s, t);
    p.digest = try digestOf(a, &p, c);
    return p;
}

fn weight(inv: node.Inventory) u64 {
    return if (inv.bandwidth > 0) inv.bandwidth else @max(inv.gpu_limit, 1);
}

fn stages(a: Allocator, p: *Plan, c: checkpoint.Checkpoint, s: *const model.Shape) !void {
    const l = p.layout;
    const layer_bytes = try a.alloc(u64, s.layers);
    @memset(layer_bytes, 0);
    for (c.tensors) |t| if (t.class.layer) |x| if (x < s.layers) {
        layer_bytes[x] += t.bytes;
    };
    const stage_weight = try a.alloc(u64, l.pipeline);
    for (stage_weight, 0..) |*w, i| {
        w.* = 0;
        for (p.nodes[i * l.tensor ..][0..l.tensor]) |inv| w.* += @max(inv.gpu_limit, 1) / node.gib;
    }
    const layer_ranges = try a.alloc(Range, l.pipeline);
    splitLayers(layer_bytes, stage_weight, layer_ranges);
    for (p.stages, 0..) |*st, i| {
        st.* = .{ .layers = layer_ranges[i], .first = @intCast(i * l.tensor), .tensor = l.tensor, .expert = l.expert, .slices = try a.alloc(Range, l.tensor), .groups = try a.alloc(Range, l.expert) };
        var ws: [256]u64 = undefined;
        for (p.nodes[st.first..][0..l.tensor], 0..) |inv, k| ws[k] = weight(inv);
        canon.partition(p.opts.slices, ws[0..l.tensor], st.slices);
        const per = l.tensor / l.expert;
        var bw: [256]u64 = undefined;
        for (0..l.expert) |b| {
            bw[b] = 0;
            for (ws[b * per ..][0..per]) |w| bw[b] += w;
        }
        if (s.experts > 0) canon.partition(p.opts.groups, bw[0..l.expert], st.groups) else @memset(st.groups, .{});
        for (st.layers.begin..st.layers.end) |x| p.stage_of[x] = @intCast(i);
    }
}

/// Contiguous layer ranges with each stage's bytes in proportion to its weight (greedy on the running sum, deterministic).
fn splitLayers(bytes: []const u64, weights: []const u64, out: []Range) void {
    var total: u128 = 0;
    for (bytes) |b| total += b;
    var wsum: u128 = 0;
    for (weights) |w| wsum += w;
    var at: u32 = 0;
    var run: u128 = 0;
    var wrun: u128 = 0;
    for (out, 0..) |*r, i| {
        wrun += weights[i];
        const target = if (wsum == 0) total else total * wrun / wsum;
        const left_stages: u32 = @intCast(out.len - i - 1);
        r.begin = at;
        while (at < bytes.len - left_stages and (run + bytes[at] <= target or at == r.begin or i == out.len - 1)) {
            run += bytes[at];
            at += 1;
        }
        r.end = at;
    }
}

fn placeOf(p: *const Plan, s: *const model.Shape, t: checkpoint.Tensor) Place {
    const role = t.class.role;
    const layer = t.class.layer;
    const stage: u8 = if (layer) |x| (if (x < s.layers) p.stage_of[x] else 0) else switch (role) {
        .head, .final_norm => @intCast(p.stages.len - 1),
        else => 0,
    };
    if (layer) |x| if (x >= s.layers) return .{ .kind = .single, .stage = 0, .owner = 0 };
    const by_streams = p.opts.mla == .streams and layer != null and s.kind(layer.?) == .mla;
    switch (role) {
        .attn_col, .attn_row, .attn_rep => if (by_streams) return .{ .kind = .replicate, .stage = stage, .owner = 0 },
        .vision => return .{ .kind = if (p.opts.vision) .single else .skip, .stage = 0, .owner = 0 },
        .draft => return .{ .kind = .single, .stage = 0, .owner = 0 },
        .expert => {
            const e = t.class.expert orelse return .{ .kind = .replicate, .stage = stage, .owner = 0 };
            const st = p.stages[stage];
            const group: u32 = @intCast(@as(u64, e) * p.opts.groups / s.experts);
            return .{ .kind = .expert, .stage = stage, .owner = @intCast(canon.owner(st.groups, group)) };
        },
        .latent_in => return .{ .kind = if (p.opts.latent_in == .column) .column else .replicate, .stage = stage, .owner = 0 },
        .latent_out => return .{ .kind = .row, .stage = stage, .owner = 0 },
        else => {},
    }
    const kind: Kind = switch (roles.split(role)) {
        .column => .column,
        .row => .row,
        else => .replicate,
    };
    if (kind == .replicate or p.stages[stage].tensor == 1) return .{ .kind = .replicate, .stage = stage, .owner = 0 };
    const dim = if (kind == .row) t.shape[t.rank -| 1] else t.shape[0];
    if (t.rank == 0 or dim % p.opts.slices != 0) return .{ .kind = .replicate, .stage = stage, .owner = 0 };
    return .{ .kind = kind, .stage = stage, .owner = 0 };
}

/// Hash the layout and every placement in canonical order (file, then offset), so tensor order never matters.
fn digestOf(a: Allocator, p: *const Plan, c: checkpoint.Checkpoint) !u64 {
    var h = std.hash.Wyhash.init(0x504c414e);
    h.update(std.mem.asBytes(&p.layout));
    h.update(std.mem.asBytes(&p.opts.slices));
    h.update(std.mem.asBytes(&p.opts.groups));
    h.update(std.mem.asBytes(&p.opts.mode));
    h.update(std.mem.asBytes(&p.opts.latent_in));
    h.update(std.mem.asBytes(&p.opts.mla));
    for (p.nodes) |n| h.update(std.mem.asBytes(&n.id));
    for (p.stages) |st| {
        h.update(std.mem.asBytes(&st.layers));
        h.update(std.mem.sliceAsBytes(st.slices));
        h.update(std.mem.sliceAsBytes(st.groups));
    }
    const order = try a.alloc(u32, c.tensors.len);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, c, struct {
        fn less(ck: checkpoint.Checkpoint, x: u32, y: u32) bool {
            const tx = ck.tensors[x];
            const ty = ck.tensors[y];
            const fx = ck.files[tx.file].name;
            const fy = ck.files[ty.file].name;
            return switch (std.mem.order(u8, fx, fy)) {
                .lt => true,
                .gt => false,
                .eq => tx.start < ty.start,
            };
        }
    }.less);
    for (order) |i| {
        const t = c.tensors[i];
        h.update(c.files[t.file].name);
        h.update(std.mem.asBytes(&t.start));
        h.update(std.mem.asBytes(&p.place[i]));
    }
    return h.final();
}

test {
    _ = @import("plan_test.zig");
}
