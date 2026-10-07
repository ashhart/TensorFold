//! What each rank must hold under a plan (weights by kind, KV and state, activations, the drafter) against its GPU's limit.
const std = @import("std");
const node = @import("node.zig");
const checkpoint = @import("checkpoint.zig");
const model = @import("model.zig");
const roles = @import("roles.zig");
const plan_mod = @import("plan.zig");

const Plan = plan_mod.Plan;
const mib: u64 = 1 << 20;

pub const Category = enum { attention, shared, dense, latent, router, experts, embed, head, norms, vision, draft, other };

pub fn category(r: roles.Role) Category {
    return switch (r) {
        .attn_col, .attn_row, .attn_rep => .attention,
        .shared_col, .shared_row => .shared,
        .dense_col, .dense_row => .dense,
        .latent_in, .latent_out, .latent_norm => .latent,
        .router => .router,
        .expert => .experts,
        .embed => .embed,
        .head => .head,
        .norm, .final_norm => .norms,
        .vision => .vision,
        .draft => .draft,
        .other => .other,
    };
}

pub const Need = struct {
    weights: [@typeInfo(Category).@"enum".field_names.len]u64 = @splat(0),
    kv: u64 = 0,
    state: u64 = 0,
    activations: u64 = 0,
    drafter: u64 = 0,

    pub fn of(n: *const Need, c: Category) u64 {
        return n.weights[@backingInt(c)];
    }

    pub fn weightBytes(n: *const Need) u64 {
        var sum: u64 = 0;
        for (n.weights) |w| sum += w;
        return sum;
    }

    pub fn total(n: *const Need) u64 {
        return n.weightBytes() + n.kv + n.state + n.activations + n.drafter;
    }
};

/// Every rank's need: weights, caches for `streams` x `context`, and buffers for a round of rows plus a prompt chunk.
pub fn needs(a: std.mem.Allocator, p: *const Plan, c: checkpoint.Checkpoint, s: *const model.Shape) ![]Need {
    const out = try a.alloc(Need, p.ranks());
    @memset(out, .{});
    for (c.tensors, 0..) |t, i| {
        const mtp = if (t.class.layer) |l| l >= s.layers else false;
        const cat = @backingInt(if (mtp) Category.draft else category(t.class.role));
        for (out, 0..) |*n, r| n.weights[cat] += p.bytesOn(t, i, @intCast(r));
    }
    for (out, 0..) |*n, r| {
        const held = caches(p, s, @intCast(r));
        n.kv = held.kv;
        n.state = held.state;
        n.activations = buffers(p, s, @intCast(r));
    }
    const o = p.opts;
    out[0].drafter = o.drafter_bytes;
    if (o.drafter_bytes > 0 and o.drafter_head) {
        for (c.tensors, 0..) |t, i| if (t.class.role == .head) {
            out[0].drafter += t.bytes - p.bytesOn(t, i, 0);
        };
    }
    return out;
}

/// Rank `r`'s caches: MLA latent and indexer keys (all streams' by heads, its own by streams), GQA and linear by heads.
pub fn caches(p: *const Plan, s: *const model.Shape, r: u32) struct { kv: u64, state: u64 } {
    const o = p.opts;
    const st = p.stages[p.stageOfRank(r)];
    const mine = st.slices[r - st.first];
    const held: u64 = if (o.mla == .streams) std.math.divCeil(u64, o.streams, st.tensor) catch unreachable else o.streams;
    var kv: u64 = 0;
    var state: u64 = 0;
    for (st.layers.begin..st.layers.end) |l| {
        const layer: u32 = @intCast(l);
        switch (s.kind(layer)) {
            .mla => kv += (s.kvPerToken(layer) + s.indexPerToken(layer)) * o.context * held,
            .gqa => kv += plan_mod.share(s.kvPerToken(layer) * o.context * o.streams, mine, o.slices),
            .linear => state += plan_mod.share(s.statePerStream(layer) * (o.streams + o.rows), mine, o.slices),
        }
    }
    return .{ .kv = kv, .state = state };
}

/// Round buffers for decode rows plus a prompt chunk: activations, expert intermediates, the exchange windows, logits.
pub fn buffers(p: *const Plan, s: *const model.Shape, r: u32) u64 {
    const o = p.opts;
    const st = p.stages[p.stageOfRank(r)];
    const rows: u64 = o.rows + o.chunk;
    const window = 2 * @as(u64, st.tensor) * rows * s.hidden * 4;
    var bytes = rows * (s.hidden * 16 + s.latent * 16 + @as(u64, s.top_k) * s.moe_inter * 8 / st.tensor) + window;
    if (r == 0) bytes += @as(u64, o.rows) * s.vocab * 4;
    return bytes;
}

pub const Fit = struct {
    rank: u32,
    need: u64,
    limit: u64,
    ok: bool,
    /// The iogpu.wired_limit_mb that would fit this rank, when a raise is needed.
    raise_mb: u64 = 0,
    /// A raise within physical memory less the OS floor can fit it.
    possible: bool = true,
};

pub fn fits(a: std.mem.Allocator, p: *const Plan, ns: []const Need) ![]Fit {
    const out = try a.alloc(Fit, ns.len);
    for (ns, p.nodes, 0..) |n, inv, r| {
        const need = n.total() + p.opts.margin;
        out[r] = .{ .rank = @intCast(r), .need = need, .limit = inv.gpu_limit, .ok = need <= inv.gpu_limit };
        if (out[r].ok) continue;
        out[r].raise_mb = std.math.divCeil(u64, need, mib) catch unreachable;
        out[r].possible = inv.backend == .metal and inv.unified and need + p.opts.os_floor <= inv.memory;
    }
    return out;
}

pub fn allFit(fs: []const Fit) bool {
    for (fs) |f| if (!f.ok) return false;
    return true;
}

/// One line per rank that does not fit, naming the wired-limit raise the owner must make, or why none can help.
pub fn refusal(w: *std.Io.Writer, p: *const Plan, fs: []const Fit) !void {
    for (fs) |f| {
        if (f.ok) continue;
        const inv = p.nodes[f.rank];
        try w.print("refused: node {s} (rank {d}) needs {d:.1} GiB but its GPU may hold {d:.1} GiB", .{ inv.name.str(), f.rank, gb(f.need), gb(f.limit) });
        if (inv.wired_limit_mb == 0 and inv.backend == .metal) try w.writeAll(" (iogpu.wired_limit_mb=0, macOS's default)");
        if (f.possible) {
            try w.print(". The owner must raise it: sudo sysctl iogpu.wired_limit_mb={d} on {s} (it resets at reboot).\n", .{ f.raise_mb, inv.name.str() });
        } else {
            try w.print(". No wired-limit raise fits it ({d:.0} GiB physical, {d:.0} GiB kept for the OS): add nodes, or lower context or streams.\n", .{ gb(inv.memory), gb(p.opts.os_floor) });
        }
    }
}

pub fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(node.gib));
}
