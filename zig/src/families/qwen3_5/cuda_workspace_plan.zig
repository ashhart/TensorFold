//! CPU workspace metadata describes borrowed arena views; carving retains the existing allocation sequence.
const std = @import("std");
const E = @import("cuda_engine.zig");
const scratch = @import("cuda_scratch.zig");

pub const Owner = enum { engine_arena };
pub const Lifetime = enum { round, delta, attention, delta_attention, dense_mlp, routed };
pub const Name = enum {
    h,
    x,
    r,
    xs,
    y,
    mix,
    z,
    v,
    q,
    k,
    qkv,
    a,
    b,
    beta,
    g_,
    qraw,
    qn,
    kraw,
    vraw,
    kn,
    pm,
    pl,
    po,
    gate,
    up,
    act,
    logits,
    amax,
    picks,
    routed_logits,
    routed_picks,
    routed_wts,
    routed_members,
    routed_items,
    routed_counts,
    routed_rank,
    routed_hist,
    routed_act,
    routed_y,
};
const names = @typeInfo(Name).@"enum".field_names.len;
pub const alignment = 256;

pub const Region = struct {
    name: Name,
    offset: usize,
    bytes: usize,
    owner: Owner = .engine_arena,
    alignment: usize = alignment,
    lifetime: Lifetime,
};
pub const Allocation = struct {
    name: ?Name = null,
    offset: usize,
    bytes: usize,
    cursor_before: usize,
    cursor_after: usize,
};
pub const Budget = struct { bytes: usize, extent: usize };

fn lifetime(name: Name) Lifetime {
    return switch (name) {
        .h, .x, .r, .xs, .logits, .amax, .picks => .round,
        .y, .mix => .delta_attention,
        .z, .v, .q, .k, .qkv, .a, .b, .beta, .g_ => .delta,
        .qraw, .qn, .kraw, .vraw, .kn, .pm, .pl, .po => .attention,
        .gate, .up, .act => .dense_mlp,
        else => .routed,
    };
}

pub const Trace = struct {
    regions: [names]Region = undefined,
    present: std.EnumSet(Name) = .empty,
    sequence: [48]Allocation = undefined,
    n_allocations: usize = 0,

    pub fn record(t: *Trace, address: u64, bytes: usize, before: usize, after: usize) !void {
        if (t.n_allocations == t.sequence.len) return error.WorkspaceAllocationLimit;
        t.sequence[t.n_allocations] = .{ .offset = @intCast(address), .bytes = bytes, .cursor_before = before, .cursor_after = after };
        t.n_allocations += 1;
    }

    fn bind(t: *Trace, name: Name, index: usize, define: bool) void {
        const entry = &t.sequence[index];
        entry.name = name;
        if (!define) return;
        t.regions[@backingInt(name)] = .{ .name = name, .offset = entry.offset, .bytes = entry.bytes, .lifetime = lifetime(name) };
        t.present.insert(name);
    }

    pub fn tagLast(t: *Trace, comptime name: []const u8, define: bool) void {
        t.bind(@field(Name, name), t.n_allocations - 1, define);
    }

    pub fn bindRouted(t: *Trace, comptime field: []const u8, address: u64, first: usize) !void {
        for (t.sequence[first..t.n_allocations], first..) |entry, i| if (entry.offset == address) {
            t.bind(@field(Name, "routed_" ++ field), i, true);
            return;
        };
        return error.WorkspaceRegionMissing;
    }
};

const Shape = struct {
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    k_heads: usize,
    v_heads: usize,
    q_heads: usize,
    kv_heads: usize,
    value_width: usize,
    attention_width: usize,
    conv_width: usize,
    experts: usize,
    slots: usize,

    fn of(g: E.Geometry) Shape {
        return .{
            .hidden = g.hidden,
            .intermediate = g.intermediate,
            .vocab = g.vocab,
            .k_heads = g.k_heads,
            .v_heads = g.v_heads,
            .q_heads = g.q_heads,
            .kv_heads = g.kv_heads,
            .value_width = g.valueWidth(),
            .attention_width = g.attnWidth(),
            .conv_width = g.convWidth(),
            .experts = g.experts,
            .slots = if (g.moe()) g.slots() else 0,
        };
    }
};

pub const WorkspacePlan = struct {
    rows: usize,
    owner: Owner = .engine_arena,
    alignment: usize = alignment,
    budget: Budget,
    shape: Shape,
    trace: Trace,

    /// A symbolic address-zero arena computes metadata only; no device, heap allocation or driver is used.
    pub fn forRows(g: E.Geometry, rows: usize) !WorkspacePlan {
        if (rows == 0 or rows > E.max_rows) return error.InvalidWorkspaceRows;
        var e: E.Engine = undefined;
        e.g = g;
        e.arena_at = 0;
        e.arena = .{ .d = undefined, .ptr = 0, .len = std.math.maxInt(usize), .borrowed = true };
        var trace: Trace = .{};
        const bytes = try scratch.layout(&e, .allocate, rows, &trace);
        return .{ .rows = rows, .budget = .{ .bytes = bytes, .extent = e.arena_at }, .shape = Shape.of(g), .trace = trace };
    }

    pub fn region(p: *const WorkspacePlan, name: Name) ?Region {
        return if (p.trace.present.contains(name)) p.trace.regions[@backingInt(name)] else null;
    }

    pub fn allocations(p: *const WorkspacePlan) []const Allocation {
        return p.trace.sequence[0..p.trace.n_allocations];
    }

    /// Region offsets are relative to the aligned start, while this includes its padding from `start`.
    pub fn requiredAt(p: *const WorkspacePlan, start: usize) !usize {
        if (p.alignment != alignment) return error.InvalidWorkspaceAlignment;
        const aligned = (try std.math.add(usize, start, alignment - 1)) & ~@as(usize, alignment - 1);
        return std.math.add(usize, aligned - start, p.budget.extent);
    }

    /// Allocation and failure timing stay in the old arena carver, including rewinds and partial assignments.
    pub fn carve(p: *const WorkspacePlan, e: *E.Engine, mode: scratch.Mode) !usize {
        if (p.rows == 0 or p.rows > E.max_rows) return error.InvalidWorkspaceRows;
        if (p.alignment != alignment) return error.InvalidWorkspaceAlignment;
        if (!std.meta.eql(p.shape, Shape.of(e.g))) return error.WorkspaceShapeChanged;
        const bytes = try scratch.layout(e, mode, p.rows, null);
        if (bytes != p.budget.bytes) return error.WorkspaceBudgetChanged;
        return bytes;
    }
};
