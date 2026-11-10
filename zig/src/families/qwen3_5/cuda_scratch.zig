//! The engine's scratch rows from its arena: the two mixers' rows and the dense MLP's share one region.
const E = @import("cuda_engine.zig");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const routed = @import("cuda_routed.zig");

pub const Mode = enum { measure, allocate };
pub const WorkspacePlan = @import("cuda_workspace_plan.zig").WorkspacePlan;
const Trace = @import("cuda_workspace_plan.zig").Trace;

/// What a carve has taken: bytes counted (`measure`) or arena allocations (`allocate`).
const Take = struct {
    e: *E.Engine,
    mode: Mode,
    total: usize = 0,
    trace: ?*Trace = null,

    const Mark = struct { total: usize, at: usize };

    pub fn alloc(t: *Take, bytes: usize) !u64 {
        const before = t.e.arena_at;
        t.total += E.aligned(bytes);
        const address = if (t.mode == .measure) 0 else try t.e.alloc(bytes);
        if (t.trace) |trace| try trace.record(address, bytes, before, t.e.arena_at);
        return address;
    }

    fn put(t: *Take, comptime name: []const u8, bytes: usize) !void {
        @field(t.e, name) = try t.alloc(bytes);
        if (t.trace) |trace| trace.tagLast(name, true);
    }

    fn alias(t: *Take, comptime name: []const u8, bytes: usize) !void {
        _ = try t.alloc(bytes);
        if (t.trace) |trace| trace.tagLast(name, false);
    }

    fn mark(t: *const Take) Mark {
        return .{ .total = t.total, .at = t.e.arena_at };
    }

    /// Back to `start` for a region that never runs at once with the last; returns where the last ended.
    fn rewind(t: *Take, start: Mark) Mark {
        const end = t.mark();
        t.total = start.total;
        t.e.arena_at = start.at;
        return end;
    }

    /// Past the furthest of the shared regions' ends.
    fn past(t: *Take, ends: []const Mark) void {
        for (ends) |m| {
            t.total = @max(t.total, m.total);
            t.e.arena_at = @max(t.e.arena_at, m.at);
        }
    }
};

/// The scratch rows' bytes (`measure`: summed, nothing taken), or the rows themselves from the arena.
pub fn scratch(e: *E.Engine, mode: Mode) !usize {
    const plan = try WorkspacePlan.forRows(e.g, E.max_rows);
    return plan.carve(e, mode);
}

/// The existing carve sequence, shared by metadata planning and the default arena adapter.
pub fn layout(e: *E.Engine, mode: Mode, R: usize, trace: ?*Trace) !usize {
    var t: Take = .{ .e = e, .mode = mode, .trace = trace };
    const g = e.g;
    const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden); // y and mix: either mixer's rows
    inline for (.{ "h", "x", "r" }) |name| try t.put(name, R * g.hidden * 2);
    try t.put("xs", R * (@max(g.intermediate, wide) / 64) * 4);
    // a layer's mixer, then its MLP, pass only h, x, xs and r (outside it): their rows share one start
    const shared = t.mark();
    inline for (.{ "y", "mix", "z", "v" }) |name| try t.put(name, R * wide * 2);
    inline for (.{ "q", "k" }) |name| try t.put(name, R * g.k_heads * geometry.dk * 2);
    try t.put("qkv", R * g.convWidth() * 2);
    inline for (.{ "a", "b", "beta" }) |name| try t.put(name, R * g.v_heads * 2);
    try t.put("g_", R * g.v_heads * 4);
    const delta = t.rewind(shared);
    inline for (.{ "y", "mix" }) |name| try t.alias(name, R * wide * 2); // the attention mixer's y and mix: the same rows
    try t.put("qraw", R * 2 * g.attnWidth() * 2);
    try t.put("qn", R * g.attnWidth() * 2);
    inline for (.{ "kraw", "vraw", "kn" }) |name| try t.put(name, R * g.kv_heads * geometry.head_dim * 2);
    // Partials stay live through attention merge, then the MLP may reuse their bytes.
    const partials = E.most_verify * g.q_heads * E.max_chunks;
    try t.put("pm", partials * 4);
    try t.put("pl", partials * 4);
    try t.put("po", partials * geometry.head_dim * 4);
    const attention = t.rewind(shared);
    inline for (.{ "gate", "up", "act" }) |name| try t.put(name, R * g.intermediate * 2);
    t.past(&.{ delta, attention });
    try t.put("logits", E.most_verify * g.vocab * 2);
    if (g.moe()) {
        const first = if (trace) |v| v.n_allocations else 0;
        e.routed = try routed.Scratch.carve(&t, g, R);
        if (trace) |v| inline for (@typeInfo(routed.Scratch).@"struct".field_names) |name| {
            try v.bindRouted(name, @field(e.routed, name), first);
        };
    }
    try t.put("amax", E.most_verify * E.argmaxSlices(g.vocab) * 8);
    try t.put("picks", E.most_verify * cuda.sample.Draws.row_scratch);
    return t.total;
}

test {
    _ = @import("cuda_scratch_test.zig");
    _ = @import("cuda_workspace_plan_test.zig");
}
