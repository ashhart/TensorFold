//! The engine's scratch rows from its arena: the two mixers' rows and the dense MLP's share one region.
const E = @import("cuda_engine.zig");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const routed = @import("cuda_routed.zig");

pub const Mode = enum { measure, allocate };

/// What a carve has taken: bytes counted (`measure`) or arena allocations (`allocate`).
const Take = struct {
    e: *E.Engine,
    mode: Mode,
    total: usize = 0,

    const Mark = struct { total: usize, at: usize };

    pub fn alloc(t: *Take, bytes: usize) !u64 {
        t.total += E.aligned(bytes);
        return if (t.mode == .measure) 0 else try t.e.alloc(bytes);
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
    var t: Take = .{ .e = e, .mode = mode };
    const g = e.g;
    const R = E.max_rows;
    const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden); // y and mix: either mixer's rows
    inline for (.{ "h", "x", "r" }) |name| @field(e, name) = try t.alloc(R * g.hidden * 2);
    e.xs = try t.alloc(R * (@max(g.intermediate, wide) / 64) * 4);
    // a layer's mixer, then its MLP, pass only h, x, xs and r (outside it): their rows share one start
    const shared = t.mark();
    inline for (.{ "y", "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(R * wide * 2);
    inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(R * g.k_heads * geometry.dk * 2);
    e.qkv = try t.alloc(R * g.convWidth() * 2);
    inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(R * g.v_heads * 2);
    e.g_ = try t.alloc(R * g.v_heads * 4);
    const delta = t.rewind(shared);
    inline for (.{ "y", "mix" }) |_| _ = try t.alloc(R * wide * 2); // the attention mixer's y and mix: the same rows
    e.qraw = try t.alloc(R * 2 * g.attnWidth() * 2);
    e.qn = try t.alloc(R * g.attnWidth() * 2);
    inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(R * g.kv_heads * geometry.head_dim * 2);
    const attention = t.rewind(shared);
    inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(R * g.intermediate * 2);
    t.past(&.{ delta, attention });
    // the decode partials: verify rows only (a prompt chunk's attention runs cuda_prompt_attention's tiles)
    const partials = E.most_verify * g.q_heads * E.max_chunks;
    e.pm = try t.alloc(partials * 4);
    e.pl = try t.alloc(partials * 4);
    e.po = try t.alloc(partials * geometry.head_dim * 4);
    e.logits = try t.alloc(E.most_verify * g.vocab * 2);
    if (g.moe()) e.routed = try routed.Scratch.carve(&t, g, R);
    e.amax = try t.alloc(E.most_verify * E.argmaxSlices(g.vocab) * 8);
    e.picks = try t.alloc(E.most_verify * cuda.sample.Draws.row_scratch);
    return t.total;
}
