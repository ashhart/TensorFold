//! The site's change, a block of ranks a lesson (Nemotron's adapters.Sites for one site: layers.44's shared-expert down_proj).
//! Host memory throughout; a, b and tau may live in GPU-shared buffers the engine's forward reads (unified memory).
const std = @import("std");
const lm = @import("lw_math.zig");
const subspace = @import("../nemotron/subspace.zig");

pub const block = lm.block;
pub const max_rank = lm.max_rank;
pub const max_blocks = lm.max_blocks;
pub const candidates = lm.candidates;
pub const avoid_dims = lm.avoid_dims;

/// The change's live memory: a [max_rank, in], b [max_rank, out], tau [max_blocks], f32.
pub const Live = struct { a: []f32, b: []f32, tau: []f32 };

pub const Site = struct {
    in: usize,
    out: usize,
    a: []f32,
    b: []f32,
    tau: []f32,
    gb: []f32, // [block, out]: the open block's gradient
    mb: []f32, // [block, out]: its Adam moments
    vb: []f32,
    kept: []f32, // [3, block, out]: the open block's b and moments as its round began
    avoid: []f32, // [avoid_dims, in]: a sketch of the inputs the next block must leave alone
    seek: []f32, // [candidates, in]: a sketch of the fact's inputs, then the candidate directions it frames

    pub fn gate(s: *const Site, k: usize) *f32 {
        return &s.tau[k];
    }
};

pub const Sites = struct {
    gpa: std.mem.Allocator,
    list: []Site, // one site
    rank: usize = 0, // ranks in use, the open block last
    steps: u64 = 0, // the open block's Adam steps
    kept_steps: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, in: usize, out: usize, live: Live) !Sites {
        std.debug.assert(live.a.len >= max_rank * in and live.b.len >= max_rank * out and live.tau.len >= max_blocks);
        const list = try gpa.alloc(Site, 1);
        errdefer gpa.free(list);
        const s = &list[0];
        s.* = .{ .in = in, .out = out, .a = live.a, .b = live.b, .tau = live.tau, .gb = &.{}, .mb = &.{}, .vb = &.{}, .kept = &.{}, .avoid = &.{}, .seek = &.{} };
        const fields = [_]*[]f32{ &s.gb, &s.mb, &s.vb, &s.kept, &s.avoid, &s.seek };
        const sizes = [_]usize{ block * out, block * out, block * out, 3 * block * out, avoid_dims * in, candidates * in };
        var got: usize = 0;
        errdefer for (fields[0..got]) |f| gpa.free(f.*);
        for (fields, sizes) |f, n| {
            f.* = try gpa.alloc(f32, n);
            @memset(f.*, 0);
            got += 1;
        }
        @memset(live.a, 0);
        @memset(live.b, 0);
        @memset(live.tau, lm.shut);
        return .{ .gpa = gpa, .list = list };
    }

    pub fn deinit(s: *Sites) void {
        for (s.list) |*site| for ([_][]f32{ site.gb, site.mb, site.vb, site.kept, site.avoid, site.seek }) |f| s.gpa.free(f);
        s.gpa.free(s.list);
    }

    /// Committed blocks from a learned folder (rows [0, ranks) of a and b): always on.
    pub fn adopt(s: *Sites, a: []const f32, b: []const f32, ranks: usize) !void {
        if (ranks % block != 0 or ranks > max_rank) return error.BadLearnedRanks;
        const site = &s.list[0];
        @memcpy(site.a[0 .. ranks * site.in], a[0 .. ranks * site.in]);
        @memcpy(site.b[0 .. ranks * site.out], b[0 .. ranks * site.out]);
        for (0..ranks / block) |k| site.tau[k] = -std.math.inf(f32);
        s.rank = ranks;
    }

    pub fn lora(s: *const Sites) lm.Lora {
        const site = &s.list[0];
        return .{ .a = site.a, .b = site.b, .tau = site.tau, .rank = s.rank, .in = site.in, .out = site.out };
    }

    pub fn first(s: *const Sites) usize {
        return s.rank - block;
    }

    /// The candidate directions: the fact's sketch with what must stay taken out, made orthonormal.
    pub fn frame(s: *Sites) void {
        for (s.list) |*site| {
            subspace.orthonormal(site.avoid, avoid_dims, site.in);
            subspace.remove(site.seek, candidates, site.avoid, avoid_dims, site.in);
            subspace.orthonormal(site.seek, candidates, site.in);
        }
    }

    fn directions(site: *Site, at: usize, choice: []const f32, l: usize) void {
        const a = site.a[at * site.in ..][0 .. block * site.in];
        @memset(a, 0);
        for (0..block) |q| for (0..candidates) |j| {
            subspace.axpy(a[q * site.in ..][0..site.in], lm.a_norm * choice[(l * block + q) * candidates + j], site.seek[j * site.in ..][0..site.in]);
        };
    }

    /// A new block, shut, its directions `choice` [sites, block, candidates] of the candidates; b zero.
    pub fn open(s: *Sites, choice: []const f32) !void {
        if (s.rank + block > max_rank) return error.LearnedChangeFull;
        const at = s.rank;
        for (s.list, 0..) |*site, l| {
            directions(site, at, choice, l);
            site.gate(at / block).* = lm.shut;
            @memset(site.b[at * site.out ..][0 .. block * site.out], 0);
            for ([_][]f32{ site.gb, site.mb, site.vb }) |buf| @memset(buf, 0);
        }
        s.rank += block;
        s.steps = 0;
    }

    /// The open block's directions rebuilt from `choice`, its gate open on every row: a plain change.
    pub fn plain(s: *Sites, choice: []const f32) void {
        const at = s.first();
        for (s.list, 0..) |*site, l| {
            const g = site.gate(at / block);
            if (g.* >= lm.shut) continue;
            directions(site, at, choice, l);
            g.* = -std.math.inf(f32);
        }
    }

    /// The open block taken back out.
    pub fn close(s: *Sites) void {
        s.rank -= block;
        for (s.list) |*site| {
            @memset(site.b[s.rank * site.out ..][0 .. block * site.out], 0);
            @memset(site.a[s.rank * site.in ..][0 .. block * site.in], 0);
            site.gate(s.rank / block).* = lm.shut;
        }
    }

    /// One Adam step on the open block's outputs; the gradient is cleared.
    pub fn adam(s: *Sites) void {
        s.steps += 1;
        for (s.list) |*site| lm.adam(site.b[s.first() * site.out ..][0 .. block * site.out], site.gb, site.mb, site.vb, s.steps);
    }

    pub fn keep(s: *Sites) void {
        s.kept_steps = s.steps;
        for (s.list) |*site| {
            const n = block * site.out;
            @memcpy(site.kept[0..n], site.b[s.first() * site.out ..][0..n]);
            @memcpy(site.kept[n .. 2 * n], site.mb);
            @memcpy(site.kept[2 * n ..], site.vb);
        }
    }

    pub fn restore(s: *Sites) void {
        s.steps = s.kept_steps;
        for (s.list) |*site| {
            const n = block * site.out;
            @memcpy(site.b[s.first() * site.out ..][0..n], site.kept[0..n]);
            @memcpy(site.mb, site.kept[n .. 2 * n]);
            @memcpy(site.vb, site.kept[2 * n ..]);
            @memset(site.gb, 0);
        }
    }

    pub fn clear(s: *Sites) void {
        for (s.list) |*site| {
            @memset(site.gb, 0);
            @memset(site.avoid, 0);
            @memset(site.seek, 0);
        }
    }

    /// Rows [0, ranks) of a and b rounded to bf16 in place: what a saved sidecar holds, so RAM and a reload agree.
    pub fn roundSaved(s: *Sites, ranks: usize) void {
        const site = &s.list[0];
        for (site.a[0 .. ranks * site.in]) |*v| v.* = lm.bf(v.*);
        for (site.b[0 .. ranks * site.out]) |*v| v.* = lm.bf(v.*);
    }
};

test "open, keep, restore, close, adopt" {
    const gpa = std.testing.allocator;
    const in = 8;
    const out = 4;
    const a = try gpa.alloc(f32, max_rank * in);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, max_rank * out);
    defer gpa.free(b);
    var tau: [max_blocks]f32 = undefined;
    var s = try Sites.init(gpa, in, out, .{ .a = a, .b = b, .tau = &tau });
    defer s.deinit();
    for (s.list[0].seek, 0..) |*v, i| v.* = if (i % (in + 1) == 0 and i < candidates * in) 1 else 0; // seek rows: unit vectors (8 of them)
    var choice: [block * candidates]f32 = @splat(0);
    choice[0] = 1; // rank 0 = candidate 0
    try s.open(&choice);
    try std.testing.expectEqual(@as(usize, block), s.rank);
    try std.testing.expectApproxEqAbs(lm.a_norm, a[0], 1e-6);
    try std.testing.expectEqual(lm.shut, tau[0]);
    s.keep();
    s.list[0].gb[0] = 1;
    s.adam();
    try std.testing.expect(b[0] < 0);
    s.restore();
    try std.testing.expectEqual(@as(f32, 0), b[0]);
    tau[0] = 0.1;
    s.plain(&choice);
    try std.testing.expectEqual(-std.math.inf(f32), tau[0]);
    s.close();
    try std.testing.expectEqual(@as(usize, 0), s.rank);
    try std.testing.expectEqual(lm.shut, tau[0]);
    const la: [block * in]f32 = @splat(0.5);
    const lb: [block * out]f32 = @splat(0.25);
    try s.adopt(&la, &lb, block);
    try std.testing.expectEqual(-std.math.inf(f32), tau[0]);
    try std.testing.expectEqual(@as(f32, 0.25), b[block * out - 1]);
    try std.testing.expectError(error.BadLearnedRanks, s.adopt(&la, &lb, 3));
}
