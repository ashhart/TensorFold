//! Each layer's learned change on CUDA, as adapters.zig on Metal, in host memory the GPU maps (one pool on a GB10).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const train_ops = @import("cuda_train_ops.zig");
const subspace = @import("subspace.zig");
const shape = @import("learn_shape.zig");
const route = @import("lesson_route.zig");

pub const block = shape.block;
pub const max_rank = shape.max_rank;
pub const max_blocks = shape.max_blocks;
const room = 8; // blocks a layer first has room for
pub const scale = shape.scale;
pub const avoid_dims = shape.avoid_dims;
pub const candidates = shape.candidates;
pub const hyper = shape.hyper;
const a_norm = shape.a_norm;
pub const unit = shape.unit;

/// Host memory the GPU maps: kernels read it at `dev`, the host through `slice`, as a Metal shared buffer's.
pub const Mapped = struct {
    host: cuda.HostBuffer,
    dev: u64,

    pub fn init(d: *const cuda.Driver, bytes: usize) !Mapped {
        var h = try cuda.HostBuffer.allocMapped(d, @max(bytes, 16));
        errdefer h.free();
        @memset(h.bytes, 0);
        return .{ .host = h, .dev = try h.device() };
    }

    pub fn deinit(m: *Mapped) void {
        m.host.free();
    }

    pub fn slice(m: Mapped, comptime T: type, n: usize) []T {
        return m.host.slice(T)[0..n];
    }
};

/// One layer's change after its output projection: y += scale (x a^T) b, f32 a [ranks, in] and b [ranks, out].
pub const Site = struct {
    layer: usize,
    in: usize,
    out: usize,
    a: Mapped, // every block's input directions, fixed when its lesson begins
    b: Mapped, // every block's outputs; only the open block's still learn
    tau: Mapped, // [cap]: each block's router offset here, its share times the other prompts' mean score (0: unread)
    gb: Mapped, // [block, out]: the open block's gradient
    mb: Mapped, // [block, out]: its Adam moments
    vb: Mapped,
    kept: Mapped, // [3, block, out]: the open block's b and moments as its round began
    avoid: Mapped, // [avoid_dims, in]: a sketch of the inputs the next block must leave alone
    seek: Mapped, // [candidates, in]: a sketch of the fact's inputs, then the candidate directions it frames

    pub fn adapter(s: *const Site, sites: *const Sites) weights.Adapter {
        return .{ .a = s.a.dev, .b = s.b.dev, .tau = s.tau.dev, .xa = sites.xa.dev, .xn = sites.xn.dev, .gates = sites.gates.dev, .need = sites.need.dev, .latch = sites.latch.dev, .score = sites.score.dev, .whole = sites.whole, .in = s.in, .out = s.out, .rank = sites.rank, .scale = scale, .unit = unit };
    }

    /// Block k's router offset.
    pub fn gate(s: *const Site, k: usize) *f32 {
        return &s.tau.slice(f32, k + 1)[k];
    }

    fn fields(s: *Site) [9]*Mapped {
        return .{ &s.a, &s.b, &s.tau, &s.gb, &s.mb, &s.vb, &s.kept, &s.avoid, &s.seek };
    }
};

pub const inWidth = shape.inWidth;
pub const maxIn = shape.maxIn;

pub const Sites = struct {
    gpa: std.mem.Allocator,
    list: []Site,
    xa: Mapped, // [rows, max_rank]: a forward's scratch, used by one site at a time
    xn: Mapped, // [rows]: each row's squared input length, the same way
    gates: Mapped, // [rows, max_blocks]: which blocks are open on each row, the same way
    need: Mapped, // f32 [max_blocks]: the router score each lesson needs at a decision row to act on the reply (0: never)
    latch: Mapped, // u32 [max_slots, max_blocks]: each sequence's lessons acting on its reply, a flag a block
    score: Mapped, // f32 [max_slots, max_blocks]: each sequence's router scores at its decision row, until they are latched
    rank: usize = 0, // ranks in use, the open block last
    saved: usize = 0, // blocks written beside the shards, a file each
    cap: usize, // blocks every layer has room for, doubled when a lesson needs more (max_blocks at most)
    whole: bool = false, // lessons edit the weights: every one acts on every row, no router deciding
    d: *const cuda.Driver,
    steps: u64 = 0, // the open block's Adam steps
    kept_steps: u64 = 0,

    /// Room for `blocks` blocks per layer (`room` at least), none in use: the forward is unchanged until one opens.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: Config, rows: usize, blocks: usize) !Sites {
        const cap = @max(blocks, room);
        const list = try gpa.alloc(Site, c.layers);
        var made: usize = 0;
        errdefer {
            for (list[0..made]) |*s| free(s);
            gpa.free(list);
        }
        for (list, 0..) |*s, i| {
            const in = inWidth(c, i);
            const out = c.hidden;
            s.* = .{ .layer = i, .in = in, .out = out, .a = undefined, .b = undefined, .tau = undefined, .gb = undefined, .mb = undefined, .vb = undefined, .kept = undefined, .avoid = undefined, .seek = undefined };
            const sizes = [_]usize{ cap * block * in, cap * block * out, cap, block * out, block * out, block * out, 3 * block * out, avoid_dims * in, candidates * in };
            const f = s.fields();
            var got: usize = 0;
            errdefer for (f[0..got]) |m| m.deinit();
            for (f, sizes) |m, n| {
                m.* = try Mapped.init(d, n * 4);
                got += 1;
            }
            made += 1;
        }
        var xa = try Mapped.init(d, rows * max_rank * 4);
        errdefer xa.deinit();
        var xn = try Mapped.init(d, rows * 4);
        errdefer xn.deinit();
        var gates = try Mapped.init(d, rows * max_blocks * 4);
        errdefer gates.deinit();
        var need = try Mapped.init(d, max_blocks * 4);
        errdefer need.deinit();
        var latch = try Mapped.init(d, route.max_slots * max_blocks * 4);
        errdefer latch.deinit();
        const score = try Mapped.init(d, route.max_slots * max_blocks * 4);
        return .{ .gpa = gpa, .list = list, .xa = xa, .xn = xn, .gates = gates, .need = need, .latch = latch, .score = score, .cap = cap, .d = d };
    }

    pub fn deinit(s: *Sites) void {
        for (s.list) |*site| free(site);
        s.gpa.free(s.list);
        inline for (.{ "xa", "xn", "gates", "need", "latch", "score" }) |f| @field(s, f).deinit();
    }

    fn free(s: *Site) void {
        for (s.fields()) |m| m.deinit();
    }

    /// Every layer's forward applies the blocks in use from here on (none: no change at all).
    pub fn attach(s: *Sites, w: *weights.Weights, on: bool) void {
        for (s.list) |*site| {
            const ad: ?weights.Adapter = if (on and s.rank > 0) site.adapter(s) else null;
            const b = &w.blocks[site.layer];
            switch (b.kind) {
                .moe => b.moe.adapter = ad,
                .mamba => b.mamba.adapter = ad,
                .attention => b.attn.adapter = ad,
            }
        }
    }

    /// Each block's router score needed at a decision row to act on the reply (0: never), as the forward reads them.
    pub fn needs(s: *const Sites) []f32 {
        return s.need.slice(f32, max_blocks);
    }

    /// The first rank of the open block.
    pub fn first(s: *const Sites) usize {
        return s.rank - block;
    }

    /// Every layer's candidate directions: the fact's sketch with what must stay taken out, made orthonormal.
    pub fn frame(s: *Sites) void {
        var threads: [16]?std.Thread = @splat(null);
        const n = @min(threads.len, s.list.len);
        for (threads[0..n], 0..) |*t, k| t.* = std.Thread.spawn(.{}, frameShare, .{ s.list, k, n }) catch null;
        for (threads[0..n], 0..) |t, k| if (t) |th| th.join() else frameShare(s.list, k, n);
    }

    /// Room for `blocks` blocks at every layer, buffers doubled as needed; the forward reads them once attached again.
    fn grow(s: *Sites, blocks: usize) !void {
        if (blocks <= s.cap) return;
        if (blocks > max_blocks) return error.LearnedChangeFull;
        var cap = s.cap;
        while (cap < blocks) cap *= 2;
        cap = @min(cap, max_blocks);
        const fresh = try s.gpa.alloc([3]Mapped, s.list.len);
        defer s.gpa.free(fresh);
        var made: usize = 0;
        errdefer for (fresh[0..made]) |*f| for (f) |*x| x.deinit();
        for (s.list, fresh) |*site, *f| {
            const per = [3]usize{ block * site.in, block * site.out, 1 }; // a block's floats in a, b and tau
            const old = [3]Mapped{ site.a, site.b, site.tau };
            var got: usize = 0;
            errdefer for (f[0..got]) |*x| x.deinit();
            for (f, per, old) |*x, n, o| {
                x.* = try Mapped.init(s.d, cap * n * 4);
                got += 1;
                @memcpy(x.slice(f32, s.cap * n), o.slice(f32, s.cap * n));
                @memset(x.slice(f32, cap * n)[s.cap * n ..], 0);
            }
            made += 1;
        }
        for (s.list, fresh) |*site, f| {
            for ([_]*Mapped{ &site.a, &site.b, &site.tau }) |x| x.deinit();
            site.a = f[0];
            site.b = f[1];
            site.tau = f[2];
        }
        s.cap = cap;
    }

    /// A new block per layer, directions `choice` [layers, block, candidates], b and need zero: it acts on no reply.
    pub fn open(s: *Sites, choice: []const f32) !void {
        try s.grow(s.rank / block + 1);
        const at = s.rank;
        for (s.list, 0..) |*site, l| {
            const a = site.a.slice(f32, s.cap * block * site.in)[at * site.in ..][0 .. block * site.in];
            const f = site.seek.slice(f32, candidates * site.in);
            @memset(a, 0);
            for (0..block) |q| for (0..candidates) |j| {
                subspace.axpy(a[q * site.in ..][0..site.in], a_norm * choice[(l * block + q) * candidates + j], f[j * site.in ..][0..site.in]);
            };
            site.gate(at / block).* = 0;
            @memset(site.b.slice(f32, s.cap * block * site.out)[at * site.out ..][0 .. block * site.out], 0);
            for ([_]Mapped{ site.gb, site.mb, site.vb }) |m| @memset(m.slice(f32, block * site.out), 0);
        }
        s.rank += block;
        s.steps = 0;
    }

    /// The open block taken back out, its lesson having left nothing in it.
    pub fn close(s: *Sites) void {
        s.rank -= block;
        for (s.list) |*site| @memset(site.b.slice(f32, s.cap * block * site.out)[s.rank * site.out ..][0 .. block * site.out], 0);
    }

    /// One Adam step on the open block's outputs (after a step's gradients are in); the gradients are cleared.
    pub fn adam(s: *Sites, o: train_ops.Ops) !void {
        s.steps += 1;
        const t: f32 = @floatFromInt(s.steps);
        const corr = [2]f32{ 1 / (1 - std.math.pow(f32, hyper[1], t)), 1 / (1 - std.math.pow(f32, hyper[2], t)) };
        for (s.list) |*site| {
            try o.adam(site.b.dev + s.first() * site.out * 4, site.gb.dev, site.mb.dev, site.vb.dev, block * site.out, shape.hyperAt(s.steps), corr);
        }
    }

    /// Aims layer l's router (the block's first rank) along `dir`: its score less `mean`, times `weight`.
    pub fn aim(s: *Sites, l: usize, dir: []const f32, weight: f32, mean: f32) void {
        const site = &s.list[l];
        const a = site.a.slice(f32, s.cap * block * site.in)[s.first() * site.in ..][0..site.in];
        const f = site.seek.slice(f32, candidates * site.in);
        @memset(a, 0);
        for (0..candidates) |j| subspace.axpy(a, a_norm * weight * dir[j], f[j * site.in ..][0..site.in]);
        site.gate(s.first() / block).* = weight * mean;
    }

    /// The open block's outputs and moments as they are now, for `restore` (read once the stream is idle).
    pub fn keep(s: *Sites) void {
        s.kept_steps = s.steps;
        for (s.list) |*site| {
            const n = block * site.out;
            const k = site.kept.slice(f32, 3 * n);
            @memcpy(k[0..n], site.b.slice(f32, s.cap * block * site.out)[s.first() * site.out ..][0..n]);
            @memcpy(k[n .. 2 * n], site.mb.slice(f32, n));
            @memcpy(k[2 * n ..], site.vb.slice(f32, n));
        }
    }

    /// Back to what `keep` saw, gradients cleared.
    pub fn restore(s: *Sites) void {
        s.steps = s.kept_steps;
        for (s.list) |*site| {
            const n = block * site.out;
            const k = site.kept.slice(f32, 3 * n);
            @memcpy(site.b.slice(f32, s.cap * block * site.out)[s.first() * site.out ..][0..n], k[0..n]);
            @memcpy(site.mb.slice(f32, n), k[n .. 2 * n]);
            @memcpy(site.vb.slice(f32, n), k[2 * n ..]);
            @memset(site.gb.slice(f32, n), 0);
        }
    }

    /// Gradients and both sketches back to zero, before a lesson's inputs are sketched.
    pub fn clear(s: *Sites) void {
        for (s.list) |*site| {
            @memset(site.gb.slice(f32, block * site.out), 0);
            @memset(site.avoid.slice(f32, avoid_dims * site.in), 0);
            @memset(site.seek.slice(f32, candidates * site.in), 0);
        }
    }
};

/// Every n-th site from k: its candidate directions.
fn frameShare(list: []Site, k: usize, n: usize) void {
    var i = k;
    while (i < list.len) : (i += n) {
        const site = &list[i];
        const avoid = site.avoid.slice(f32, avoid_dims * site.in);
        const seek = site.seek.slice(f32, candidates * site.in);
        subspace.orthonormal(avoid, avoid_dims, site.in);
        subspace.remove(seek, candidates, avoid, avoid_dims, site.in);
        subspace.orthonormal(seek, candidates, site.in);
    }
}
