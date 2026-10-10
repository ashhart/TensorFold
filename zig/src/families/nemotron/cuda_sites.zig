//! Each layer's learned change on CUDA, as adapters.zig on Metal, in host memory the GPU maps (one pool on a GB10).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("cuda_weights.zig");
const train_ops = @import("cuda_train_ops.zig");
const subspace = @import("subspace.zig");
const shape = @import("learn_shape.zig");

pub const block = shape.block;
pub const max_rank = shape.max_rank;
pub const max_blocks = shape.max_blocks;
pub const scale = shape.scale;
pub const avoid_dims = shape.avoid_dims;
pub const candidates = shape.candidates;
pub const hyper = shape.hyper;
const a_norm = shape.a_norm;
pub const unit = shape.unit;
pub const shut = shape.shut;

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

/// One layer's change: y += scale (x a^T) b after its output projection; a [max_rank, in], b [max_rank, out], f32.
pub const Site = struct {
    layer: usize,
    in: usize,
    out: usize,
    a: Mapped, // every block's input directions, fixed when its lesson begins
    b: Mapped, // every block's outputs; only the open block's still learn
    tau: Mapped, // [max_blocks]: each block's gate, the share of a row's input it needs to act on that row
    gb: Mapped, // [block, out]: the open block's gradient
    mb: Mapped, // [block, out]: its Adam moments
    vb: Mapped,
    kept: Mapped, // [3, block, out]: the open block's b and moments as its round began
    avoid: Mapped, // [avoid_dims, in]: a sketch of the inputs the next block must leave alone
    seek: Mapped, // [candidates, in]: a sketch of the fact's inputs, then the candidate directions it frames

    pub fn adapter(s: *const Site, sites: *const Sites) weights.Adapter {
        return .{ .a = s.a.dev, .b = s.b.dev, .tau = s.tau.dev, .xa = sites.xa.dev, .xn = sites.xn.dev, .gates = sites.gates.dev, .in = s.in, .out = s.out, .rank = sites.rank, .scale = scale, .unit = unit };
    }

    /// Block k's gate.
    pub fn gate(s: *const Site, k: usize) *f32 {
        return &s.tau.slice(f32, max_blocks)[k];
    }

    fn fields(s: *Site) [9]*Mapped {
        return .{ &s.a, &s.b, &s.tau, &s.gb, &s.mb, &s.vb, &s.kept, &s.avoid, &s.seek };
    }
};

pub const inWidth = shape.inWidth;

pub const Sites = struct {
    gpa: std.mem.Allocator,
    list: []Site,
    xa: Mapped, // [rows, max_rank]: a forward's scratch, used by one site at a time
    xn: Mapped, // [rows]: each row's squared input length, the same way
    gates: Mapped, // [rows, max_blocks]: which blocks are open on each row, the same way
    rank: usize = 0, // ranks in use, the open block last
    steps: u64 = 0, // the open block's Adam steps
    kept_steps: u64 = 0,

    /// Room for every block at every layer, none in use: the forward is unchanged until a block opens.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: Config, rows: usize) !Sites {
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
            const sizes = [_]usize{ max_rank * in, max_rank * out, max_blocks, block * out, block * out, block * out, 3 * block * out, avoid_dims * in, candidates * in };
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
        const gates = try Mapped.init(d, rows * max_blocks * 4);
        return .{ .gpa = gpa, .list = list, .xa = xa, .xn = xn, .gates = gates };
    }

    pub fn deinit(s: *Sites) void {
        for (s.list) |*site| free(site);
        s.gpa.free(s.list);
        inline for (.{ "xa", "xn", "gates" }) |f| @field(s, f).deinit();
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

    /// A new block at every layer, shut, its directions `choice` [layers, block, candidates] of the candidates; b zero.
    pub fn open(s: *Sites, choice: []const f32) !void {
        if (s.rank + block > max_rank) return error.LearnedChangeFull;
        const at = s.rank;
        for (s.list, 0..) |*site, l| {
            const a = site.a.slice(f32, max_rank * site.in)[at * site.in ..][0 .. block * site.in];
            const f = site.seek.slice(f32, candidates * site.in);
            @memset(a, 0);
            for (0..block) |q| for (0..candidates) |j| {
                subspace.axpy(a[q * site.in ..][0..site.in], a_norm * choice[(l * block + q) * candidates + j], f[j * site.in ..][0..site.in]);
            };
            site.gate(at / block).* = shut;
            @memset(site.b.slice(f32, max_rank * site.out)[at * site.out ..][0 .. block * site.out], 0);
            for ([_]Mapped{ site.gb, site.mb, site.vb }) |m| @memset(m.slice(f32, block * site.out), 0);
        }
        s.rank += block;
        s.steps = 0;
    }

    /// The open block's directions rebuilt from `choice` and its gate open on every row: a plain change of the weights.
    pub fn plain(s: *Sites, choice: []const f32) void {
        const at = s.first();
        for (s.list, 0..) |*site, l| {
            const g = site.gate(at / block);
            if (g.* >= shut) continue;
            const a = site.a.slice(f32, max_rank * site.in)[at * site.in ..][0 .. block * site.in];
            const f = site.seek.slice(f32, candidates * site.in);
            @memset(a, 0);
            for (0..block) |q| for (0..candidates) |j| {
                subspace.axpy(a[q * site.in ..][0..site.in], a_norm * choice[(l * block + q) * candidates + j], f[j * site.in ..][0..site.in]);
            };
            g.* = -std.math.inf(f32);
        }
    }

    /// The open block taken back out, its lesson having left nothing in it.
    pub fn close(s: *Sites) void {
        s.rank -= block;
        for (s.list) |*site| @memset(site.b.slice(f32, max_rank * site.out)[s.rank * site.out ..][0 .. block * site.out], 0);
    }

    /// One Adam step on the open block's outputs (after a step's gradients are in); the gradients are cleared.
    pub fn adam(s: *Sites, o: train_ops.Ops) !void {
        s.steps += 1;
        const t: f32 = @floatFromInt(s.steps);
        const corr = [2]f32{ 1 / (1 - std.math.pow(f32, hyper[1], t)), 1 / (1 - std.math.pow(f32, hyper[2], t)) };
        for (s.list) |*site| try o.adam(site.b.dev + s.first() * site.out * 4, site.gb.dev, site.mb.dev, site.vb.dev, block * site.out, hyper, corr);
    }

    /// The open block's outputs and moments as they are now, for `restore` (read once the stream is idle).
    pub fn keep(s: *Sites) void {
        s.kept_steps = s.steps;
        for (s.list) |*site| {
            const n = block * site.out;
            const k = site.kept.slice(f32, 3 * n);
            @memcpy(k[0..n], site.b.slice(f32, max_rank * site.out)[s.first() * site.out ..][0..n]);
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
            @memcpy(site.b.slice(f32, max_rank * site.out)[s.first() * site.out ..][0..n], k[0..n]);
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
