//! Each layer's learned change, a block of ranks a lesson: inputs fixed away from what must stay, outputs learned.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const ops = @import("train_ops.zig");
const pl = @import("prefill_launch.zig");
const subspace = @import("subspace.zig");
const dims = @import("slide_dims.zig");

const Buffer = mtl.Buffer;
const At = pl.At;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const block = dims.block;
pub const max_rank = dims.max_rank;
pub const max_blocks = dims.max_blocks;
pub const scale = dims.scale;
pub const avoid_dims = dims.avoid_dims;
pub const candidates = dims.candidates;
pub const hyper = dims.hyper;
const a_norm = dims.a_norm;
pub const unit = dims.unit;
pub const shut = dims.shut;

/// One layer's change: y += scale (x a^T) b after its output projection; a [max_rank, in], b [max_rank, out], f32.
pub const Site = struct {
    layer: usize,
    in: usize,
    out: usize,
    a: Buffer, // every block's input directions, fixed when its lesson begins
    b: Buffer, // every block's outputs; only the open block's still learn
    tau: Buffer, // [max_blocks]: each block's gate, the share of a row's input it needs to act on that row
    gb: Buffer, // [block, out]: the open block's gradient
    mb: Buffer, // [block, out]: its Adam moments
    vb: Buffer,
    kept: Buffer, // [3, block, out]: the open block's b and moments as its round began
    avoid: Buffer, // [avoid_dims, in]: a sketch of the inputs the next block must leave alone
    seek: Buffer, // [candidates, in]: a sketch of the fact's inputs, then the candidate directions it frames

    pub fn adapter(s: *const Site, sites: *const Sites) wts.Adapter {
        return .{ .a = s.a, .b = s.b, .tau = s.tau, .xa = sites.xa, .xn = sites.xn, .gates = sites.gates, .in = s.in, .out = s.out, .rank = sites.rank, .scale = scale, .unit = unit };
    }

    /// Block k's gate.
    pub fn gate(s: *const Site, k: usize) *f32 {
        return &s.tau.slice(f32, max_blocks)[k];
    }
};

pub const Sites = struct {
    gpa: std.mem.Allocator,
    list: []Site,
    xa: Buffer, // [rows, max_rank]: the forward's scratch, used by one site at a time
    xn: Buffer, // [rows]: each row's squared input length, the same way
    gates: Buffer, // [rows, max_blocks]: which blocks are open on each row, the same way
    rank: usize = 0, // ranks in use, the open block last
    steps: u64 = 0, // the open block's Adam steps
    kept_steps: u64 = 0,

    /// Room for every block at every layer, none in use: the forward is unchanged until a block opens.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, c: cfg.Config, rows: usize) !Sites {
        const list = try gpa.alloc(Site, c.layers);
        var made: usize = 0;
        errdefer {
            for (list[0..made]) |*s| free(s);
            gpa.free(list);
        }
        for (list, 0..) |*s, i| {
            const in: usize = switch (c.kinds[i]) {
                .moe => c.shared_width,
                .mamba => c.inner(),
                .attention => c.heads * c.head_dim,
            };
            const out = c.hidden;
            s.* = .{ .layer = i, .in = in, .out = out, .a = undefined, .b = undefined, .tau = undefined, .gb = undefined, .mb = undefined, .vb = undefined, .kept = undefined, .avoid = undefined, .seek = undefined };
            const sizes = [_]usize{ max_rank * in, max_rank * out, max_blocks, block * out, block * out, block * out, 3 * block * out, avoid_dims * in, candidates * in };
            const fields = [_]*Buffer{ &s.a, &s.b, &s.tau, &s.gb, &s.mb, &s.vb, &s.kept, &s.avoid, &s.seek };
            var got: usize = 0;
            errdefer for (fields[0..got]) |f| f.deinit();
            for (fields, sizes) |f, n| {
                f.* = try device.buffer(n * 4, opts);
                @memset(f.slice(f32, n), 0);
                got += 1;
            }
            made += 1;
        }
        const xa = try device.buffer(rows * max_rank * 4, opts);
        errdefer xa.deinit();
        const xn = try device.buffer(rows * 4, opts);
        errdefer xn.deinit();
        const gates = try device.buffer(rows * max_blocks * 4, opts);
        return .{ .gpa = gpa, .list = list, .xa = xa, .xn = xn, .gates = gates };
    }

    pub fn deinit(s: *Sites) void {
        for (s.list) |*site| free(site);
        s.gpa.free(s.list);
        inline for (.{ "xa", "xn", "gates" }) |f| @field(s, f).deinit();
    }

    fn free(s: *Site) void {
        for ([_]*Buffer{ &s.a, &s.b, &s.tau, &s.gb, &s.mb, &s.vb, &s.kept, &s.avoid, &s.seek }) |f| f.deinit();
    }

    /// Every layer's forward applies the blocks in use from here on (none: no change at all).
    pub fn attach(s: *Sites, w: *wts.Weights, on: bool) void {
        for (s.list) |*site| {
            const ad: ?wts.Adapter = if (on and s.rank > 0) site.adapter(s) else null;
            switch (w.layers[site.layer]) {
                .moe => |*m| m.adapter = ad,
                .mamba => |*m| m.adapter = ad,
                .attention => |*a| a.adapter = ad,
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
            for ([_]Buffer{ site.gb, site.mb, site.vb }) |buf| @memset(buf.slice(f32, block * site.out), 0);
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
    pub fn adam(s: *Sites, o: ops.Ops) void {
        s.steps += 1;
        const t: f32 = @floatFromInt(s.steps);
        const corr = [2]f32{ 1 / (1 - std.math.pow(f32, hyper[1], t)), 1 / (1 - std.math.pow(f32, hyper[2], t)) };
        for (s.list) |*site| o.adam(At.of(site.b).plus(s.first() * site.out * 4), site.gb, site.mb, site.vb, block * site.out, hyper, corr);
    }

    /// The open block's outputs and moments as they are now, for `restore` (read after every command buffer landed).
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
