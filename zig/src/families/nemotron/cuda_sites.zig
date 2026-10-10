//! Each layer's learned change on CUDA (--slide): a block of ranks a lesson, in mapped host memory the GPU reads too.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const dims = @import("slide_dims.zig");
const subspace = @import("subspace.zig");
const weights = @import("cuda_weights.zig");
const ops = @import("cuda_train_ops.zig");

pub const block = dims.block;
pub const max_rank = dims.max_rank;
pub const max_blocks = dims.max_blocks;
pub const avoid_dims = dims.avoid_dims;
pub const candidates = dims.candidates;

/// Pinned host pages mapped into the device's address space: the host writes and reads them, kernels too.
pub const Shared = struct {
    buf: cuda.HostBuffer,
    dev: u64,

    /// `n` zeroed values of T.
    pub fn init(d: *const cuda.Driver, comptime T: type, n: usize) !Shared {
        var buf = try cuda.HostBuffer.allocMapped(d, @max(n * @sizeOf(T), 16));
        errdefer buf.free();
        @memset(buf.bytes, 0);
        return .{ .buf = buf, .dev = try buf.device() };
    }

    pub fn free(s: *Shared) void {
        s.buf.free();
    }

    pub fn slice(s: Shared, comptime T: type, n: usize) []T {
        return s.buf.slice(T)[0..n];
    }
};

/// One layer's change: y += scale (x a^T) b after its output projection; a [max_rank, in], b [max_rank, out], f32.
pub const Site = struct {
    layer: usize,
    in: usize,
    out: usize,
    a: Shared, // every block's input directions, fixed when its lesson begins
    b: Shared, // every block's outputs; only the open block's still learn
    tau: Shared, // [max_blocks]: each block's gate, the share of a row's input it needs to act on that row
    gb: Shared, // [block, out]: the open block's gradient
    mb: Shared, // [block, out]: its Adam moments
    vb: Shared,
    avoid: Shared, // [avoid_dims, in]: a sketch of the inputs the next block must leave alone
    seek: Shared, // [candidates, in]: a sketch of the fact's inputs, then the candidate directions it frames
    kept: []f32, // host: [3, block, out], the open block's b and moments as its round began
    ad: weights.Adapter = undefined, // the descriptor the forward points at while attached

    pub fn adapter(s: *const Site, sites: *const Sites) weights.Adapter {
        return .{ .a = s.a.dev, .b = s.b.dev, .tau = s.tau.dev, .rank = sites.word.dev, .xa = sites.xa.ptr, .xn = sites.xn.ptr, .in = s.in, .out = s.out };
    }

    /// Block k's gate.
    pub fn gate(s: *const Site, k: usize) *f32 {
        return &s.tau.slice(f32, max_blocks)[k];
    }

    fn fields(s: *Site) [8]*Shared {
        return .{ &s.a, &s.b, &s.tau, &s.gb, &s.mb, &s.vb, &s.avoid, &s.seek };
    }
};

pub const Sites = struct {
    gpa: std.mem.Allocator,
    list: []Site,
    word: Shared, // u32: the ranks in use as the forward's kernels read them (0 while detached)
    xa: cuda.DeviceBuffer, // [rows, max_rank]: the forward's scratch, used by one site at a time
    xn: cuda.DeviceBuffer, // [rows]: each row's squared input length, the same way
    rank: usize = 0, // ranks in use, the open block last
    steps: u64 = 0, // the open block's Adam steps
    kept_steps: u64 = 0,

    /// Room for every block at every layer, none in use: the forward is unchanged until a block opens.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: cfg.Config, rows: usize) !Sites {
        const list = try gpa.alloc(Site, c.layers);
        var made: usize = 0;
        errdefer {
            for (list[0..made]) |*s| free(gpa, s);
            gpa.free(list);
        }
        for (list, 0..) |*s, i| {
            const in: usize = switch (c.kinds[i]) {
                .moe => c.shared_width,
                .mamba => c.inner(),
                .attention => c.heads * c.head_dim,
            };
            const out = c.hidden;
            s.* = .{ .layer = i, .in = in, .out = out, .a = undefined, .b = undefined, .tau = undefined, .gb = undefined, .mb = undefined, .vb = undefined, .avoid = undefined, .seek = undefined, .kept = &.{} };
            const sizes = [_]usize{ max_rank * in, max_rank * out, max_blocks, block * out, block * out, block * out, avoid_dims * in, candidates * in };
            var got: usize = 0;
            errdefer for (s.fields()[0..got]) |f| f.free();
            for (s.fields(), sizes) |f, n| {
                f.* = try Shared.init(d, f32, n);
                got += 1;
            }
            s.kept = try gpa.alloc(f32, 3 * block * out);
            made += 1;
        }
        var word = try Shared.init(d, u32, 1);
        errdefer word.free();
        var xa = try cuda.DeviceBuffer.alloc(d, rows * max_rank * 4);
        errdefer xa.free();
        const xn = try cuda.DeviceBuffer.alloc(d, rows * 4);
        return .{ .gpa = gpa, .list = list, .word = word, .xa = xa, .xn = xn };
    }

    pub fn deinit(s: *Sites) void {
        for (s.list) |*site| free(s.gpa, site);
        s.gpa.free(s.list);
        s.word.free();
        s.xa.free();
        s.xn.free();
    }

    fn free(gpa: std.mem.Allocator, s: *Site) void {
        for (s.fields()) |f| f.free();
        gpa.free(s.kept);
    }

    /// Every layer's forward carries its change (graphs captured after keep it); the ranks in use apply when on.
    pub fn attach(s: *Sites, w: *weights.Weights, on: bool) void {
        for (s.list) |*site| {
            site.ad = site.adapter(s);
            const ad: ?*const weights.Adapter = if (on) &site.ad else null;
            switch (w.blocks[site.layer].kind) {
                .moe => w.blocks[site.layer].moe.adapter = ad,
                .mamba => w.blocks[site.layer].mamba.adapter = ad,
                .attention => w.blocks[site.layer].attn.adapter = ad,
            }
        }
        s.publish(on);
    }

    /// The ranks the kernels apply: those in use, or none.
    fn publish(s: *Sites, on: bool) void {
        s.word.slice(u32, 1)[0] = if (on) @intCast(s.rank) else 0;
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
            directions(site, at, l, choice);
            site.gate(at / block).* = dims.shut;
            @memset(site.b.slice(f32, max_rank * site.out)[at * site.out ..][0 .. block * site.out], 0);
            for ([_]Shared{ site.gb, site.mb, site.vb }) |buf| @memset(buf.slice(f32, block * site.out), 0);
        }
        s.rank += block;
        s.steps = 0;
        s.publish(true);
    }

    /// The open block's directions rebuilt from `choice` and its gate open on every row: a plain change of the weights.
    pub fn plain(s: *Sites, choice: []const f32) void {
        const at = s.first();
        for (s.list, 0..) |*site, l| {
            const g = site.gate(at / block);
            if (g.* >= dims.shut) continue;
            directions(site, at, l, choice);
            g.* = -std.math.inf(f32);
        }
    }

    /// The open block taken back out, its lesson having left nothing in it.
    pub fn close(s: *Sites) void {
        s.rank -= block;
        for (s.list) |*site| @memset(site.b.slice(f32, max_rank * site.out)[s.rank * site.out ..][0 .. block * site.out], 0);
        s.publish(true);
    }

    /// One Adam step on the open block's outputs (after a step's gradients are in); the gradients are cleared.
    pub fn adam(s: *Sites, t: ops.Train) !void {
        s.steps += 1;
        const step: f32 = @floatFromInt(s.steps);
        const h = dims.hyper;
        const corr = [2]f32{ 1 / (1 - std.math.pow(f32, h[1], step)), 1 / (1 - std.math.pow(f32, h[2], step)) };
        for (s.list) |*site| try t.adam(site.b.dev + s.first() * site.out * 4, site.gb.dev, site.mb.dev, site.vb.dev, block * site.out, h, corr);
    }

    /// The open block's outputs and moments as they are now, for `restore` (read once the stream has finished).
    pub fn keep(s: *Sites) void {
        s.kept_steps = s.steps;
        for (s.list) |*site| {
            const n = block * site.out;
            @memcpy(site.kept[0..n], site.b.slice(f32, max_rank * site.out)[s.first() * site.out ..][0..n]);
            @memcpy(site.kept[n .. 2 * n], site.mb.slice(f32, n));
            @memcpy(site.kept[2 * n ..], site.vb.slice(f32, n));
        }
    }

    /// Back to what `keep` saw, gradients cleared.
    pub fn restore(s: *Sites) void {
        s.steps = s.kept_steps;
        for (s.list) |*site| {
            const n = block * site.out;
            @memcpy(site.b.slice(f32, max_rank * site.out)[s.first() * site.out ..][0..n], site.kept[0..n]);
            @memcpy(site.mb.slice(f32, n), site.kept[n .. 2 * n]);
            @memcpy(site.vb.slice(f32, n), site.kept[2 * n ..]);
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

/// Block `at`'s input directions at layer l: its `choice` rows' combinations of the candidates, each a_norm long.
fn directions(site: *Site, at: usize, l: usize, choice: []const f32) void {
    const a = site.a.slice(f32, max_rank * site.in)[at * site.in ..][0 .. block * site.in];
    const f = site.seek.slice(f32, candidates * site.in);
    @memset(a, 0);
    for (0..block) |q| for (0..candidates) |j| {
        subspace.axpy(a[q * site.in ..][0..site.in], dims.a_norm * choice[(l * block + q) * candidates + j], f[j * site.in ..][0..site.in]);
    };
}

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
