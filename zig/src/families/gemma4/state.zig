//! Gemma 4 dense decode state: each stream's caches (a ring a sliding layer, a buffer a global one) and scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

const opts = mtl.ResourceOptions.private | mtl.ResourceOptions.untracked;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A verify window's rows at most (the exact width), and a prompt window's (the reg projection's widest).
pub const max_rows = 16;
pub const prompt_rows = 128;
/// Ints a row's attention meta takes: first chunk, chunks, ring, rows, first new key, padding.
pub const meta_ints = 8;
/// Rows a prompt chunk takes (prompt.zig), and a sliding layer's ring: the window plus a chunk written past it.
pub const chunk_rows = 512;
/// The 31B's and 12B's ring (the MTP assistant reads theirs).
pub const ring = cfg.window + chunk_rows;

/// A geometry's sliding ring.
pub fn ringOf(geo: cfg.Geometry) usize {
    return geo.sliding_window + chunk_rows;
}

/// The layer whose cache layer `i` reads: itself, or (the E-series' trailing layers) the last earlier one of its kind.
pub fn kvSource(geo: cfg.Geometry, i: usize) usize {
    return cfg.cacheOwner(i, geo.cacheLayers()).?;
}

/// Where a stream's latest tapped rows sit: `rows` rows from cache position `start`, current while `gen` matches.
pub const Tap = struct { buf: ?mtl.Buffer = null, off: usize = 0, start: u64 = 0, rows: usize = 0, gen: ?u64 = null };

/// One stream's keys and values, per layer [kv heads, slots, head dim] bf16, and how many positions it holds.
pub const Cache = struct {
    k: [cfg.max_layers]mtl.Buffer,
    v: [cfg.max_layers]mtl.Buffer,
    geo: cfg.Geometry,
    capacity: usize,
    len: usize = 0,
    /// The cache length the last verify started from (keep rolls back to it plus the kept rows).
    from: usize = 0,
    /// The last row's logits, bf16 [vocab]: the prompt's (first) or a one-token round's (queue).
    logits: mtl.Buffer,
    /// The drafter's rows: the prompt's last (in `last_tap`), then each verify's in the scratch.
    tap: Tap = .{},
    last_tap: ?mtl.Buffer = null,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(device: mtl.Device, geo: cfg.Geometry, capacity: usize) !Cache {
        var c: Cache = .{ .k = undefined, .v = undefined, .geo = geo, .capacity = capacity, .logits = try device.buffer(cfg.vocab * 2, shared) };
        errdefer c.logits.deinit();
        var made: usize = 0;
        errdefer for (0..made) |i| { // the owned layers come first
            c.k[i].deinit();
            c.v[i].deinit();
        };
        for (0..geo.layers) |i| {
            if (kvSource(geo, i) != i) { // a shared layer reads an earlier layer's buffers, never writes them
                c.k[i] = c.k[kvSource(geo, i)];
                c.v[i] = c.v[kvSource(geo, i)];
                continue;
            }
            const slots = slotsOfGeo(geo, i, capacity);
            const bytes = geo.kvHeads(i) * slots * cfg.headDim(i) * 2;
            c.k[i] = try device.buffer(bytes, opts);
            c.v[i] = device.buffer(bytes, opts) catch |e| {
                c.k[i].deinit();
                return e;
            };
            made += 1;
        }
        return c;
    }

    pub fn deinit(c: *Cache) void {
        c.setResident(null);
        c.logits.deinit();
        if (c.last_tap) |b| b.deinit();
        for (0..c.geo.layers) |i| if (kvSource(c.geo, i) == i) {
            c.k[i].deinit();
            c.v[i].deinit();
        };
    }

    pub fn setResident(c: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        var buffers: [2 * cfg.max_layers + 2]mtl.Buffer = undefined;
        buffers[0] = c.logits;
        var count: usize = 1;
        if (c.last_tap) |b| {
            buffers[count] = b;
            count += 1;
        }
        for (0..c.geo.layers) |i| if (kvSource(c.geo, i) == i) {
            buffers[count] = c.k[i];
            buffers[count + 1] = c.v[i];
            count += 2;
        };
        if (c.follow) |f| for (buffers[0..count]) |b| f.leave(f.ctx, b);
        c.follow = hook;
        if (hook) |f| for (buffers[0..count]) |b| f.join(f.ctx, b);
    }

    pub fn setLastTap(c: *Cache, buffer: mtl.Buffer) void {
        if (c.last_tap) |old| {
            if (c.follow) |f| f.leave(f.ctx, old);
            old.deinit();
        }
        c.last_tap = buffer;
        if (c.follow) |f| f.join(f.ctx, buffer);
    }
};

/// Bytes a stream's caches take at `capacity` (Cache.init's buffers).
pub fn cacheBytes(geo: cfg.Geometry, capacity: usize) u64 {
    var total: u64 = cfg.vocab * 2;
    for (0..geo.layers) |i| if (kvSource(geo, i) == i) {
        total += 2 * geo.kvHeads(i) * slotsOfGeo(geo, i, capacity) * cfg.headDim(i) * 2;
    };
    return total;
}

/// Cache slots for layer `i` of the 31B or 12B: the ring on a sliding layer, the whole capacity on a global one.
pub fn slotsOf(i: usize, capacity: usize) usize {
    return if (cfg.global(i)) capacity else ring;
}

/// Cache slots for layer `i` of `geo`.
pub fn slotsOfGeo(geo: cfg.Geometry, i: usize, capacity: usize) usize {
    return if (cfg.global(i)) capacity else ringOf(geo);
}

/// Activations for one window of up to `max_rows` rows, shared by every layer.
pub const Scratch = struct {
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    /// Per layer kind (0 sliding, 1 global): each row's first key, and at a sub-batch's first row its meta.
    lo: [2]mtl.Buffer,
    meta: [2]mtl.Buffer,
    h: [2]mtl.Buffer,
    normed: [2]mtl.Buffer,
    qkv: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    attn: mtl.Buffer,
    o: mtl.Buffer,
    hn: mtl.Buffer,
    n_mlp: mtl.Buffer,
    gate_up: mtl.Buffer,
    act: mtl.Buffer,
    y: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    logits: mtl.Buffer,
    /// The next projection's group sums, written by the kernel that made its input (reg lane kernel order).
    sums: mtl.Buffer,
    /// The drafter's tapped rows [rows, taps x hidden] bf16, once a drafter asked for them.
    taps: ?mtl.Buffer = null,
    /// Each row's greedy token by the GPU argmax (tree_round Pick: token, nonfinite flag).
    picks: mtl.Buffer,
    /// Attention partial slots: heads x rows x chunks (256 keys a chunk on global layers; sliding ones in the window).
    slots: usize,
    /// The E-series' per-layer inputs for the window's rows, and its per-layer gate's scratch.
    ple: ?Ple = null,

    pub const Ple = struct {
        /// The table's rows [rows, layers x width], the embeddings' projection, and the combined inputs layers read.
        tok: mtl.Buffer,
        prj: mtl.Buffer,
        pli: mtl.Buffer,
        /// A layer's gate [rows, width] and the gelu(gate) x input that its projection reads.
        gate: mtl.Buffer,
        mid: mtl.Buffer,
        /// bf16 1.0: the MLP's dense tail leaves the residual unscaled; the layer scalar follows the per-layer input.
        one: mtl.Buffer,

        pub fn init(device: mtl.Device, geo: cfg.Geometry, rows: usize) !Ple {
            const wide = rows * geo.layers * geo.layer_input * 2;
            var p: Ple = undefined;
            var made: usize = 0;
            errdefer inline for (.{ "tok", "prj", "pli", "gate", "mid", "one" }, 0..) |f, i| if (i < made) @field(p, f).deinit();
            inline for (.{ .{ "tok", wide }, .{ "prj", wide }, .{ "pli", wide }, .{ "gate", rows * geo.layer_input * 2 }, .{ "mid", rows * geo.layer_input * 2 }, .{ "one", 2 } }) |f| {
                @field(p, f[0]) = try device.buffer(f[1], if (std.mem.eql(u8, f[0], "one")) shared else opts);
                made += 1;
            }
            p.one.slice(u16, 1)[0] = 0x3f80;
            return p;
        }

        pub fn deinit(p: Ple) void {
            inline for (.{ "tok", "prj", "pli", "gate", "mid", "one" }) |f| @field(p, f).deinit();
        }
    };

    pub fn init(device: mtl.Device, geo: cfg.Geometry, capacity: usize) !Scratch {
        const r = prompt_rows;
        const d = geo.hidden;
        const qd = geo.queryWidth();
        const kv = geo.kvWidth();
        const slots = geo.query_heads * max_rows * @max(capacity / 256 + 2, (geo.sliding_window + max_rows) / 128 + 2);
        var s: Scratch = undefined;
        s.slots = slots;
        s.taps = null;
        s.ple = null;
        var made: usize = 0;
        errdefer s.release(made);
        inline for (.{ .{ "ids", r * 4, true }, .{ "pos", r * 4, true }, .{ "qkv", r * (qd + 2 * kv) * 2, false }, .{ "q", r * qd * 2, false }, .{ "k", r * kv * 2, false }, .{ "v", r * kv * 2, false }, .{ "attn", r * qd * 2, false }, .{ "o", r * d * 2, false }, .{ "hn", r * d * 2, false }, .{ "n_mlp", r * d * 2, false }, .{ "gate_up", r * 2 * geo.intermediate * 2, false }, .{ "act", r * geo.intermediate * 2, false }, .{ "y", r * d * 2, false }, .{ "pm", slots * 4, false }, .{ "pl", slots * 4, false }, .{ "po", slots * cfg.global_dim * 4, false }, .{ "logits", r * cfg.vocab * 2, true }, .{ "sums", r * (geo.intermediate / 64) * 4, false }, .{ "picks", r * 8, true } }) |f| {
            @field(s, f[0]) = try device.buffer(f[1], if (f[2]) shared else opts);
            made += 1;
        }
        for (&s.lo) |*b| {
            b.* = try device.buffer(r * 4, shared);
            made += 1;
        }
        for (&s.meta) |*b| {
            b.* = try device.buffer(r * meta_ints * 4, shared);
            made += 1;
        }
        for ([_][]mtl.Buffer{ &s.h, &s.normed }) |pair| for (pair) |*b| {
            b.* = try device.buffer(r * d * 2, opts);
            made += 1;
        };
        if (geo.layer_input != 0) s.ple = try Ple.init(device, geo, r);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        s.release(std.math.maxInt(usize));
        if (s.taps) |b| b.deinit();
        if (s.ple) |p| p.deinit();
    }

    /// Releases the first `made` buffers in init's order, so a failed init frees what it made.
    fn release(s: *Scratch, made: usize) void {
        var i: usize = 0;
        inline for (.{ "ids", "pos", "qkv", "q", "k", "v", "attn", "o", "hn", "n_mlp", "gate_up", "act", "y", "pm", "pl", "po", "logits", "sums", "picks" }) |f| {
            if (i < made) @field(s, f).deinit();
            i += 1;
        }
        for (s.lo ++ s.meta ++ s.h ++ s.normed) |b| {
            if (i < made) b.deinit();
            i += 1;
        }
    }
};
