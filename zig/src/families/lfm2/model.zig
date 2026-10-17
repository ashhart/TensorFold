//! LFM2's prompt pass over a packed decision: one shared trunk, then a branch a question, read at each branch's labels.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Weights = @import("weights.zig").Weights;
const Kernels = @import("kernels.zig").Kernels;
const Attention = @import("attention.zig").Attention;
const Linear = cuda.cublaslt.Linear;
const Allocator = std.mem.Allocator;

const workspace_bytes = 32 << 20;
pub const labels_most = 4096;
pub const branches_most = 256;

/// Picture features over the trunk's placeholder rows: each run `[row, count]` takes the next `count` rows.
pub const Placed = struct { features: u64, runs: []const [2]usize };

pub const Model = struct {
    gpa: Allocator,
    cfg: *const Config,
    w: *const Weights,
    k: Kernels,
    at: Attention,
    embed: cuda.embed_rows.EmbedRows,
    lt: cuda.cublaslt.Library,
    stream: cuda.Stream,
    rows_most: usize,
    scratch: cuda.DeviceBuffer,
    workspace: cuda.DeviceBuffer,
    host: cuda.HostBuffer, // pinned: the index rows in, the label logits and the logsumexp out
    linears: std.AutoHashMapUnmanaged([3]u64, Linear) = .empty,
    b: Buffers,

    const Buffers = struct { ids: u64, pos: u64, prev: u64, cu: u64, labels: u64, h: u64, x: u64, wide: u64, y: u64, q: u64, kk: u64, v: u64, a: u64, g: u64, logits: u64, lse: u64 };

    /// Scratch for packed decisions of up to `rows_most` rows.
    pub fn init(gpa: Allocator, d: *const cuda.Driver, cfg: *const Config, w: *const Weights, rows_most: usize) !Model {
        var m: Model = .{ .gpa = gpa, .cfg = cfg, .w = w, .rows_most = rows_most, .k = try Kernels.load(d), .at = undefined, .embed = undefined, .lt = undefined, .stream = undefined, .scratch = undefined, .workspace = undefined, .host = undefined, .b = undefined };
        errdefer m.k.unload();
        m.at = try Attention.load(d);
        errdefer m.at.unload();
        m.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer m.embed.unload();
        m.lt = try cuda.cublaslt.Library.open();
        errdefer m.lt.close();
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();
        const r = rows_most;
        const hd = cfg.heads * cfg.dim;
        const kvd = cfg.kv_heads * cfg.dim;
        const wide = @max(3 * cfg.hidden, hd + 2 * kvd, 2 * cfg.ffn);
        // i32 index rows, bf16 activations, then f32 logits (a vocabulary's worth for the logsumexp)
        const sizes = [_]usize{ 4 * r, 4 * r, 4 * r * (cfg.taps - 1), 4 * (branches_most + 1), 4 * labels_most, 2 * r * cfg.hidden, 2 * r * cfg.hidden, 2 * r * wide, 2 * r * cfg.hidden, 2 * r * hd, 2 * r * kvd, 2 * r * kvd, 2 * r * hd, 2 * r * cfg.ffn, 4 * @max(cfg.vocab, labels_most), 4 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        m.scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer m.scratch.free();
        var at: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &at) |s, *p| {
            p.* = try m.scratch.at(off);
            off += std.mem.alignForward(usize, s, 256);
        }
        m.b = .{ .ids = at[0], .pos = at[1], .prev = at[2], .cu = at[3], .labels = at[4], .h = at[5], .x = at[6], .wide = at[7], .y = at[8], .q = at[9], .kk = at[10], .v = at[11], .a = at[12], .g = at[13], .logits = at[14], .lse = at[15] };
        m.workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        errdefer m.workspace.free();
        m.host = try cuda.HostBuffer.alloc(d, total);
        return m;
    }

    pub fn deinit(m: *Model) void {
        var it = m.linears.valueIterator();
        while (it.next()) |l| l.deinit();
        m.linears.deinit(m.gpa);
        m.host.free();
        m.workspace.free();
        m.scratch.free();
        m.stream.deinit();
        m.lt.close();
        m.embed.unload();
        m.at.unload();
        m.k.unload();
    }

    /// One pass over trunk ++ branches: each branch's label logits (f32) at its last row, in branch order.
    pub fn decide(m: *Model, trunk: []const u32, branches: []const []const u32, counts: []const usize, labels: []const u32, logits: []f64, placed: ?Placed) !void {
        const rows = try m.upload(trunk, branches, labels);
        var leaves: [branches_most]usize = undefined;
        var row = trunk.len;
        for (branches, leaves[0..branches.len]) |br, *leaf| {
            row += br.len;
            leaf.* = row - 1;
        }
        try m.forward(trunk.len, branches.len, rows, leaves[0..branches.len], placed);
        var at: usize = 0;
        for (leaves[0..branches.len], counts) |leaf, n| {
            try m.k.labelLogits(m.stream, m.b.x + 2 * leaf * m.cfg.hidden, m.w.embed, m.b.labels + 4 * at, m.b.logits + 4 * at, n, m.cfg.hidden);
            at += n;
        }
        const out = m.host.slice(f32)[0..labels.len];
        try m.get(m.b.logits, std.mem.sliceAsBytes(out));
        try m.stream.synchronize();
        for (logits, out) |*x, v| x.* = v;
    }

    /// `labels`' logits at the row after `prompt`, as a branch of its last token; the logsumexp when `mass`.
    pub fn score(m: *Model, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) !f64 {
        if (prompt.len == 0) return error.PromptLength;
        try m.decide(prompt[0 .. prompt.len - 1], &.{prompt[prompt.len - 1 ..]}, &.{labels.len}, labels, logits, null);
        if (!mass) return 0;
        const c = m.cfg;
        try m.linear(m.b.x + 2 * (prompt.len - 1) * c.hidden, m.w.embed, m.b.logits, 1, c.vocab, c.hidden, .f32);
        try m.k.logsumexp(m.stream, m.b.logits, m.b.lse, c.vocab);
        const out = m.host.slice(f32)[0..1];
        try m.get(m.b.lse, std.mem.sliceAsBytes(out));
        try m.stream.synchronize();
        return out[0];
    }

    /// The packed rows' ids, positions, conv taps and branch offsets and the labels, uploaded.
    fn upload(m: *Model, trunk: []const u32, branches: []const []const u32, labels: []const u32) !usize {
        var rows = trunk.len;
        for (branches) |br| {
            if (br.len == 0) return error.EmptyBranch;
            rows += br.len;
        }
        if (rows > m.rows_most) return error.PromptLength;
        if (branches.len == 0 or branches.len > branches_most) return error.Branches;
        if (labels.len > labels_most) return error.TooManyLabels;
        const taps = m.cfg.taps - 1;
        const host = m.host.slice(i32);
        const ids = host[0..rows];
        const pos = host[rows..][0..rows];
        const prev = host[2 * rows ..][0 .. rows * taps];
        const cu = host[(2 + taps) * rows ..][0 .. branches.len + 1];
        const lab = host[(2 + taps) * rows + branches.len + 1 ..][0..labels.len];
        const p = trunk.len;
        var r: usize = 0;
        cu[0] = 0;
        for (0..branches.len + 1) |b| {
            const tokens = if (b == 0) trunk else branches[b - 1];
            for (tokens, 0..) |t, j| {
                ids[r] = @intCast(t);
                const at: usize = if (b == 0) j else p + j;
                pos[r] = @intCast(at);
                // each tap back in the row's own sequence: its branch, then the trunk (-1: before it)
                for (0..taps) |k| prev[r * taps + k] = if (at < k + 1) -1 else if (at - k - 1 >= p) @intCast(r - k - 1) else @intCast(at - k - 1);
                r += 1;
            }
            if (b > 0) cu[b] = @intCast(r - p);
        }
        for (labels, lab) |l, *x| x.* = @intCast(l);
        try m.put(m.b.ids, std.mem.sliceAsBytes(ids));
        try m.put(m.b.pos, std.mem.sliceAsBytes(pos));
        try m.put(m.b.prev, std.mem.sliceAsBytes(prev));
        try m.put(m.b.cu, std.mem.sliceAsBytes(cu));
        try m.put(m.b.labels, std.mem.sliceAsBytes(lab));
        return rows;
    }

    /// y [rows, n] = x [rows, k] . w [n, k]^T, a plan kept a shape.
    fn linear(m: *Model, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize, out: cuda.cublaslt.DataType) !void {
        const key = [3]u64{ rows, n, k | (@as(u64, @intCast(@backingInt(out))) << 48) };
        const got = try m.linears.getOrPut(m.gpa, key);
        if (!got.found_existing) got.value_ptr.* = Linear.init(&m.lt, rows, n, k, out, workspace_bytes) catch |e| {
            m.linears.removeByPtr(got.key_ptr);
            return e;
        };
        try got.value_ptr.run(x, w, y, m.workspace.ptr, workspace_bytes, m.stream.handle);
    }

    /// Pinned host bytes to a scratch address, on the model's stream.
    fn put(m: *Model, dst: u64, bytes: []const u8) !void {
        try m.scratch.uploadAsync(dst - m.scratch.ptr, bytes, m.stream.handle);
    }

    fn get(m: *Model, src: u64, out: []u8) !void {
        try m.scratch.downloadAsync(src - m.scratch.ptr, out, m.stream.handle);
    }

    /// Where picture features may be written before a pass: free until the first layer, `rows_most` rows.
    pub fn pictureRows(m: *const Model) u64 {
        return m.b.y;
    }

    /// The layers over the uploaded rows; leaves each branch's final-normed last row in x.
    fn forward(m: *Model, trunk: usize, branches: usize, rows: usize, leaves: []const usize, placed: ?Placed) !void {
        const c = m.cfg;
        const s = m.stream;
        const b = m.b;
        const hd = c.heads * c.dim;
        const kvd = c.kv_heads * c.dim;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dim)));
        try m.embed.bf16(s, b.ids, m.w.embed, b.h, rows, c.hidden);
        if (placed) |p| {
            var from = p.features;
            for (p.runs) |run| {
                const n = 2 * run[1] * c.hidden;
                try m.scratch.copyFrom(b.h + 2 * run[0] * c.hidden - m.scratch.ptr, from, n, s.handle);
                from += n;
            }
        }
        try m.k.addRms(s, b.h, null, m.w.layers[0].op_norm, b.x, rows, c.hidden, c.eps);
        for (m.w.layers, 0..) |l, i| {
            switch (l.kind) {
                .conv => {
                    try m.linear(b.x, l.proj, b.wide, rows, 3 * c.hidden, c.hidden, .bf16);
                    try m.k.shortConv(s, b.wide, l.conv, b.prev, b.a, rows, c.hidden, c.taps);
                    try m.linear(b.a, l.out, b.y, rows, c.hidden, c.hidden, .bf16);
                },
                .attention => {
                    try m.linear(b.x, l.proj, b.wide, rows, hd + 2 * kvd, c.hidden, .bf16);
                    try m.k.qkNormRope(s, b.wide, l.q_norm, l.k_norm, b.pos, b.q, b.kk, b.v, rows, c.heads, c.kv_heads, c.dim, c.theta, c.eps);
                    try m.at.tree(s, b.q, b.kk, b.v, b.a, trunk, b.cu, branches, rows, c.heads, c.kv_heads, c.dim, scale);
                    try m.linear(b.a, l.out, b.y, rows, c.hidden, hd, .bf16);
                },
            }
            try m.k.addRms(s, b.h, b.y, l.ffn_norm, b.x, rows, c.hidden, c.eps);
            try m.linear(b.x, l.up, b.wide, rows, 2 * c.ffn, c.hidden, .bf16);
            try m.k.siluMul(s, b.wide, b.g, rows, c.ffn);
            try m.linear(b.g, l.down, b.y, rows, c.hidden, c.ffn, .bf16);
            if (i + 1 < m.w.layers.len) try m.k.addRms(s, b.h, b.y, m.w.layers[i + 1].op_norm, b.x, rows, c.hidden, c.eps);
        }
        // the final norm at each branch's last row alone: the head reads no other row
        for (leaves) |row| {
            const at = 2 * row * c.hidden;
            try m.k.addRms(s, b.h + at, b.y + at, m.w.norm, b.x + at, 1, c.hidden, c.eps);
        }
    }
};
