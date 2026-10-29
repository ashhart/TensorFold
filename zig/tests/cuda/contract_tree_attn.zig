//! tree_attention.cu (#746) under the contract on CUDA: paths byte for byte, attention head by head against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const tree = cuda.tree_attention;

/// A tree per stream: its rows (at most 16, the 27B's tree), each a path through earlier rows of its stream.
const width = 16;

/// The P x V chain of the longest row: every committed key, its path, a few combines.
fn keyChain(s: c.Shape) usize {
    return s.k + tree.max_depth + 16;
}

/// Blocks: four warps (sms 0) or one warp with 2 (sms 1) or 3 stages; `pages`: the _pg kernels, `swap`: their control.
pub const Variant = struct { d: usize, h: usize, hk: usize, sms: usize, name: []const u8, pages: usize = 0, swap: bool = false };

pub const variants = [_]Variant{
    .{ .d = 128, .h = 48, .hk = 4, .sms = 0, .name = "cuda tree attention dim 128, four-warp blocks (tf_tree_shared, tf_tree_part, tf_tree_merge)" },
    .{ .d = 128, .h = 48, .hk = 4, .sms = 1, .name = "cuda tree attention dim 128, one-warp blocks (tf_tree_shared_w1, tf_tree_part_w1, sliced merge)" },
    .{ .d = 128, .h = 48, .hk = 4, .sms = 1 << 20, .name = "cuda tree attention dim 128, one-warp blocks, 3 stages (tf_tree_shared_w1s3, tf_tree_part_w1s3)" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 0, .name = "cuda tree attention dim 256, four-warp blocks (tf_tree_shared_256, tf_tree_part_256, tf_tree_merge)" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 1, .name = "cuda tree attention dim 256, one-warp blocks (tf_tree_shared_256_w1, tf_tree_part_256_w1)" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 1 << 20, .name = "cuda tree attention dim 256, one-warp blocks, 3 stages (tf_tree_shared_256_w1s3)" },
    .{ .d = 128, .h = 48, .hk = 4, .sms = 0, .pages = 16, .name = "cuda tree attention dim 128 over pages of 16, four-warp blocks (tf_tree_shared_pg, tf_tree_part_pg)" },
    .{ .d = 128, .h = 48, .hk = 4, .sms = 1, .pages = 37, .name = "cuda tree attention dim 128 over pages of 37, one-warp blocks (tf_tree_shared_w1_pg, tf_tree_part_w1_pg)" },
    .{ .d = 128, .h = 48, .hk = 4, .sms = 1 << 20, .pages = 256, .name = "cuda tree attention dim 128 over pages of 256, 3 stages (tf_tree_shared_w1s3_pg, tf_tree_part_w1s3_pg)" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 0, .pages = 37, .name = "cuda tree attention dim 256 over pages of 37, four-warp blocks (tf_tree_shared_256_pg, tf_tree_part_256_pg)" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 1, .pages = 16, .name = "cuda tree attention dim 256 over pages of 16, one-warp blocks (tf_tree_shared_256_w1_pg, tf_tree_part_256_w1_pg)" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 1 << 20, .pages = 256, .name = "cuda tree attention dim 256 over pages of 256, 3 stages (tf_tree_shared_256_w1s3_pg, tf_tree_part_256_w1s3_pg)" },
    .{ .d = 128, .h = 48, .hk = 4, .sms = 0, .pages = 16, .swap = true, .name = "control: dim 128 paged tree attention whose first page id is an unheld page" },
    .{ .d = 256, .h = 24, .hk = 4, .sms = 1, .pages = 16, .swap = true, .name = "control: dim 256 one-warp paged tree attention whose first page id is an unheld page" },
};

/// Shape: drafted rows (trees of 16 over the streams), query heads (n) and stream 0's committed keys (k).
pub fn treeSpec(v: Variant) c.Spec {
    // Kolibri (dim 128, 48 heads over 4) and Qwen3.8-27B (dim 256, 24 over 4): one stream's tree, then four streams'
    const served: []const c.Shape = if (v.d == 128) &.{ .{ .rows = 16, .n = 48, .k = 2100 }, .{ .rows = 64, .n = 48, .k = 700 } } else &.{ .{ .rows = 16, .n = 24, .k = 2100 }, .{ .rows = 64, .n = 24, .k = 700 } };
    return .{ .name = v.name, .accum = .f32, .depth = keyChain, .ops = 1, .out = .bf16, .tiles = .{ .{ .size = width }, .{}, .{ .size = tree.span } }, .served = served, .sums = .{ .why = attn.why } };
}

pub const Tree = struct {
    rig: rig_.Rig,
    v: Variant,
    ta: tree.TreeAttention,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    scale: f32 = 0,
    ng: usize = 0,
    committed: std.ArrayList(usize) = .empty,
    row_stream: std.ArrayList(i32) = .empty,
    paths: std.ArrayList(i32) = .empty,
    depths: std.ArrayList(i32) = .empty,
    offs: std.ArrayList(i64) = .empty,
    q: std.ArrayList(u16) = .empty,
    kn: std.ArrayList(u16) = .empty,
    vn: std.ArrayList(u16) = .empty,
    cache: std.ArrayList(u16) = .empty,
    dev: [9]u64 = @splat(0),
    pages: ?tree.Pages = null,
    sc: tree.Scratch = undefined,
    out: u64 = 0,

    pub fn deinit(self: *Tree) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "committed", "row_stream", "paths", "depths", "offs", "q", "kn", "vn", "cache" }) |f| @field(self, f).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Tree) g.Kernel {
        return .{ .spec = treeSpec(self.v), .ctx = self, .vt = &vt, .control = if (self.v.swap) .shapes else null };
    }

    const vt = g.hooks(Tree);

    pub fn prepare(self: *Tree, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const v = self.v;
        if (shape.n != v.h or shape.rows == 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.shape = shape;
        const n = shape.rows;
        const ns = (n + width - 1) / width;
        try self.committed.resize(a, ns);
        for (self.committed.items, 0..) |*p, si| p.* = @max(1, shape.k - si * shape.k / ns);
        try self.row_stream.resize(a, n);
        try self.depths.resize(a, n);
        try self.paths.resize(a, n * tree.max_depth);
        @memset(self.paths.items, -1);
        var streams = try a.alloc(i32, 4 * ns);
        defer a.free(streams);
        const parents = try a.alloc(i32, n);
        defer a.free(parents);
        for (0..ns) |si| {
            const start = si * width;
            const w = @min(width, n - start);
            for (0..w) |j| {
                self.row_stream.items[start + j] = @intCast(si);
                parents[start + j] = if (j == 0 or rng.uintLessThan(u32, 6) == 0) -1 else @intCast(start + rng.uintLessThan(usize, j));
            }
            for ([_]usize{ start, w, self.committed.items[si], 0 }, 0..) |x, k| streams[4 * si + k] = @intCast(x);
        }
        self.ng = 0;
        var most: usize = 0;
        for (0..n) |r| {
            var depth: usize = 0;
            var cur: i32 = @intCast(r);
            while (cur >= 0) : (cur = parents[@intCast(cur)]) depth += 1;
            self.depths.items[r] = @intCast(depth);
            cur = @intCast(r);
            var slot = depth;
            while (slot > 0) : (cur = parents[@intCast(cur)]) {
                slot -= 1;
                self.paths.items[r * tree.max_depth + slot] = cur;
            }
            const keys = self.committed.items[@intCast(self.row_stream.items[r])] + depth;
            self.ng = @max(self.ng, tree.TreeAttention.groups(keys));
            most = @max(most, keys);
        }
        // one-warp blocks take only grids that fit their slice area; anything larger runs four-warp blocks
        if (v.sms > 0 and n * self.ng * v.hk > tree.slice_items) return false;
        self.ta.sms = v.sms;
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, v.d, most);
        const kv = v.hk * v.d;
        var total: usize = 0;
        for (self.committed.items) |p| total += p;
        try self.offs.resize(a, 2 * ns);
        var at: usize = 0;
        for (self.committed.items, 0..) |p, si| {
            self.offs.items[2 * si] = @intCast(at * kv);
            self.offs.items[2 * si + 1] = @intCast((total + at) * kv);
            at += p;
        }
        try self.cache.resize(a, 2 * total * kv);
        try self.q.resize(a, n * v.h * v.d);
        try self.kn.resize(a, n * kv);
        try self.vn.resize(a, n * kv);
        for ([_][]u16{ self.cache.items, self.q.items, self.kn.items, self.vn.items }) |x| for (x) |*e| {
            e.* = attn.normal(rng);
        };
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng, total),
        }
        self.scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(v.d))));
        for ([_][]const u8{ std.mem.sliceAsBytes(self.q.items), std.mem.sliceAsBytes(self.kn.items), std.mem.sliceAsBytes(self.vn.items), std.mem.sliceAsBytes(self.cache.items), std.mem.sliceAsBytes(self.offs.items), std.mem.sliceAsBytes(self.row_stream.items), std.mem.sliceAsBytes(streams), std.mem.sliceAsBytes(self.paths.items), std.mem.sliceAsBytes(self.depths.items) }, &self.dev) |bytes, *p| p.* = try self.rig.input(bytes);
        self.pages = if (v.pages > 0) try self.paged(rng) else null;
        const base = try self.rig.scratch(tree.TreeAttention.scratchBytes(n, v.h, v.hk, self.ng, v.d));
        self.sc = tree.TreeAttention.scratch(base, n, v.h, v.hk, self.ng, v.d);
        self.out = try self.rig.output(n * v.h * v.d * 2);
        return true;
    }

    /// The committed keys copied to shuffled `pages`-position pages (NaN in the rest): origin, offs and the table.
    fn paged(self: *Tree, rng: std.Random) !tree.Pages {
        const a = self.rig.gpu.gpa;
        const v = self.v;
        const kv = v.hk * v.d;
        const tok = v.pages;
        var stride: usize = 0;
        var npages: usize = 2; // pages no stream holds
        for (self.committed.items) |p| {
            stride = @max(stride, (p + tok - 1) / tok);
            npages += (p + tok - 1) / tok;
        }
        const ids = try a.alloc(i32, npages);
        defer a.free(ids);
        for (ids, 0..) |*x, i| x.* = @intCast(i);
        rng.shuffle(i32, ids);
        const ns = self.committed.items.len;
        const table = try a.alloc(i32, ns * stride);
        defer a.free(table);
        @memset(table, -1);
        const page = tok * kv;
        const pool = try a.alloc(u16, 2 * npages * page);
        defer a.free(pool);
        @memset(pool, c.toBf16(std.math.nan(f32)));
        var next: usize = 0;
        for (self.committed.items, 0..) |p, si| {
            for (0..(p + tok - 1) / tok) |j| {
                table[si * stride + j] = ids[next];
                next += 1;
            }
            for (0..p) |j| for (0..2) |part| {
                const src: usize = @as(usize, @intCast(self.offs.items[2 * si + part])) + j * kv;
                const dst = part * npages * page + @as(usize, @intCast(table[si * stride + j / tok])) * page + j % tok * kv;
                @memcpy(pool[dst..][0..kv], self.cache.items[src..][0..kv]);
            };
        }
        if (v.swap) table[0] = ids[next]; // the control: stream 0's first page is one no stream holds (NaN)
        const offs = try a.alloc(i64, 2 * ns);
        defer a.free(offs);
        for (0..ns) |si| {
            offs[2 * si] = 0;
            offs[2 * si + 1] = @intCast(npages * page);
        }
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(pool));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(offs));
        return .{ .table = try self.rig.input(std.mem.sliceAsBytes(table)), .page = page, .tok = tok, .stride = stride };
    }

    /// NaN in a query, +inf and -inf in values (committed, on a path, in one column), a NaN key, +inf at tiny weights.
    fn plant(self: *Tree) void {
        const d = self.v.d;
        const kv = self.v.hk * d;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const p0 = self.committed.items[0];
        const ko: usize = @intCast(self.offs.items[0]);
        const vo: usize = @intCast(self.offs.items[1]);
        self.q.items[3] = nan; // row 0, head 0
        self.cache.items[vo + (p0 - 1) * kv + 0 * d + 7] = inf;
        self.vn.items[1 * kv + 1 * d + 9] = ninf; // path row 1
        if (p0 >= 3) {
            self.cache.items[vo + (p0 - 2) * kv + 2 * d + 11] = inf;
            self.cache.items[vo + (p0 - 3) * kv + 2 * d + 11] = ninf;
        }
        self.cache.items[ko + (p0 - 1) * kv + 3 * d + 0] = nan;
        self.under(1, 21);
    }

    /// Key head kh, row 0: key p0 - 1 all 8, its other keys -8 with +inf at e; queries 2 (inf) and 8 (weight 0: NaN).
    fn under(self: *Tree, kh: usize, e: usize) void {
        const d = self.v.d;
        const kv = self.v.hk * d;
        const G = self.v.h / self.v.hk;
        const p0 = self.committed.items[0];
        const ko: usize = @intCast(self.offs.items[0]);
        const vo: usize = @intCast(self.offs.items[1]);
        const inf = c.toBf16(std.math.inf(f32));
        @memset(self.q.items[(kh * G) * d ..][0..d], c.toBf16(2));
        @memset(self.q.items[(kh * G + 1) * d ..][0..d], c.toBf16(8));
        for (0..p0) |j| {
            @memset(self.cache.items[ko + j * kv + kh * d ..][0..d], c.toBf16(if (j == p0 - 1) 8 else -8));
            if (j != p0 - 1) self.cache.items[vo + j * kv + kh * d + e] = inf;
        }
        @memset(self.kn.items[kh * d ..][0..d], c.toBf16(-8)); // row 0's path: itself
        self.vn.items[kh * d + e] = inf;
    }

    /// Gate 4: every key alike (each p is 1), committed values mirrored per column, path values 0.
    fn stress(self: *Tree, rng: std.Random, total: usize) !void {
        const a = self.rig.gpu.gpa;
        const v = self.v;
        const kv = v.hk * v.d;
        const key = try a.alloc(u16, kv);
        defer a.free(key);
        for (key) |*e| e.* = attn.normal(rng);
        for (0..total) |j| @memcpy(self.cache.items[j * kv ..][0..kv], key);
        for (0..self.shape.rows) |j| @memcpy(self.kn.items[j * kv ..][0..kv], key);
        @memset(self.vn.items, 0);
        const col = try a.alloc(f32, total);
        defer a.free(col);
        for (self.committed.items, 0..) |p, si| for (0..kv) |e| {
            attn.mirrored(rng, col[0..p]);
            const vo: usize = @intCast(self.offs.items[2 * si + 1]);
            for (col[0..p], 0..) |x, j| self.cache.items[vo + j * kv + e] = c.toBf16(x);
        };
    }

    pub fn run(self: *Tree) !void {
        const v = self.v;
        const rws: tree.Rows = .{ .row_stream = self.dev[5], .streams = self.dev[6], .offs = self.dev[4], .origin = self.dev[3], .paths = self.dev[7], .depths = self.dev[8], .pages = self.pages };
        try self.ta.run(self.rig.s, self.dev[0], self.dev[1], self.dev[2], rws, self.sc, self.out, self.shape.rows, v.h, v.hk, v.d, self.ng, self.scale);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Tree) usize {
        return self.shape.rows * self.v.h * self.v.d;
    }

    /// Output head i (row i / h, head i % h): its stream's committed keys, then its path root first.
    pub fn load(self: *Tree, i: usize, hd: *attn.Head) !void {
        const v = self.v;
        const d = v.d;
        const kv = v.hk * d;
        const r = i / v.h;
        const head = i % v.h;
        const kh = head / (v.h / v.hk);
        const si: usize = @intCast(self.row_stream.items[r]);
        const p = self.committed.items[si];
        const ko: usize = @intCast(self.offs.items[2 * si]);
        const vo: usize = @intCast(self.offs.items[2 * si + 1]);
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[(r * v.h + head) * d + e]);
        for (0..p) |j| for (0..d) |e| {
            hd.k[j * d + e] = c.fromBf16(self.cache.items[ko + j * kv + kh * d + e]);
            hd.v[j * d + e] = c.fromBf16(self.cache.items[vo + j * kv + kh * d + e]);
        };
        const depth: usize = @intCast(self.depths.items[r]);
        for (0..depth) |j| {
            const node: usize = @intCast(self.paths.items[r * tree.max_depth + j]);
            for (0..d) |e| {
                hd.k[(p + j) * d + e] = c.fromBf16(self.kn.items[node * kv + kh * d + e]);
                hd.v[(p + j) * d + e] = c.fromBf16(self.vn.items[node * kv + kh * d + e]);
            }
        }
        hd.n = p + depth;
    }

    fn heads(self: *Tree) usize {
        return self.shape.rows * self.v.h;
    }

    pub fn check(self: *Tree, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), self.scale, try self.rig.outputBytes(), at);
    }

    pub fn reference(self: *Tree, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), self.scale, want, mag);
    }

    pub fn simulate(self: *Tree, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), self.scale, acc, order, out);
    }
};

fn noSums(_: c.Shape) usize {
    return 0;
}

const paths_rule = "integers only: each row's depth (at most 128) and its nearest 128 ancestors root first; other slots untouched";

pub fn pathsSpec() c.Spec {
    return .{ .name = "cuda tree attention tf_tree_paths", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = paths_rule }, .tiles = .{ .{ .size = 128 }, .{}, .{} }, .served = &.{ .{ .rows = 16, .n = 1, .k = 1 }, .{ .rows = 128, .n = 1, .k = 1 }, .{ .rows = 600, .n = 1, .k = 1 } } };
}

/// tf_tree_paths over a forest of `rows` rows: random trees, and from 300 rows a 200-deep chain past the 128 cap.
pub const Paths = struct {
    rig: rig_.Rig,
    ta: *const tree.TreeAttention,
    n: usize = 0,
    parents: std.ArrayList(i32) = .empty,
    dev: [3]u64 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Paths) void {
        self.parents.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Paths) g.Kernel {
        return .{ .spec = pathsSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Paths);

    pub fn prepare(self: *Paths, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0) return false;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.n = shape.rows;
        try self.parents.resize(self.rig.gpu.gpa, self.n);
        for (self.parents.items, 0..) |*p, r| p.* = if (r == 0) -1 else if (self.n >= 300 and r < 200) @intCast(r - 1) else if (rng.uintLessThan(u32, 5) == 0) -1 else @intCast(rng.uintLessThan(usize, r));
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.parents.items));
        self.dev[1] = try self.rig.output(self.n * tree.max_depth * 4);
        self.dev[2] = try self.rig.output(self.n * 4);
        return true;
    }

    pub fn run(self: *Paths) !void {
        try self.ta.paths(self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.n);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Paths) usize {
        return self.n * (tree.max_depth + 1) * 4;
    }

    pub fn reference(_: *Paths, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Paths, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try self.rig.outputBytes();
        const want = try a.alloc(u8, got.len);
        defer a.free(want);
        @memset(want, 0xff);
        const paths = std.mem.bytesAsSlice(i32, want[0 .. self.n * tree.max_depth * 4]);
        const depths = std.mem.bytesAsSlice(i32, want[self.n * tree.max_depth * 4 ..]);
        for (0..self.n) |r| {
            var depth: usize = 0;
            var cur: i32 = @intCast(r);
            while (cur >= 0 and depth < tree.max_depth) : (cur = self.parents.items[@intCast(cur)]) depth += 1;
            depths[r] = @intCast(depth);
            cur = @intCast(r);
            var slot = depth;
            while (slot > 0) : (cur = self.parents.items[@intCast(cur)]) {
                slot -= 1;
                paths[r * tree.max_depth + slot] = cur;
            }
        }
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} is 0x{x:0>2} where the host gives 0x{x:0>2}", .{ at, got.len, got[at], want[at] });
        return null;
    }
};
