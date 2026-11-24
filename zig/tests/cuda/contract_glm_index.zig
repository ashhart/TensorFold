//! GLM's DSA index kernels on KV pages under the contract on CUDA: the pool scores and the token lists' page rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");
const select = @import("contract_glm_select.zig");
const dsel = @import("contract_glm_dsel.zig");
const full_k = @import("contract_glm_full.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sparse = glm.sparse;
const dk = pages.dk;
const heads = 32;
const w_stride = dk + heads; // the head weights sit after the index keys in a row, as the block passes them

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A key's 128 products, then the 32 heads' weighted sum.
fn scoreDepth(_: c.Shape) usize {
    return dk + heads;
}

const nan_relu = "relu is fmaxf(x, 0), as Triton's tl.maximum: a NaN dot scores 0 where float64 keeps the NaN";

/// A decode window's served shapes: decodeLoop's 2, 4 and 1 tiles a CTA, as cuda_sparse launches Set.dscore.
const decode_served = [_]c.Shape{ .{ .rows = 8, .n = 4096, .k = dk }, .{ .rows = 16, .n = 51200, .k = dk }, .{ .rows = 4, .n = 131072, .k = dk } };

/// scores: rows x pool columns (k: an index key's width); tokens: rows x K pools a row; full: full GLM-5.3's indexer.
pub fn spec(comptime which: enum { scores, tokens, scores_full, tokens_full }) c.Spec {
    return switch (which) {
        .scores => .{ .name = "cuda glm tf_glm_scores on pages out of order", .accum = .f32, .depth = scoreDepth, .ops = 2, .out = .f32, .served = &.{ .{ .rows = 8, .n = 2048, .k = dk }, .{ .rows = 64, .n = 1024, .k = dk } }, .known = .{ .gate = .nonfinite, .finding = nan_relu } },
        .tokens => .{ .name = "cuda glm tf_glm_dsa_tokens on pages out of order", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "integers only: pool ids, positions and page ids" }, .served = &.{ .{ .rows = 8, .n = sparse.topk_pools, .k = 1 }, .{ .rows = 64, .n = sparse.topk_pools, .k = 1 } } },
        .scores_full => .{ .name = "cuda glm tf_glm_scores, pools of 1 and 64 roped query dims", .accum = .f32, .depth = scoreDepth, .ops = 2, .out = .f32, .served = &.{ .{ .rows = 8, .n = 1024, .k = dk }, .{ .rows = 64, .n = 2048, .k = dk } }, .known = .{ .gate = .nonfinite, .finding = nan_relu } },
        .tokens_full => .{ .name = "cuda glm tf_glm_dsa_tokens, pools of 1", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "integers only: pool ids, positions and page ids" }, .served = &.{ .{ .rows = 8, .n = 2048, .k = 1 }, .{ .rows = 64, .n = 2048, .k = 1 } } },
    };
}

/// A planted fault on full GLM-5.3's indexer: queries or keys left unroped, pools read as 4, rows one position late.
pub const Fault = enum { none, unroped, pool_of_4, late };

const maxNan = pages.maxNan;
const held = pages.held;
const sumIn = pages.sumIn;

/// sparse._scores: each row's visible pools scored by 32 heads (a weighted sum of relu(q . key)), the rest -inf.
pub const Scores = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    /// tf_glm_index_scores, which must equal `f`'s bytes (`f` is then the baseline it replaces).
    fast: ?cuda.Function = null,
    base: u64 = 0,
    wrong_table: bool = false,
    decode: bool = false, // a decode window's tiles a CTA (glm.index.decodeLoop), as cuda_sparse launches Set.dscore
    short: bool = false, // the control: the decode tiling is told it has one 64-pool tile fewer
    /// Rows a pooled key sums (1: each key) and the queries' roped dims (full GLM-5.3: 1, 64), a fault, a name.
    pool: usize = sparse.pool,
    rope: usize = 0,
    fault: Fault = .none,
    label: ?[]const u8 = null,
    freqs: [32]f32 = @splat(0),
    qr: []f64 = &.{}, // the queries roped in float64
    extra: []f64 = &.{}, // per score: the roped queries' bf16 rounding and sincosf
    inv: u64 = 0,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    w: []u16 = &.{},
    keys: []u16 = &.{}, // the pooled keys in pool order, as the pages hold them
    ids: []i32 = &.{},
    at: usize = 0,
    seen: usize = 0, // the last row's visible pools
    scale: f32 = 0, // the dots' scale and the head weights' (served: 1/sqrt(128), 1/sqrt(32); stress: powers of two)
    wscale: f32 = 0,
    in: [3]u64 = @splat(0),
    pos: u64 = 0,
    pt: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Scores) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Scores) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "q", "w", "keys" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
        a.free(self.ids);
        self.ids = &.{};
        a.free(self.qr);
        self.qr = &.{};
        a.free(self.extra);
        self.extra = &.{};
    }

    pub fn kernel(self: *Scores) g.Kernel {
        var declared = if (self.pool == 1) spec(.scores_full) else spec(.scores);
        if (self.fast != null) {
            declared.name = "cuda glm tf_glm_index_scores on pages out of order";
            declared.replaces = "tf_glm_scores";
        }
        if (self.decode) {
            declared.name = "cuda glm decode scores tf_glm_index_scores on pages out of order";
            declared.served = &decode_served;
        }
        if (self.short) declared.name = "control: GLM decode scores write one tile short";
        if (self.wrong_table) declared.name = "control: GLM paged scores use a swapped table";
        if (self.label) |name| declared.name = name;
        const planted = self.wrong_table or self.short or self.fault != .none;
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (planted) .shapes else null };
    }

    const vt = g.hooks(Scores);

    fn visible(self: *const Scores, r: usize) usize {
        return (self.at + r + 1) / self.pool;
    }

    /// A page's step in bytes: the test's two-cache page, or past its 512 keys at pools of 1.
    fn pageStep(self: *const Scores) usize {
        return if (self.pool == 1) pages.part_bytes + pages.rows_a_page * (pages.lw + dk) * 2 else pages.page_bytes;
    }

    /// Queries and keys in [-2, 2); stress: integers in [-8, 8] and power-of-two scales; non-finite: NaN and inf.
    pub fn prepare(self: *Scores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dk or shape.n % 64 != 0 or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // decode: up to 8,192 visible pools (64 pages)
        self.at = if (self.decode) @as(usize, @min(shape.n, 8192)) * self.pool - shape.rows - 4 else if (self.pool == 1) (if (shape.rows <= 8) @as(usize, 1000) else 1900) else if (shape.rows <= 8) 4100 else 2000;
        self.scale = if (fill == .stress) 0.125 else @floatCast(std.math.pow(f64, dk, -0.5));
        self.wscale = if (fill == .stress) 0.25 else sparse.headScale(heads);
        self.seen = self.visible(shape.rows - 1);
        if (self.seen > shape.n) return false;
        self.q = try a.alloc(u16, shape.rows * heads * dk);
        self.w = try a.alloc(u16, shape.rows * w_stride);
        self.keys = try a.alloc(u16, self.seen * dk);
        for ([_][]u16{ self.q, self.keys }) |x| for (x) |*v| {
            v.* = c.toBf16(if (fill == .stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8) else rng.float(f32) * 4 - 2);
        };
        for (self.w) |*v| v.* = c.toBf16(if (fill == .stress) 1 else rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            self.keys[2 * dk + 5] = c.toBf16(std.math.inf(f32));
        }
        try self.ropeQueries(fill);
        const pp = pages.rows_a_page / self.pool; // pools a page holds
        const n_pages = (self.seen + pp - 1) / pp;
        self.ids = try pages.shuffled(a, rng, n_pages);
        const arena = try a.alloc(u8, (n_pages + 3) * self.pageStep());
        defer a.free(arena);
        @memset(arena, 0);
        const pk = pages.part_bytes + pages.rows_a_page * pages.lw * 2; // the cache's pooled keys in page 0
        for (0..self.seen) |p| {
            const at = pk + @as(usize, @intCast(self.ids[p / pp])) * self.pageStep() + (p % pp) * dk * 2;
            @memcpy(arena[at..][0 .. dk * 2], std.mem.sliceAsBytes(self.keys[p * dk ..][0..dk]));
        }
        const p: i32 = @intCast(self.at);
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.in[2] = try self.rig.input(arena) + pk;
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try pages.uploadTable(&self.rig, self.ids, self.wrong_table);
        self.inv = if (self.rope > 0) try self.rig.input(std.mem.sliceAsBytes(&self.freqs)) else 0;
        self.out = try self.rig.output(shape.rows * shape.n * 4);
        self.base = try self.rig.scratch(shape.rows * shape.n * 4);
        return true;
    }

    /// Each row's queries roped at its position in float64 (stress: frequencies 0, so the rope is exact).
    fn ropeQueries(self: *Scores, fill: g.Fill) !void {
        const a = self.rig.gpu.gpa;
        self.freqs = if (self.rope > 0 and fill != .stress) cuda.mla.ropeFreqs(8.0e6) else @splat(0);
        self.qr = try a.alloc(f64, self.q.len);
        for (self.qr, self.q) |*y, x| y.* = c.fromBf16(x);
        self.extra = try a.alloc(f64, self.q.len);
        @memset(self.extra, 0);
        for (0..self.shape.rows) |r| for (0..heads) |h| for (0..self.rope / 2) |i| {
            const at = (r * heads + h) * dk + 2 * i;
            const x0 = self.qr[at];
            const x1 = self.qr[at + 1];
            const p: f32 = @floatFromInt(self.at + r);
            const angle: f64 = self.freqs[i] * p; // the fp32 product, as the kernel forms it
            const co = @cos(angle);
            const si = @sin(angle);
            self.qr[at] = x0 * co - x1 * si;
            self.qr[at + 1] = x1 * co + x0 * si;
            // bf16's half ulp of the roped value, then sincosf's 2 ulp, two products and the add (stress: exact)
            const turn = 8 * 0x1p-24 * (@abs(x0) + @abs(x1));
            const exact = fill == .stress or !std.math.isFinite(turn);
            for ([2]usize{ at, at + 1 }) |d| self.extra[d] = if (exact) 0 else 0x1p-8 * @abs(self.qr[d]) + turn;
        };
    }

    pub fn run(self: *Scores) !void {
        try self.launch(self.fast orelse self.f, self.fast != null, self.out);
        try self.rig.s.synchronize();
    }

    fn launch(self: *Scores, f: cuda.Function, fast: bool, out: u64) !void {
        const rows = self.shape.rows;
        const nb = if (fast and self.short) self.shape.n / 64 - 1 else self.shape.n / 64;
        const loop: usize = if (fast and self.decode) glm.index.decodeLoop(rows, self.shape.n / 64) else if (fast) glm.index.score_tiles else if (rows >= sparse.score_loop_from) sparse.score_loop else 1;
        const pool: c_int = if (self.fault == .pool_of_4) 4 else @intCast(self.pool);
        const rope: c_int = if (self.fault == .unroped) 0 else @intCast(self.rope);
        const ps: i64 = @intCast(self.pageStep());
        var a: cuda.Args = .{};
        if (fast) {
            inline for (.{ self.in[0], self.in[1], @as(i64, w_stride), self.in[2], out, self.pos, @as(i64, @intCast(rows)), @as(i64, @intCast(self.shape.n)), @as(i64, @intCast(nb)), self.scale, self.wscale, @as(i64, heads), @as(c_int, @intCast(loop)), self.pt, ps, pool, rope, self.inv }) |v| a.add(v);
            var g2 = glm.index.scoresGrid(rows, nb);
            // cuda_sparse's decode grid: nb tiles in runs of loop
            if (self.decode) g2[1] = @intCast(@max(1, (nb + loop - 1) / loop));
            try cuda.launch.launch(f, .{ .grid = .{ .x = g2[0], .y = g2[1] }, .block = .{ .x = glm.index.threads } }, self.rig.s, &a);
        } else {
            inline for (.{ self.in[0], self.in[1], @as(i64, w_stride), self.in[2], out, self.pos, @as(i64, @intCast(self.shape.n)), @as(i64, @intCast(nb)), self.scale, self.wscale, @as(i64, heads), @as(c_int, @intCast(loop)), self.pt, ps, pool, rope, self.inv }) |v| a.add(v);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((nb + loop - 1) / loop) }, .block = .{ .x = 128 } }, self.rig.s, &a);
        }
    }

    /// tf_glm_scores on the same inputs, as cuda_sparse launches it.
    pub fn replaced(self: *Scores) ![]const u8 {
        try self.launch(self.f, false, self.base);
        return self.rig.bytes(self.base, self.shape.rows * self.shape.n * 4);
    }

    pub fn count(self: *Scores) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Row r's score of pool p with every sum in `acc` (null: float64), its terms' magnitude into `mag`.
    fn score(self: *const Scores, r: usize, p: usize, acc: ?c.Accum, order: g.Order, mag: *f64) f64 {
        mag.* = 0;
        if (p >= self.visible(r)) return -std.math.inf(f64);
        const scale: f64 = self.scale;
        const wscale: f64 = self.wscale;
        var dots: [dk]f64 = undefined;
        var terms: [heads]f64 = undefined;
        for (&terms, 0..) |*t, h| {
            var abs: f64 = 0;
            for (&dots, 0..) |*x, d| {
                x.* = self.qr[(r * heads + h) * dk + d] * @as(f64, c.fromBf16(self.keys[p * dk + d]));
                abs += @abs(x.*);
            }
            const wv = c.fromBf16(self.w[r * w_stride + h]) * wscale;
            const dot = if (acc) |k| sumIn(k, order, &dots) else sumIn(.none, .sequential, &dots);
            t.* = wv * maxNan(dot * scale, 0);
            mag.* += if (std.math.isFinite(abs)) @abs(wv) * scale * abs else 0;
        }
        return if (acc) |k| sumIn(k, order, &terms) else sumIn(.none, .sequential, &terms);
    }

    pub fn reference(self: *Scores, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| for (0..self.shape.n) |p| {
            want[r * self.shape.n + p] = self.score(r, p, null, .sequential, &mag[r * self.shape.n + p]);
        };
    }

    /// The roped queries' rounding reaches a score through each head's weight, the scale and its keys; -inf is exact.
    pub fn bounds(self: *Scores, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow) |*l| if (!std.math.isFinite(l.*)) {
            l.* = 0;
        };
        if (self.rope == 0) return;
        for (0..self.shape.rows) |r| for (0..self.visible(r)) |p| {
            var e: f64 = 0;
            for (0..heads) |h| {
                var s: f64 = 0;
                for (0..self.rope) |d| s += self.extra[(r * heads + h) * dk + d] * @abs(@as(f64, c.fromBf16(self.keys[p * dk + d])));
                const w = @abs(@as(f64, c.fromBf16(self.w[r * w_stride + h]))) * @as(f64, self.wscale) * @as(f64, self.scale);
                if (std.math.isFinite(s)) e += w * s;
            }
            allow[r * self.shape.n + p] += e * (1 + 0x1p-7);
        };
    }

    pub fn simulate(self: *Scores, acc: c.Accum, order: g.Order, out: []f64) !void {
        var mag: f64 = 0;
        for (0..self.shape.rows) |r| for (0..self.shape.n) |p| {
            out[r * self.shape.n + p] = held(.f32, self.score(r, p, acc, order, &mag));
        };
    }
};

/// sparse.select_tokens' assembly on pages: each row's K pools' tokens and the visible tail as rows in the pages.
pub const Tokens = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    wrong_table: bool = false,
    /// Tokens a pool holds (full GLM-5.3: 1), a fault (late: launched one position on), a name.
    pool: usize = sparse.pool,
    fault: Fault = .none,
    label: ?[]const u8 = null,
    shape: c.Shape = undefined,
    pools: []i64 = &.{},
    ids: []i32 = &.{},
    at: usize = 0,
    in: u64 = 0,
    pos: u64 = 0,
    pt: u64 = 0,
    tokens: u64 = 0,
    counts: u64 = 0,
    msg: [200]u8 = undefined,

    const page_rows = pages.page_bytes / (pages.lw * 2); // a page's step in latent rows

    pub fn deinit(self: *Tokens) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Tokens) void {
        const a = self.rig.gpu.gpa;
        a.free(self.pools);
        a.free(self.ids);
        self.pools = &.{};
        self.ids = &.{};
    }

    pub fn kernel(self: *Tokens) g.Kernel {
        var declared = if (self.pool == 1) spec(.tokens_full) else spec(.tokens);
        if (self.wrong_table) declared.name = "control: GLM paged token assembly uses a swapped table";
        if (self.label) |name| declared.name = name;
        const planted = self.wrong_table or self.fault != .none;
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (planted) .shapes else null };
    }

    const vt = g.hooks(Tokens);

    fn width(self: *const Tokens) usize {
        return self.pool * self.shape.n + self.pool - 1;
    }

    /// From row 2046 on, so the first rows still see K pools or fewer (count 0) and the rest select K of theirs.
    pub fn prepare(self: *Tokens, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.n == 0 or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = 2046;
        self.pools = try a.alloc(i64, shape.rows * shape.n);
        for (0..shape.rows) |r| {
            const seen = (self.at + r + 1) / self.pool;
            for (self.pools[r * shape.n ..][0..shape.n]) |*v| v.* = rng.uintLessThan(u32, @intCast(seen));
        }
        self.ids = try pages.shuffled(a, rng, (self.at + shape.rows + @intFromBool(self.fault == .late) + pages.rows_a_page - 1) / pages.rows_a_page);
        const p: i32 = @intCast(self.at + @intFromBool(self.fault == .late));
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.pools));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try pages.uploadTable(&self.rig, self.ids, self.wrong_table);
        self.tokens = try self.rig.output(shape.rows * self.width() * 4);
        self.counts = try self.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(self: *Tokens) !void {
        var a: cuda.Args = .{};
        a.add(self.in);
        a.add(self.pos);
        a.add(@as(c_int, @intCast(self.shape.n)));
        a.add(self.tokens);
        a.add(self.counts);
        a.add(self.pt);
        a.add(@as(i64, page_rows));
        a.add(@as(c_int, @intCast(self.pool)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Tokens) usize {
        return self.shape.rows * (self.width() + 1) * 4;
    }

    pub fn reference(_: *Tokens, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Token t's row in the pages: row t % 512 of page ids[t / 512], pages page_rows rows apart.
    fn row(self: *const Tokens, t: i64) i32 {
        const at: usize = @intCast(t);
        return @intCast(@as(i64, self.ids[at / pages.rows_a_page]) * page_rows + @as(i64, @intCast(at % pages.rows_a_page)));
    }

    /// Every row's list against the pools and the table on the host, and its count (0 while K pools cover it).
    pub fn check(self: *Tokens, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const w = self.width();
        const k = self.shape.n;
        const n_a = self.pool; // tokens a pool
        const got = try a.dupe(u8, try self.rig.bytes(self.tokens, self.shape.rows * w * 4));
        defer a.free(got);
        const counts = try self.rig.bytes(self.counts, self.shape.rows * 4);
        for (0..self.shape.rows) |r| {
            const q: i64 = @intCast(self.at + r);
            const seen = @divFloor(q + 1, @as(i64, @intCast(n_a)));
            var tail: i32 = 0;
            for (0..w) |i| {
                const want: i32 = if (i < n_a * k) self.row(self.pools[r * k + i / n_a] * @as(i64, @intCast(n_a)) + @as(i64, @intCast(i % n_a))) else blk: {
                    const t = seen * @as(i64, @intCast(n_a)) + @as(i64, @intCast(i - n_a * k));
                    tail += @intFromBool(t <= q);
                    break :blk if (t <= q) self.row(t) else -1;
                };
                const v = std.mem.readInt(i32, got[(r * w + i) * 4 ..][0..4], .little);
                if (v != want) return try std.fmt.bufPrint(&self.msg, "row {d} token {d}: {d} where the table gives {d}", .{ r, i, v, want });
            }
            const n: i32 = if (seen > k) @as(i32, @intCast(n_a * k)) + tail else 0;
            const v = std.mem.readInt(i32, counts[r * 4 ..][0..4], .little);
            if (v != n) return try std.fmt.bufPrint(&self.msg, "row {d}: count {d} where {d} is due", .{ r, v, n });
        }
        return null;
    }
};

/// The index scores and the decode selects (and the decode select's control), one runner entry for GLM's index.
pub const Index = struct {
    scores: Scores,
    select: select.Select,
    dsel: dsel.Dsel,
    dsel_short: dsel.Dsel,
    fns: glm.glue.Kernels,
    full: ?*full_k.Full = null, // full GLM-5.3's indexer cases, made by append
    a: ?std.mem.Allocator = null,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, fns: glm.glue.Kernels) Index {
        return .{ .scores = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fns.scores }, .select = .{ .rig = .{ .gpu = gpu, .s = s }, .wide = fns.select_rows_wide, .narrow = fns.select_rows }, .dsel = .{ .base = .{ .rig = .{ .gpu = gpu, .s = s }, .wide = fns.select_rows_wide, .narrow = fns.select_rows }, .fns = fns }, .dsel_short = .{ .base = .{ .rig = .{ .gpu = gpu, .s = s }, .wide = fns.select_rows_wide, .narrow = fns.select_rows }, .fns = fns, .short = true }, .fns = fns };
    }

    /// After init's value sits at its final address: each kernel's context points into it.
    pub fn append(self: *Index, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        try kernels.append(a, self.scores.kernel());
        try kernels.append(a, self.select.kernel());
        try kernels.append(a, self.dsel.kernel());
        try kernels.append(a, self.dsel_short.kernel());
        self.full = try full_k.Full.create(a, self.scores.rig.gpu, self.scores.rig.s, self.fns, kernels);
        self.a = a;
    }

    pub fn deinit(self: *Index) void {
        if (self.full) |f| f.destroy(self.a.?);
        self.dsel_short.deinit();
        self.dsel.deinit();
        self.select.deinit();
        self.scores.deinit();
    }
};
