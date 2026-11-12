//! The 64-row prompt attention tile (prefill.hip) under the contract on HIP: over pages, and flat bit-equal to it.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const page = 64;

/// A caller's query heads (a shape's n) over its kv heads at its head dim.
const Geometry = struct { heads: usize, kv: usize, d: usize };

/// The 27B's and the 35B-A3B's full-attention layers, and Nemotron's.
const geometries = [_]Geometry{
    .{ .heads = 24, .kv = 4, .d = 256 },
    .{ .heads = 16, .kv = 2, .d = 256 },
    .{ .heads = 32, .kv = 2, .d = 128 },
};

/// Prompt chunks (rows from position k): a first chunk with partial pages, later chunks over whole tiles before them.
const served = [_]c.Shape{
    .{ .rows = 200, .n = 24, .k = 0 },
    .{ .rows = 150, .n = 24, .k = 320 },
    .{ .rows = 96, .n = 24, .k = 100 },
    .{ .rows = 64, .n = 16, .k = 2048 },
    .{ .rows = 128, .n = 32, .k = 1024 },
};

const names = [2][2][]const u8{
    .{
        "hip prompt tile tf_fa_wide_f16 (flat fp16 cache, no served caller)",
        "hip prompt tile tf_fa_wide_bf16 (flat bf16 cache, no served caller)",
    },
    .{ "hip prompt tile tf_fa_paged_f16 (fp16 cache pages)", "hip prompt tile tf_fa_paged_bf16 (bf16 cache pages)" },
};

/// The edge entries: non-finite values only at keys past some rows' edges; those rows must ignore them, later rows not.
const edge_names = [2][2][]const u8{
    .{
        "hip prompt tile tf_fa_wide_f16 (flat fp16 cache), non-finite values only past a row's edge",
        "hip prompt tile tf_fa_wide_bf16 (flat bf16 cache), non-finite values only past a row's edge",
    },
    .{
        "hip prompt tile tf_fa_paged_f16 (fp16 cache pages), non-finite values only past a row's edge",
        "hip prompt tile tf_fa_paged_bf16 (bf16 cache pages), non-finite values only past a row's edge",
    },
};

/// Their planted controls: the tile with no edge pass and no mask, so 0 x inf past a row's edge gives NaN there.
const control_names = [2][2][]const u8{
    .{ "control: prompt tile tf_fa_wide_f16 without its edge pass", "control: prompt tile tf_fa_wide_bf16 without its edge pass" },
    .{ "control: prompt tile tf_fa_paged_f16 without its edge pass", "control: prompt tile tf_fa_paged_bf16 without its edge pass" },
};

/// The served entry, or an edge entry: its own plant and a span a page past its last row.
pub const Mode = enum { served, edge };

/// What gate 1 finds on the tile: NaN scores leave its max, and its P x V takes a seen key whose weight underflows.
const finding = "a row whose every score is NaN writes 0 where float64 gives NaN (its running max is fmaxf, " ++
    "which drops NaN, and a row at max -inf takes p = 0), and an infinite value at a key the row sees under " ++
    "fp32's least weight gives NaN where float64 gives inf, as 0 x inf enters P x V";

/// A row's P x V chain: a dot2 a key pair and a fold a tile, under its keys plus a tile.
fn keyChain(s: c.Shape) usize {
    return s.k + s.rows + page;
}

/// Rows of a prompt chunk (rows), query heads (n) and the chunk's first position (k); the flat tile equals the paged.
pub fn spec(k: kit.Kind, paged: bool, mode: Mode, planted: bool) c.Spec {
    const bf16 = k == .bf16;
    const p = @intFromBool(paged);
    const b = @intFromBool(bf16);
    return .{
        .name = if (planted) control_names[p][b] else if (mode == .edge) edge_names[p][b] else names[p][b],
        .accum = .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .f32,
        .replaces = if (paged or planted) null else if (bf16) "tf_fa_paged_bf16" else "tf_fa_paged_f16",
        .tiles = .{ .{ .size = 64 }, .{}, .{ .size = page, .min = 0 } },
        .served = &served,
        .sums = .{ .why = attn.why },
        .known = if (mode == .served) .{ .gate = .nonfinite, .finding = finding } else null,
    };
}

/// shape.rows prompt rows of shape.n query heads at positions shape.k on, each over every key up to its own.
pub const Prefill = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    /// tf_fa_paged_* over a permuted page table; else tf_fa_wide_* over flat keys, the paged tile its baseline.
    paged: bool,
    mode: Mode = .served,
    /// A planted control (tf_ctl_fa_*_unmasked_*) launched in place of the served dispatch.
    planted: ?hip.Function = null,
    arena: std.heap.ArenaAllocator,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    geo: Geometry = undefined,
    q: []f64 = &.{}, // [heads][rows][d]
    kc: []f64 = &.{}, // [kv][span][d]
    vc: []f64 = &.{},
    q32: u64 = 0,
    flat: [2]u64 = @splat(0), // k and v as [kv][span][d]
    pool: [2]u64 = @splat(0), // k and v as decode/pages.hpp lays a pool out
    table: u64 = 0,
    pages: usize = 0,
    out: u64 = 0,
    base: u64 = 0, // the paged tile's output on the flat tile's inputs

    pub fn init(
        a: std.mem.Allocator,
        rt: *const hip.Runtime,
        s: hip.Stream,
        l: *const hip.Launcher,
        paged: bool,
    ) Prefill {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .paged = paged, .arena = .init(a) };
    }

    pub fn deinit(self: *Prefill) void {
        if (self.r) |*r| r.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Prefill) g.Kernel {
        const planted = self.planted != null;
        const control: ?g.Gate = if (planted) .nonfinite else null;
        return .{ .spec = spec(self.k, self.paged, self.mode, planted), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Prefill);

    fn span(self: *const Prefill) usize {
        return self.shape.k + self.shape.rows + @as(usize, if (self.mode == .edge) page else 0);
    }

    /// Element e of kv head kh's key `key` in kc and vc.
    fn at(self: *const Prefill, kh: usize, key: usize, e: usize) usize {
        return (kh * self.span() + key) * self.geo.d + e;
    }

    /// Row r's position.
    fn pos(self: *const Prefill, r: usize) usize {
        return self.shape.k + r;
    }

    pub fn prepare(self: *Prefill, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const geo = for (geometries) |x| {
            if (x.heads == shape.n) break x;
        } else return false;
        if (shape.rows == 0 or (!self.paged and !self.l.wide)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.geo = geo;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = self.k;
        const d = geo.d;
        const keys = self.span();
        self.q = try a.alloc(f64, geo.heads * shape.rows * d);
        self.kc = try a.alloc(f64, geo.kv * keys * d);
        self.vc = try a.alloc(f64, self.kc.len);
        // q exact in the cache type, so the tile's rounding of it is exact
        for ([_][]f64{ self.q, self.kc, self.vc }) |x| for (x) |*v| {
            v.* = kit.round(k, rng.floatNorm(f64));
        };
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(self.rig.a, d, keys);
        switch (fill) {
            .random => {},
            .nonfinite => if (self.mode == .edge) self.edgePlant() else self.plant(),
            .stress => try self.stress(rng),
        }
        self.q32 = try self.rig.input(try kit.f32Bytes(a, self.q));
        // the pages under a permuted table, slots past the span NaN so a read of one shows
        self.pages = (keys + page - 1) / page;
        const n = self.pages;
        const table = try a.alloc(u32, n);
        for (table, 0..) |*p, i| p.* = @intCast((i * 5 + 3) % n);
        if (n % 5 == 0) for (table, 0..) |*p, i| {
            p.* = @intCast(n - 1 - i);
        };
        for ([_][]const f64{ self.kc, self.vc }, &self.pool) |src, *dev| {
            const bytes = try a.alloc(u8, geo.kv * n * page * d * 2);
            for (0..bytes.len / 2) |i| kit.put(k, bytes, i, std.math.nan(f64));
            for (0..geo.kv) |h| for (0..keys) |p| for (0..d) |e| {
                kit.put(k, bytes, ((h * n + table[p / page]) * page + p % page) * d + e, src[self.at(h, p, e)]);
            };
            dev.* = try self.rig.input(bytes);
        }
        self.table = try self.rig.input(std.mem.sliceAsBytes(table));
        if (!self.paged) for ([_][]const f64{ self.kc, self.vc }, &self.flat) |src, *dev| {
            const bytes = try a.alloc(u8, src.len * 2);
            for (src, 0..) |v, i| kit.put(k, bytes, i, v);
            dev.* = try self.rig.input(bytes);
        };
        const len = self.count() * 4;
        self.out = try self.rig.output(len);
        if (!self.paged) self.base = try self.rig.scratch(len);
        return true;
    }

    /// NaN query; +inf and -inf values, both in one column; NaN key; +inf at a row's first key and at tiny weights.
    fn plant(self: *Prefill) void {
        const n = self.shape.rows;
        const hk = self.geo.kv;
        const inf = std.math.inf(f64);
        self.q[3] = std.math.nan(f64); // row 0, head 0
        self.vc[self.at(0, self.pos(0), 7)] = inf;
        if (n > 1) self.vc[self.at(1 % hk, self.pos(1), 9)] = -inf;
        if (n > 2) {
            self.vc[self.at(2 % hk, self.pos(2), 11)] = inf;
            self.vc[self.at(2 % hk, self.pos(2) - 1, 11)] = -inf;
        }
        if (n > 3) self.kc[self.at(3 % hk, self.pos(3), 0)] = std.math.nan(f64);
        if (n > 4) self.vc[self.at(4 % hk, 0, 13)] = inf;
        // +inf at float64 weight 0 (row 5's own key: NaN), and under fp32's least (inf)
        if (n > 6) {
            if (hk > 2) self.reach(5, 1);
            self.under(n - 1, 2 % hk);
        }
    }

    /// Non-finite values only past rows' edges: +inf, -inf, NaN, one column seeing both infinities, one past every row.
    fn edgePlant(self: *Prefill) void {
        const n = self.shape.rows;
        if (n < 8) return;
        const hk = self.geo.kv;
        const inf = std.math.inf(f64);
        self.vc[self.at(0, self.pos(0) + 1, 7)] = inf;
        self.vc[self.at(1 % hk, self.pos(2) + 2, 9)] = -inf;
        self.vc[self.at(0, self.pos(5) + 1, 11)] = std.math.nan(f64);
        self.vc[self.at(1 % hk, self.pos(0) + 1, 13)] = inf;
        self.vc[self.at(1 % hk, self.pos(0) + 2, 13)] = -inf;
        self.vc[self.at(0, self.pos(n - 1) + 1, 15)] = inf;
    }

    /// Query row r of kv head kh's first head.
    fn query(self: *Prefill, r: usize, kh: usize) []f64 {
        const head = kh * (self.geo.heads / self.geo.kv);
        return self.q[(head * self.shape.rows + r) * self.geo.d ..][0..self.geo.d];
    }

    fn keyRow(self: *Prefill, kh: usize, key: usize) []f64 {
        return self.kc[self.at(kh, key, 0)..][0..self.geo.d];
    }

    /// Row r, key head kh's first head: query all 8, key p - 1 all 8, key p all -8 with +inf at element 17.
    fn reach(self: *Prefill, r: usize, kh: usize) void {
        const p = self.pos(r);
        @memset(self.query(r, kh), 8);
        @memset(self.keyRow(kh, p - 1), 8);
        @memset(self.keyRow(kh, p), -8);
        self.vc[self.at(kh, p, 17)] = std.math.inf(f64);
    }

    /// Row r, key head kh's first head: query all 2, key p - 1 all 8, its other keys all -8 with +inf at element 21.
    fn under(self: *Prefill, r: usize, kh: usize) void {
        const p = self.pos(r);
        @memset(self.query(r, kh), 2);
        for (0..p + 1) |key| {
            @memset(self.keyRow(kh, key), @as(f64, if (key == p - 1) 8 else -8));
            if (key != p - 1) self.vc[self.at(kh, key, 21)] = std.math.inf(f64);
        }
    }

    /// Gate 4: every key alike (each p is 1); values mirrored over the keys every row sees, 0 elsewhere.
    fn stress(self: *Prefill, rng: std.Random) !void {
        const a = self.arena.allocator();
        const d = self.geo.d;
        for (0..self.geo.kv) |kh| {
            const key = self.keyRow(kh, 0);
            for (1..self.span()) |p| @memcpy(self.keyRow(kh, p), key);
        }
        @memset(self.vc, 0);
        const col = try a.alloc(f64, self.pos(0) + 1);
        for (0..self.geo.kv) |kh| for (0..d) |e| {
            attn.mirrored(rng, col);
            for (col, 0..) |y, key| self.vc[self.at(kh, key, e)] = kit.round(self.k, y);
        };
    }

    /// pagedCausal's cache kinds are 0 fp16 and 1 bf16, not kit.Kind's.
    fn cacheKind(self: *const Prefill) c_int {
        return if (self.k == .bf16) 1 else 0;
    }

    fn scale(self: *const Prefill) f32 {
        return 1 / @sqrt(@as(f32, @floatFromInt(self.geo.d)));
    }

    fn launchPaged(self: *Prefill, out: u64) !void {
        const s = self.shape;
        const geo = self.geo;
        try self.l.pagedCausal(
            self.q32,
            self.pool[0],
            self.pool[1],
            self.table,
            out,
            s.rows,
            self.span(),
            geo.heads,
            geo.kv,
            geo.d,
            self.scale(),
            s.k,
            self.pages,
            self.cacheKind(),
            self.rig.s.handle,
        );
    }

    pub fn run(self: *Prefill) !void {
        if (self.planted) |f| return self.runPlanted(f);
        if (self.paged) {
            try self.launchPaged(self.out);
            return self.rig.s.synchronize();
        }
        const s = self.shape;
        const geo = self.geo;
        const ss: c_longlong = @intCast(geo.d);
        const sh: c_longlong = @intCast(self.span() * geo.d);
        // tf_causal takes the 64-row tile for a 16-bit cache of whole 64-wide heads
        try self.l.tf_causal(
            @ptrFromInt(self.q32),
            @ptrFromInt(self.flat[0]),
            @ptrFromInt(self.flat[1]),
            @ptrFromInt(self.out),
            1,
            @intCast(s.rows),
            @intCast(self.span()),
            @intCast(geo.heads),
            @intCast(geo.kv),
            @intCast(geo.d),
            self.scale(),
            @intCast(s.k),
            0,
            sh,
            ss,
            0,
            sh,
            ss,
            self.cacheKind(),
            null,
            null,
            null,
            self.rig.s.handle,
            null,
        );
        try self.rig.s.synchronize();
    }

    /// The planted control on the served symbol's arguments and grid, with no edge pass after it.
    fn runPlanted(self: *Prefill, f: hip.Function) !void {
        const s = self.shape;
        const geo = self.geo;
        var a: hip.launch.Args = .{};
        for ([_]u64{ self.q32, if (self.paged) self.pool[0] else self.flat[0], if (self.paged) self.pool[1] else self.flat[1], self.out }) |p| try a.add(p);
        for ([_]usize{ s.rows, self.span(), geo.heads, geo.kv, geo.d }) |x| try a.add(@as(c_int, @intCast(x)));
        try a.add(self.scale());
        try a.add(@as(c_int, @intCast(s.k)));
        if (self.paged) {
            try a.add(self.table);
            try a.add(@as(c_int, @intCast(self.pages)));
        } else {
            const ss: c_longlong = @intCast(geo.d);
            const sh: c_longlong = @intCast(self.span() * geo.d);
            for ([_]c_longlong{ 0, sh, ss, 0, sh, ss }) |x| try a.add(x);
        }
        const cfg: hip.launch.Config = .{ .grid = .{ .x = @intCast((s.rows + 63) / 64), .y = @intCast(geo.heads) }, .block = .{ .x = 256 } };
        try hip.launch.launch(f, cfg, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    /// The paged tile on the same keys, laid out in pages.
    pub fn replaced(self: *Prefill) ![]const u8 {
        try self.launchPaged(self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Prefill) usize {
        return self.geo.heads * self.shape.rows * self.geo.d;
    }

    /// Output head i (head i / rows, row i % rows): its query and keys 0 to its position, at their kv head.
    pub fn load(self: *Prefill, i: usize, hd: *attn.Head) !void {
        const d = self.geo.d;
        const r = i % self.shape.rows;
        const kh = (i / self.shape.rows) / (self.geo.heads / self.geo.kv);
        hd.n = self.pos(r) + 1;
        hd.q = self.q[i * d ..][0..d];
        hd.k = self.kc[self.at(kh, 0, 0)..][0 .. hd.n * d];
        hd.v = self.vc[self.at(kh, 0, 0)..][0 .. hd.n * d];
    }

    fn heads(self: *Prefill) usize {
        return self.shape.rows * self.geo.heads;
    }

    pub fn check(self: *Prefill, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), self.scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Prefill, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), self.scale(), want, mag);
    }

    pub fn simulate(self: *Prefill, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), self.scale(), acc, order, out);
    }
};
