//! The eight served paged symbols' FP32 sums, exact integer stress and accumulator replays.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const sums = core.contract.rows;

const hd = 256;
const qh = 24;
const kh = 2;
const ih = 4;
const di = 128;
const pw = 13952;
const ptok = 48;
const idw = 2052;
const keys = 2048;
const blocks = 600;
const eps: f32 = 1e-6;

pub const Mode = enum { prep, prep_media, pool, pool_media, scores, ahead, one, fused };
const modes = std.enums.values(Mode);
const vt = g.hooks(Runner);
const rms_why = "BF16 RMS output rounding can hide a BF16 sum error; FP16 overflow is proved; BF16 margins are printed";

fn depth(s: c.Shape) usize {
    return s.k + 32;
}

pub fn spec(mode: Mode) c.Spec {
    const norm = mode == .prep or mode == .prep_media or mode == .pool or mode == .pool_media;
    return .{
        .name = switch (mode) {
            .prep => "cuda paged fn_attn_prep FP32 norms",
            .prep_media => "cuda paged fn_attn_prep_media FP32 norms",
            .pool => "cuda paged fn_qsa_pool FP32 pool and norms",
            .pool_media => "cuda paged fn_qsa_pool_media FP32 pool and norms",
            .scores => "cuda paged fn_qsa_scores FP32 dots",
            .ahead => "cuda paged fn_attn_chunks_ahead FP32 attention",
            .one => "cuda paged fn_attn_chunks_one FP32 attention",
            .fused => "cuda paged fn_attn_fused FP32 attention",
        },
        .accum = .f32,
        .depth = depth,
        .ops = if (norm) 8 else if (mode == .scores) 6 else 1,
        .out = if (mode == .scores) .f32 else .bf16,
        .served = if (norm) &.{.{ .rows = 4, .n = qh, .k = hd }} else if (mode == .scores)
            &.{.{ .rows = 4, .n = blocks, .k = di }}
        else
            &.{.{ .rows = 17, .n = qh, .k = keys }},
        .sums = if (norm) .{ .bf16 = 0, .why = rms_why } else .{},
    };
}

fn attentionValues(out: []f32) void {
    const half = out.len / 2;
    for (0..half) |i| {
        const v: f32 = if (i % 2 == 0) 127 else 16320;
        out[i] = v;
        out[half + i] = (if (i < 8 and i % 2 == 0) @as(f32, 1) else 0) - v;
    }
}

fn dotTerm(i: usize) f64 {
    return if (i < di / 2) 127 * 127 else -128 * 126;
}

fn normValue(i: usize) f64 {
    return @floatFromInt(64 + i % 32);
}

fn normalized(a: std.mem.Allocator, width: usize, acc: c.Accum, order: g.Order) ![]f64 {
    const terms = try a.alloc(f64, width);
    defer a.free(terms);
    for (terms, 0..) |*v, i| v.* = normValue(i) * normValue(i);
    const ss = sums.sumIn(acc, order, terms);
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
    const out = try a.alloc(f64, width);
    for (out, 0..) |*v, i| v.* = normValue(i) * inv;
    return out;
}

/// This is independent of accumulator rounding and of the device's page lookup.
fn normReference(width: usize, i: usize) f64 {
    var ss: f64 = 0;
    for (0..width) |j| ss += normValue(j) * normValue(j);
    return normValue(i) / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
}

const Runner = struct {
    rig: rig_.Rig,
    glue: *const flash.Glue,
    mode: Mode,
    arena: ?std.heap.ArenaAllocator = null,
    r: ?attn.Ref = null,
    rows: usize = 0,
    q: []u16 = &.{},
    kv: []u16 = &.{},
    terms: []f64 = &.{},
    dev: [16]u64 = @splat(0),
    pg: flash.Pg = .{},

    fn deinit(self: *Runner) void {
        cuda.completion.drain(self.rig.gpu.d, self.rig.s, "paged sum checks did not drain before buffer release");
        self.rig.deinit();
        if (self.r) |*r| r.deinit();
        if (self.arena) |*a| a.deinit();
    }

    pub fn prepare(self: *Runner, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (fill != .stress) return error.StressOnlyAdapter;
        self.rig.reset();
        if (self.r) |*r| r.deinit();
        self.r = null;
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.rows = shape.rows;
        self.dev = @splat(0);
        const pos: i32 = if (self.mode == .scores) 4 * blocks - 1 else if (self.isAttention()) 4095 else 0;
        self.dev[0] = try self.rig.input(std.mem.asBytes(&pos));
        const cap = if (self.mode == .scores) 4 * blocks else if (self.isAttention()) 4096 + self.rows else ptok;
        const np = (cap + ptok - 1) / ptok;
        const table = try a.alloc(u32, np);
        for (table, 0..) |*id, i| id.* = @intCast(np - 1 - i);
        const page_elems = ptok * (2 * kh * hd + di) + ptok / 4 * di;
        self.kv = try a.alloc(u16, np * page_elems);
        @memset(self.kv, 0);
        self.pg = .{ .pt = try self.rig.input(std.mem.sliceAsBytes(table)), .ps = @intCast(page_elems), .ptok = ptok };
        if (self.isAttention()) {
            self.q = try a.alloc(u16, self.rows * qh * hd);
            @memset(self.q, 0);
            _ = seed;
            const col = try a.alloc(f32, keys);
            for (0..kh) |h| for (0..hd) |d| {
                attentionValues(col);
                for (col, 0..) |v, j| self.kv[self.pageAt(table, j, .value) + h * hd + d] = c.toBf16(v);
            };
            self.r = try attn.Ref.init(self.rig.gpu.gpa, hd, keys);
            self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.q));
            const ids = try a.alloc(i32, self.rows * idw);
            for (ids, 0..) |*v, i| v.* = @intCast(i % idw % keys);
            self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(ids));
            const nk = try a.alloc(i32, self.rows);
            @memset(nk, keys);
            self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(nk));
            @memset(nk, 1);
            self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(nk));
            self.dev[5] = try self.rig.scratch(self.rows * 5 * qh * hd * 4);
            self.dev[6] = try self.rig.scratch(self.rows * 5 * qh * 4);
            self.dev[7] = try self.rig.scratch(self.rows * 5 * qh * 4);
            self.dev[8] = try self.rig.output(self.count() * 2);
        } else if (self.mode == .scores) {
            self.q = try a.alloc(u16, self.rows * ih * di);
            for (self.q, 0..) |*v, i| v.* = c.toBf16(if (i % di < di / 2) 127 else 126);
            self.terms = try a.alloc(f64, di);
            for (self.terms, 0..) |*t, i| t.* = dotTerm(i);
            for (0..blocks) |j| {
                for (0..di) |d| self.kv[self.pageAt(table, j, .pooled) + d] = c.toBf16(if (d < di / 2) 127 else -128);
            }
            self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.q));
            self.dev[8] = try self.rig.output(self.count() * 4);
        } else try self.prepareNorm(a, table);
        self.dev[9] = try self.rig.scratch(self.kv.len * 2);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.dev[9], .len = self.kv.len * 2 }).uploadAsync(0, std.mem.sliceAsBytes(self.kv), self.rig.s.handle);
        return true;
    }

    const Part = enum { key, value, index, pooled };
    fn pageAt(_: *const Runner, table: []const u32, row: usize, part: Part) usize {
        const stride = ptok * (2 * kh * hd + di) + ptok / 4 * di;
        const per: usize = if (part == .pooled) ptok / 4 else ptok;
        const row_len: usize = if (part == .key or part == .value) kh * hd else di;
        const off: usize = switch (part) {
            .key => 0,
            .value => ptok * kh * hd,
            .index => 2 * ptok * kh * hd,
            .pooled => ptok * (2 * kh * hd + di),
        };
        return table[row / per] * stride + off + row % per * row_len;
    }

    fn isAttention(self: *const Runner) bool {
        return self.mode == .ahead or self.mode == .one or self.mode == .fused;
    }

    fn prepareNorm(self: *Runner, a: std.mem.Allocator, table: []const u32) !void {
        const proj = try a.alloc(u16, self.rows * pw);
        @memset(proj, 0);
        for (0..self.rows) |row| {
            for (0..qh) |h| {
                for (0..hd) |d| proj[row * pw + h * 2 * hd + d] = c.toBf16(@floatCast(normValue(d)));
            }
            for (0..kh) |h| {
                for (0..hd) |d| proj[row * pw + qh * 2 * hd + h * hd + d] = c.toBf16(@floatCast(normValue(d)));
            }
            for (0..ih) |h| {
                for (0..di) |d| proj[row * pw + qh * 2 * hd + 2 * kh * hd + h * di + d] = c.toBf16(@floatCast(normValue(d)));
            }
        }
        if (self.mode == .pool or self.mode == .pool_media) {
            for (0..4) |j| {
                for (0..di) |d| self.kv[self.pageAt(table, j, .index) + d] = c.toBf16(@floatCast(normValue(d)));
            }
        }
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(proj));
        const w = try a.alloc(f32, hd);
        @memset(w, 1);
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(w));
        const inv: [32]f32 = @splat(0);
        self.dev[3] = try self.rig.input(std.mem.asBytes(&inv));
        const rope: [12]i32 = @splat(0);
        self.dev[4] = try self.rig.input(std.mem.asBytes(&rope));
        if (self.mode == .prep or self.mode == .prep_media) {
            self.dev[8] = try self.rig.output(self.rows * qh * hd * 2);
            self.dev[10] = try self.rig.output(self.rows * ih * di * 2);
        } else {
            self.dev[8] = try self.rig.output(di * 2);
        }
    }

    fn launch(self: *Runner, f: cuda.Function, grid: [3]usize, threads: usize, shared: usize, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(threads) }, .shared = @intCast(shared) }, self.rig.s, args);
    }

    pub fn run(self: *Runner) !void {
        const p = self.dev;
        const kc = p[9];
        const vc = kc + ptok * kh * hd * 2;
        const ikc = kc + 2 * ptok * kh * hd * 2;
        const pooled = ikc + ptok * di * 2;
        if (self.mode == .scores) {
            try self.glue.qsaScores(self.rig.s, p[1], pooled, p[0], p[8], blocks, self.rows, blocks, self.pg);
        } else if (self.isAttention()) {
            if (self.mode == .fused) {
                try self.glue.attnFused(self.rig.s, p[1], kc, vc, p[0], p[8], p[2], p[3], p[4], self.rows, kh, self.pg);
            } else {
                var args: cuda.Args = .{};
                for ([_]u64{ p[1], kc, vc, p[0], p[5], p[6], p[7], p[2], p[3], p[4] }) |v| args.add(v);
                for ([_]c_int{ qh, kh, qh / kh, 512, 5 }) |v| args.add(v);
                args.add(@as(f32, 0.0625));
                args.add(@as(c_int, idw));
                args.add(@as(c_int, 1));
                args.add(self.pg);
                const ahead = self.mode == .ahead;
                const shared: usize = (16 + (if (ahead) @as(usize, 128) else 32)) * (hd + 8) * 2 + 16 * 40 * 2;
                try self.launch(if (ahead) self.glue.f.chunks_ahead_paged else self.glue.f.chunks_one_paged, .{ self.rows, kh, 4 }, 128, shared, &args);
                try self.glue.attnMerge(self.rig.s, p[5], p[6], p[7], p[0], p[8], p[3], p[4], self.rows, kh, true);
            }
        } else {
            var args: cuda.Args = .{};
            const prep = self.mode == .prep or self.mode == .prep_media;
            const media = self.mode == .prep_media or self.mode == .pool_media;
            if (prep) {
                for ([_]u64{ p[1], p[0], p[2], p[2], p[2], p[3], p[8], kc, vc, p[10], ikc }) |v| args.add(v);
                args.add(eps);
                for ([_]c_int{ pw, qh, kh, ih, di, 32 }) |v| args.add(v);
            } else {
                for ([_]u64{ ikc, pooled, p[0], p[2], p[3] }) |v| args.add(v);
                args.add(eps);
                for ([_]c_int{ 4, di, 32, 4 }) |v| args.add(v);
            }
            if (media) {
                args.add(p[4]);
                args.add(@as(c_int, 4));
                args.add(@as(c_int, -1));
            }
            args.add(self.pg);
            const f = if (prep) (if (media) self.glue.f.prep_media_paged else self.glue.f.prep_paged) else (if (media) self.glue.f.pool_media_paged else self.glue.f.pool_paged);
            try self.launch(f, if (prep) .{ self.rows, qh + kh + ih + 1, 1 } else .{ 1, 1, 1 }, if (prep) hd else di, 0, &args);
            if (!prep) {
                const np = self.kv.len / @as(usize, @intCast(self.pg.ps));
                const at = pooled + (np - 1) * @as(usize, @intCast(self.pg.ps)) * 2;
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = p[8], .len = di * 2 }).copyFrom(0, at, di * 2, self.rig.s.handle);
            }
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Runner) usize {
        return if (self.isAttention()) self.rows * qh * hd else if (self.mode == .scores) self.rows * blocks else if (self.mode == .pool or self.mode == .pool_media) di else self.rows * (qh * hd + ih * di);
    }

    pub fn load(self: *Runner, i: usize, head: *attn.Head) !void {
        const h = i % qh / (qh / kh);
        const a = self.arena.?.allocator();
        const np = self.kv.len / @as(usize, @intCast(self.pg.ps));
        const table = try a.alloc(u32, np);
        for (table, 0..) |*id, j| id.* = @intCast(np - 1 - j);
        @memset(head.q, 0);
        @memset(head.k[0 .. keys * hd], 0);
        for (0..keys) |j| {
            for (0..hd) |d| head.v[j * hd + d] = c.fromBf16(self.kv[self.pageAt(table, j, .value) + h * hd + d]);
        }
        head.n = keys;
    }

    pub fn reference(self: *Runner, want: []f64, mag: []f64) !void {
        if (self.isAttention()) return self.r.?.reference(self, self.rows * qh, 0.0625, want, mag);
        if (self.mode == .scores) {
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (self.terms) |v| {
                dot += v;
                abs += @abs(v);
            }
            @memset(want, @max(dot, 0) * ih / @sqrt(@as(f64, di)));
            @memset(mag, abs * ih / @sqrt(@as(f64, di)));
            return;
        }
        const prep = self.mode == .prep or self.mode == .prep_media;
        for (want, mag, 0..) |*w, *m, i| {
            const width: usize = if (!prep or i >= self.rows * qh * hd) di else hd;
            w.* = normReference(width, i % width);
            m.* = @abs(w.*);
        }
    }

    pub fn simulate(self: *Runner, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.isAttention()) return self.r.?.simulate(self, self.rows * qh, 0.0625, acc, order, out);
        if (self.mode == .scores) {
            const dot = sums.sumIn(acc, order, self.terms);
            const terms: [ih]f64 = @splat(@max(dot, 0));
            @memset(out, sums.sumIn(acc, order, &terms) / @sqrt(@as(f64, di)));
            return;
        }
        const a = self.arena.?.allocator();
        const wide = try normalized(a, hd, acc, order);
        const narrow = try normalized(a, di, acc, order);
        const prep = self.mode == .prep or self.mode == .prep_media;
        for (out, 0..) |*o, i| o.* = if (!prep or i >= self.rows * qh * hd) narrow[i % di] else wide[i % hd];
    }
};

pub fn run(gpu: check.Gpu, glue: *const flash.Glue) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    for (modes) |mode| {
        var r: Runner = .{ .rig = .{ .gpu = gpu, .s = stream }, .glue = glue, .mode = mode };
        defer r.deinit();
        const kernel: g.Kernel = .{ .spec = spec(mode), .ctx = &r, .vt = &vt };
        try check.expect(try g.sums(gpu.gpa, kernel, 0x517a) == .pass, "paged {t}: FP32 sum gate failed", .{mode});
    }
}

test "paged norm stress has exact FP32 square sums and refuses FP16 overflow in every order" {
    const a = std.testing.allocator;
    for ([_]usize{ di, hd }) |width| for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        const fp32 = try normalized(a, width, .f32, order);
        defer a.free(fp32);
        const half = try normalized(a, width, .f16, order);
        defer a.free(half);
        for (fp32, half, 0..) |got, bad, i| {
            const want = normReference(width, i);
            const bound = c.boundExact(spec(.prep), want);
            try std.testing.expect(@abs(got - want) <= bound);
            try std.testing.expect(@abs(bad - want) > 100 * bound);
        }
    };
}

test "paged dot stress detects FP16 and BF16 accumulators under every order" {
    var terms: [di]f64 = undefined;
    var want: f64 = 0;
    for (&terms, 0..) |*t, i| {
        t.* = dotTerm(i);
        want += t.*;
    }
    want = @max(want, 0) * ih / @sqrt(@as(f64, di));
    const bound = c.boundExact(spec(.scores), want);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| for ([_]c.Accum{ .f16, .bf16 }) |acc| {
        const dot = sums.sumIn(acc, order, &terms);
        const heads: [ih]f64 = @splat(@max(dot, 0));
        const bad = sums.sumIn(acc, order, &heads) / @sqrt(@as(f64, di));
        try std.testing.expect(!std.math.isFinite(bad) or @abs(bad - want) > 100 * bound);
    };
}

test "paged attention mirrored values detect FP16 and BF16 sum controls in every order" {
    var col: [keys]f32 = undefined;
    attentionValues(&col);
    var terms: [keys]f64 = undefined;
    const ones: [keys]f64 = @splat(1);
    var total: f64 = 0;
    for (col, &terms) |v, *t| {
        t.* = c.fromBf16(c.toBf16(v));
        total += t.*;
    }
    const want = total / keys;
    const bound = c.boundExact(spec(.ahead), want);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        const good = sums.sumIn(.f32, order, &terms) / sums.sumIn(.f32, order, &ones);
        try std.testing.expect(@abs(good - want) <= bound);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const bad = sums.sumIn(acc, order, &terms) / sums.sumIn(acc, order, &ones);
            try std.testing.expect(!std.math.isFinite(bad) or @abs(bad - want) > 100 * bound);
        }
    }
}
