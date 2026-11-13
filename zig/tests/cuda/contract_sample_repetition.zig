//! Request-owned repetition at BF16 loads, before temperature; tokens and full-row probability have separate gates.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const contract = @import("core").contract;
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const hf = @import("repetition_hf_fixture.zig");
const c = contract.spec;
const g = contract.gates;

const Mode = enum { greedy, pick, probability };
const Fault = enum { none, shared_history, identity, scribble, rounded, nan_wins };
const served = [_]c.Shape{
    .{ .rows = 8, .n = 1, .k = 154880 },
    .{ .rows = 32, .n = 1, .k = 154880 },
    .{ .rows = 2, .n = 1, .k = 31 },
    .{ .rows = 3, .n = 1, .k = 32 },
    .{ .rows = 4, .n = 1, .k = 33 },
};

fn depth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 37;
}

const rule = "NaNs never win; greedy chooses the lowest ID strictly above -inf and otherwise returns 0; " ++
    "full-row draws retain the keyed race, infinities and all-NaN rows have NaN probability";

pub const Case = struct {
    rig: rig_.Rig,
    mode: Mode,
    f: cuda.Function,
    plain: cuda.Function,
    fault: Fault = .none,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    sampling: lanes.Sampling = .{ .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0, .seed = 0x7265706574 },
    host: []u16 = &.{},
    bits: []u32 = &.{},
    factors: []f32 = &.{},
    corrected: []f32 = &.{},
    x: u64 = 0,
    mask: u64 = 0,
    fac: u64 = 0,
    rl: u64 = 0,
    meta: u64 = 0,
    token: u64 = 0,
    prob: u64 = 0,
    base_token: u64 = 0,
    base_prob: u64 = 0,
    unity: u64 = 0,
    rounded: u64 = 0,
    msg: [200]u8 = undefined,

    const vt = blk: {
        var v = g.hooks(Case);
        v.output = struct {
            fn f(ctx: *anyopaque) anyerror![]const u8 {
                const self: *Case = @ptrCast(@alignCast(ctx));
                return self.output();
            }
        }.f;
        break :blk v;
    };

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const name = switch (self.fault) {
            .shared_history => "control: repetition reads first row history",
            .identity => "control: repetition identity factor",
            .scribble => "control: repetition overwrites logits",
            .rounded => "control: repetition rounds corrected logits to BF16",
            .nan_wins => "control: repetition promotes NaN to maximum",
            .none => switch (self.mode) {
                .greedy => "cuda sample tf_greedy_repetition tokens",
                .pick => "cuda sample tf_draw_repetition tokens",
                .probability => "cuda sample tf_draw_repetition FP32 probability",
            },
        };
        return .{
            .spec = .{
                .name = name,
                .accum = if (self.mode == .probability) .f32 else .none,
                .depth = depth,
                .ops = 8,
                .out = if (self.mode == .probability) .f32 else .raw,
                .nan = .{ .stated = rule },
                .tiles = .{ .{ .size = 4 }, .{}, .{ .size = 1024, .min = 1 } },
                .served = &served,
            },
            .ctx = self,
            .vt = &vt,
            .control = switch (self.fault) {
                .none => null,
                .shared_history, .identity, .rounded => .shapes,
                .scribble => .repeat,
                .nan_wins => .nonfinite,
            },
        };
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        self.fill = fill;
        self.sampling.seed = seed;
        const w = (shape.k + 31) / 32;
        self.host = try a.alloc(u16, shape.rows * shape.k);
        self.corrected = try a.alloc(f32, self.host.len);
        self.bits = try a.alloc(u32, shape.rows * w);
        self.factors = try a.alloc(f32, shape.rows);
        @memset(self.bits, 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            self.factors[r] = ([_]f32{ 1, 2, 0.75, 1.1 })[r % 4];
            for (0..shape.k) |i| {
                const v: i32 = @as(i32, @intCast(rng.uintLessThan(u32, 128))) - 64;
                self.host[r * shape.k + i] = c.toBf16(@as(f32, @floatFromInt(v)) / 8);
                if ((i + r * 3) % 7 == 0) self.bits[r * w + i / 32] |= @as(u32, 1) << @intCast(i % 32);
            }
            if (fill == .stress) {
                for (0..shape.k) |i| self.host[r * shape.k + i] = c.toBf16(if (i % 8 == 0) -std.math.inf(f32) else 0);
            } else if (fill == .nonfinite) {
                const row = self.host[r * shape.k ..][0..shape.k];
                switch (r % 8) {
                    0 => row[shape.k / 2] = 0x7fc0,
                    1 => row[shape.k - 1] = 0x7f80,
                    2 => row[shape.k / 2] = 0xff80,
                    3 => {
                        row[0] = 0xff80;
                        row[shape.k - 1] = 0x7f80;
                    },
                    4 => @memset(row, 0x7fc0),
                    5 => {
                        @memset(row, 0xff80);
                        row[0] = 0x7fc0;
                    },
                    6 => @memset(row, 0xff80),
                    else => {},
                }
            } else if (self.mode == .greedy and shape.k > 1) {
                const row = self.host[r * shape.k ..][0..shape.k];
                @memset(row, c.toBf16(-60));
                row[0] = c.toBf16(1.5);
                row[shape.k - 1] = c.toBf16(2);
                @memset(self.bits[r * w ..][0..w], 0);
                if (r % 4 != 0) self.bits[r * w + (shape.k - 1) / 32] |= @as(u32, 1) << @intCast((shape.k - 1) % 32);
                if (r % 4 == 3) {
                    row[0] = c.toBf16(1);
                    row[shape.k - 1] = c.toBf16(1.1015625);
                } else if (r % 4 == 2) {
                    row[0] = c.toBf16(-1.25);
                    row[shape.k - 1] = c.toBf16(-1);
                }
            }
        }
        for (0..shape.rows) |r| for (0..shape.k) |i| {
            var x = c.fromBf16(self.host[r * shape.k + i]);
            if ((self.bits[r * w + i / 32] >> @intCast(i % 32)) & 1 != 0)
                x = if (x < 0) x * self.factors[r] else x / self.factors[r];
            self.corrected[r * shape.k + i] = x;
        };
        self.x = try self.rig.input(std.mem.sliceAsBytes(self.host));
        if (self.fault == .rounded or self.fault == .nan_wins) {
            const rounded = try a.alloc(u16, self.host.len);
            for (rounded, self.corrected, self.host) |*h, x, raw| h.* = if (self.fault == .rounded)
                c.toBf16(x)
            else if (std.math.isNan(c.fromBf16(raw)))
                0x7f80
            else
                raw;
            self.rounded = try self.rig.input(std.mem.sliceAsBytes(rounded));
        }
        self.mask = try self.rig.input(std.mem.sliceAsBytes(self.bits));
        self.fac = try self.rig.input(std.mem.sliceAsBytes(self.factors));
        const ones = try a.alloc(f32, shape.rows);
        @memset(ones, 1);
        self.unity = try self.rig.input(std.mem.sliceAsBytes(ones));
        const rl = cuda.sample.Rule.from(lanes.gpu_rule.rule(self.sampling));
        self.rl = try self.rig.input(std.mem.asBytes(&rl));
        self.meta = try self.rig.input(std.mem.asBytes(&[4]i32{ 17, 0, 0, 0 }));
        self.token = try self.rig.output(shape.rows * 4);
        if (self.mode != .greedy) self.prob = try self.rig.output(shape.rows * 4);
        self.base_token = try self.rig.scratch(shape.rows * 4);
        self.base_prob = try self.rig.scratch(shape.rows * 4);
        return true;
    }

    fn launch(self: *Case, baseline: bool, factors: u64, tok: u64, prob: u64) !void {
        var a: cuda.Args = .{};
        a.add(if (self.fault == .rounded or self.fault == .nan_wins) self.rounded else self.x);
        a.add(@as(u32, @intCast(self.shape.k)));
        if (self.mode != .greedy) {
            inline for (.{ self.rl, self.meta, @as(i32, 2), tok, prob }) |v| a.add(v);
        }
        if (!baseline) {
            a.add(self.mask);
            a.add(@as(u32, @intCast(if (self.fault == .shared_history) 0 else (self.shape.k + 31) / 32)));
            a.add(factors);
            if (self.mode == .greedy) a.add(tok);
        }
        try cuda.launch.launch(if (baseline) self.plain else self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 1024 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Case) !void {
        const factors = if (self.fault == .identity or self.fault == .rounded) self.unity else self.fac;
        try self.launch(false, factors, if (self.fault == .scribble) self.x else self.token, self.prob);
    }

    pub fn output(self: *Case) ![]const u8 {
        if (self.mode == .probability) return self.rig.bytes(self.prob, self.shape.rows * 4);
        return self.rig.outputBytes();
    }

    pub fn count(self: *Case) usize {
        return if (self.mode == .probability) self.shape.rows else self.shape.rows * 4;
    }

    fn greedy(row: []const f32) u32 {
        var m = -std.math.inf(f32);
        for (row) |x| if (x > m) {
            m = x;
        };
        if (m == -std.math.inf(f32)) return 0;
        for (row, 0..) |x, i| if (x == m) return @intCast(i);
        return 0;
    }

    fn tokenWant(self: *Case, r: usize) !u32 {
        const row = self.corrected[r * self.shape.k ..][0..self.shape.k];
        if (self.mode == .greedy) return greedy(row);
        if (self.fill == .nonfinite) {
            var m = -std.math.inf(f32);
            for (row) |x| if (x > m) {
                m = x;
            };
            if (std.math.isInf(m)) {
                for (row, 0..) |x, i| if (x == m) return @intCast(i);
                return @intCast(row.len - 1);
            }
            const clean = try self.arena.allocator().dupe(f32, row);
            for (clean) |*x| if (std.math.isNan(x.*)) {
                x.* = -std.math.inf(f32);
            };
            return lanes.gpu_full.draw(self.arena.allocator(), clean, self.sampling, @intCast(20 + r), null);
        }
        return lanes.gpu_full.draw(self.arena.allocator(), row, self.sampling, @intCast(20 + r), null);
    }

    fn probability(self: *Case, r: usize, pick: usize) !f64 {
        const row = self.corrected[r * self.shape.k ..][0..self.shape.k];
        const inv_t = lanes.gpu_rule.rule(self.sampling).inv_t;
        var m = -std.math.inf(f32);
        for (row) |x| if (x * inv_t > m) {
            m = x * inv_t;
        };
        var z: f64 = 0;
        if (self.sampling.top_k != 0 and self.sampling.top_k < row.len) {
            const sorted = try self.arena.allocator().alloc(lanes.gpu_rule.Cand, row.len);
            for (sorted, row, 0..) |*item, x, i| item.* = .{ .v = x * inv_t, .i = @intCast(i) };
            std.mem.sort(lanes.gpu_rule.Cand, sorted, {}, lanes.gpu_rule.better);
            for (sorted[0..self.sampling.top_k]) |item| z += @exp(@as(f64, item.v) - m);
        } else for (row) |x| if (!std.math.isNan(x)) {
            z += @exp(@as(f64, x * inv_t) - m);
        };
        return @exp(@as(f64, row[pick] * inv_t) - m) / z;
    }

    pub fn check(self: *Case, gate: g.Gate) !?[]const u8 {
        const a = self.arena.allocator();
        const got = try a.dupe(u8, try self.rig.bytes(self.token, self.shape.rows * 4));
        const probs = if (self.mode != .greedy) try a.dupe(u8, try self.rig.bytes(self.prob, self.shape.rows * 4)) else &.{};
        for (0..self.shape.rows) |r| {
            const id = std.mem.readInt(u32, got[r * 4 ..][0..4], .little);
            const want = try self.tokenWant(r);
            if (id != want) return try std.fmt.bufPrint(&self.msg, "row {d}: token {d}, independent corrected-F32 rule gives {d}", .{ r, id, want });
            if (self.mode != .greedy) {
                const p = c.read(.f32, probs, r);
                const ref = try self.probability(r, id);
                if (c.classOf(p) != c.classOf(ref)) return "probability class differs from independent FP64 formula";
                const k = self.sampling.top_k;
                const chain = if (k > 0 and k <= 1024) k else depth(self.shape);
                const allowance = (c.gamma(@floatFromInt(chain), c.unit(.f32)) + 24 * 0x1p-23) * @abs(ref);
                if (std.math.isFinite(ref) and @abs(p - ref) > allowance)
                    return try std.fmt.bufPrint(&self.msg, "row {d}: probability {e}, FP64 {e}, derived allowance {e}", .{ r, p, ref, allowance });
            }
        }
        if (self.mode != .greedy and gate == .shapes) {
            try self.launch(true, self.unity, self.base_token, self.base_prob);
            const base_tok = try a.dupe(u8, try self.rig.bytes(self.base_token, self.shape.rows * 4));
            const base_prob = try a.dupe(u8, try self.rig.bytes(self.base_prob, self.shape.rows * 4));
            try self.launch(false, self.unity, self.base_token, self.base_prob);
            if (!std.mem.eql(u8, base_tok, try self.rig.bytes(self.base_token, self.shape.rows * 4))) return "identity-factor tokens differ from original tf_draw ABI";
            if (!std.mem.eql(u8, base_prob, try self.rig.bytes(self.base_prob, self.shape.rows * 4))) return "identity-factor probability bytes differ from original tf_draw ABI";
        }
        return null;
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        if (self.mode != .probability) {
            @memset(want, 0);
            @memset(mag, 0);
            return;
        }
        for (want, mag, 0..) |*w, *m, r| {
            w.* = try self.probability(r, if (self.fill == .stress) 1 else try self.tokenWant(r));
            m.* = @abs(w.*);
        }
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.arena.allocator().alloc(f64, self.shape.k);
        for (terms, 0..) |*x, i| x.* = if (i % 8 == 0) 0 else 1;
        const z = contract.rows.sumIn(acc, order, terms);
        @memset(out, 1 / z);
    }
};

pub const Group = struct {
    cases: [8]Case,
    reference: cuda.Function,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, module: cuda.Module, controls: cuda.Module) !Group {
        const draw = try module.function("tf_draw_repetition");
        const greedy_fn = try module.function("tf_greedy_repetition");
        const plain = try module.function("tf_draw");
        var self: Group = undefined;
        for (&self.cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .mode = if (i == 0 or i >= 3) .greedy else if (i == 1) .pick else .probability,
            .f = if (i == 0 or i >= 3) greedy_fn else draw,
            .plain = plain,
            .fault = if (i < 3) .none else @fromBackingInt(@intCast(i - 2)),
            .arena = .init(gpu.gpa),
        };
        self.reference = try controls.function("tf_contract_repetition_values");
        return self;
    }

    pub fn append(self: *Group, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        for (&self.cases) |*item| try kernels.append(a, item.kernel());
    }

    pub fn deinit(self: *Group) void {
        for (&self.cases) |*item| item.deinit();
    }

    pub fn pinnedCases(self: *Group) !void {
        const gpu = self.cases[0].rig.gpu;
        var rig: rig_.Rig = .{ .gpu = gpu, .s = self.cases[0].rig.s };
        defer rig.deinit();
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        for (hf.cases) |fixture| {
            rig.reset();
            const logits = try a.alloc(u16, fixture.logits_bits.len);
            for (logits, fixture.logits_bits) |*value, bits| value.* = c.toBf16(@as(f32, @bitCast(bits)));
            const words = (fixture.logits_bits.len + 31) / 32;
            const membership = try a.alloc(u32, words);
            @memset(membership, 0);
            for (fixture.history) |token| {
                if (token >= fixture.logits_bits.len) return error.InvalidRepetitionFixtureToken;
                membership[token / 32] |= @as(u32, 1) << @intCast(token % 32);
            }
            const x = try rig.input(std.mem.sliceAsBytes(logits));
            const mask = try rig.input(std.mem.sliceAsBytes(membership));
            const out = try rig.output(fixture.expected_bits.len * 4);
            const factor: f32 = @bitCast(fixture.penalty_bits);
            var args: cuda.Args = .{};
            inline for (.{ x, @as(u32, @intCast(logits.len)), mask, factor, out }) |value| args.add(value);
            try cuda.launch.launch(self.reference, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, rig.s, &args);
            const got = try rig.bytes(out, fixture.expected_bits.len * 4);
            var best = -std.math.inf(f32);
            var greedy: u32 = 0;
            for (fixture.expected_bits, 0..) |expected_bits, token| {
                const actual_bits = std.mem.readInt(u32, got[token * 4 ..][0..4], .little);
                const actual: f32 = @bitCast(actual_bits);
                const expected: f32 = @bitCast(expected_bits);
                if (std.math.isNan(expected)) {
                    if (!std.math.isNan(actual)) {
                        std.debug.print("FAIL repetition HF fixture {s}, token {d}: expected NaN, got 0x{x}\n", .{ fixture.name, token, actual_bits });
                        return error.TestFailed;
                    }
                } else if (actual_bits != expected_bits) {
                    std.debug.print("FAIL repetition HF fixture {s}, token {d}: expected 0x{x}, got 0x{x}\n", .{ fixture.name, token, expected_bits, actual_bits });
                    return error.TestFailed;
                }
                if (actual > best) {
                    best = actual;
                    greedy = @intCast(token);
                }
            }
            if (best == -std.math.inf(f32)) greedy = 0;
            if (fixture.expected_greedy) |expected| if (greedy != expected) {
                std.debug.print("FAIL repetition HF fixture {s}: expected greedy {d}, got {d}\n", .{ fixture.name, expected, greedy });
                return error.TestFailed;
            };
            if (!try rig.intact()) return error.RepetitionFixtureGuardChanged;
        }
        std.debug.print("PASS repetition Hugging Face {s}: {d} source-pinned FP32 cases through production arithmetic\n", .{ hf.provenance.transformers, hf.cases.len });
    }

    pub fn finiteRules(self: *Group) !void {
        const Rules = struct { k: u32, p: f64, min: f64, t: f64 = 0.7 };
        const rules = [_]Rules{
            .{ .k = 1, .p = 1, .min = 0 },
            .{ .k = 40, .p = 0.9, .min = 0.05 },
            .{ .k = 50, .p = 0.95, .min = 0, .t = 1 },
            .{ .k = 1024, .p = 1, .min = 0 },
            .{ .k = 1500, .p = 0.95, .min = 0.05 },
            .{ .k = 0, .p = 0.9, .min = 0 },
        };
        const item = &self.cases[1];
        defer item.sampling = .{ .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0, .seed = 0x7265706574 };
        var total: usize = 0;
        for ([_]usize{ 31, 1025, 154880 }) |vocab| for (rules) |rl| {
            item.sampling.top_k = rl.k;
            item.sampling.temperature = rl.t;
            item.sampling.top_p = rl.p;
            item.sampling.min_p = rl.min;
            _ = try item.prepare(.{ .rows = 4, .n = 1, .k = vocab }, .random, 0x726570 + total);
            try item.run();
            if (try item.check(.shapes)) |why| {
                std.debug.print("FAIL repetition finite rules V {d}, top_k {d}, top_p {d}: {s}\n", .{ vocab, rl.k, rl.p, why });
                return error.TestFailed;
            }
            total += 4;
        };
        std.debug.print("PASS repetition finite filters: {d} row/rule cases, host token and FP64 probability, identity bytes equal original ABI\n", .{total});
    }

    pub fn verifyExtraCases(self: *Group) !void {
        try self.pinnedCases();
        try self.finiteRules();
    }
};
