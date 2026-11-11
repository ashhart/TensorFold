//! The Qwen3.8-27B glue's top-k under the contract on XPU: the candidates a sampled row draws from, by its stated rule.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref;

const PARTS = 256; // topk_part's most groups a row, one per 1024 ids
const SEG = 1024; // ids a group takes
const KMAX = 64; // candidates a row keeps at most
const WORDS = 132; // a row's output: KMAX values (fp32 bits), KMAX ids, the flag, padding
const VOCAB = 248320; // Qwen3.8-27B's logits a row
const MAX_ROWS = 16; // decode and verification rows

/// The real merge, and the two planted ones of `contract_gate_controls.cl`.
pub const Which = enum { real, ties, cand_flag };

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "a row holding a NaN or an infinity anywhere raises the flag, whatever the candidates hold; else the flag " ++
    "is clear and the K candidates are the K largest values, the lower id first at a tie, signed zeros equal";

const served = [_]c.Shape{
    .{ .rows = 1, .n = VOCAB, .k = 20 },
    .{ .rows = 16, .n = VOCAB, .k = 64 },
    .{ .rows = 4, .n = 100003, .k = 7 },
    .{ .rows = 2, .n = PARTS * SEG, .k = 33 },
};

pub fn spec(w: Which) c.Spec {
    return .{
        .name = switch (w) {
            .real => "xpu top-k (the candidates)",
            .ties => "control: top-k merge breaking ties the wrong way",
            .cand_flag => "control: top-k flag read off the candidates only",
        },
        .accum = .none,
        .depth = none,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 16, .min = 1 }, .{ .size = SEG, .min = KMAX }, .{ .size = KMAX, .min = 1 } },
        .served = &served,
    };
}

/// shape.rows rows of shape.n logits of fp16, shape.k candidates each: the merged words of every row.
pub const Topk = struct {
    rig: rig_.Rig,
    part: xpu.Kernel,
    merge: xpu.Kernel,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    inb: []u8 = &.{},
    xin: ?*anyopaque = null,
    pv: ?*anyopaque = null,
    pi: ?*anyopaque = null,
    pf: ?*anyopaque = null,
    out: ?*anyopaque = null,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, which: Which) !Topk {
        return .{
            .rig = .{ .a = a, .dev = dev },
            .part = try dev.kernel(.glue, "topk_part", .{ 256, 1, 1 }),
            .merge = switch (which) {
                .real => try dev.kernel(.glue, "topk_merge", .{ 256, 1, 1 }),
                .ties => try dev.kernel(.gate, "tf_ctl_topk_merge_ties", .{ 256, 1, 1 }),
                .cand_flag => try dev.kernel(.gate, "tf_ctl_topk_merge_cand_flag", .{ 256, 1, 1 }),
            },
            .which = which,
            .arena = .init(a),
        };
    }

    pub fn deinit(self: *Topk) void {
        self.merge.deinit();
        self.part.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Topk) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = switch (self.which) {
            .real => null,
            .ties => .shapes,
            .cand_flag => .nonfinite,
        } };
    }

    const vt = g.hooks(Topk);

    pub fn prepare(self: *Topk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > MAX_ROWS or shape.k == 0 or shape.k > KMAX) return false;
        if (shape.n < shape.k or shape.n > PARTS * SEG) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(f64, shape.rows * shape.n);
        for (0..shape.rows) |r| {
            const row = self.x[r * shape.n ..][0..shape.n];
            for (row, 0..) |*v, i| {
                const level = @as(f64, @floatFromInt(rng.uintLessThan(u32, 256))) / 8 - 16;
                const tiny = @as(f64, @floatFromInt(1 + rng.uintLessThan(u32, 1023))) * 0x1p-24; // an fp16 subnormal
                if (r % 8 == 5) { // fp16 subnormals of both signs on top of a -1 floor
                    v.* = if (rng.uintLessThan(u32, 4) == 0) (if (i % 2 == 1) -tiny else tiny) else -1.0;
                    continue;
                }
                v.* = switch (r % 4) {
                    2 => 1.5, // one value everywhere: only the tie order picks the candidates
                    3 => if (level == -16) (if (i % 2 == 1) -0.0 else 0.0) else -@abs(level), // zeros of both signs on top
                    else => kit.round(.f16, level),
                };
            }
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, shape.n / 3 }, .{ 0, shape.n - 1 }, .{ 1, shape.n / 2 }, .{ 2, 0 }, .{ 3, 7 } };
            const put = [_]f64{ std.math.nan(f64), std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64) };
            for (at, put) |p, v| if (p[0] < shape.rows and p[1] < shape.n) {
                self.x[p[0] * shape.n + p[1]] = v;
            };
        }
        self.inb = try kit.bytes(a, self.x, .f16);
        self.xin = try self.rig.input(self.inb);
        self.pv = try self.rig.scratch(shape.rows * PARTS * KMAX * 4);
        self.pi = try self.rig.scratch(shape.rows * PARTS * KMAX * 4);
        self.pf = try self.rig.scratch(shape.rows * PARTS * 4);
        self.out = try self.rig.output(shape.rows * WORDS * 4);
        return true;
    }

    pub fn run(self: *Topk) !void {
        const s = self.shape;
        const parts: u32 = @intCast((s.n + SEG - 1) / SEG);
        const v: u32 = @intCast(s.n);
        const k: u32 = @intCast(s.k);
        try self.rig.launch(self.part, .{ parts, @intCast(s.rows), 1 }, .{ self.xin, v, k, self.pv, self.pi, self.pf });
        try self.rig.launch(self.merge, .{ @intCast(s.rows), 1, 1 }, .{ self.pv, self.pi, self.pf, k, parts, self.out });
    }

    pub fn count(self: *Topk) usize {
        return self.shape.rows * WORDS * 4;
    }

    pub fn reference(_: *Topk, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each row's flag and candidates against the stated rule over the row's values as fp16 holds them.
    pub fn check(self: *Topk, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const got = try self.rig.bytes(self.out, s.rows * WORDS * 4);
        var vals: [KMAX]f64 = undefined;
        var ids: [KMAX]u32 = undefined;
        for (0..s.rows) |r| {
            const w = got[r * WORDS * 4 ..][0 .. WORDS * 4];
            const flag = std.mem.readInt(u32, w[4 * 2 * KMAX ..][0..4], .little);
            const row = self.x[r * s.n ..][0..s.n];
            const bad = ref.hasNonfinite(row);
            if ((flag != 0) != bad) return try std.fmt.bufPrint(&self.msg, "row {d} flag {d} where the row {s} a value that is not finite", .{ r, flag, if (bad) "holds" else "holds no" });
            if (bad) continue;
            ref.topkPick(row, s.k, &vals, &ids);
            for (0..s.k) |i| {
                const v: f32 = @bitCast(std.mem.readInt(u32, w[4 * i ..][0..4], .little));
                const id = std.mem.readInt(u32, w[4 * (KMAX + i) ..][0..4], .little);
                const want: f32 = @floatCast(vals[i]);
                if (id != ids[i] or @as(u32, @bitCast(v)) != @as(u32, @bitCast(want))) {
                    return try std.fmt.bufPrint(&self.msg, "row {d} candidate {d} is {d} at id {d}, the rule gives {d} at id {d}", .{ r, i, v, id, want, ids[i] });
                }
            }
        }
        return null;
    }
};
