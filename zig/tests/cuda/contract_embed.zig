//! embed_rows.cu's MLX affine rows under the contract (4-bit, 2 to 8 bits with a bf16 scale, head tiles), controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ec = @import("embed_check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;

/// A registration: code width, scaled (tf_embed_rows_affine) or not, a control, or `tiles` (tf_embed_rows_tiles4).
pub const Entry = struct { name: []const u8, bits: u4, scaled: bool = true, keep: ?i32 = null, tiles: bool = false };

const round = 1; // embed_affine.cuh's ROUND
const fold = 2; // and its FOLD
const mlx_order = 4; // and its MLX_ORDER

pub const entries = [_]Entry{
    .{ .name = "cuda embed_rows tf_embed_rows_affine4 (MLX 4-bit rows)", .bits = 4, .scaled = false },
    .{ .name = "cuda embed_rows tf_embed_rows_affine (2-bit, bf16 scale)", .bits = 2 },
    .{ .name = "cuda embed_rows tf_embed_rows_affine (3-bit, bf16 scale)", .bits = 3 },
    .{ .name = "cuda embed_rows tf_embed_rows_affine (4-bit, bf16 scale)", .bits = 4 },
    .{ .name = "cuda embed_rows tf_embed_rows_affine (5-bit, bf16 scale)", .bits = 5 },
    .{ .name = "cuda embed_rows tf_embed_rows_affine (6-bit, bf16 scale)", .bits = 6 },
    .{ .name = "cuda embed_rows tf_embed_rows_affine (8-bit, bf16 scale)", .bits = 8 },
    .{ .name = "control: scaled rows without the dequantized value's bf16 rounding", .bits = 4, .keep = 0 },
    .{ .name = "control: scaled rows with the scale folded into the group's scale and bias", .bits = 4, .keep = fold },
    .{ .name = "cuda embed_rows tf_embed_rows_tiles4 (4-bit head tiles, bf16 scale)", .bits = 4, .tiles = true },
    .{ .name = "control: tile rows with each word's codes in MLX's order", .bits = 4, .tiles = true, .keep = round | mlx_order },
};

/// affine4_pack.cu's two packers, which Affine4Kernels.pack launches at load (tf_pack_dense, tf_transpose_pad16).
pub const Pack = struct { words: cuda.Function, scales: cuda.Function };

fn depth(_: c.Shape) usize {
    return 0;
}

/// Shape: rows x row width (n); the group is 64.
pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .none, // no sums: every output is checked bit for bit against float64 at its rounding points
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 2048, .step = 64, .min = 64 }, .{} }, // 256 threads of 8 values a block
        // Gemma 4 12B's, 31B's and E4B's hidden
        .served = &.{ .{ .rows = 17, .n = 3840, .k = 1 }, .{ .rows = 3, .n = 5376, .k = 1 }, .{ .rows = 9, .n = 2560, .k = 1 } },
        // the tile rows are the stored table's rows, byte for byte; check() still holds them to float64
        .replaces = if (e.tiles) "tf_embed_rows_affine (the stored 4-bit table, bf16 scale)" else null,
    };
}

const vocab = 97;

pub const Embed = struct {
    rig: rig_.Rig,
    entry: Entry,
    e: *const cuda.embed_rows.EmbedRows,
    ctl: ?cuda.Function = null,
    pack: ?Pack = null,
    npad: usize = 0,
    mlx: [3]u64 = undefined, // the tile entries' MLX words, scales and biases, the replaced gather's table
    base: u64 = 0, // and its rows
    arena: ?std.heap.ArenaAllocator = null,
    k: ec.Case = undefined,
    dev: [5]u64 = undefined,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Embed) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Embed);

    pub fn kernel(self: *Embed) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.keep != null) .shapes else null };
    }

    pub fn prepare(self: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const d = shape.n;
        if (d % 64 != 0 or d == 0 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const bits = self.entry.bits;
        const codes = try a.alloc(u32, vocab * d);
        for (codes) |*q| q.* = rng.int(u32) & ((@as(u32, 1) << bits) - 1);
        const s = try a.alloc(u16, vocab * d / 64);
        const b = try a.alloc(u16, vocab * d / 64);
        for (s, b) |*x, *y| { // the checkpoint's ranges, one group in 16 far past them
            const big: f32 = if (rng.uintLessThan(u32, 16) == 0) 0x1p40 else 1;
            x.* = bfr.bits((rng.float(f32) * 0.5 - 0.18) * big);
            y.* = bfr.bits((rng.float(f32) * 6.8 - 4.1) * big);
        }
        const ids = try a.alloc(i32, shape.rows);
        for (ids, 0..) |*x, i| x.* = @intCast(if (i == 0) vocab - 1 else rng.uintLessThan(usize, vocab));
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, u16 }{ .{ 0, bfr.bits(std.math.nan(f32)) }, .{ 1, bfr.bits(std.math.inf(f32)) }, .{ 2, bfr.bits(-std.math.inf(f32)) } };
            for (planted, 0..) |p, i| if (i < shape.rows) {
                const t: usize = @intCast(ids[i]);
                (if (i == 1) b else s)[t * d / 64 + p[0] + 1] = p[1];
            };
        }
        if (fill == .stress) for (s) |*x| { // fp32's largest scales: q s + b and its scaled value past bf16's range
            x.* = bfr.bits(0x1p126);
        };
        const scale: ?f32 = if (self.entry.scaled) bfr.round32(@sqrt(@as(f32, @floatFromInt(d)))) else null;
        const w = try ec.pack(a, codes, bits);
        self.k = .{ .rows = shape.rows, .d = d, .bits = bits, .group = 64, .scale = scale, .ids = ids, .w = w, .s = s, .b = b };
        for ([_][]const u8{ std.mem.sliceAsBytes(ids), std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(s), std.mem.sliceAsBytes(b) }, 0..) |bytes, i|
            self.dev[i] = try self.rig.input(bytes);
        if (self.entry.tiles) try self.packTiles(d);
        self.dev[4] = try self.rig.output(shape.rows * d * 2);
        return true;
    }

    /// The uploaded MLX table packed into a tied head's tiles (Affine4.layout), which the call then reads in its place.
    fn packTiles(self: *Embed, d: usize) !void {
        const p = self.pack.?;
        const st = self.rig.s;
        const l = cuda.qlinear.Affine4.layout(vocab, d);
        self.npad = l.npad;
        const raw = [3]u64{ self.dev[1], self.dev[2], self.dev[3] };
        self.mlx = raw;
        self.base = try self.rig.scratch(self.k.rows * d * 2);
        const a = self.arena.?.allocator();
        // every byte of the tiles is written by the packers below
        for ([_]usize{ l.words, l.scales, l.scales }, 1..) |len, i| self.dev[i] = try self.rig.input(try a.alloc(u8, len));
        const kg = d / 64;
        const total: u64 = @as(u64, l.npad / 64) * kg * 512;
        var wa: cuda.Args = .{};
        wa.add(raw[0]);
        for ([_]usize{ vocab, d / 8, kg }) |v| wa.add(@as(c_int, @intCast(v)));
        wa.add(self.dev[1]);
        wa.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(p.words, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, st, &wa);
        for (raw[1..], self.dev[2..4]) |src, out| {
            var ta: cuda.Args = .{};
            ta.add(src);
            for ([_]usize{ vocab, kg, l.npad }) |v| ta.add(@as(c_int, @intCast(v)));
            ta.add(out);
            try cuda.launch.launch(p.scales, .{ .grid = .{ .x = @intCast((kg * l.npad + 255) / 256) }, .block = .{ .x = 256 } }, st, &ta);
        }
        try st.synchronize();
    }

    fn tiles(self: *Embed) cuda.qlinear.Affine4 {
        return .{ .w = self.dev[1], .s = self.dev[2], .b = self.dev[3], .n = vocab, .k = self.k.d, .npad = self.npad };
    }

    pub fn run(self: *Embed) !void {
        const k = self.k;
        const st = self.rig.s;
        const v = self.dev;
        if (self.ctl != null and self.entry.tiles) {
            var args: cuda.Args = .{};
            for (v) |p| args.add(p);
            for ([_]usize{ k.d, self.npad }) |x| args.add(@as(c_int, @intCast(x)));
            args.add(k.scale.?);
            args.add(@as(c_int, self.entry.keep.?));
            try cuda.launch.launch(self.ctl.?, .{ .grid = .{ .x = @intCast(k.rows), .y = @intCast((k.d / 8 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, st, &args);
        } else if (self.ctl) |f| {
            var args: cuda.Args = .{};
            for (v) |p| args.add(p);
            for ([_]usize{ k.d, k.bits, k.group }) |x| args.add(@as(c_int, @intCast(x)));
            args.add(k.scale.?);
            args.add(@as(c_int, self.entry.keep.?));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(k.rows), .y = @intCast((k.d / 8 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, st, &args);
        } else if (self.entry.tiles) {
            try self.e.tiles4(st, v[0], self.tiles(), v[4], k.rows, k.scale);
        } else if (self.entry.scaled) {
            try self.e.affine(st, v[0], v[1], v[2], v[3], v[4], k.rows, k.d, .{ .bits = k.bits, .scale = k.scale });
        } else try self.e.affine4(st, v[0], v[1], v[2], v[3], v[4], k.rows, k.d);
        try st.synchronize();
    }

    pub fn count(self: *Embed) usize {
        return self.k.rows * self.k.d;
    }

    pub fn reference(self: *Embed, out: []f64, mag: []f64) !void {
        for (0..self.k.rows) |row| for (0..self.k.d) |i| {
            out[row * self.k.d + i] = ec.want(self.k, row, i);
        };
        @memset(mag, 0);
    }

    /// Every output bit-equal to float64's value at the rounding points (a dropped or moved point leaves it).
    /// The replaced kernel's bytes: tf_embed_rows_affine on the stored table, first held to float64 by `verify`.
    pub fn replaced(self: *Embed) ![]const u8 {
        const k = self.k;
        try self.e.affine(self.rig.s, self.dev[0], self.mlx[0], self.mlx[1], self.mlx[2], self.base, k.rows, k.d, .{ .bits = k.bits, .scale = k.scale });
        const n = k.rows * k.d;
        const want = try self.arena.?.allocator().alloc(u16, n);
        @memcpy(std.mem.sliceAsBytes(want), try self.rig.bytes(self.base, n * 2));
        var stats: ec.Stats = .{};
        if (try ec.verify(k, want, &stats, &self.msg)) |why| {
            std.debug.print("baseline {s}: {s}\n", .{ self.entry.name, why });
            return error.BaselineOffItsRoundingPoints;
        }
        return std.mem.sliceAsBytes(want);
    }

    pub fn check(self: *Embed, _: g.Gate) !?[]const u8 {
        const n = self.count();
        const out = try self.arena.?.allocator().alloc(u16, n);
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.bytes(self.dev[4], n * 2));
        var stats: ec.Stats = .{};
        return ec.verify(self.k, out, &stats, &self.msg);
    }
};

test {
    _ = ec;
}

test "the controls drop the one rounding point, fold the scale past it, or read tile words in MLX's order" {
    var seen: usize = 0;
    for (entries) |e| if (e.keep) |k| {
        try std.testing.expect(e.scaled and if (e.tiles) k == round | mlx_order else k & round == 0);
        seen += 1;
    };
    try std.testing.expectEqual(@as(usize, 3), seen);
}
