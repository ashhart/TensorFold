//! The unquantized embedding gather under the contract on HIP: tf_embed_dense on BF16 and fp32 tables, two controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const bf16 = core.quant.convert.bf16;

/// The tables a gather reads; the controls read a BF16 table the next row down, or as fp16.
pub const Which = enum { bf16_table, f32_table, wrong_row, wrong_kind };

const vocab = 4099;

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "a NaN stays a NaN; infinities and finite values round once to the output's type, to nearest even";

/// Rows a prompt pass gathers at Clef-Flash's width: one, a fill's piece, a chunk and a prefill step; and a narrow one.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 4096, .k = 0 },
    .{ .rows = 512, .n = 4096, .k = 0 },
    .{ .rows = 1024, .n = 4096, .k = 0 },
    .{ .rows = 2048, .n = 4096, .k = 0 },
    .{ .rows = 7, .n = 1024, .k = 0 },
};

pub fn spec(w: Which) c.Spec {
    const name = switch (w) {
        .bf16_table => "hip tf_embed_dense (bf16 table)",
        .f32_table => "hip tf_embed_dense (fp32 table)",
        .wrong_row => "control: tf_embed_dense gathers the next row",
        .wrong_kind => "control: tf_embed_dense reads a bf16 table as fp16",
    };
    return .{
        .name = name,
        .accum = .none,
        .depth = none,
        .out = .raw,
        .replaces = if (w == .f32_table) "tf_embed_dense before its table kind" else null,
        .nan = .{ .stated = rule },
        .served = &served,
    };
}

/// shape.rows ids into a table of `vocab` rows of shape.n values: each output row a table row in the activation kind.
pub const Embed = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    /// The gather before its table kind, from the controls' code object: the fp32 table's baseline.
    base: hip.Function,
    which: Which,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    table: []f32 = &.{},
    ids: []i32 = &.{},
    tab: u64 = 0,
    ids_dev: u64 = 0,
    next_dev: u64 = 0,
    out: u64 = 0,
    base_out: u64 = 0,
    msg: [192]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, ctl: hip.Module, w: Which) !Embed {
        const base = try ctl.function("tf_base_embed_dense");
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .base = base, .which = w, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Embed) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Embed) g.Kernel {
        const control: ?g.Gate = if (self.which == .wrong_row or self.which == .wrong_kind) .shapes else null;
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Embed);

    fn bf16Table(self: *const Embed) bool {
        return self.which != .f32_table;
    }

    fn outSize(self: *const Embed) usize {
        return kit.size(self.k);
    }

    /// Values over forty binades, exact in the table's type; non-finite fills plant NaN, infinities and bf16's edges.
    pub fn prepare(self: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.table = try a.alloc(f32, vocab * shape.n);
        for (self.table) |*v| {
            const x: f32 = @floatCast(rng.floatNorm(f64) * std.math.pow(f64, 2, @floatFromInt(rng.intRangeAtMost(i32, -20, 20))));
            v.* = if (self.bf16Table()) c.fromBf16(bf16(x)) else x;
        }
        self.ids = try a.alloc(i32, shape.rows);
        for (self.ids, 0..) |*id, r| id.* = switch (r) {
            0 => vocab - 1,
            1 => 0,
            else => rng.intRangeLessThan(i32, 0, vocab),
        };
        if (fill == .nonfinite) {
            const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), 3.4e38, 1e-39 };
            const planted_n: usize = if (self.bf16Table()) 3 else planted.len;
            for (planted[0..planted_n], 0..) |v, j| {
                const row: usize = @intCast(self.ids[j % shape.rows]);
                self.table[row * shape.n + (j * 37) % shape.n] = v;
            }
        }
        const bytes = try a.alloc(u8, self.table.len * @as(usize, if (self.bf16Table()) 2 else 4));
        for (self.table, 0..) |v, i| if (self.bf16Table()) kit.put(.bf16, bytes, i, v) else kit.put(.f32, bytes, i, v);
        self.tab = try self.rig.input(bytes);
        self.ids_dev = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        const next = try a.alloc(i32, shape.rows);
        for (next, self.ids) |*x, id| x.* = @mod(id + 1, vocab);
        self.next_dev = try self.rig.input(std.mem.sliceAsBytes(next));
        self.out = try self.rig.output(self.count());
        self.base_out = try self.rig.scratch(self.count());
        return true;
    }

    pub fn run(self: *Embed) !void {
        const s = self.shape;
        const kind: c_int = switch (self.which) {
            .f32_table => 0,
            .wrong_kind => 1,
            else => 2,
        };
        const ids = if (self.which == .wrong_row) self.next_dev else self.ids_dev;
        try self.l.tf_embed_dense(@ptrFromInt(self.tab), kind, @ptrFromInt(ids), @intCast(s.rows), @intCast(s.n), @ptrFromInt(self.out), @backingInt(self.k), self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The old fp32 gather's bytes on the same table, ids and output kind.
    pub fn replaced(self: *Embed) ![]const u8 {
        const s = self.shape;
        var args: hip.launch.Args = .{};
        try args.add(self.tab);
        try args.add(self.ids_dev);
        try args.add(@as(c_int, @intCast(s.rows)));
        try args.add(@as(c_int, @intCast(s.n)));
        try args.add(self.base_out);
        try args.add(@as(c_int, @backingInt(self.k)));
        const total = s.rows * s.n;
        const cfg: hip.launch.Config = .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } };
        try hip.launch.launch(self.base, cfg, self.rig.s, &args);
        return self.rig.bytes(self.base_out, self.count());
    }

    pub fn count(self: *Embed) usize {
        return self.shape.rows * self.shape.n * self.outSize();
    }

    pub fn reference(_: *Embed, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every output element against its table value rounded once to the output's kind, by the stated rule.
    pub fn check(self: *Embed, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const got = try self.rig.bytes(self.out, self.count());
        for (0..s.rows) |r| for (0..s.n) |col| {
            const v = self.table[@as(usize, @intCast(self.ids[r])) * s.n + col];
            const i = r * s.n + col;
            const out = kit.get(self.k, got, i);
            const want = kit.round(self.k, v);
            const same = if (std.math.isNan(v)) std.math.isNan(out) else out == want and std.math.signbit(out) == std.math.signbit(want);
            if (!same) return try std.fmt.bufPrint(&self.msg, "row {d} (id {d}) column {d} is {e}, the table's {e} rounds to {e}", .{ r, self.ids[r], col, out, v, want });
        };
        return null;
    }
};
