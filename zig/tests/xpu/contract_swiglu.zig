//! The Qwen3.8-27B glue's SwiGLU under the contract on XPU: the activation, and the row sums it writes beside it.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const c = core.spec;
const g = core.gates;
const ref = core.xpu_ref;

/// The activation the model reads, or the row sums the quantizer reads beside it.
pub const Part = enum { act, sums };

const FF = 17408; // Q27_FF, the SwiGLU's width
const GU = 2 * FF; // the gate and up columns in one row
const GS: usize = 64;
const BLOCKS = FF / 1024; // the grid's second dimension
const MAX_ROWS = 8320;

const why = "the f16 output's own rounding is 2^-11 of the value, and a f16 sum's error over a group of 64 is a few " ++
    "of those ulps, so its miss stays under the 100x the gate asks";

fn zero(_: c.Shape) usize {
    return 0;
}

const served = [_]c.Shape{
    .{ .rows = 1, .n = FF, .k = FF },
    .{ .rows = 16, .n = FF, .k = FF },
    .{ .rows = 1152, .n = FF, .k = FF },
};

pub fn spec(p: Part) c.Spec {
    var s = c.Spec{ .name = "", .accum = .none, .depth = zero, .ops = 16, .mids = 16, .out = .f16, .served = &served };
    s.tiles = .{ .{ .size = 16, .min = 1 }, .{}, .{} };
    switch (p) {
        .act => s.name = "xpu swiglu (the activation, fp16)",
        .sums => {
            s.name = "xpu swiglu (the row sums, fp16)";
            s.accum = .f32;
            s.depth = ref.sumsDepth;
            s.ops = 0;
            s.mids = 0;
            s.sums = .{ .f16 = 0, .bf16 = 0, .why = why };
        },
    }
    return s;
}

/// shape.rows rows of shape.n: the activation, or the row sums of the groups it covers.
pub const Swiglu = struct {
    rig: rig_.Rig,
    k: xpu.Kernel,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    gu: []f64 = &.{},
    act: []f64 = &.{},
    gub: []u8 = &.{},
    actb: []u8 = &.{},
    gin: ?*anyopaque = null,
    hout: ?*anyopaque = null,
    ahi: ?*anyopaque = null,
    alo: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, part: Part) !Swiglu {
        return .{ .rig = .{ .a = a, .dev = dev }, .k = try dev.kernelFor(.glue, "swiglu", a), .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Swiglu) void {
        self.k.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Swiglu) g.Kernel {
        return .{ .spec = spec(self.part), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Swiglu);

    pub fn prepare(self: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > MAX_ROWS or shape.n != FF) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        self.gu = try a.alloc(f64, rows * GU);
        for (self.gu) |*v| v.* = kit.round(.f16, -4 + rng.float(f64) * 8);
        if (fill == .stress) {
            // silu(4) · 64 a term: a group's sum is 16,088, the output 64,352, in fp16's range and exact in fp32
            for (0..rows) |r| {
                for (0..FF) |j| {
                    self.gu[r * GU + j] = 4;
                    self.gu[r * GU + FF + j] = 64;
                }
            }
        }
        if (fill == .nonfinite) {
            self.gu[5] = std.math.nan(f64);
            self.gu[GU + 3] = std.math.inf(f64);
            self.gu[2 * GU + 1] = -std.math.inf(f64);
            self.gu[3 * GU + 2] = std.math.inf(f64);
        }
        self.gub = try kit.bytes(a, self.gu, .f16);
        self.act = try a.alloc(f64, rows * FF);
        self.gin = try self.rig.input(self.gub);
        self.ahi = if (self.part == .sums) try self.rig.output(rows * (FF / GS) * 2) else try self.rig.scratch(rows * (FF / GS) * 2);
        self.alo = try self.rig.scratch(rows * (FF / GS) * 2);
        self.hout = if (self.part == .act) try self.rig.output(rows * FF * 2) else try self.rig.scratch(rows * FF * 2);
        if (self.part == .sums) {
            // the sums cover the activation the card wrote: the reference reads those fp16 bytes, not a model
            try self.run();
            const h = try self.rig.bytes(self.hout, rows * FF * 2);
            for (self.act, 0..) |*v, i| v.* = kit.get(.f16, h, i);
        }
        return true;
    }

    pub fn run(self: *Swiglu) !void {
        const s = self.shape;
        try self.rig.launch(self.k, .{ @intCast(s.rows), BLOCKS, 1 }, .{
            self.gin,     self.hout,   self.ahi,    self.alo,
            @as(u32, GS), @as(u32, 0), @as(u32, 0),
        });
    }

    pub fn count(self: *Swiglu) usize {
        return if (self.part == .act) self.shape.rows * FF else self.shape.rows * (FF / GS);
    }

    pub fn reference(self: *Swiglu, want: []f64, mag: []f64) !void {
        switch (self.part) {
            .act => ref.swigluAct(self.gu, FF, want, mag),
            .sums => ref.swigluSums(self.act, FF, want, mag),
        }
    }

    pub fn simulate(self: *Swiglu, acc: c.Accum, order: g.Order, out: []f64) !void {
        switch (self.part) {
            .act => @memset(out, 0),
            .sums => try ref.simulateSums(self.arena.allocator(), self.act, FF, acc, order, out),
        }
    }
};
