//! The XPU gate controls: five kernels built to break one gate each, which that gate must report as failing.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.spec;
const g = core.gates;
const rows = core.rows;

/// The five planted kernels of `zig/kernels/xpu/contract_gate_controls.cl`.
pub const Which = enum { fmax_max, atomic_sum, scribble, drop_tail, fp16_sum };

const K = 4096; // the rows' width, as contract_rows.zig's controls use

fn laneDepth(s: c.Shape) usize {
    return s.k / 256 + 8;
}
fn noSums(_: c.Shape) usize {
    return 0;
}

fn symbol(w: Which) []const u8 {
    return switch (w) {
        .fmax_max => "tf_ctl_row_max_fmax",
        .atomic_sum => "tf_ctl_atomic_sum",
        .scribble => "tf_ctl_scribble",
        .drop_tail => "tf_ctl_drop_tail",
        .fp16_sum => "tf_ctl_fp16_sum",
    };
}

fn opOf(w: Which) rows.Op {
    return switch (w) {
        .fmax_max => .max,
        .fp16_sum => .dot,
        else => .sum,
    };
}

/// The gate each control must be red at.
pub fn control(w: Which) g.Gate {
    return switch (w) {
        .fmax_max => .nonfinite,
        .atomic_sum, .scribble => .repeat,
        .drop_tail => .shapes,
        .fp16_sum => .sums,
    };
}

pub fn spec(w: Which) c.Spec {
    const ctl = [_]c.Shape{.{ .rows = 8, .n = 1, .k = K }};
    var s = c.Spec{ .name = "", .accum = .f32, .depth = laneDepth, .out = .f32, .served = &ctl };
    switch (w) {
        .fmax_max => {
            s.name = "control: fmax row max";
            s.accum = .none;
            s.depth = noSums;
        },
        .atomic_sum => s.name = "control: atomic-add row sum",
        .scribble => s.name = "control: row sum writing its input",
        .drop_tail => {
            s.name = "control: row sum losing a tail input";
            s.tiles = .{ .{}, .{}, .{ .size = 32 } };
        },
        .fp16_sum => s.name = "control: dot summed in half",
    }
    return s;
}

/// shape.rows bf16 rows of shape.k, as the contract's rows family builds them.
pub const Control = struct {
    rig: rig_.Rig,
    k: xpu.Kernel,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    r: rows.Rows = undefined,
    xb: []u8 = &.{},
    wb: []u8 = &.{},
    zeros: []u8 = &.{},
    xat: ?*anyopaque = null,
    wat: ?*anyopaque = null,
    out: ?*anyopaque = null,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, which: Which) !Control {
        const sym = try a.allocSentinel(u8, symbol(which).len, 0);
        defer a.free(sym);
        @memcpy(sym, symbol(which));
        return .{ .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.gate, sym, .{ 256, 1, 1 }), .which = which, .arena = .init(a) };
    }

    pub fn deinit(self: *Control) void {
        self.k.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Control) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    const vt = g.hooks(Control);

    pub fn prepare(self: *Control, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        self.r = try rows.Rows.init(a, opOf(self.which), shape);
        self.r.fill(fill, seed);
        const xb = try a.alloc(u16, self.r.x.len);
        @memcpy(xb, self.r.x);
        self.xb = try a.alloc(u8, self.r.x.len * 2);
        @memcpy(self.xb, std.mem.sliceAsBytes(xb));
        self.wb = try a.alloc(u8, self.r.w.len * 2);
        @memcpy(self.wb, std.mem.sliceAsBytes(self.r.w));
        self.zeros = try a.alloc(u8, shape.rows * 4);
        @memset(self.zeros, 0);
        self.xat = try self.rig.input(self.xb);
        self.wat = try self.rig.input(self.wb);
        self.out = try self.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(self: *Control) !void {
        const s = self.shape;
        const k: u32 = @intCast(s.k);
        switch (self.which) {
            .fmax_max, .scribble, .drop_tail => try self.rig.launch(self.k, .{ @intCast(s.rows), 1, 1 }, .{ self.xat, self.out, k }),
            .atomic_sum => {
                try self.rig.put(self.out, self.zeros); // every add lands on a zeroed sum
                try self.rig.launch(self.k, .{ (k + 255) / 256, @intCast(s.rows), 1 }, .{ self.xat, self.out, k });
            },
            .fp16_sum => try self.rig.launch(self.k, .{ @intCast(s.rows), 1, 1 }, .{ self.xat, self.wat, self.out, k }),
        }
    }

    pub fn count(self: *Control) usize {
        return self.shape.rows;
    }

    pub fn reference(self: *Control, want: []f64, mag: []f64) !void {
        try self.r.reference(want, mag);
    }

    pub fn simulate(self: *Control, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.r.simulate(acc, order, out);
    }
};
