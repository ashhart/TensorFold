//! Level Zero TP={2,4} collective kernels in tp.cl under the XPU kernel contract (gates 1 to 4).
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const c = core.spec;
const g = core.gates;
const rows = core.rows;

/// All nine P2P collective kernels in zig/kernels/xpu/tp.cl.
pub const Which = enum { ar2_f16, ar4_f16, rs2_f16, rs4_f16, ar2_bf16, ar4_bf16, ar2_f32, ar4_f32, rs4_f32 };

const why_k2_f16 = "two fp16 inputs sum once directly to fp16 without an intermediate narrowing loss";
const why_k2_bf16 = "two bf16 inputs sum once directly to bf16, while fp16 sums overflow at 65536";

fn symbol(w: Which) []const u8 {
    return switch (w) {
        .ar2_f16 => "tp2_allreduce_f16",
        .ar4_f16 => "tp4_allreduce_f16",
        .rs2_f16 => "tp2_rs_f16",
        .rs4_f16 => "tp4_rs_f16",
        .ar2_bf16 => "tp2_allreduce_bf16",
        .ar4_bf16 => "tp4_allreduce_bf16",
        .ar2_f32 => "tp2_allreduce_f32",
        .ar4_f32 => "tp4_allreduce_f32",
        .rs4_f32 => "tp4_rs_f32",
    };
}

fn ranksOf(w: Which) usize {
    return switch (w) {
        .ar2_f16, .rs2_f16, .ar2_bf16, .ar2_f32 => 2,
        else => 4,
    };
}

fn kindOf(w: Which) kit.Kind {
    return switch (w) {
        .ar2_f16, .ar4_f16, .rs2_f16, .rs4_f16 => .f16,
        .ar2_bf16, .ar4_bf16 => .bf16,
        .ar2_f32, .ar4_f32, .rs4_f32 => .f32,
    };
}

fn isRs(w: Which) bool {
    return w == .rs2_f16 or w == .rs4_f16 or w == .rs4_f32;
}

fn depth2(_: c.Shape) usize {
    return 1;
}

fn depth4(_: c.Shape) usize {
    return 3;
}

const served2 = [_]c.Shape{
    .{ .rows = 1, .n = 4, .k = 2 },
    .{ .rows = 1, .n = 1024, .k = 2 },
    .{ .rows = 1, .n = 4096, .k = 2 },
};

const served4 = [_]c.Shape{
    .{ .rows = 1, .n = 4, .k = 4 },
    .{ .rows = 1, .n = 1024, .k = 4 },
    .{ .rows = 1, .n = 4096, .k = 4 },
};

pub fn spec(w: Which) c.Spec {
    const tp = ranksOf(w);
    const kind = kindOf(w);
    const sums: c.Sums = switch (w) {
        .ar2_f16, .rs2_f16 => .{ .f16 = 0, .bf16 = 0, .why = why_k2_f16 },
        .ar2_bf16 => .{ .bf16 = 0, .why = why_k2_bf16 },
        else => .{},
    };
    return .{
        .name = symbol(w),
        .accum = .f32,
        .depth = if (tp == 2) depth2 else depth4,
        .out = kit.elem(kind),
        .tiles = .{ .{}, .{ .size = 1024, .step = 4, .min = 4 }, .{} },
        .served = if (tp == 2) &served2 else &served4,
        .sums = sums,
    };
}

/// Device buffers and host reference inputs for one tp.cl collective kernel.
pub const Tp = struct {
    rig: rig_.Rig,
    k: xpu.Kernel,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    inputs: [4][]f64 = .{ &.{}, &.{}, &.{}, &.{} },
    devs: [4]?*anyopaque = .{ null, null, null, null },
    out: ?*anyopaque = null,
    slice_off: usize = 0,

    pub fn init(a: std.mem.Allocator, dev: *rig_.Dev, which: Which) !Tp {
        return .{ .rig = .{ .a = a, .dev = dev }, .k = try dev.kernelFor(.tp, symbol(which), a), .which = which, .arena = .init(a) };
    }

    pub fn deinit(self: *Tp) void {
        self.k.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Tp) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Tp);

    fn stressVals(w: Which) [4]f64 {
        return switch (w) {
            .ar2_f16, .rs2_f16 => .{ 4096, 16, 0, 0 },
            .ar2_bf16 => .{ 65536, 512, 0, 0 },
            .ar2_f32 => .{ 1048576, 4096, 0, 0 },
            .ar4_f16, .rs4_f16 => .{ 40000, 40000, -40000, -39936 },
            .ar4_bf16 => .{ 131072, 256, -131072, 0 },
            .ar4_f32, .rs4_f32 => .{ 1048576, 4096, -1048576, 0 },
        };
    }

    pub fn prepare(self: *Tp, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const tp = ranksOf(self.which);
        if (shape.rows == 0 or shape.n == 0 or shape.n % 4 != 0 or shape.k != tp) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        const kind = kindOf(self.which);
        const total = shape.rows * shape.n;
        const full_n = if (isRs(self.which)) tp * total else total;
        self.slice_off = if (isRs(self.which)) total else 0;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..tp) |r| {
            self.inputs[r] = try a.alloc(f64, full_n);
            if (fill == .stress) {
                const sv = stressVals(self.which)[r];
                for (self.inputs[r]) |*v| v.* = kit.round(kind, sv);
            } else {
                for (self.inputs[r]) |*v| v.* = kit.round(kind, -4 + rng.float(f64) * 8);
            }
        }
        if (fill == .nonfinite) {
            const off = self.slice_off;
            self.inputs[0][off] = std.math.nan(f64);
            if (shape.rows > 1) self.inputs[1 % tp][off + shape.n] = std.math.inf(f64);
            if (shape.rows > 2) self.inputs[0][off + 2 * shape.n] = -std.math.inf(f64);
            if (shape.rows > 3) {
                self.inputs[1 % tp][off + 3 * shape.n] = std.math.inf(f64);
                self.inputs[0][off + 3 * shape.n] = -std.math.inf(f64);
            }
        }
        for (0..tp) |r| {
            const b = try kit.bytes(a, self.inputs[r], kind);
            self.devs[r] = try self.rig.input(b);
        }
        self.out = try self.rig.output(total * kit.size(kind));
        return true;
    }

    pub fn run(self: *Tp) !void {
        const n4: i32 = @intCast((self.shape.rows * self.shape.n) / 4);
        const off4: i32 = @intCast(self.slice_off / 4);
        const groups: [3]u32 = .{ @intCast((@as(usize, @intCast(n4)) + 255) / 256), 1, 1 };
        const tp = ranksOf(self.which);
        if (isRs(self.which)) {
            if (tp == 2) {
                try self.rig.launch(self.k, groups, .{ self.devs[0], self.devs[1], self.out, off4, n4 });
            } else {
                try self.rig.launch(self.k, groups, .{ self.devs[0], self.devs[1], self.devs[2], self.devs[3], self.out, off4, n4 });
            }
        } else if (tp == 2) {
            try self.rig.launch(self.k, groups, .{ self.devs[0], self.devs[1], self.out, n4 });
        } else {
            try self.rig.launch(self.k, groups, .{ self.devs[0], self.devs[1], self.devs[2], self.devs[3], self.out, n4 });
        }
    }

    pub fn count(self: *Tp) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Tp, want: []f64, mag: []f64) !void {
        const tp = ranksOf(self.which);
        const kind = kindOf(self.which);
        for (0..self.count()) |i| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (0..tp) |r| {
                const v = self.inputs[r][self.slice_off + i];
                s += v;
                abs += @abs(v);
            }
            want[i] = kit.round(kind, s);
            mag[i] = abs;
        }
    }

    pub fn simulate(self: *Tp, acc: c.Accum, order: g.Order, out: []f64) !void {
        const tp = ranksOf(self.which);
        var terms: [4]f64 = undefined;
        for (0..self.count()) |i| {
            for (0..tp) |r| terms[r] = self.inputs[r][self.slice_off + i];
            out[i] = rows.sumIn(acc, order, terms[0..tp]);
        }
    }
};
