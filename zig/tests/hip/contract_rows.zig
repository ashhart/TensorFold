//! Row kernels under the contract on HIP: the RMS norm rows (a wave a row, and a block a row) and the controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Which = enum {
    rows_f32,
    rows_f16,
    rows_bf16,
    block_f16,
    block_bf16,
    fmax_max,
    atomic_sum,
    scribble_sum,
    tail_sum,
    half_dot,
};

/// Every 32nd square a lane, its fp32 product, then the wave's xor tree.
fn laneDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6;
}

fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

const why = "a 16-bit output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps " ++
    "(the fp32-out form proves fp32 sums)";
const short = [_]c.Shape{
    .{ .rows = 32, .n = 256, .k = 256 },
    .{ .rows = 16, .n = 128, .k = 128 },
    .{ .rows = 1, .n = 1024, .k = 1024 },
};
const long = [_]c.Shape{
    .{ .rows = 17, .n = 2048, .k = 2048 },
    .{ .rows = 3, .n = 4096, .k = 4096 },
    .{ .rows = 2, .n = 5120, .k = 5120 },
    .{ .rows = 1, .n = 8192, .k = 8192 },
};
const row_tiles = [3]c.Tile{ .{ .size = 8 }, .{}, .{ .size = 32 } };
const block_tiles = [3]c.Tile{ .{}, .{}, .{ .size = 256 } };

pub fn spec(which: Which) c.Spec {
    var s = c.Spec{ .name = "", .accum = .f32, .depth = laneDepth, .ops = 8, .out = .f32, .served = &short };
    s.tiles = row_tiles;
    switch (which) {
        .rows_f32 => s.name = "hip RMS norm tf_rms_rows (fp32)",
        .rows_f16 => s.name = "hip RMS norm tf_rms_rows (fp16)",
        .rows_bf16 => s.name = "hip RMS norm tf_rms_rows (bf16)",
        .block_f16 => s.name = "hip RMS norm rms_kernel<__half>",
        .block_bf16 => s.name = "hip RMS norm rms_kernel<hip_bfloat16>",
        else => return controlSpec(which),
    }
    if (which == .rows_f32) return s;
    s.out = if (kindOf(which) == 1) .f16 else .bf16;
    s.sums = .{ .bf16 = 0, .why = why };
    if (which == .block_f16 or which == .block_bf16) {
        s.served = &long;
        s.tiles = block_tiles;
    }
    return s;
}

/// The controls': a row max, row sums and a dot over bf16 rows of 4,096.
fn controlSpec(which: Which) c.Spec {
    const ctl = &[_]c.Shape{.{ .rows = 8, .n = 1, .k = 4096 }};
    var s = c.Spec{ .name = "", .accum = .f32, .depth = laneChain, .out = .f32, .served = ctl };
    switch (which) {
        .fmax_max => {
            s.name = "control: fmax row max";
            s.accum = .none;
        },
        .atomic_sum => s.name = "control: atomic-add row sum",
        .scribble_sum => s.name = "control: row sum writing its input",
        .tail_sum => {
            s.name = "control: row sum losing a tail input";
            s.tiles = .{ .{}, .{}, .{ .size = 32 } };
        },
        else => s.name = "control: dot summed in half",
    }
    return s;
}

pub fn control(which: Which) ?g.Gate {
    return switch (which) {
        .fmax_max => .nonfinite,
        .atomic_sum, .scribble_sum => .repeat,
        .tail_sum => .shapes,
        .half_dot => .sums,
        else => null,
    };
}

fn opOf(w: Which) rows.Op {
    return switch (w) {
        .fmax_max => .max,
        .half_dot => .dot,
        .atomic_sum, .scribble_sum, .tail_sum => .sum,
        else => .rms,
    };
}

/// The kernels' activation kind: 0 fp32, 1 fp16, 2 bf16 (the controls read bf16).
fn kindOf(w: Which) c_int {
    return switch (w) {
        .rows_f32 => 0,
        .rows_f16, .block_f16 => 1,
        else => 2,
    };
}

pub const Row = struct {
    rig: rig_.Rig,
    which: Which,
    f: hip.Function,
    r: ?rows.Rows = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    /// The kernel `which` names: the launcher's for the norms, the controls' code object's for the rest.
    pub fn init(
        a: std.mem.Allocator,
        rt: *const hip.Runtime,
        s: hip.Stream,
        l: *const hip.Launcher,
        ctl: hip.Module,
        which: Which,
    ) !Row {
        const f = switch (which) {
            .rows_f32, .rows_f16, .rows_bf16 => l.op.rms_rows,
            .block_f16 => l.rms[1],
            .block_bf16 => l.rms[2],
            inline else => |w| try ctl.function("tf_ctl_" ++ @tagName(w)),
        };
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = which, .f = f };
    }

    pub fn deinit(self: *Row) void {
        if (self.r) |r| r.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Row) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    const vt = g.hooks(Row);

    pub fn prepare(self: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const block = self.which == .block_f16 or self.which == .block_bf16;
        const most: usize = if (block) 8192 else 1024;
        if (opOf(self.which) == .rms and shape.k > most) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.a;
        const r = try rows.Rows.init(a, opOf(self.which), shape);
        self.r = r;
        r.fill(fill, seed);
        const kind = kindOf(self.which);
        if (kind == 1) for (r.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0; // below fp16's normals a bf16 value may not be exact
        };
        const x = try a.alloc(u8, r.x.len * @as(usize, if (kind == 0) 4 else 2));
        defer a.free(x);
        for (r.x, 0..) |b, i| switch (kind) {
            0 => std.mem.writeInt(u32, x[4 * i ..][0..4], @bitCast(c.fromBf16(b)), .little),
            1 => std.mem.writeInt(u16, x[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(c.fromBf16(b)))), .little),
            else => std.mem.writeInt(u16, x[2 * i ..][0..2], b, .little),
        };
        self.x = try self.rig.input(x);
        if (opOf(self.which) == .rms) {
            const wf = try a.alloc(f32, r.w.len);
            defer a.free(wf);
            for (wf, r.w) |*o, b| o.* = c.fromBf16(b);
            self.w = try self.rig.input(std.mem.sliceAsBytes(wf));
        } else if (self.which == .half_dot) self.w = try self.rig.input(std.mem.sliceAsBytes(r.w));
        self.out = try self.rig.output(r.outputs() * c.size(spec(self.which).out));
        return true;
    }

    pub fn run(self: *Row) !void {
        const r = self.r.?;
        const k: c_int = @intCast(r.shape.k);
        const n: c_int = @intCast(r.shape.rows);
        var a: hip.launch.Args = .{};
        var cfg: hip.launch.Config = .{ .grid = .{ .x = @intCast(r.shape.rows) }, .block = .{ .x = 32 } };
        switch (self.which) {
            .rows_f32, .rows_f16, .rows_bf16 => {
                for ([_]u64{ self.x, self.w, self.out }, 0..) |p, i| {
                    try a.add(p);
                    if (i == 0) try a.add(kindOf(self.which));
                }
                try a.add(n);
                try a.add(k);
                try a.add(r.eps);
                cfg = .{ .grid = .{ .x = @intCast((r.shape.rows + 7) / 8) }, .block = .{ .x = 256 } };
            },
            .block_f16, .block_bf16 => {
                for ([_]u64{ self.x, self.w, self.out }) |p| try a.add(p);
                try a.add(k);
                try a.add(r.eps);
                cfg.block.x = 256;
            },
            .atomic_sum => {
                const zero = try self.rig.a.alloc(u8, r.outputs() * 4);
                defer self.rig.a.free(zero);
                @memset(zero, 0);
                try self.rig.put(self.out, zero);
                try a.add(self.x);
                try a.add(k);
                try a.add(self.out);
                const grid: hip.launch.Dim3 = .{ .x = @intCast((r.shape.k + 255) / 256), .y = @intCast(r.shape.rows) };
                cfg = .{ .grid = grid, .block = .{ .x = 256 } };
            },
            else => {
                try a.add(self.x);
                try a.add(k);
                try a.add(self.out);
                if (self.which == .half_dot) try a.add(self.w);
            },
        }
        try hip.launch.launch(self.f, cfg, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Row) usize {
        return self.r.?.outputs();
    }

    pub fn reference(self: *Row, want: []f64, mag: []f64) !void {
        return self.r.?.reference(want, mag);
    }

    pub fn simulate(self: *Row, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(acc, order, out);
    }
};
