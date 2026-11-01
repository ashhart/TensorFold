//! nemo_audio.cu's Sortformer head glue under the contract: im2col, padding and relu rows, exact.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const na = @import("nemotron").nemo_audio.kernels;
const rig_ = @import("contract_rig.zig");
const base = @import("contract_nemo_audio.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const val = base.val;
const randomBf16 = base.randomBf16;
const planted = base.planted;
const controls = base.controls;
const noSums = base.noSums;
/// The Sortformer head's width and its upsampled rows (8 frames a row).
const tf_d = 192;

pub const Glue = struct {
    rig: rig_.Rig,
    k: *const na.Kernels,
    ctl: ?cuda.Function = null, // relu's fmaxf control
    name: []const u8,
    op: Op,
    /// The im2col and zero_rows controls: one padding row read as valid.
    late: bool = false,
    control: ?g.Gate = null,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = undefined,
    len: usize = 0,
    x: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub const Op = enum { im2col3, relu, zero_rows };

    pub fn spec(t: *const Glue) c.Spec {
        return .{
            .name = t.name,
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .in_place = t.op != .im2col3,
            .tiles = .{ .{ .size = 256 }, .{}, .{} },
            // rows: the head's rows (cache, FIFO and window; relu: 8 frames each), n: its width
            .served = switch (t.op) {
                .relu => &.{ .{ .rows = 828 * 8, .n = tf_d, .k = 0 }, .{ .rows = 268 * 8, .n = tf_d, .k = 0 } },
                else => &.{ .{ .rows = 828, .n = tf_d, .k = 0 }, .{ .rows = 268, .n = tf_d, .k = 0 } },
            },
        };
    }

    pub fn deinit(t: *Glue) void {
        if (t.arena) |*a| a.deinit();
        t.rig.deinit();
    }

    const vt = g.hooks(Glue);
    pub fn kernel(t: *Glue) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }

    /// x [rows, n] in [-4, 4), the valid rows three quarters of them (at least one).
    pub fn prepare(t: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        if (t.arena) |*a| a.deinit();
        t.arena = std.heap.ArenaAllocator.init(t.rig.gpu.gpa);
        const a = t.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        t.shape = shape;
        t.len = @max(1, shape.rows * 3 / 4);
        t.x = try a.alloc(u16, shape.rows * shape.n);
        randomBf16(prng.random(), t.x, -4, 4);
        if (fill == .nonfinite) for (planted, 0..) |p, i| {
            t.x[(i * 131 + 7) % t.x.len] = bfr.bits(p);
        };
        if (t.op == .im2col3) {
            t.in = try t.rig.input(std.mem.sliceAsBytes(t.x));
            t.out = try t.rig.output(shape.rows * 3 * shape.n * 2);
        } else t.out = try t.rig.output(t.x.len * 2);
        return true;
    }

    pub fn run(t: *Glue) !void {
        const s = t.rig.s;
        const rows = t.shape.rows;
        const n = t.shape.n;
        const len = if (t.late) t.len + 1 else t.len;
        if (t.op != .im2col3) {
            const buf: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.out, .len = t.x.len * 2 };
            try buf.upload(0, std.mem.sliceAsBytes(t.x));
        }
        switch (t.op) {
            .im2col3 => try t.k.im2col3(s, t.in, t.out, rows, n, len),
            .zero_rows => try t.k.zeroRows(s, t.out, rows, n, len),
            .relu => if (t.ctl) |f| {
                var args: cuda.Args = .{};
                args.add(t.out);
                args.add(@as(i64, @intCast(t.x.len)));
                try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((t.x.len + 255) / 256) }, .block = .{ .x = 256 } }, s, &args);
            } else try t.k.reluInPlace(s, t.out, t.x.len),
        }
        try s.synchronize();
    }

    pub fn count(t: *Glue) usize {
        return if (t.op == .im2col3) 3 * t.x.len else t.x.len;
    }

    /// Copies, zeros and relu(x) = x < 0 ? 0 : x (NaN stays): no sums, no magnitude.
    pub fn reference(t: *Glue, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        switch (t.op) {
            .im2col3 => for (0..t.shape.rows) |r| for (0..3) |j| for (0..n) |ch| {
                const src = @as(isize, @intCast(r + j)) - 1;
                want[(r * 3 + j) * n + ch] = if (src >= 0 and src < t.len) val(t.x[@as(usize, @intCast(src)) * n + ch]) else 0;
            },
            .zero_rows => for (want, t.x, 0..) |*w, x, i| {
                w.* = if (i / n < t.len) val(x) else 0;
            },
            .relu => for (want, t.x) |*w, x| {
                w.* = if (val(x) < 0) 0 else val(x);
            },
        }
        @memset(mag, 0);
    }
};
