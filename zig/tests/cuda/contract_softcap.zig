//! act_rows.cu's tf_softcap_rows under the contract (a soft-capped head's logits), and controls each dropping a point.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const aref = core.act_ref;

/// tanhf within 2 ulps of its value (2^-22), twice that either way, as act_round_check's spread.
const spread = 0x1p-21;
const cap = 30;
const all = 3;

/// A registration: the served kernel (keep null), or a control keeping all rounding points but one (act_round.cuh).
pub const Entry = struct { name: []const u8, keep: ?i32 = null };

pub const entries = [_]Entry{
    .{ .name = "cuda act tf_softcap_rows (MLX's bf16 steps, cap 30)" },
    .{ .name = "control: soft cap without x / cap's rounding", .keep = all & ~@as(i32, 1) },
    .{ .name = "control: soft cap without tanh's rounding", .keep = all & ~@as(i32, 2) },
};

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .none,
        .depth = depth,
        .mids = 16, // gate 3 is the kernel's own check: float64's rounding points, ranged over tanhf's ulps
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{ .size = 2048, .step = 8, .min = 8 }, .{} },
        // Gemma 4's head over its 262,144-token vocabulary: one row, a verify window, a prompt's last rows
        .served = &.{ .{ .rows = 1, .n = 262144, .k = 1 }, .{ .rows = 16, .n = 262144, .k = 1 }, .{ .rows = 3, .n = 4096, .k = 1 } },
    };
}

pub const SoftCap = struct {
    rig: rig_.Rig,
    entry: Entry,
    act: *const cuda.act_rows.ActRows,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    rows: usize = 0,
    n: usize = 0,
    x: []const u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *SoftCap) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(SoftCap);

    pub fn kernel(self: *SoftCap) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.keep != null) .shapes else null };
    }

    /// Logits spread over [-80, 80) (some past the cap, many near 0); non-finite: NaN and both infinities planted.
    pub fn prepare(self: *SoftCap, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % 8 != 0 or shape.n == 0 or shape.rows == 0) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const x = try a.alloc(u16, shape.rows * shape.n);
        for (x, 0..) |*v, i| {
            const wide: f32 = if (i % 16 == 3) 80 else if (i % 16 == 9) 0.5 else 30;
            v.* = bfr.bits((rng.float(f32) * 2 - 1) * wide);
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, f32 }{ .{ 0, std.math.nan(f32) }, .{ shape.n / 2, std.math.inf(f32) }, .{ shape.n - 1, -std.math.inf(f32) } };
            for (planted) |p| x[p[0]] = bfr.bits(p[1]);
        }
        self.rows = shape.rows;
        self.n = shape.n;
        self.x = x;
        self.in = try self.rig.input(std.mem.sliceAsBytes(x));
        self.out = try self.rig.output(x.len * 2);
        return true;
    }

    pub fn run(self: *SoftCap) !void {
        const s = self.rig.s;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(self.in);
            args.add(self.out);
            args.add(@as(c_int, @intCast(self.n)));
            args.add(@as(f32, cap));
            args.add(@as(c_int, self.entry.keep.?));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.rows), .y = @intCast((self.n / 8 + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else try self.act.softcap(s, self.in, self.out, self.rows, self.n, cap);
        try s.synchronize();
    }

    pub fn count(self: *SoftCap) usize {
        return self.rows * self.n;
    }

    pub fn reference(self: *SoftCap, out: []f64, mag: []f64) !void {
        for (self.x, out) |v, *o| o.* = aref.softcap(bfr.value(v), cap, 0);
        @memset(mag, 0);
    }

    /// Every output within float64's ends at the rounding points (a dropped point leaves them); non-finite as float64.
    pub fn check(self: *SoftCap, _: g.Gate) !?[]const u8 {
        const a = self.arena.?.allocator();
        const out = try a.alloc(u16, self.count());
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.bytes(self.out, out.len * 2));
        for (self.x, out, 0..) |v, got_bits, i| {
            const x: f64 = bfr.value(v);
            const r = aref.softcapRange(x, cap, spread);
            const got: f64 = bfr.value(got_bits);
            const ok = if (std.math.isFinite(r[0]) and std.math.isFinite(r[1]))
                std.math.isFinite(got) and r[0] <= got and got <= r[1]
            else
                c.classOf(got) == c.classOf(aref.softcap(x, cap, 0));
            if (!ok) return try std.fmt.bufPrint(&self.msg, "out[{d}] = {e} for x {e} outside float64's [{e}, {e}]", .{ i, got, x, r[0], r[1] });
        }
        return null;
    }
};

test "every control keeps all but one rounding point, each point once" {
    var seen: i32 = 0;
    for (entries) |e| if (e.keep) |k| {
        const dropped = all & ~k;
        try std.testing.expectEqual(@as(u32, 1), @popCount(dropped));
        seen |= dropped;
    };
    try std.testing.expectEqual(@as(i32, all), seen);
}
