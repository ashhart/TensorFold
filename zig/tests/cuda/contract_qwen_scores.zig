//! The production FP32 stats launcher through all five CUDA contract gates, with a real omitted-tail control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ref = @import("qwen_scores_ref.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
pub fn spec(control: bool) c.Spec {
    return .{ .name = if (control) "control: Qwen stats omit vocabulary tail" else "cuda qwen35_bf16_rows tf_q35_score_stats", .accum = .f32, .depth = ref.reductionDepth, .ops = 12, .out = .f32, .nan = .{ .stated = "NaN propagates to LSE; top ignores NaNs; +/-Inf normalizers retained; lower-ID ties; unfilled words zero" }, .tiles = .{ .{ .size = 128 }, .{ .size = 256 }, .{ .size = 20 } }, .served = &.{ .{ .rows = 4, .n = 33, .k = 2 }, .{ .rows = 128, .n = 257, .k = 20 }, .{ .rows = 1, .n = 248320, .k = 1 }, .{ .rows = 17, .n = 65537, .k = 0 } }, .sums = .{ .exact = false } };
}
pub const Case = struct {
    const vt = g.hooks(@This());
    rig: Rig,
    kernels: *const cuda.qwen35_bf16_rows.Kernels,
    ctl: ?cuda.Function = null,
    shape: c.Shape = undefined,
    arena: ?std.heap.ArenaAllocator = null,
    input: []f32 = &.{},
    picks: []u32 = &.{},
    xd: u64 = 0,
    pd: u64 = 0,
    out: u64 = 0,
    pub fn deinit(self: *Case) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.ctl != null), .ctx = self, .vt = &vt, .control = if (self.ctl != null) .shapes else null };
    }
    pub fn prepare(self: *Case, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 128 or s.n == 0 or s.k > @min(20, s.n)) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = s;
        self.input = try a.alloc(f32, s.rows * s.n);
        self.picks = try a.alloc(u32, s.rows);
        var random = std.Random.DefaultPrng.init(seed);
        for (self.input) |*x| x.* = random.random().float(f32) * 20 - 10;
        for (self.picks, 0..) |*pick, r| {
            pick.* = @intCast(r % s.n);
            const row = self.input[r * s.n ..][0..s.n];
            row[s.n - 1] = 20;
            if (fill == .stress) ref.stress(row);
            if (fill == .nonfinite) switch (r % 5) {
                0 => row[0] = std.math.nan(f32),
                1 => row[0] = std.math.inf(f32),
                2 => @memset(row, -std.math.inf(f32)),
                3 => @memset(row, std.math.nan(f32)),
                else => {
                    row[0] = std.math.inf(f32);
                    row[s.n - 1] = -std.math.inf(f32);
                },
            };
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.input));
        self.pd = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        self.out = try self.rig.output(s.rows * ref.words * 4);
        return true;
    }
    pub fn run(self: *Case) !void {
        var kernels = self.kernels.*;
        if (self.ctl) |f| kernels.stats_fn = f;
        try kernels.stats(self.rig.s, self.xd, self.shape.n, self.pd, @intCast(self.shape.k), self.out, self.shape.rows);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Case) usize {
        return self.shape.rows * ref.words;
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        for (0..self.shape.rows) |r| {
            const expected = ref.row(self.input[r * self.shape.n ..][0..self.shape.n], self.picks[r], self.shape.k);
            for (expected.bits, 0..) |x, i| want[r * ref.words + i] = @as(f32, @bitCast(x));
            want[r * ref.words] = expected.lse;
            mag[r * ref.words] = 1;
        }
    }
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const raw = try self.rig.outputBytes();
        for (0..self.shape.rows) |r| {
            const expected = ref.row(self.input[r * self.shape.n ..][0..self.shape.n], self.picks[r], self.shape.k);
            var got: [ref.words]u32 = undefined;
            for (&got, 0..) |*x, i| x.* = std.mem.readInt(u32, raw[(r * ref.words + i) * 4 ..][0..4], .little);
            if (ref.check(&got, expected, spec(false), self.shape)) |why| return why;
        }
        return null;
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, want: []f64) !void {
        const mag = try self.arena.?.allocator().alloc(f64, want.len);
        try self.reference(want, mag);
        const terms = try self.arena.?.allocator().alloc(f64, self.shape.n);
        for (0..self.shape.rows) |r| {
            const row = self.input[r * self.shape.n ..][0..self.shape.n];
            var max: f64 = -std.math.inf(f64);
            for (row) |x| max = @max(max, x);
            for (terms, row) |*t, x| t.* = @exp(@as(f64, x) - max);
            want[r * ref.words] = max + @log(core.contract.rows.sumIn(acc, order, terms));
        }
    }
};
test {
    _ = ref;
}
