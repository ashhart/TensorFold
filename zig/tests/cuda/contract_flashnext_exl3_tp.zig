//! Flash Next EXL3 expert parallel: fn_exl3_tp_mask keeps the picks a rank holds and turns the others into -1.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

/// Slots a row (ten routed experts and the shared one) and the ids a table holds (512 routed, the shared one as 512).
const slots = 11;
const table_n = 513;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// The ids [lo, hi) of rank `rank` of two (cuda_exl3_tp.Rank).
fn range(rank: usize) [2]i32 {
    return if (rank == 0) .{ 0, 256 } else .{ 256, table_n };
}

fn spec(rank: usize) c.Spec {
    return .{
        .name = if (rank == 0) "cuda Flash Next EXL3 fn_exl3_tp_mask (rank 0)" else "cuda Flash Next EXL3 fn_exl3_tp_mask (rank 1)",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = "integer picks only: a pick in the rank's [lo, hi) keeps its value and every other one, negatives and the int32 extremes included, becomes -1" },
        // the masks run over rows * 11 picks in blocks of 256: 24 rows is the first count that crosses one
        .tiles = .{ .{ .size = 24, .step = 1, .min = 1 }, .{}, .{} },
        .served = &.{ .{ .rows = 1, .n = slots, .k = 0 }, .{ .rows = 8, .n = slots, .k = 0 }, .{ .rows = 64, .n = slots, .k = 0 }, .{ .rows = 4096, .n = slots, .k = 0 } },
    };
}

pub const Mask = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    rank: usize,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    picks: []i32 = &.{},
    din: u64 = 0,
    dout: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Mask) void {
        self.rig.gpu.gpa.free(self.picks);
        self.rig.deinit();
    }

    const vt = g.hooks(Mask);

    pub fn kernel(self: *Mask) g.Kernel {
        return .{ .spec = spec(self.rank), .ctx = self, .vt = &vt };
    }

    pub fn prepare(self: *Mask, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 8192) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.picks);
        self.picks = &.{};
        self.shape = shape;
        self.picks = try a.alloc(i32, shape.rows * slots);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d61736b);
        const rng = prng.random();
        for (self.picks) |*p| p.* = rng.intRangeLessThan(i32, 0, table_n);
        // the edges of both ranks' ranges, and values no router makes (the kernel masks them like any other id)
        const edge = [_]i32{ 0, 255, 256, 511, 512, 513, -1, -2, std.math.minInt(i32), std.math.maxInt(i32) };
        if (fill != .random) for (self.picks, 0..) |*p, i| {
            if (i % 3 == 0) p.* = edge[(i / 3) % edge.len];
        };
        self.din = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        self.dout = try self.rig.output(self.picks.len * 4);
        return true;
    }

    pub fn run(self: *Mask) !void {
        const r = range(self.rank);
        var args: cuda.Args = .{};
        args.add(self.din);
        args.add(self.dout);
        args.add(@as(c_int, @intCast(self.picks.len)));
        args.add(@as(c_int, r[0]));
        args.add(@as(c_int, r[1]));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((self.picks.len + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Mask) usize {
        return self.picks.len;
    }

    fn expected(self: *Mask, p: i32) i32 {
        const r = range(self.rank);
        return if (p >= r[0] and p < r[1]) p else -1;
    }

    pub fn reference(self: *Mask, want: []f64, mag: []f64) !void {
        for (self.picks, want, mag) |p, *w, *m| {
            w.* = @floatFromInt(self.expected(p));
            m.* = @abs(w.*);
        }
    }

    pub fn check(self: *Mask, _: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a payload guard changed";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.picks), try self.rig.inputBytes())) return "the picks changed";
        const out = try self.rig.outputBytes();
        for (self.picks, 0..) |p, i| {
            const got = std.mem.readInt(i32, out[i * 4 ..][0..4], .little);
            if (got != self.expected(p)) return try std.fmt.bufPrint(&self.msg, "pick {d} ({d}) became {d}, not {d}", .{ i, p, got, self.expected(p) });
        }
        return null;
    }
};

/// Both ranks' masks over one loaded image, for the runner's list.
pub const Masks = struct {
    mod: cuda.Module,
    each: [2]Mask,

    pub fn init(gpu: check.Gpu, s: cuda.Stream) !Masks {
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_exl3_tp);
        errdefer mod.unload();
        const f = try mod.function("fn_exl3_tp_mask");
        return .{ .mod = mod, .each = .{ .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .rank = 0 }, .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .rank = 1 } } };
    }

    pub fn deinit(self: *Masks) void {
        for (&self.each) |*m| m.deinit();
        self.mod.unload();
    }

    pub fn add(self: *Masks, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        for (&self.each) |*m| try list.append(a, m.kernel());
    }
};
