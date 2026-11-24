//! TF_QMMF_HALF_BN=1: 32-column half tiles for FP4, FP8 and FP8G too, below two blocks an SM by default (opt-in).
const std = @import("std");
const q = @import("qmmf.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Function = @import("module.zig").Function;

/// The formats this adds; MXFP8 and BF16 keep their own half tiles (Lane.half).
pub const modes = [_]q.Mode{ .fp4, .fp8, .fp8g };

/// Per added format, Out, then plain / cluster.
pub const Set = [modes.len][2][2]Function;

pub fn index(mode: q.Mode) ?usize {
    return std.mem.indexOfScalar(q.Mode, &modes, mode);
}

/// The switch; default off, which leaves every lane's tiles and threshold as before.
pub fn on() !bool {
    return @import("sys").environment.read(bool, "TF_QMMF_HALF_BN", false);
}

/// The block count below which a 16-row call halves its tiles: TF_QMMF_HALF_BELOW when set, else two blocks an SM.
pub fn below(sms: usize) !usize {
    return @import("sys").environment.read(usize, "TF_QMMF_HALF_BELOW", 2 * sms);
}

/// The added instances from the lane's module.
pub fn load(l: *const q.Lane) !Set {
    var set: Set = undefined;
    inline for (modes, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ false, true }, 0..) |cl, ci| {
        const f = try l.mod.function(q.halfSymbol(mode, cl, wide));
        try f.allowDynamicShared(q.halfShared(mode));
        set[mi][oi][ci] = f;
    };
    return set;
}

/// With the switch on: the added instances, and the threshold from the current device's SM count.
pub fn apply(l: *q.Lane, d: *const Driver) !void {
    if (!try on()) return;
    const set = try load(l);
    var dev: abi.Device = 0;
    try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
    var sms: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, dev), "cuDeviceGetAttribute");
    l.half_n = set;
    l.half_below = try below(@intCast(@max(sms, 1)));
}

test "half tiles for FP4, FP8 and FP8G: qmmf.cu's names and shared bytes, and the 16-row calls that take them" {
    const t = std.testing;
    try t.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi0ELi16ELi32ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", q.halfSymbol(.fp4, false, false));
    try t.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi3ELi16ELi32ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", q.halfSymbol(.fp8g, true, true));
    // Tile<MODE, 16, 32, 1, 4, 4>::SMEM: four stages of x (2048), half the weights and half the scales
    try t.expectEqual(@as(u32, 12800), q.halfShared(.fp4)); // 2048 + 1024 + 128
    try t.expectEqual(@as(u32, 16384), q.halfShared(.fp8)); // 2048 + 2048
    try t.expectEqual(@as(u32, 16896), q.halfShared(.fp8g)); // 2048 + 2048 + 128
    try t.expectEqual(@as(u32, 16896), q.halfShared(.mxfp8)); // unchanged: 2048 + 2048 + 64
    try t.expectEqual(@as(u32, 24576), q.halfShared(.bf16)); // unchanged
    try t.expectEqual(@as(?usize, 0), index(.fp4));
    try t.expectEqual(@as(?usize, 2), index(.fp8g));
    try t.expectEqual(@as(?usize, null), index(.mxfp8));
    try t.expectEqual(@as(?usize, null), index(.bf16));

    const Mock = struct {
        var id: usize = 0;
        var grid: c_uint = 0;
        var smem: u32 = 0;
        fn run(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
            id = @intFromPtr(f.?);
            grid = gx;
            smem = sm;
            return 0;
        }
    };
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Mock.run;
    var l: q.Lane = undefined;
    l.clusters = true;
    l.pdl = false;
    l.fuse_split_n = 0;
    l.prompt = null;
    l.half_n = null;
    l.half_below = 340; // two blocks an SM on 170 SMs
    for (&l.fns, 0..) |*set, mi| for (set, 0..) |*f, i| {
        f.* = .{ .d = &d, .handle = @ptrFromInt(1000 + 10 * mi + i) };
    };
    var added: Set = undefined;
    for (&added, 0..) |*a, mi| for (a, 0..) |*o, oi| for (o, 0..) |*f, ci| {
        f.* = .{ .d = &d, .handle = @ptrFromInt(2000 + 100 * mi + 10 * oi + ci) };
    };
    const s: @import("stream.zig").Stream = .{ .d = &d, .handle = null };
    // one K slice (8 groups), 16 column tiles: the plain launch, so cuLaunchKernel
    for ([_]q.Mode{ .fp4, .fp8, .fp8g }, 0..) |mode, mi| for ([_]usize{ 1, 16 }) |rows| {
        const w: q.Weight = .{ .mode = mode, .codes = 1, .scales = 2, .n = 1024, .k = 512, .npad = 1024 };
        l.half_n = null; // off: the 64-column 16-row tile, as before
        try l.matmul(s, 3, 512, rows, w, 4, 0);
        try t.expectEqual(1000 + 10 * @as(usize, @backingInt(mode)), Mock.id);
        try t.expectEqual(@as(c_uint, 16), Mock.grid);
        try t.expectEqual(q.sharedBytes(mode, 16), Mock.smem);
        l.half_n = added; // on: the half tile, twice the blocks
        try l.matmul(s, 3, 512, rows, w, 4, 0);
        try t.expectEqual(2000 + 100 * mi, Mock.id);
        try t.expectEqual(@as(c_uint, 32), Mock.grid);
        try t.expectEqual(q.halfShared(mode), Mock.smem);
        // 17 rows take the 32-row tile either way; a grid at the threshold stays on 64 columns
        try l.matmul(s, 3, 512, 17, w, 4, 0);
        try t.expectEqual(1000 + 10 * @as(usize, @backingInt(mode)) + 2, Mock.id);
        var wide = w;
        wide.n = 340 * 64;
        wide.npad = 340 * 64;
        try l.matmul(s, 3, 512, rows, wide, 4, 0);
        try t.expectEqual(1000 + 10 * @as(usize, @backingInt(mode)), Mock.id);
    };
}
