//! Decode fusion admission and actual launch arguments without a GPU.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const hc = @import("cuda_batch_hc.zig");
const wt = @import("cuda_prompt_wt.zig");
const Forms = @import("cuda_batch_kernels.zig").Forms;

const Tri = struct {
    owned: bool = true,
    pub fn writebackOwned(self: Tri) bool {
        return self.owned;
    }
};

test "decode fusion: HC opt-in retains ownership native keep master and prompt gates" {
    const Pass = struct { tri: Tri = .{}, k: struct { forms: Forms = .{} } = .{}, hooks: struct { prompt: bool = false } = .{} };
    var p: Pass = .{};
    var w: wt.HcW = undefined;
    w.down.native = cuda.modelopt.bf16Weight(1, 324, 10240);
    try t.expect(!p.k.forms.hc_decode_fuse and !p.k.forms.shared_down8);
    for (1..64) |n| {
        try t.expect(!hc.fuses(&p, w, n, .{}));
        p.k.forms.hc_decode_fuse = true;
        try t.expect(hc.fuses(&p, w, n, .{}));
        try t.expect(!hc.fuses(&p, w, n, .{ .h = 1 }));
        try t.expect(!hc.fuses(&p, w, n, .{ .normed = 1 }));
        try t.expect(!hc.fuses(&p, w, n, .{ .v = 1 }));
        try t.expect(!hc.fuses(&p, w, n, .{ .up = 1 }));
        try t.expect(!hc.fuses(&p, w, n, .{ .inject = 1 }));
        p.k.forms.hc_fuse = false;
        try t.expect(!hc.fuses(&p, w, n, .{}));
        p.k.forms.hc_fuse = true;
        p.tri.owned = false;
        try t.expect(!hc.fuses(&p, w, n, .{}));
        p.tri.owned = true;
        w.down.native = null;
        try t.expect(!hc.fuses(&p, w, n, .{}));
        w.down.native = cuda.modelopt.bf16Weight(1, 324, 10240);
        p.k.forms.hc_decode_fuse = false;
    }
    try t.expect(hc.fuses(&p, w, 64, .{}));
    p.k.forms.hc_decode_fuse = true;
    try t.expect(!hc.fuses(&p, w, 0, .{}));
}

const Mock = struct {
    var launches: usize = 0;
    var id: usize = 0;
    var args_seen: [12]u64 = undefined;
    var shared: u32 = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, sm: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        launches += 1;
        id = @intFromPtr(f);
        shared = sm;
        for (0..5) |i| args_seen[i] = if (i == 3) @as(*const u32, @ptrCast(@alignCast(args.?[i].?))).* else @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
        for (5..12) |i| args_seen[i] = @intCast(@as(*const i32, @ptrCast(@alignCast(args.?[i].?))).*);
        std.debug.assert(gx == (2560 / (if (id == 1) @as(u32, 32) else 64)) and bx == 128);
        return 0;
    }
};
const Glue = struct {
    var scatters: usize = 0;
    pub fn sharedScatter(_: *Glue, _: cuda.Stream, _: u64, _: u64, _: usize, _: usize, _: usize) !void {
        scatters += 1;
    }
};
const K = struct {
    forms: Forms = .{},
    native_lane: cuda.qmmf.Lane,
    var contiguous: usize = 0;
    pub fn nativeMatmul(_: *K, _: cuda.Stream, _: u64, _: usize, _: usize, _: cuda.qmmf.Weight, _: u64, _: u64, _: cuda.qmmf.Out) !void {
        contiguous += 1;
    }
};

test "decode fusion: MXFP8 real launcher preserves addresses strides scales tiles and fallback" {
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Mock.launch;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var lane: cuda.qmmf.Lane = undefined;
    lane.clusters = true;
    lane.half_below = 96;
    lane.shared_down8 = .{ .{ .d = &driver, .handle = @ptrFromInt(1) }, .{ .d = &driver, .handle = @ptrFromInt(2) } };
    var k: K = .{ .native_lane = lane };
    var glue: Glue = .{};
    var scratch = .{ .y = @as(u64, 0x100000), .shared_out = @as(u64, 0x200000), .native_part = @as(u64, 0x300000) };
    var p = .{ .k = &k, .s = &scratch, .tri = .{ .own = @as(?*Glue, &glue) } };
    var native: struct { shared: [3]struct { native: ?cuda.qmmf.Weight } } = undefined;
    native.shared[2].native = cuda.mxfp8.weight(11, 22, 2560, 640);
    native.shared[2].native.?.scale = 0.75;
    for (1..17) |n| for ([_]bool{ false, true }) |half| {
        k.native_lane.half_below = if (half) 96 else 0;
        k.forms.shared_down8 = true;
        Mock.launches = 0;
        K.contiguous = 0;
        Glue.scatters = 0;
        try @import("cuda_batch_side.zig").down(&p, stream, 0x400000, &native, n);
        try t.expectEqual(@as(usize, 1), Mock.launches);
        try t.expectEqual(@as(usize, if (half) 1 else 2), Mock.id);
        try t.expectEqual([12]u64{ 0x400000, 11, 22, @as(u32, @bitCast(@as(f32, 0.75))), scratch.y + 10 * 2560 * 4, n, 2560, 640, 2560, 640, 1, 11 * 2560 }, Mock.args_seen);
        try t.expectEqual(if (half) cuda.qmmf.halfShared(.mxfp8) else cuda.qmmf.sharedBytes(.mxfp8, 16), Mock.shared);
        try t.expectEqual(@as(usize, 0), Glue.scatters + K.contiguous);
        k.forms.shared_down8 = false;
        try @import("cuda_batch_side.zig").down(&p, stream, 0x400000, &native, n);
        try t.expectEqual(@as(usize, 1), Glue.scatters);
        try t.expectEqual(@as(usize, 1), K.contiguous);
    };
    k.forms.shared_down8 = true;
    const original = native.shared[2].native.?;
    for (0..5) |which| {
        native.shared[2].native = original;
        var rows: usize = 16;
        switch (which) {
            0 => native.shared[2].native.?.sk = 2,
            1 => native.shared[2].native.?.npad = 2561,
            2 => rows = 17,
            3 => native.shared[2].native.?.mode = .fp8,
            else => k.native_lane.shared_down8 = null,
        }
        Mock.launches = 0;
        K.contiguous = 0;
        Glue.scatters = 0;
        try @import("cuda_batch_side.zig").down(&p, stream, 0x400000, &native, rows);
        try t.expectEqual(@as(usize, 0), Mock.launches);
        try t.expectEqual(@as(usize, 1), Glue.scatters);
        try t.expectEqual(@as(usize, 1), K.contiguous);
    }
}
