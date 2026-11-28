//! Production prompt launch bookkeeping and rank agreement at the fake CUDA driver boundary.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;
const Rec = struct {
    var ids: [128]usize = undefined;
    var rows: [128]usize = undefined;
    var out: [128]u64 = undefined;
    var src: [128]u64 = undefined;
    var used: usize = 0;
    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        ids[used] = id;
        rows[used] = switch (id) {
            1 => @intCast(arg(c_int, args, 7)),
            3 => @intCast(arg(c_int, args, 2)),
            7 => @intCast(arg(i32, args, 5)),
            else => gx,
        };
        out[used] = switch (id) {
            1 => arg(u64, args, 4),
            2, 3 => arg(u64, args, 1),
            7 => arg(u64, args, 4),
            else => 0,
        };
        src[used] = arg(u64, args, 0);
        used += 1;
        return 0;
    }
    fn function(d: *cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = glm.pdl.forwardEx(launch);
        return d;
    }
};
fn glue(d: *cuda.Driver) glm.glue.Kernels {
    var k: glm.glue.Kernels = undefined;
    k.swiglu_512 = Rec.function(d, 2);
    k.combine = Rec.function(d, 4);
    return k;
}
fn layer() glm.moe.Layer {
    return .{ .router = 0, .bias = 0, .routed = .{ .format = .nvfp4, .up = 0, .down = 0, .width = 1024, .dims = 4096, .experts = 288, .limit = 10 }, .count = 288, .shared_gu = .{ .bf16 = 0x1000000 }, .shared_down = .{ .bf16 = 0x2000000 }, .shared = 1024, .top_k = 8, .scale = 1, .norm = true };
}
fn scratch() glm.moe.Scratch {
    var s: glm.moe.Scratch = undefined;
    s.sgu = 0x3000000;
    s.sact = 0x4000000;
    s.sxs = 0x5000000;
    s.sy = 0x6000000;
    s.mm_part = 0x7000000;
    s.ey = 0x8000000;
    s.wts = 0x9000000;
    return s;
}
test "GLM shared prefill: actual shared prepare then pieces preserve offsets and only put/combine per piece" {
    for ([_]usize{ 1, 17, 127, 128, 129, 2048, 2051 }) |total| for ([_]bool{ false, true }) |flag| {
        var d = Rec.driver();
        const stream: cuda.Stream = .{ .d = &d, .handle = null };
        var g = glue(&d);
        var mm: glm.mm.Kernels = undefined;
        mm.prompt = Rec.function(&d, 1);
        mm.prompt_fixed_k = false;
        mm.pdl = false;
        const k = glm.moe.Kernels{ .shared_once = flag, .set = .{ .glue = &g, .mm = &mm, .prompt = true }, .experts = undefined, .exl3 = undefined, .put_slot = Rec.function(&d, 3) };
        const l = layer();
        const sc = scratch();
        Rec.used = 0;
        try glm.moe.prepareShared(k, stream, l, sc, 0xa000000, 4096, total, true);
        if (flag) {
            try t.expectEqualSlices(usize, &.{ 1, 2, 1 }, Rec.ids[0..Rec.used]);
            try t.expectEqual(total, Rec.rows[0]);
        } else try t.expectEqual(@as(usize, 0), Rec.used);
        var cut: [glm.forward.max_pieces][2]usize = undefined;
        const pieces = glm.forward.ranges(total, 3, &cut);
        for (cut[0..pieces]) |piece| {
            const lo = piece[0];
            const n = piece[1] - lo;
            const begin = Rec.used;
            try glm.moe.fill(k, stream, l, sc, 0xa000000, 4096, lo, piece[1], true, 0xb000000 + lo * 4096 * 4);
            try t.expectEqualSlices(usize, if (flag) &.{ 3, 4 } else &.{ 1, 2, 1, 3, 4 }, Rec.ids[begin..Rec.used]);
            const put = Rec.used - 2;
            try t.expectEqual(n, Rec.rows[put]);
            try t.expectEqual(sc.sy + lo * 4096 * 4, Rec.src[put]);
            try t.expectEqual(sc.ey + lo * 9 * 4096 * 2, Rec.out[put]);
            if (!flag) try t.expectEqual(n, Rec.rows[begin]);
        }
        try t.expectEqual(if (flag) 3 + 2 * pieces else 5 * pieces, Rec.used);
    };
}
test "GLM shared prefill: legacy experimental formats decode and unset flag retain old eligibility" {
    var d = Rec.driver();
    var g = glue(&d);
    var mm: glm.mm.Kernels = undefined;
    mm.prompt_fixed_k = false;
    var k = glm.moe.Kernels{ .shared_once = true, .set = .{ .glue = &g, .mm = &mm, .prompt = true }, .experts = undefined, .exl3 = undefined, .put_slot = Rec.function(&d, 3) };
    var l = layer();
    try t.expect(glm.moe.sharedOnce(k, l, true));
    try t.expect(!glm.moe.sharedOnce(k, l, false));
    mm.prompt_fixed_k = true;
    try t.expect(!glm.moe.sharedOnce(k, l, true));
    mm.prompt_fixed_k = false;
    k.set.mm = null;
    try t.expect(!glm.moe.sharedOnce(k, l, true));
    k.set.mm = &mm;
    k.shared_once = false;
    try t.expect(!glm.moe.sharedOnce(k, l, true));
    k.shared_once = true;
    l.shared_gu = .{ .mxfp8 = undefined };
    try t.expect(!glm.moe.sharedOnce(k, l, true));
}
test "GLM shared prefill: dense clamp uses one pair launch while default and another clamp keep the five calls" {
    for ([_]usize{ 1, 17, 129, 2048 }) |rows| for ([_]bool{ false, true }) |flag| for ([_]f32{ 10, 7 }) |limit| {
        var d = Rec.driver();
        var g = glue(&d);
        var p: cuda.nvfp4.Prompt = undefined;
        p.lean = Rec.function(&d, 7);
        p.lean_glm_clamp = Rec.function(&d, 7);
        p.sms = 48;
        const gate = cuda.nvfp4.weight(0x100000, 0x200000, 1, 6144, 4096);
        const up = cuda.nvfp4.weight(0x300000, 0x400000, 1, 6144, 4096);
        const k = glm.mlp.Kernels{ .shared_once = flag, .set = .{ .glue = &g }, .linear = .{ .nvfp4_prompt = &p }, .copy_rows = Rec.function(&d, 6) };
        Rec.used = 0;
        try glm.mlp.body(k, .{ .d = &d, .handle = null }, .{ .w = .{ .nvfp4 = .{ .gate = gate, .up = up, .down = gate } }, .width = 6144, .dims = 4096, .limit = limit }, .{ .tmp = 0x500000, .gu = 0x600000, .act = 0x700000, .xs_act = 0x800000, .mm_part = 0 }, 0x900000, 4096, rows, true);
        try t.expectEqualSlices(usize, if (flag and limit == 10) &.{7} else &.{ 7, 6, 7, 6, 2 }, Rec.ids[0..Rec.used]);
        try t.expectEqual(rows, Rec.rows[0]);
        if (flag and limit == 10) try t.expectEqual(@as(u64, 0x700000), Rec.out[0]);
    };
}
test "GLM shared prefill: strict settings default off and disagreement stops the ranks" {
    const F = struct {
        var raw: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_PREFILL_SHARED_ONCE")) raw else null;
        }
    };
    var why: []const u8 = "";
    for ([_]?[]const u8{ null, "", " ", "0", "1" }) |raw| {
        F.raw = raw;
        const s = try glm.ranks.Settings.read(F.get, &why);
        try t.expectEqual(raw != null and std.mem.eql(u8, raw.?, "1"), s.prefill_shared_once);
    }
    for ([_][]const u8{ "yes", "true", "2", " 1" }) |raw| {
        F.raw = raw;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(F.get, &why));
    }
    const hash: [32]u8 = @splat(0);
    var all = [_]glm.ranks.Fingerprint{ glm.ranks.Fingerprint.of(2, 0, .{}, 8192, true, hash, hash, hash), glm.ranks.Fingerprint.of(2, 1, .{ .prefill_shared_once = true }, 8192, true, hash, hash, hash) };
    try t.expectEqual(glm.ranks.Fingerprint.Field.prefill_shared_once, glm.ranks.mismatch(&all).?.field);
}
