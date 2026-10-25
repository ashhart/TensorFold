//! Existing row launch ABIs recorded against the release runtime, without a driver or kernel image.
const std = @import("std");
const cuda = @import("cuda");
const t = std.testing;
const R = struct {
    var hash: std.crypto.hash.sha2.Sha256 = undefined;
    var count: usize = 0;
    fn sizes(id: usize) []const u8 {
        return switch (id) {
            1 => &.{ 8, 4, 8, 8, 4, 4, 4 },
            2 => &.{ 8, 4, 8, 8, 8, 8, 4, 4, 4 },
            3 => &.{ 8, 8, 8, 4, 4 },
            4 => &.{ 8, 8, 8, 8, 8, 4, 4 },
            5, 6 => &.{ 8, 8, 8, 8, 8, 8, 4, 4 },
            7, 8, 9 => &.{ 8, 8, 8, 8, 8, 8, 8, 4, 4 },
            10 => &.{ 8, 8, 8, 8, 8, 4, 4, 4 },
            11, 12 => &.{ 8, 8, 8, 8, 8, 8, 8, 8, 4, 4 },
            20, 24, 25 => &.{ 8, 8, 8, 8, 8, 8, 4 },
            21 => &.{ 8, 8, 8, 8, 8, 8, 4, 4 },
            22 => &.{ 8, 8, 8, 4 },
            23 => &.{ 8, 8, 8, 8, 8, 4, 4, 4, 4 },
            26 => &.{ 8, 8, 4, 4 },
            30, 31 => &.{ 72, 20, 8, 8, 8, 4, 8, 8, 72, 8, 8, 8, 4, 4, 4 },
            32, 33 => &.{ 72, 20, 8, 8, 8, 4, 8, 8, 72, 8, 8, 8, 4, 4, 4, 4 },
            40...43 => &.{ 8, 8, 8, 8, 8, 4, 4, 4 },
            else => unreachable,
        };
    }
    fn launch(kernel: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: cuda.abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(kernel);
        std.debug.assert(extra == null);
        const header = [_]u64{ id, gx, gy, gz, bx, by, bz, shared, @intFromPtr(s) };
        hash.update(std.mem.sliceAsBytes(&header));
        for (sizes(id), 0..) |n, i| {
            const p: [*]const u8 = @ptrCast(args.?[i].?);
            hash.update(p[0..n]); // QkW's final four padding bytes carry no kernel parameter.
        }
        count += 1;
        return 0;
    }
};
fn f(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn norms(d: *const cuda.Driver) cuda.norms.Norms {
    var n: cuda.norms.Norms = undefined;
    n.rms_fn = f(d, 1);
    n.add_rms_fn = f(d, 2);
    n.rms_bf16_fn = f(d, 3);
    n.add_rms_bf16_fn = f(d, 4);
    n.add_rms_h_fn = f(d, 5);
    n.add_rms_h_f32w_fn = f(d, 6);
    for (&n.gated_fns, 0..) |*v, i| v.* = f(d, 7 + i);
    n.rms_sums_fn = f(d, 10);
    n.tail_fns = .{ f(d, 11), f(d, 12) };
    return n;
}
fn acts(d: *const cuda.Driver) cuda.act_rows.ActRows {
    var a: cuda.act_rows.ActRows = undefined;
    a.swiglu_fn = f(d, 20);
    a.gate_fn = f(d, 21);
    a.sums_fn = f(d, 22);
    a.dconv_fn = f(d, 23);
    a.geglu_fn = f(d, 24);
    a.swiglu_round_fn = f(d, 25);
    a.softcap_fn = f(d, 26);
    return a;
}
test "shared row launches retain the release argument bytes and geometry" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = R.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x1800) };
    R.hash = .init(.{});
    R.count = 0;
    const n = norms(&d);
    const a = acts(&d);
    var q: cuda.qk_rows.QkRows = undefined;
    q.fn_ = f(&d, 30);
    q.warp_fn = f(&d, 31);
    q.once_fn = f(&d, 32);
    q.round_fn = f(&d, 33);
    var dense: cuda.dense_rows.DenseRows = undefined;
    for (&dense.fns, 0..) |*v, i| v.* = f(&d, 40 + i);
    for ([_]usize{ 1, 16, 17, 128 }) |rows| {
        try n.rms(s, 0x1000, .f32, null, 0x2000, .bf16, rows, 1024, 1e-5);
        try n.rms(s, 0x1000, .bf16, 0x3000, 0x2000, .bf16, rows, 2560, 1e-6);
        try n.addRms(s, 0x1000, .f16, 0x4000, null, 0x3000, 0x2000, .f32, rows, 1024, 1e-5);
        try n.addRms(s, 0x1000, .bf16, 0x4000, 0x5000, 0x3000, 0x2000, .bf16, rows, 2560, 1e-6);
        for ([_]cuda.norms.Kind{ .bf16, .f32 }) |kind|
            try n.addRmsH(s, 0x1000, 0x4000, 0x3000, kind, 0x6000, 0x2000, 0x7000, rows, 2560, 1e-6);
        try n.addRmsTail(s, 0x1000, null, 0x3000, 0x6000, 0x2000, null, rows, 2560, 1e-6, .{});
        for ([_]bool{ false, true }) |round|
            try n.addRmsTail(s, 0x1000, 0x4000, 0x3000, 0x6000, 0x2000, 0x7000, rows, 2560, 1e-6, .{ .round = round, .branch = 0x5000, .scalar = 0x8000 });
        try n.rmsSums(s, 0x1000, 2624, 0x3000, 0x2000, 0x7000, rows, 2560, 64, 1e-6);
        for ([_]usize{ 64, 128, 256 }) |dv|
            try n.gatedRms(s, 0x1000, 8192, 0x3000, 8256, 0x4000, 0x2000, null, rows, 32, dv, 1e-6);
        try a.swiglu(s, 0x1000, 2048, 0x3000, 2112, 0x2000, null, rows, 2048);
        try a.swigluRound(s, 0x1000, 2048, 0x3000, 2112, 0x2000, 0x4000, rows, 2048);
        try a.geglu(s, 0x1000, 2048, 0x3000, 2112, 0x2000, null, rows, 2048);
        try a.outputGate(s, 0x1000, 0x3000, 8192, 256, 0x2000, 0x4000, rows, 16, 128);
        try a.groupSums(s, 0x1000, 2112, 0x4000, rows, 2048);
        try a.softcap(s, 0x1000, 0x2000, rows, 4096, 30);
        try a.dconv(s, 0x1000, 0x3000, 0x4000, null, 0x2000, rows, 2048, 64, 1, 16);
        for ([_]usize{ 64, 128, 256, 512 }) |dim| {
            const hs: cuda.qk_rows.Heads = .{ .h = 16, .hk = 4, .dim = dim };
            const out: cuda.qk_rows.Out = .{ .q = 0x2000, .k = 0x3000, .v = 0x4000, .kc = 0x5000, .vc = 0x6000, .len = 4096 };
            try q.rows(s, 0x1000, hs, .{}, 0x7000, null, out, rows);
            try q.rows(s, 0x1000, hs, .{ .norms = 0x8000, .norms_bf16 = true, .inv = 0x9000, .rot = dim, .round = true }, 0x7000, 0xa000, out, rows);
            try q.rows(s, 0x1000, hs, .{ .norms = 0x8000, .norm_k = 0xb000, .cos = 0x9000, .sin = 0xc000, .rot = dim / 2, .v_norm = true }, 0x7000, 0xa000, out, rows);
        }
    }
    for ([_]usize{ 1, 2, 4, 5, 16, 17, 32, 33 }) |rows|
        try dense.rows(s, 0x1000, 272, 0x3000, 0x2000, 137, rows, 129, 256);
    const before = R.count;
    try dense.rows(s, 0x1000, 272, 0x3000, 0x2000, 137, 0, 129, 256);
    try t.expectEqual(before, R.count);
    var digest: [32]u8 = undefined;
    R.hash.final(&digest);
    try t.expectEqualStrings("f562c5d490b5d1aed8cb10a28a1162d9b30fed969adec3727cfbd8fedaf06a69", &std.fmt.bytesToHex(digest, .lower));
    try t.expectEqual(@as(usize, 136), R.count);
}
