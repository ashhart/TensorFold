//! Real row-selection paths with fake CUDA reductions and downloads; no device is opened.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const ops = @import("cuda_mtp_ops.zig");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const t = std.testing;
const gpa = t.allocator;

const Device = struct {
    var best: [4]f32 = undefined;
    var launches: usize = 0;
    fn launch(_: cuda.abi.Function, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        launches += 1;
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn read(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) c_int {
        const bytes = @as([*]u8, @ptrCast(dst.?))[0..n];
        if (src == 0x1000) {
            std.debug.assert(n == 16);
            @memcpy(bytes, std.mem.asBytes(&best));
        } else @memset(bytes, 0);
        return 0;
    }
    fn context(out: *c_int) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) c_int {
        out.* = 0;
        return 0;
    }
    fn driver() cuda.Driver {
        cuda.download_stage.forget();
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuLaunchKernel = launch;
        d.api.cuStreamSynchronize = wait;
        d.api.cuMemcpyDtoH_v2 = read;
        d.api.cuCtxGetDevice = context;
        d.api.cuDeviceGetAttribute = attribute;
        launches = 0;
        return d;
    }
};

fn reduction(value: f32, lse: f32, col: i32) [4]f32 {
    return .{ value, lse, @bitCast(col), 0 };
}
fn draft(d: *cuda.Driver, host: []u16) mtp.Draft {
    var x: mtp.Draft = undefined;
    x.label = "fixture-stream";
    x.len = 2051;
    x.cap = 4096;
    x.pg = .{};
    x.pooled_lane = false;
    x.host = host;
    x.rows = 1;
    x.fed0 = 2050;
    x.best = .{ .d = d, .ptr = 0x1000, .len = 16 };
    x.out = .{ .d = d, .ptr = 0x2000, .len = wt.wide * 2 };
    x.dlogits = .{ .d = d, .ptr = 0x3000, .len = 6 };
    x.logits = .{ .d = d, .ptr = 0x4000, .len = wt.vocab * 2 };
    return x;
}

fn head(d: *cuda.Driver) !mtp.Head {
    var h: mtp.Head = undefined;
    h.artifact = try core.proposal_vocab.legacy.artifact(gpa, &.{ 0, 3, 7 }, .{ .vocab = wt.vocab, .hidden = wt.dims, .format = "bf16", .group = 0, .scale_recipe = "fixture" });
    h.ids = h.artifact.ids();
    h.service = try @import("lanes").proposal_head.Service.legacy(gpa, &h.artifact);
    h.best_fn = .{ .d = d, .handle = @ptrFromInt(1) };
    h.topk_fn = .{ .d = d, .handle = @ptrFromInt(2) };
    return h;
}

test "Flash Next nonfinite: single and batched head selections reject sentinels and preserve finite ID mapping" {
    var d = Device.driver();
    defer cuda.download_stage.closeDriver(&d);
    var h = try head(&d);
    defer h.artifact.deinit();
    var host: [3]u16 = undefined;
    var x = draft(&d, &host);
    const st: cuda.Stream = .{ .d = &d, .handle = null };
    var e: prompt.Engine = undefined;
    e.stream = @constCast(&st);
    e.gpa = gpa;
    const o: ops.Ops(struct {}) = .{ .e = &e, .h = &h, .tri = .{} };
    const invalid = [_][4]f32{
        reduction(-std.math.inf(f32), std.math.nan(f32), std.math.maxInt(i32)),
        reduction(std.math.nan(f32), 2, 1),
        reduction(std.math.inf(f32), std.math.nan(f32), 1),
        reduction(1, std.math.nan(f32), 1),
        reduction(1, 2, -1),
        reduction(1, 2, 3),
    };
    for (invalid) |bad| {
        Device.best = bad;
        try t.expectError(error.ProposalNonfinite, o.pick(&x, 12, null));
        for ([_]bool{ false, true }) |batched| {
            try t.expectError(error.ProposalNonfinite, draw.pick(&h, st, gpa, &x, bad, 12, null, batched));
            try t.expectError(error.ProposalNonfinite, draw.pick(&h, st, gpa, &x, bad, 12, .{ .seed = 7, .temperature = 1, .top_k = 2 }, batched));
        }
    }
    for (h.ids, 0..) |id, col| {
        Device.best = reduction(2, 3, @intCast(col));
        const solo = try o.pick(&x, 12, null);
        const batched = try draw.pick(&h, st, gpa, &x, Device.best, 12, null, true);
        try t.expectEqual(id, solo.token);
        try t.expectEqualDeep(solo, batched);
        try t.expectEqual(@exp(@as(f64, -1)), solo.prob);
    }
}

test "Flash Next nonfinite: main-model greedy sampled and masked rows fail before using invalid columns" {
    var d = Device.driver();
    defer cuda.download_stage.closeDriver(&d);
    var h = try head(&d);
    defer h.artifact.deinit();
    const host = try gpa.alloc(u16, wt.vocab);
    defer gpa.free(host);
    var x = draft(&d, host);
    const st: cuda.Stream = .{ .d = &d, .handle = null };
    const sampled: mtp.Sampling = .{ .seed = 7, .temperature = 1, .top_k = 2 };
    for ([_]?mtp.Sampling{ null, sampled }) |sm| for ([_][]const u32{ &.{}, &.{1} }) |masks| {
        Device.best = reduction(-std.math.inf(f32), std.math.nan(f32), std.math.maxInt(i32));
        var out = [_]u32{42};
        try t.expectError(error.ProposalNonfinite, draw.draws(&h, st, gpa, &x, sm, masks, &out));
        try t.expectEqual(@as(u32, 42), out[0]);
    };
    Device.best = reduction(2, 3, 17);
    var out: [1]u32 = undefined;
    try draw.draws(&h, st, gpa, &x, null, &.{}, &out);
    try t.expectEqual(@as(u32, 17), out[0]);
}

test "Flash Next nonfinite: top-k rejects NaNs infinities and invalid columns before mapping or sampling" {
    const s: mtp.Sampling = .{ .seed = 7, .temperature = 1, .top_k = 2 };
    for ([_]u16{ 0x7fc1, 0xffc1, 0x7f80, 0xff80 }) |bad| {
        try t.expectError(error.ProposalNonfinite, draw.drawTopk(gpa, &.{ 0x3f80, bad }, &.{ 0, 1 }, &.{ 0, 3 }, 12, s));
    }
    for ([_]i32{ -1, 2, std.math.maxInt(i32) }) |col| {
        try t.expectError(error.ProposalNonfinite, draw.drawTopk(gpa, &.{ 0x3f80, 0 }, &.{ 0, col }, &.{ 0, 3 }, 12, s));
    }
    try t.expectError(error.ProposalNonfinite, draw.drawTopk(gpa, &.{0x3f80}, &.{wt.vocab}, null, 12, s));
    const row = [_]u16{ 0x3f80, 0, 0xbf80 };
    const ids = [_]u32{ 0, 3, 7 };
    for (0..20) |position| {
        try t.expectEqual(try mtp.drawRow(gpa, &row, &ids, position, s), try draw.drawTopk(gpa, row[0..2], &.{ 0, 1 }, &ids, position, s));
    }
}
