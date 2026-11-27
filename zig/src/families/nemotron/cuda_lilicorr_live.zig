//! The live LiLiCorr drafter audit (TF_LILICORR_DRAFTER=<dir>): the real checkpoint through the body/head loaders.
const std = @import("std");
const core = @import("core");
const head = @import("cuda_lilicorr_head.zig");
const body = @import("cuda_lilicorr_body.zig");
const cuda = @import("cuda");

test "the real drafter checkpoint loads and stays sane" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const gpa = arena.allocator();
    const dir = std.c.getenv("TF_LILICORR_DRAFTER") orelse return error.SkipZigTest;
    const d = std.mem.span(dir);
    if (d.len == 0) return error.SkipZigTest;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, d);
    defer core.checkpoint.cache.close(&ck);
    const g: body.Geo = .{ .layers = 5, .hidden = 4096, .intermediate = 12288, .q_width = 4096, .kv_width = 256 };
    var w = try body.load(gpa, &ck, g);
    try std.testing.expect(body.sane(&w));
    // the head loads from the same checkpoint through the fixed widen (its slot embedding

    // must read sub-4.0 magnitudes; the f16 corruption class reads them into the thousands)
    const hw = try head.load(gpa, &ck);
    for (hw.slot_emb) |x| try std.testing.expect(@abs(x) < 4.0);
    // the conv projections are the dynamic-conv width the frame's launches expect
    for (w.layers) |l| {
        try std.testing.expectEqual(@as(usize, body.conv_proj_width * 4096), l.attn_proj.len);
        try std.testing.expectEqual(@as(usize, body.conv_base_rows * body.conv_base_rows * 4096), l.attn_base.len);
    }
}

test "the head's device staging round-trips the real checkpoint" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const gpa = arena.allocator();
    const dir = std.c.getenv("TF_LILICORR_DRAFTER") orelse return error.SkipZigTest;
    const d = std.mem.span(dir);
    if (d.len == 0) return error.SkipZigTest;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, d);
    defer core.checkpoint.cache.close(&ck);
    const hw = try head.load(gpa, &ck);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var dw = try head.loadDevice(gpa, &ctx, &hw);
    defer dw.deinit(gpa);
}
