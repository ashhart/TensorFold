//! Times the HC read-out on the row kernels and the prompt tiles at the rows around each cut-over.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const glue_k = @import("contract_flashnext_glue.zig");

const rows = [_]usize{ 32, 63, 64, 65, 96, 126, 127, 128, 129, 130, 192, 256 };
const reps = 25;

fn median(gpu: check.Gpu, m: *glue_k.Glue, tiles: bool) !f64 {
    var t: [reps]i96 = undefined;
    _ = try m.timed(tiles, gpu.io); // the first launch loads the program
    for (&t) |*x| x.* = try m.timed(tiles, gpu.io);
    std.mem.sort(i96, &t, {}, std.sort.asc(i96));
    return @as(f64, @floatFromInt(t[reps / 2])) / std.time.ns_per_ms;
}

pub fn run(gpu: check.Gpu) !void {
    const kk = cuda.kernels;
    const ftk = flash.tile_kernels;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var batch = try cuda.Module.load(gpu.d, kk.flashnext_batch);
    defer batch.unload();
    var attn = try cuda.Module.load(gpu.d, kk.flashnext_attn);
    defer attn.unload();
    var hcf = try cuda.Module.load(gpu.d, kk.flashnext_hcf);
    defer hcf.unload();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    const f: glue_k.Fns = .{
        .tpr = try batch.function("fn_gdn_chain_tpr"),
        .rows_to = try batch.function("fn_gdn_rows_to"),
        .tile = try attn.function("_ZN8qsa_tile6scoresILi8ELi4EEEvPK13__nv_bfloat16S3_PKiPfiiii"),
        .scores = try attn.function("fn_qsa_scores"),
        .hcf_down = try hcf.function(ftk.cubin_hcf_down),
        .hcf_up = try hcf.function(ftk.cubin_hcf_up),
        .front = try batch.function("fn_gdn_front"),
        .norm = try batch.function("fn_gdn_norm"),
        .fused = try batch.function("fn_gdn_fused"),
    };
    try f.fused.allowDynamicShared(flash.batch_state.gdn_fused_smem);
    try f.tpr.allowDynamicShared(flash.batch_state.chain_tpr_smem);
    try f.hcf_down.allowDynamicShared(ftk.hcf_down_smem);
    try f.hcf_up.allowDynamicShared(ftk.hcf_up_smem);
    std.debug.print("rows row_ms tile_ms (median of {d}, alternated)\n", .{reps * 2});
    for (rows) |n| {
        var m: glue_k.Glue = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = .readout, .f = &f, .glue = &g };
        defer m.deinit();
        try check.expect(try m.prepare(.{ .rows = n, .n = 2560, .k = 10240 }, .random, 0xc07 + n), "read-out cut: the shape was refused", .{});
        const r1 = try median(gpu, &m, false);
        const t1 = try median(gpu, &m, true);
        const r2 = try median(gpu, &m, false);
        const t2 = try median(gpu, &m, true);
        std.debug.print("CUT {d} {d:.2} {d:.2}\n", .{ n, (r1 + r2) / 2, (t1 + t2) / 2 });
    }
}
