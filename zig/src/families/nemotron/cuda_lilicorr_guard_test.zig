//! The absorb-guard battery: the chunk clamp keeps every gathered row at staging_rows (128).

// ---------------------------------------------------------------------------

// absorb-guard (PR 5102 owner round-2 ask 5, part B): absorbRows' chunk clamp

// `const rows = @min(n, staging_rows);` (cure ecc1f7f3) keeps EVERY gathered

// row <= staging_rows(128) and the chunk sizes summing to n, so a 129-row (or

// 256-row) prompt ingest never overruns the 128-row staging carve (taps_in/th/

// thn/kabs/vabs/kn/xs). A recording driver fake drives all three launch seams

// with zero real GPU calls: the family `run` helper (cuda.launch), the affine

// gemv path (d.affine via projSums, e.ops().dense via projDense), and norm's

// e.k.norms.addRmsTail; the anchor DtoH copy is faked too. RED side: at the

// pre-cure clamp `@min(n, max_rows)` (reverted for the fail-before capture in

// a throwaway worktree) the n=129 case records a single 129-row gather launch

// (the guard's grid-cap assert fires on it); the pre-cure absorb also dies at a

// downstream seam there (the mock's wide-verify face tops out at 128 rows), so

// the harness records that error instead of `try`-ing it and asserts on the

// launches FIRST — at HEAD the error arm never fires (asserted null).
const guard_lc = @import("cuda_lilicorr.zig");
const guard_wts = @import("cuda_lilicorr_weights.zig");
const guard_absorb = @import("cuda_lilicorr_absorb.zig");
const guard_cuda = @import("cuda");
const std = @import("std");

const AbsorbGuard = struct {
    const lc = guard_lc;
    const cuda = guard_cuda;
    const abi = cuda.abi;

    var f_handle: [96]abi.Function = undefined;
    var f_grid: [96]usize = undefined;
    var f_block: [96]usize = undefined;
    var f_arg0: [96]u64 = undefined; // param 0's value: a norm-tail's input row pointer
    var f_rows_n: [96]i32 = undefined; // the gather_rows by-value TapRows' n, as the kernel reads it
    var f_rows: [96][64]u32 = undefined; // ...and its first 64 row entries (the kernel's int rows[64])
    var count: usize = 0;
    var ex_count: usize = 0;

    // handle values the harness fixture pins: gather_rows = 0x1101, norm tails = 0x9100/0x9108
    const gather_rows_handle = 0x1101;
    const tail_handles = [_]usize{ 0x9100, 0x9108 };

    fn fake(dv: *const cuda.Driver, h: usize) cuda.Function {
        return .{ .d = dv, .handle = @ptrFromInt(h) };
    }

    const Fake = struct {
        fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
            _ = gy;
            _ = gz;
            _ = by;
            _ = bz;
            _ = shared;
            _ = stream;
            _ = extra;
            if (count < f_handle.len) {
                f_handle[count] = f;
                f_grid[count] = gx;
                f_block[count] = bx;
                const hv = @intFromPtr(f.?);
                if (args) |ap| {
                    const ps: [*]?*anyopaque = @ptrCast(ap);
                    if (blk: {
                        for (tail_handles) |th| if (hv == th) break :blk true;
                        break :blk false;
                    }) {
                        f_arg0[count] = @as(*u64, @ptrCast(@alignCast(ps[0].?))).*;
                    }
                    if (hv == gather_rows_handle) {
                        const b: [*]u8 = @ptrCast(@alignCast(ps[3].?));
                        f_rows_n[count] = std.mem.readInt(i32, b[0..4], .little);
                        for (0..64) |j| f_rows[count][j] = std.mem.readInt(u32, b[4 + j * 4 ..][0..4], .little);
                    }
                }
                count += 1;
            }
            return abi.success;
        }
        fn launchEx(cfg: *const abi.LaunchConfig, f: abi.Function, args: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
            _ = args;
            _ = extra;
            if (count < f_handle.len) {
                f_handle[count] = f;
                f_grid[count] = cfg.grid_x;
                f_block[count] = cfg.block_x;
                ex_count += 1; // the wide/pdl affine faces launch through here
                count += 1;
            }
            return abi.success;
        }
        fn memcpyH(_: ?*anyopaque, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) abi.Result {
            return abi.success;
        }
    };

    fn reset() void {
        count = 0;
        ex_count = 0;
    }
};

test "the absorb chunk cap keeps every gathered row at staging_rows (owner ask-5 B, absorb-guard)" {
    const t = std.testing;
    const g = AbsorbGuard;
    const lc = g.lc;
    const cuda = g.cuda;

    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = g.Fake.launch;
    driver.api.cuLaunchKernelEx = g.Fake.launchEx;
    driver.api.cuMemcpyDtoHAsync_v2 = g.Fake.memcpyH;

    var e: lc.E = undefined;
    var ctx_dev: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    e.ctx = &ctx_dev;
    e.stream = .{ .d = &driver, .handle = null };
    e.b = undefined;
    e.k = undefined;
    e.k.wide = true;
    e.k.modelopt = null;
    e.k.ck = null;
    e.k.ggml = null;
    const fade = g.fake(&driver, 0x9000);
    e.k.affine = .{
        .group_fn = fade,
        .mlp128_fn = fade,
        .gemv_fn = fade,
        .verify_fn = fade,
        .verify64_fn = fade,
        .verify128_fn = fade,
        .split_fn = fade,
        .seg_fn = fade,
        .many_fn = fade,
        .prefill_fn = fade,
        .prefill22_fn = fade,
        .prefill_many_fn = fade,
        .words_fn = fade,
        .scales_fn = fade,
        .gemv_blocks = 1,
        .split = null,
        .pdl = true, // the production GB10 face: 128-row wide verify tiles, Ex launches
        .wide_verify = false,
    };
    e.k.norms = undefined;
    e.k.norms.tail_fns = .{ g.fake(&driver, 0x9100), g.fake(&driver, 0x9108) };

    var d: lc.Draft = undefined;
    d.e = &e;
    d.c = .{
        .hidden = 128,
        .intermediate = 128,
        .layers = 1,
        .vocab = 16,
        .heads = 2,
        .kv_heads = 1,
        .head_dim = 32,
        .block = 4,
        .mask = 0,
        .window = 4096, // keep = min(window, tap_rows - max_rows) = min(4096, 512-256) = 256: no trim at n=256
        .group = 8,
        .rank = 4,
        .topk = 2,
        .eps = 1e-5,
        .theta = 10000,
        .taps = .{ 1, 2, 3, 4, 5, 6, 7, 8 },
        .n_taps = 4,
    };
    d.gpa = t.allocator;
    d.tap_rows = 512;
    d.taps_in = 0x1a0000;
    d.th = 0x1b0000;
    d.thn = 0x1c0000;
    d.kabs = 0x1d0000;
    d.vabs = 0x1e0000;
    d.kn = 0x1f0000;
    d.xs = 0x200000;
    d.anchor = 0x210000;
    var staged_b: [8192]u8 align(16) = @splat(0); // anchor copy lands at anchor_at(2080)+256 bytes, faked
    d.staged = .{ .d = &driver, .bytes = staged_b[0..] };
    d.w = undefined;
    // fc/k shape (n=64, k=1024): splitK=2 with a.split=null, so decode takes the gemv face and

    // the 128-row chunk takes the wide verify face (pdl=true: cfg.plain() false => cuLaunchKernelEx).
    d.w.fc = .{ .w = 0x30000, .s = 0x31000, .b = 0x32000, .n = 64, .k = 1024, .npad = 128 };
    d.w.hidden_norm = 0x220000;
    const wq: cuda.qlinear.Weight = .{ .affine4 = .{ .w = 0x33000, .s = 0x34000, .b = 0x35000, .n = 64, .k = 1024, .npad = 128 } };
    var lz: [1]guard_wts.Layer = .{.{
        .input_norm = 0x40000,
        .post_norm = 0x40040,
        .q = wq,
        .k = wq,
        .v = wq,
        .o = wq,
        .q_norm = 0x40080,
        .k_norm = 0x400c0,
        .sink = 0x40100,
        .gate = wq,
        .up = wq,
        .down = wq,
        .attn_conv = wq,
        .attn_base = 0x40140,
        .mlp_conv = wq,
        .mlp_base = 0x40180,
    }};
    d.w.layers = lz[0..];
    d.fns = .{
        .gather_taps = g.fake(&driver, 0x1100),
        .gather_rows = g.fake(&driver, 0x1101),
        .norm_rope = g.fake(&driver, 0x1102),
        .ring_put = g.fake(&driver, 0x1103),
        .sums = g.fake(&driver, 0x1104),
        .norm_rope_batch = g.fake(&driver, 0x1105),
        .attn_partial = g.fake(&driver, 0x1106),
        .attn_merge = g.fake(&driver, 0x1107),
        .mlp_act = g.fake(&driver, 0x1108),
        .mask_rows = g.fake(&driver, 0x1109),
        .dyn_conv_batch = g.fake(&driver, 0x110a),
        .token_proj = g.fake(&driver, 0x110b),
        .lat_attn = g.fake(&driver, 0x110c),
        .factor = g.fake(&driver, 0x110d),
        .edge_linear = g.fake(&driver, 0x110e),
        .l2norm = g.fake(&driver, 0x110f),
        .dots = g.fake(&driver, 0x1110),
        .rms = g.fake(&driver, 0x1111),
        .lin_f32 = g.fake(&driver, 0x1112),
        .silu_bf16 = g.fake(&driver, 0x1113),
        .add_bf16 = g.fake(&driver, 0x1114),
        .lattice = g.fake(&driver, 0x1115),
        .rms_f32 = g.fake(&driver, 0x1116),
        .widen = g.fake(&driver, 0x1117),
        .narrow = g.fake(&driver, 0x1118),
    };
    d.affine = e.k.affine;

    const gather_handle = d.fns.gather_taps.handle;
    const tail0 = e.k.norms.tail_fns[0];
    const tail1 = e.k.norms.tail_fns[1];
    const ring_buf: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 0x5a000000, .len = 1 << 24 };

    for ([_]usize{ 129, 256 }) |rows_in| {
        g.reset();
        var ctx_s: lc.Ctx = .{ .ring = ring_buf };
        // The call may abort (RED side: the 129-row chunk overruns the next 128-row wide-verify

        // window with WindowTooWide before the anchor); the guard grades the RECORDED launches.
        var absorb_err: ?anyerror = null;
        guard_absorb.absorbRows(&d, &ctx_s, 0x6a000000, rows_in - 1, rows_in, null, true) catch |err| {
            absorb_err = err;
        };
        try t.expect(g.count > 0);
        try t.expect(g.f_handle[0] == gather_handle); // harness liveness: the gather IS the first launch
        var g_n: usize = 0;
        var g_sum: usize = 0;
        for (0..g.count) |i| {
            if (g.f_handle[i] == gather_handle) {
                g_n += 1;
                g_sum += g.f_grid[i];
                std.debug.print("[absorb-guard] gather #{d}: grid={d} block={d}\n", .{ g_n, g.f_grid[i], g.f_block[i] });
                try t.expect(g.f_grid[i] <= lc.staging_rows); // THE GUARD: rows <= staging_rows(128)
                try t.expectEqual(@as(usize, 256), g.f_block[i]);
            }
        }
        std.debug.print("[absorb-guard] n={d}: launches={d} gather={d} chunkrows={d} pdl_ex={d} err={any}\n", .{ rows_in, g.count, g_n, g_sum, g.ex_count, absorb_err });
        const seen_ex = g.ex_count >= 1; // seam 2 coverage: the qlinear affine face (projSums/projDense) launches through cuLaunchKernelEx
        var seen_norms = false; // seam 3 coverage: norm -> e.k.norms.addRmsTail records its tail launch
        for (0..g.count) |i| {
            if (g.f_handle[i] == tail0.handle or g.f_handle[i] == tail1.handle) seen_norms = true;
        }
        try t.expect(seen_ex);
        try t.expect(seen_norms);
        try t.expectEqual(@as(usize, if (rows_in <= lc.staging_rows) 1 else 2), g_n);
        try t.expectEqual(rows_in, g_sum);
        try t.expect(absorb_err == null); // the cured absorb completes its whole pass: no seam dies mid-pass
    }

    // PR 5102 verdict asks: explicit row lists, 129/256 rows, PERMUTED. Faces graded by the

    // tally: chunk k gathers rs[done..done+rows] (disjoint slices covering rs exactly); the

    // by-value arg byte-matches qwen35_draft.cu's TapRows (int n + int rows[64], TapRows.n == grid);

    // the anchor tail's input row = the LAST chunk's final staging row, never the cumulative done.
    const rows_h = @intFromPtr(d.fns.gather_rows.handle.?);
    const tail0_h = @intFromPtr(tail0.handle.?);
    const tail1_h = @intFromPtr(tail1.handle.?);
    const kernel_cap = 64; // qwen35_draft.cu:47-55's by-value row list (int rows[64])
    const fail = struct {
        fn fail(ok: bool, tally: *usize, comptime what: []const u8, a: anytype) void {
            if (ok) return;
            tally.* += 1;
            std.debug.print("[absorb-rows] FAIL " ++ what ++ "\n", a);
        }
    }.fail;
    var fail_n: usize = 0;
    fail(@sizeOf(guard_absorb.TapRows) == 4 + kernel_cap * 4, &fail_n, "TapRows byte layout: @sizeOf {d} != 260 (int n + int rows[64])", .{@sizeOf(guard_absorb.TapRows)});
    fail(@offsetOf(guard_absorb.TapRows, "rows") == 4, &fail_n, "TapRows rows offset {d} != 4", .{@offsetOf(guard_absorb.TapRows, "rows")});
    for ([_]usize{ 129, 256 }) |rows_in| {
        g.reset();
        var ctx_s: lc.Ctx = .{ .ring = ring_buf };
        var rs_b: [lc.max_rows]u32 = undefined;
        // a 137-stride walk mod rows_in is a bijective, non-monotonic permutation of the rows
        for (0..rows_in) |i| rs_b[i] = @intCast((i * 137 + 11) % rows_in);
        var absorb_err: ?anyerror = null;
        guard_absorb.absorbRows(&d, &ctx_s, 0x6a000000, 1234, rows_in, rs_b[0..rows_in], true) catch |err| {
            absorb_err = err;
        };
        try t.expect(absorb_err == null);
        var gathers: usize = 0;
        var exp_done: usize = 0;
        var last_gather_grid: usize = 0;
        for (0..g.count) |li| {
            if (@intFromPtr(g.f_handle[li].?) != rows_h) continue;
            gathers += 1;
            last_gather_grid = g.f_grid[li];
            fail(g.f_grid[li] <= kernel_cap, &fail_n, "n={d} gather #{d}: grid {d} exceeds the kernel's row cap {d}", .{ rows_in, gathers, g.f_grid[li], kernel_cap });
            fail(g.f_rows_n[li] == @as(i32, @intCast(g.f_grid[li])), &fail_n, "n={d} gather #{d}: TapRows.n {d} != grid {d}", .{ rows_in, gathers, g.f_rows_n[li], g.f_grid[li] });
            const cap = @min(g.f_grid[li], kernel_cap);
            for (0..cap) |j| {
                fail(g.f_rows[li][j] == rs_b[exp_done + j], &fail_n, "n={d} gather #{d} row {d}: recorded {d}, want rs[{d}]={d}", .{ rows_in, gathers, j, g.f_rows[li][j], exp_done + j, rs_b[exp_done + j] });
            }
            exp_done += g.f_grid[li];
        }
        try t.expectEqual(rows_in, exp_done);
        var tail_last: ?usize = null;
        for (0..g.count) |li| {
            const hv = @intFromPtr(g.f_handle[li].?);
            if (hv == tail0_h or hv == tail1_h) tail_last = li;
        }
        try t.expect(tail_last != null);
        const want_input = d.th + @as(u64, @intCast(last_gather_grid - 1)) * @as(u64, d.c.hidden) * 2;
        fail(g.f_arg0[tail_last.?] == want_input, &fail_n, "n={d}: anchor input 0x{x}, want 0x{x} (last chunk holds {d} rows)", .{ rows_in, g.f_arg0[tail_last.?], want_input, last_gather_grid });
        std.debug.print("[absorb-rows] n={d}: launches={d} gathers={d} exp_done={d} pdl_ex={d} fail_n={d}\n", .{ rows_in, g.count, gathers, exp_done, g.ex_count, fail_n });
    }
    try t.expectEqual(@as(usize, 0), fail_n);
}
