//! Living-memory recurrent delta compression: parallel per-layer SVD of body minus prefix recurrent state.
const std = @import("std");
const quant = @import("quant.zig");
const types = @import("types.zig");
const Allocator = std.mem.Allocator;
const ArchSpec = types.ArchSpec;
const RawState = types.RawState;
const RecLayerData = types.RecLayerData;

/// Parallelized per-layer SVD of recurrent state deltas (`S_after - S_prefix`), its F32 factors kept as computed.
pub fn compressRecurrentLayers(
    gpa: Allocator,
    arch: ArchSpec,
    pref: *const RawState,
    body: *const RawState,
    rank: usize,
) ![]RecLayerData {
    const rec = try gpa.alloc(RecLayerData, arch.num_rec);
    var made: usize = 0;
    errdefer {
        for (rec[0..made]) |*r| r.deinit(gpa);
        gpa.free(rec);
    }
    for (0..arch.num_rec) |r| {
        const fu = try gpa.alloc(f32, arch.rec_heads * arch.rec_rows * rank);
        errdefer gpa.free(fu);
        const fv = try gpa.alloc(f32, arch.rec_heads * rank * arch.rec_cols);
        errdefer gpa.free(fv);
        const cv = if (arch.keep_conv) try gpa.dupe(u16, body.rec_conv[r]) else try gpa.alloc(u16, 0);
        errdefer gpa.free(cv);
        rec[r] = .{ .rank = @intCast(rank), .u = fu, .v = fv, .conv = cv };
        made += 1;
    }
    if (arch.num_rec == 0 or rank == 0) return rec;

    const WorkerCtx = struct {
        arch: ArchSpec,
        pref: *const RawState,
        body: *const RawState,
        rec: []RecLayerData,
        rank: usize,
        next_layer: std.atomic.Value(usize),
        failed: std.atomic.Value(bool),

        fn run(ctx: *@This()) void {
            const alloc = std.heap.page_allocator;
            const head_mat = ctx.arch.rec_rows * ctx.arch.rec_cols;
            const delta_buf = alloc.alloc(f32, head_mat) catch {
                ctx.failed.store(true, .release);
                return;
            };
            defer alloc.free(delta_buf);

            var scratch_arena = std.heap.ArenaAllocator.init(alloc);
            defer scratch_arena.deinit();

            while (true) {
                const r = ctx.next_layer.fetchAdd(1, .acq_rel);
                if (r >= ctx.arch.num_rec) break;
                const rl = &ctx.rec[r];
                for (0..ctx.arch.rec_heads) |h| {
                    _ = scratch_arena.reset(.retain_capacity);
                    const p_st = ctx.pref.rec_state[r][h * head_mat ..][0..head_mat];
                    const b_st = ctx.body.rec_state[r][h * head_mat ..][0..head_mat];
                    for (p_st, b_st, delta_buf) |pv, bv, *dv| dv.* = bv - pv;
                    _ = quant.randomizedSvdBalanced(
                        scratch_arena.allocator(),
                        delta_buf,
                        ctx.arch.rec_rows,
                        ctx.arch.rec_cols,
                        ctx.rank,
                        rl.u[h * ctx.arch.rec_rows * ctx.rank ..][0 .. ctx.arch.rec_rows * ctx.rank],
                        rl.v[h * ctx.rank * ctx.arch.rec_cols ..][0 .. ctx.rank * ctx.arch.rec_cols],
                    ) catch {
                        ctx.failed.store(true, .release);
                        return;
                    };
                }
            }
        }
    };

    var ctx = WorkerCtx{
        .arch = arch,
        .pref = pref,
        .body = body,
        .rec = rec,
        .rank = rank,
        .next_layer = .init(0),
        .failed = .init(false),
    };
    const n_threads = @min(arch.num_rec, @max(@as(usize, 1), std.Thread.getCpuCount() catch 4));
    if (n_threads <= 1) {
        WorkerCtx.run(&ctx);
    } else {
        const threads = try gpa.alloc(std.Thread, n_threads - 1);
        defer gpa.free(threads);
        var spawned: usize = 0;
        for (0..n_threads - 1) |t| {
            threads[t] = std.Thread.spawn(.{}, WorkerCtx.run, .{&ctx}) catch break;
            spawned += 1;
        }
        WorkerCtx.run(&ctx);
        for (threads[0..spawned]) |th| th.join();
    }
    if (ctx.failed.load(.acquire)) return error.OutOfMemory;
    return rec;
}
