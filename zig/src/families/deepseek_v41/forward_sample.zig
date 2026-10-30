//! The forward's greedy pick: each rank's best column (host or GPU), all-gathered, lower rank wins.

const std = @import("std");
const cuda = @import("cuda");
const run = @import("run.zig");
const hooks = @import("forward_hooks.zig");
const ph = @import("phases.zig");
const fwd = @import("forward.zig");
const fwd_follow = @import("forward_follow.zig");
const Forward = fwd.Forward;
const kops = fwd.kops;
const op_greedy = fwd_follow.op_greedy;

/// Greedy choices of the last window's rows: each rank's max (lowest column), all-gathered (lower rank wins).
pub fn greedy(f: *Forward, out: []u32) !void {
    if (f.leads()) try f.link.?.send(&.{ op_greedy, @intCast(out.len) });
    const tc = ph.start(f.phases, .candidates);
    defer tc.stop();
    const r = f.runner;
    const n = out.len;
    const W: usize = f.comm.world();
    const vocab_part = try @import("partition.zig").share(f.cfg.vocab, f.comm.rank(), W);
    const V: usize = vocab_part.len;
    // pinned: the logits [n, V], then the pairs [2n] and every rank's [W][2n] (reused, no allocation a window)
    const lb = std.mem.alignForward(usize, 4 * n * V, 8);
    const need = lb + 16 * n * (W + 1);
    if (f.pick_host == null or f.pick_host.?.bytes.len < need) {
        if (f.pick_host) |*b| b.free();
        f.pick_host = try cuda.HostBuffer.alloc(r.d, @max(need, 1 << 20));
    }
    if (f.pick_dev == null or f.pick_dev.?.len < 16 * n * (W + 1)) {
        if (f.pick_dev) |*b| b.free();
        f.pick_dev = try cuda.DeviceBuffer.alloc(r.d, @max(16 * n * (W + 1), 4096));
    }
    const hb = f.pick_host.?.bytes;
    const host: []f32 = @alignCast(std.mem.bytesAsSlice(f32, hb[0 .. 4 * n * V]));
    const pairs: []f64 = @alignCast(std.mem.bytesAsSlice(f64, hb[lb..need]));
    const dev = f.pick_dev.?;
    const logits = r.addressOf("w.logits") orelse return error.Unbound;
    if (f.grammar) |g| try g.apply(g.ptr, 0, @intCast(n));
    if (f.greedy_gpu == null) f.greedy_gpu = if (std.c.getenv("TF_DSV41_GREEDY_GPU")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    if (f.greedy_gpu.? and V <= 1 << 16) {
        // the device pick: topk_keys' best column a row, [n] fp32 values + [n] int64 columns back, not logits
        const vals_dev = try f.greedyScratch(r, n);
        const cols_dev = vals_dev + std.mem.alignForward(u64, 4 * n, 16);
        const k = r.kernels.others(r.stream);
        try k.topKeys(logits, V, n, V, 1, vals_dev, cols_dev);
        f.pick_last = null;
        if (k.f.pick_pack != null and k.f.pick_merge != null and n <= kops.glue.pick_max_rows) {
            // the pairs, their exchange and the merge on the device; one copy back (glue.cu pick_*)
            const ids_dev = cols_dev + 8 * n;
            const out_dev = ids_dev + 8 * n;
            try k.pickPack(vals_dev, cols_dev, n, vocab_part.lo, dev.ptr);
            try f.comm.allGather(dev.ptr, dev.ptr + 16 * n, 2 * n, .f64, r.stream.handle);
            const chain = if (f.slot.pending) |p| p.n == n and f.ids.len >= n else false;
            const start: i64 = if (f.slot.pending) |p| @intCast(p.start) else 0;
            const ih: []i64 = @alignCast(std.mem.bytesAsSlice(i64, hb[0 .. 8 * n]));
            if (chain) {
                for (ih, f.ids[0..n]) |*x, t| x.* = t;
                try cuda.copy_source.enqueue(r.d, ids_dev, std.mem.sliceAsBytes(ih[0..n]), r.stream);
            }
            try k.pickMerge(dev.ptr + 16 * n, W, n, if (chain) ids_dev else 0, start, out_dev);
            const oh: []i64 = @alignCast(std.mem.bytesAsSlice(i64, hb[std.mem.alignForward(usize, 8 * n, 16)..][0 .. 8 * (n + 3)]));
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(oh.ptr, out_dev, 8 * (n + 3), r.stream.handle), "cuMemcpyDtoHAsync");
            f.pick_last = .{ .out = out_dev, .n = @intCast(n), .chain = chain, .start = @intCast(start), .host = oh.ptr };
            if (f.after_pick) |h| {
                // work from the device pick (speculative DSpark pass) follows the copy back; the host waits on it
                if (f.pick_done == null) f.pick_done = try cuda.Event.init(r.d, false);
                try f.pick_done.?.record(r.stream);
                try h.run(h.ctx, f.pick_last.?);
                try f.pick_done.?.synchronize();
            } else try r.stream.synchronize();
            for (out, oh[0..n]) |*o, x| o.* = @intCast(x);
            return;
        }
        const vh: []f32 = @alignCast(std.mem.bytesAsSlice(f32, hb[0 .. 4 * n]));
        const ch: []i64 = @alignCast(std.mem.bytesAsSlice(i64, hb[std.mem.alignForward(usize, 4 * n, 16)..][0 .. 8 * n]));
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(vh.ptr, vals_dev, 4 * n, r.stream.handle), "cuMemcpyDtoHAsync");
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(ch.ptr, cols_dev, 8 * n, r.stream.handle), "cuMemcpyDtoHAsync");
        try r.stream.synchronize();
        for (0..n) |row| {
            pairs[2 * row] = vh[row];
            pairs[2 * row + 1] = @floatFromInt(@as(usize, @intCast(ch[row])) + vocab_part.lo);
        }
    } else {
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(host.ptr, logits, 4 * n * V, r.stream.handle), "cuMemcpyDtoHAsync");
        try r.stream.synchronize();
        const argmax = (try hooks.greedy()).argmax;
        for (0..n) |row| {
            const vals = host[row * V ..][0..V];
            const best = argmax(vals);
            pairs[2 * row] = vals[best];
            pairs[2 * row + 1] = @floatFromInt(best + vocab_part.lo);
        }
    }
    try dev.uploadAsync(0, std.mem.sliceAsBytes(pairs[0 .. 2 * n]), r.stream.handle);
    try f.comm.allGather(dev.ptr, dev.ptr + 16 * n, 2 * n, .f64, r.stream.handle);
    const all = pairs[2 * n ..];
    try dev.downloadAsync(16 * n, std.mem.sliceAsBytes(all), r.stream.handle);
    try r.stream.synchronize();
    for (out, 0..) |*o, row| {
        var bk: usize = 0;
        for (1..W) |k| if (all[2 * (k * n + row)] > all[2 * (bk * n + row)]) {
            bk = k;
        };
        o.* = @intFromFloat(all[2 * (bk * n + row) + 1]);
    }
}

/// The GPU pick's device scratch: [n] fp32 values, [n] int64 columns, [n] int64 window ids, [n + 3] int64 out.
pub fn greedyScratch(f: *Forward, r: *run.Runner, n: usize) !u64 {
    const bytes = std.mem.alignForward(usize, 4 * n, 16) + 8 * n + 8 * n + 8 * (n + 3);
    if (f.greedy_dev == null or f.greedy_dev.?.len < bytes) {
        if (f.greedy_dev) |*b| b.free();
        f.greedy_dev = null;
        f.greedy_dev = try cuda.DeviceBuffer.alloc(r.d, @max(bytes, 4096));
    }
    return f.greedy_dev.?.ptr;
}
