//! exl3-kvpage kernel: kvp_plan against the host's LRU model and replayed.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;
const kvp = flash.exl3.kvpage;

/// The host's model of kvp_plan: exact LRU at call granularity, ties to the lower slot.
const Model = struct {
    stamp: []u32,
    slot_of: std.AutoHashMap(u32, u32),
    blk: []i64,
    epoch: u32 = 0,

    fn use(mo: *Model, a: std.mem.Allocator, need: []const u32) !usize {
        mo.epoch += 1;
        var misses: std.ArrayList(u32) = .empty;
        defer misses.deinit(a);
        for (need) |b| {
            if (mo.slot_of.get(b)) |sl| mo.stamp[sl] = mo.epoch else try misses.append(a, b);
        }
        const order = try a.alloc(u32, mo.stamp.len);
        defer a.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Ctx = struct {
            st: []const u32,
            ep: u32,
            fn key(c: @This(), i: u32) u64 {
                const t = c.st[i];
                return if (t == c.ep) std.math.maxInt(u64) else (@as(u64, t) << 32) | i;
            }
            fn less(c: @This(), x: u32, y: u32) bool {
                return c.key(x) < c.key(y);
            }
        };
        std.mem.sort(u32, order, Ctx{ .st = mo.stamp, .ep = mo.epoch }, Ctx.less);
        for (misses.items, 0..) |b, i| {
            const sl = order[i];
            if (mo.stamp[sl] == mo.epoch) return error.ModelOverflow;
            if (mo.blk[sl] >= 0) _ = mo.slot_of.remove(@intCast(mo.blk[sl]));
            mo.blk[sl] = b;
            try mo.slot_of.put(b, sl);
            mo.stamp[sl] = mo.epoch;
        }
        return misses.items.len;
    }
};

pub fn kernelRun(gpu: Gpu, slots: usize) !void {
    const a = gpu.gpa;
    const idw = flash.exl3.prompt_wt.index_idw;
    const row = kvp.row_bytes;
    const cap: usize = 60_000;
    const w0: usize = 40_000;
    const nb_before = w0 / 4;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const pg = try kvp.Paged.init(a, gpu.d, cap, .{ .window = 2052, .slots = slots, .rows = 256 });
    defer pg.destroy();
    pg.setWindow(0, w0);
    // each host row's 256 words unique to (position, K or V)
    for (0..2) |v| {
        const h = std.mem.bytesAsSlice(u32, pg.hostMut(0, v == 1));
        for (0..cap) |p| for (0..row / 4) |i| {
            h[p * (row / 4) + i] = @intCast((p *% 2654435761 +% i *% 40503 +% v *% 977) & 0xFFFF_FFFF);
        };
    }
    var ids_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * idw * 4);
    defer ids_d.free();
    var nk_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 4);
    defer nk_d.free();
    var sp_d = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 4);
    defer sp_d.free();
    const ids = try a.alloc(i32, 256 * idw);
    defer a.free(ids);
    const back = try a.alloc(i32, 256 * idw);
    defer a.free(back);
    const nk = try a.alloc(i32, 256);
    defer a.free(nk);
    const sp = try a.alloc(i32, 256);
    defer a.free(sp);
    @memset(sp, 1);
    const lay = pg.slotRows(0);
    const dk = try a.alloc(u8, lay[0] * row);
    defer a.free(dk);
    const dv = try a.alloc(u8, lay[0] * row);
    defer a.free(dv);
    var model: Model = .{ .stamp = try a.alloc(u32, slots), .slot_of = .init(a), .blk = try a.alloc(i64, slots) };
    defer {
        a.free(model.stamp);
        a.free(model.blk);
        model.slot_of.deinit();
    }
    @memset(model.stamp, 0);
    @memset(model.blk, -1);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    // a drifting selection: 512 distinct blocks of [0, nb_before), 15% replaced a call
    var base: [512]u32 = undefined;
    var used = std.AutoHashMap(u32, void).init(a);
    defer used.deinit();
    var k: usize = 0;
    while (k < 512) {
        const b = rng.uintLessThan(u32, @intCast(nb_before));
        if ((try used.getOrPut(b)).found_existing) continue;
        base[k] = b;
        k += 1;
    }
    var gpu_miss: usize = 0;
    var model_miss: usize = 0;
    var bad: usize = 0;
    var keys_checked: usize = 0;
    var plan_ns: i96 = 0;
    const calls: usize = 60;
    for (0..calls) |c| {
        const m: usize = if (c % 10 == 9) 64 else 1;
        var need = std.AutoHashMap(u32, void).init(a);
        defer need.deinit();
        for (0..m) |r| {
            // drift: replace ~15% of the base blocks
            for (0..77) |_| {
                const j = rng.uintLessThan(usize, 512);
                while (true) {
                    const b = rng.uintLessThan(u32, @intCast(nb_before));
                    if ((try used.getOrPut(b)).found_existing) continue;
                    _ = used.remove(base[j]);
                    base[j] = b;
                    break;
                }
            }
            var sorted = base;
            std.mem.sort(u32, &sorted, {}, std.sort.asc(u32));
            const out = ids[r * idw ..][0..idw];
            for (sorted, 0..) |b, j| for (0..4) |t| {
                out[j * 4 + t] = @intCast(b * 4 + t);
            };
            // 3 tail keys inside the window
            for (0..3) |t| out[2048 + t] = @intCast(w0 + 100 + c * 4 + t);
            nk[r] = 2051;
            for (sorted) |b| try need.put(b, {});
        }
        var need_list: std.ArrayList(u32) = .empty;
        defer need_list.deinit(a);
        var it = need.keyIterator();
        while (it.next()) |b| try need_list.append(a, b.*);
        model_miss += try model.use(a, need_list.items);
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(ids_d.ptr, ids.ptr, m * idw * 4), "cuMemcpyHtoD");
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(nk_d.ptr, nk.ptr, m * 4), "cuMemcpyHtoD");
        try gpu.d.check(gpu.d.api.cuMemcpyHtoD_v2(sp_d.ptr, sp.ptr, m * 4), "cuMemcpyHtoD");
        const before = try pg.stats(0);
        const t0 = check.now(gpu.io);
        try pg.page(stream, 0, ids_d.ptr, nk_d.ptr, sp_d.ptr, m);
        try stream.synchronize();
        plan_ns += check.now(gpu.io) - t0;
        const after = try pg.stats(0);
        try check.expect(after.epoch == c + 1 and after.calls == c + 1, "pager device epoch/calls advance exactly once on launch {d}", .{c + 1});
        gpu_miss += after.misses - before.misses;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(back.ptr, ids_d.ptr, m * idw * 4), "cuMemcpyDtoH");
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(dk.ptr, pg.devRows(0, false), dk.len), "cuMemcpyDtoH");
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(dv.ptr, pg.devRows(0, true), dv.len), "cuMemcpyDtoH");
        const hk = pg.hostRows(0, false);
        const hv = pg.hostRows(0, true);
        for (0..m) |r| for (0..2051) |ki| {
            const p: usize = @intCast(ids[r * idw + ki]);
            const q: usize = @intCast(back[r * idw + ki]);
            keys_checked += 1;
            if (p >= pg.win0[0]) {
                bad += @intFromBool(q != p - pg.win0[0]);
                continue;
            }
            if (q < lay[1] or q >= lay[0]) {
                bad += 1;
                continue;
            }
            bad += @intFromBool(!std.mem.eql(u8, dk[q * row ..][0..row], hk[p * row ..][0..row]) or !std.mem.eql(u8, dv[q * row ..][0..row], hv[p * row ..][0..row]));
        };
    }
    const st = try pg.stats(0);
    std.debug.print("kernel: {d} calls ({d} keys checked): bad {d}; misses GPU {d}, host LRU model {d}; needed {d}; overflow {}; plan+fetch+remap {d:.1} us a call (incl. launch + sync)\n", .{ calls, keys_checked, bad, gpu_miss, model_miss, st.needed, st.overflow, @as(f64, @floatFromInt(plan_ns)) / 1e3 / @as(f64, @floatFromInt(calls)) });
    try check.expect(bad == 0 and !st.overflow, "every remapped key reads its own host row", .{});
    const diff = @as(f64, @floatFromInt(@max(gpu_miss, model_miss) - @min(gpu_miss, model_miss))) / @as(f64, @floatFromInt(@max(model_miss, 1)));
    try check.expect(diff < 0.05, "misses within 5% of the LRU model ({d:.3})", .{diff});
    check.pass("exl3-kvpage kernel", .{});
}
