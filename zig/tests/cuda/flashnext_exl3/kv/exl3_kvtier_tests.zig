//! exl3-kvtier: the host tier slots (kvt_plan, kvt_fetch, kvt_remap) against the host LRU model and store bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const Gpu = check.Gpu;
const kvt = flash.exl3.kvtier;
const pages = flash.exl3.pages;
const host_tier = flash.exl3.kvhost;
const kv = core.kv_pool;

const ratio = 4;
const idw = flash.exl3.prompt_wt.index_idw;

/// The host's model of kvt_plan: exact LRU at call granularity, ties to the lower slot.
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

/// `slot_frames` lent frames of a one-layer tier over `npages` pages: drifting 512-block selections.
pub fn kernelRun(gpu: Gpu, slot_frames: usize) !void {
    const a = gpu.gpa;
    const npages: usize = 200;
    const w0: usize = 40_000;
    const nb_before = w0 / ratio;
    const nslots = slot_frames * 64;
    const kv_page = pages.kvPageBytes(kvt.page_rows, 1);
    const row = pages.rowBytes(.key);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var host = try host_tier.Host.create(a, gpu.d, npages, 1 + slot_frames, kv_page);
    defer host.destroy(a);
    var lease: kv.tier.Lease = .{};
    defer host.tier.finish(&lease);
    try host.tier.reserve(&lease, npages, 1 + slot_frames);
    try host.tier.cover(&lease, npages, null);
    try host.tier.lend(&lease, slot_frames);
    var arena = try cuda.DeviceBuffer.alloc(gpu.d, (1 + slot_frames) * kv_page);
    defer arena.free();
    var table = try cuda.DeviceBuffer.alloc(gpu.d, (npages + slot_frames) * 4);
    defer table.free();
    var itable = try cuda.DeviceBuffer.alloc(gpu.d, npages * 4);
    defer itable.free();
    try itable.upload(0, std.mem.sliceAsBytes(lease.pages.ids.items));
    var fids: std.ArrayList(u32) = .empty;
    defer fids.deinit(a);
    try fids.appendNTimes(a, std.math.maxInt(u32), npages);
    var views: [1]cuda.DeviceBuffer = .{.{ .d = gpu.d, .ptr = arena.ptr + pages.planeOffset(kvt.page_rows, 0, .key), .len = 0, .borrowed = true }};
    var vviews: [1]cuda.DeviceBuffer = .{.{ .d = gpu.d, .ptr = arena.ptr + pages.planeOffset(kvt.page_rows, 0, .value), .len = 0, .borrowed = true }};
    var kern = try kvt.Kernels.load(gpu.d, nslots);
    defer kern.unload();
    const sel = try kvt.LaneSel.init(a, gpu.d, &kern, npages * kvt.page_rows, npages, slot_frames, 1, .{ .table = table, .itable = itable, .host = host, .arena = arena, .tl = &lease, .fids = &fids, .kc = &views, .vc = &vviews });
    defer sel.deinit();
    try sel.reset(stream);
    try sel.nameSlots(stream);
    try stream.synchronize();
    sel.win0 = w0;
    // each host row's words unique to (position, K or V)
    for (0..npages) |pg| {
        const id = lease.pages.ids.items[pg];
        const words = std.mem.bytesAsSlice(u32, host.tier.host_page(id));
        for (0..kvt.page_rows) |r| for (0..2) |v| for (0..row / 4) |i| {
            words[(v * kvt.page_rows + r) * (row / 4) + i] = @intCast(((pg * 256 + r) *% 2654435761 +% i *% 40503 +% v *% 977) & 0xFFFF_FFFF);
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
    const tbl = try a.alloc(i32, npages + slot_frames);
    defer a.free(tbl);
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(tbl.ptr, table.ptr, tbl.len * 4), "cuMemcpyDtoH");
    var model: Model = .{ .stamp = try a.alloc(u32, nslots), .slot_of = .init(a), .blk = try a.alloc(i64, nslots) };
    defer {
        a.free(model.stamp);
        a.free(model.blk);
        model.slot_of.deinit();
    }
    @memset(model.stamp, 0);
    @memset(model.blk, -1);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    const per_plan = sel.rowsPerPlan();
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
    const calls: usize = 60;
    for (0..calls) |c| {
        const m: usize = @min(per_plan, @as(usize, if (c % 10 == 9) 64 else 1));
        var need = std.AutoHashMap(u32, void).init(a);
        defer need.deinit();
        for (0..m) |r| {
            for (0..77) |_| { // drift: ~15% of the selection replaced a row
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
            for (0..3) |t| out[2048 + t] = @intCast(w0 + 100 + c * 4 + t); // keys inside the window
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
        const before = try sel.stats();
        try sel.page(stream, 0, ids_d.ptr, nk_d.ptr, sp_d.ptr, m);
        try stream.synchronize();
        const after = try sel.stats();
        gpu_miss += after.fetched - before.fetched;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(back.ptr, ids_d.ptr, m * idw * 4), "cuMemcpyDtoH");
        for (0..m) |r| for (0..2051) |ki| {
            const p: usize = @intCast(ids[r * idw + ki]);
            const q: usize = @intCast(back[r * idw + ki]);
            keys_checked += 1;
            if (p >= w0) {
                bad += @intFromBool(q != p);
                continue;
            }
            // row q of the table: entry q / 256 is a lent frame, the row is where the slot's bytes sit in the arena
            const fr: usize = @intCast(tbl[q / kvt.page_rows]);
            if (q < npages * kvt.page_rows or q / kvt.page_rows >= npages + slot_frames) {
                bad += 1;
                continue;
            }
            var got: [2][1024]u8 = undefined;
            for ([_]pages.Part{ .key, .value }, 0..) |part, v| {
                const at = fr * kv_page + pages.planeOffset(kvt.page_rows, 0, part) + (q % kvt.page_rows) * row;
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(&got[v], arena.ptr + at, row), "cuMemcpyDtoH");
                const hp = host.tier.host_page(lease.pages.ids.items[p / kvt.page_rows]);
                const want = hp[pages.planeOffset(kvt.page_rows, 0, part) + (p % kvt.page_rows) * row ..][0..row];
                bad += @intFromBool(!std.mem.eql(u8, &got[v], want));
            }
        };
    }
    const st = try sel.stats();
    std.debug.print("kvtier kernel ({d} slot frames, {d} slots): {d} calls ({d} keys): bad {d}; misses GPU {d}, host model {d}; needed {d}; overflow {}\n", .{ slot_frames, nslots, calls, keys_checked, bad, gpu_miss, model_miss, st.needed, st.overflow });
    try check.expect(bad == 0 and !st.overflow, "every remapped key reads its own host row", .{});
    const diff = @as(f64, @floatFromInt(@max(gpu_miss, model_miss) - @min(gpu_miss, model_miss))) / @as(f64, @floatFromInt(@max(model_miss, 1)));
    try check.expect(diff < 0.05, "misses within 5% of the LRU model ({d:.3})", .{diff});
    check.pass("exl3-kvtier kernel, {d} slot frames", .{slot_frames});
}

pub fn run(gpu: Gpu, args: []const []const u8) !void {
    _ = args;
    for ([_]usize{ 128, 16, 9 }) |frames| try kernelRun(gpu, frames);
}
