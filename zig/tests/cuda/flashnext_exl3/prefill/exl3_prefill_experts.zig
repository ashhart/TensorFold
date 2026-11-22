//! exl3-prefill experts and prompt: the grouped, slot and expert-major prefill kernels timed on one layer.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
const flash = @import("flashnext");
const pk_ = flash.exl3.prefill_k;
const exl3e = cuda.exl3_experts;
const Gpu = check.Gpu;
const packs = @import("exl3_prefill_pack.zig");

const dims = 2560;
const width = 640;
const routed_n = 512;
const table_n = 513;
const slots = 11;
const max_call_rows: usize = 1408;

const LayerData = packs.LayerData;
const Timer = packs.Timer;
const bf16Bits = packs.bf16Bits;
const genPicks = packs.genPicks;

pub const SlotsCall = struct {
    ex: *const exl3e.Experts,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    picks: u64,
    rows: usize,
    sc: exl3e.Scratch,
    pub fn go(c: SlotsCall) anyerror!void {
        try c.ex.runSlots(c.s, c.l, c.x, dims, .bf16, c.picks, c.rows, c.sc);
    }
};

/// Expert-major passes over one block: pass j runs the block's picks of experts [j*per, (j+1)*per) (others -1).
const PassesCall = struct {
    ex: *const exl3e.Experts,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    masked: u64, // int32 [passes, rows * slots]
    rows: usize,
    passes: usize,
    sc: exl3e.Scratch,
    pub fn go(c: PassesCall) anyerror!void {
        for (0..c.passes) |j| try c.ex.runSlots(c.s, c.l, c.x, dims, .bf16, c.masked + j * c.rows * slots * 4, c.rows, c.sc);
    }
};

pub fn expertsRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const a = gpu.gpa;
    var t = try Timer.init(gpu.d);
    defer t.deinit();
    const max_rows = max_call_rows;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rng = prng.random();
    const xh = try a.alloc(u16, max_rows * dims);
    defer a.free(xh);
    for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    const picks = try a.alloc(i32, max_rows * slots);
    defer a.free(picks);
    var pk = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * slots * 4);
    defer pk.free();
    var sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(ld.res, max_rows, slots));
    defer sbuf.free();
    const sc = try exl3e.scratch(sbuf.ptr, ld.res, max_rows, slots);
    std.debug.print("  runSlots, one resident layer (10 routed + shared a row), ms a call (best of 5 x 5):\n", .{});
    for ([_]f64{ 0.8, 0 }) |zs| {
        genPicks(rng, zs, max_rows, picks);
        try pk.upload(0, std.mem.sliceAsBytes(picks));
        for ([_]usize{ 16, 64, 256, 512, 1024, max_rows }) |rows| {
            var seen: [table_n]bool = @splat(false);
            var uniq: usize = 0;
            for (picks[0 .. rows * slots]) |p| if (!seen[@intCast(p)]) {
                seen[@intCast(p)] = true;
                uniq += 1;
            };
            const ms = try t.best(s, 5, SlotsCall{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc }, SlotsCall.go);
            std.debug.print("    {s:<8} rows {d:5}: {d:8.3} ms  {d:7.2} us/row  {d:6.0} rows/s/layer  ({d} experts incl. shared, {d:.1} member rows each)\n", .{ if (zs == 0) "uniform" else "zipf0.8", rows, ms, 1e3 * ms / @as(f64, @floatFromInt(rows)), @as(f64, @floatFromInt(rows)) / ms * 1e3, uniq, @as(f64, @floatFromInt(rows * slots)) / @as(f64, @floatFromInt(uniq)) });
        }
    }
    // expert-major passes over one 1024-row block (uniform picks): bytes equal to one call, and the time
    const rows: usize = 1024;
    genPicks(rng, 0, rows, picks);
    try pk.upload(0, std.mem.sliceAsBytes(picks[0 .. rows * slots]));
    const one_bytes = try a.alloc(u8, rows * slots * dims * 4);
    defer a.free(one_bytes);
    try (SlotsCall{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc }).go();
    try s.synchronize();
    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(one_bytes.ptr, sc.y, one_bytes.len), "cuMemcpyDtoH");
    const base_ms = try t.best(s, 5, SlotsCall{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pk.ptr, .rows = rows, .sc = sc }, SlotsCall.go);
    std.debug.print("  expert-major passes over one {d}-row block (uniform picks; pass j: experts [j*512/P, (j+1)*512/P) plus the shared one in pass 0):\n", .{rows});
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |passes| {
        const per = routed_n / passes;
        const host = try a.alloc(i32, passes * rows * slots);
        defer a.free(host);
        for (0..passes) |j| for (picks[0 .. rows * slots], host[j * rows * slots ..][0 .. rows * slots]) |p, *o| {
            const in = if (p == 512) j == 0 else @as(usize, @intCast(p)) / per == j;
            o.* = if (in) p else -1;
        };
        var masked = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer masked.free();
        try gpu.d.check(gpu.d.api.cuMemsetD8_v2(sc.y, 0xff, rows * slots * dims * 4), "cuMemsetD8");
        const call: PassesCall = .{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .masked = masked.ptr, .rows = rows, .passes = passes, .sc = sc };
        try call.go();
        try s.synchronize();
        const got = try a.alloc(u8, one_bytes.len);
        defer a.free(got);
        try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(got.ptr, sc.y, got.len), "cuMemcpyDtoH");
        const same = std.mem.eql(u8, got, one_bytes);
        const ms = try t.best(s, 3, call, PassesCall.go);
        std.debug.print("    {d:2} passes of {d:3} experts: {d:8.3} ms (+{d:.3} ms vs one call, {d:.3} ms a pass)  y {s}\n", .{ passes, per, ms, ms - base_ms, ms / @as(f64, @floatFromInt(passes)), if (same) "bit-identical to one call" else "DIFFERS" });
        try check.expect(same, "masked passes reproduce one call's bytes", .{});
    }
    check.pass("layer {d}: runSlots timed at 16..{d} rows; expert-major masked passes are bit-identical to one call", .{ ld.layer, max_rows });
}

// ---------------------------------------------------------------------------------------------------- prompt kernel

const PromptCall = struct {
    k: *const pk_.Kernels,
    s: cuda.Stream,
    l: exl3e.Layer,
    x: u64,
    picks: u64,
    n_members: usize,
    n_items: usize,
    dv: pk_.Device,
    y: u64,
    pub fn go(c: PromptCall) anyerror!void {
        try pk_.run(c.k, c.s, c.l, c.x, dims, c.picks, slots, c.n_members, c.n_items, c.dv, c.y);
    }
};

/// P expert-major passes, each its own uploaded plan (members, items) over the same picks.
const PromptPasses = struct {
    calls: []const PromptCall,
    pub fn go(c: PromptPasses) anyerror!void {
        for (c.calls) |call| try call.go();
    }
};

pub fn promptRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData) !void {
    const a = gpu.gpa;
    var k = try pk_.Kernels.load(gpu.d);
    defer k.unload();
    var t = try Timer.init(gpu.d);
    defer t.deinit();
    const max_rows: usize = 8192;
    const max_pairs = max_rows * slots;
    var prng = std.Random.DefaultPrng.init(0x9a11);
    const rng = prng.random();
    const xh = try a.alloc(u16, max_rows * dims);
    defer a.free(xh);
    for (xh) |*b| b.* = bf16Bits((rng.float(f32) - 0.5) * 2.0);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xh));
    defer x.free();
    const picks = try a.alloc(i32, max_pairs);
    defer a.free(picks);
    genPicks(rng, 0.8, max_rows, picks);
    var pkd = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer pkd.free();
    var plan = try pk_.Plan.init(a, max_pairs, table_n);
    defer plan.deinit(a);
    var counts: [table_n]u32 = undefined;
    var mem_d = try cuda.DeviceBuffer.alloc(gpu.d, max_pairs * 4);
    defer mem_d.free();
    var items_d = try cuda.DeviceBuffer.alloc(gpu.d, 3 * pk_.maxItems(max_pairs, table_n) * 4);
    defer items_d.free();
    var scr = try cuda.DeviceBuffer.alloc(gpu.d, pk_.scratchBytes(max_pairs, dims, width));
    defer scr.free();
    var yk = try cuda.DeviceBuffer.alloc(gpu.d, max_pairs * dims * 4);
    defer yk.free();
    // runSlots' reference (at most max_call_rows rows a call)
    var sbuf = try cuda.DeviceBuffer.alloc(gpu.d, try exl3e.scratchBytes(ld.res, max_call_rows, slots));
    defer sbuf.free();
    const sc = try exl3e.scratch(sbuf.ptr, ld.res, max_call_rows, slots);
    std.debug.print("  prefill experts kernel (64-member tiles, compact pairs) vs runSlots, one resident layer, zipf0.8 picks:\n", .{});
    for ([_]usize{ 16, 64, 256, 1024, 1408, 2048, 4096, 8192 }) |rows| {
        const pairs = rows * slots;
        pk_.build(picks[0..pairs], null, &counts, &plan);
        try mem_d.upload(0, plan.memberBytes());
        try items_d.upload(0, plan.itemBytes());
        const dv: pk_.Device = .{ .members = mem_d.ptr, .items = items_d.ptr, .scratch = scr.ptr };
        const call: PromptCall = .{ .k = &k, .s = s, .l = ld.res, .x = x.ptr, .picks = pkd.ptr, .n_members = plan.n_members, .n_items = plan.n_items, .dv = dv, .y = yk.ptr };
        try call.go();
        try s.synchronize();
        const ms_k = try t.best(s, 3, call, PromptCall.go);
        var ms_r: f64 = std.math.nan(f64);
        var rel: f64 = std.math.nan(f64);
        var same_words: usize = 0;
        if (rows <= max_call_rows) {
            const rc: SlotsCall = .{ .ex = ex, .s = s, .l = ld.res, .x = x.ptr, .picks = pkd.ptr, .rows = rows, .sc = sc };
            try rc.go();
            try s.synchronize();
            const want = try a.alloc(f32, pairs * dims);
            defer a.free(want);
            const got = try a.alloc(f32, pairs * dims);
            defer a.free(got);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(want.ptr, sc.y, want.len * 4), "cuMemcpyDtoH");
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(got.ptr, yk.ptr, got.len * 4), "cuMemcpyDtoH");
            var dmax: f64 = 0;
            var peak: f64 = 0;
            for (got, want) |g, w| {
                dmax = @max(dmax, @abs(@as(f64, g) - w));
                peak = @max(peak, @abs(@as(f64, w)));
                same_words += @intFromBool(@as(u32, @bitCast(g)) == @as(u32, @bitCast(w)));
            }
            rel = dmax / peak;
            ms_r = try t.best(s, 3, rc, SlotsCall.go);
            try check.expect(same_words == pairs * dims, "prefill kernel's y is runSlots' bytes at {d} rows ({d} of {d} words differ)", .{ rows, pairs * dims - same_words, pairs * dims });
        }
        std.debug.print("    rows {d:5}: prefill kernel {d:8.3} ms ({d:6.2} us/row, {d} items)  runSlots {d:8.3} ms ({d:6.2} us/row)  speedup {d:.2}x  max|dy|/peak {e:.2}, {d:.1}% of words bit-equal\n", .{ rows, ms_k, 1e3 * ms_k / @as(f64, @floatFromInt(rows)), plan.n_items, ms_r, 1e3 * ms_r / @as(f64, @floatFromInt(rows)), ms_r / ms_k, rel, 100 * @as(f64, @floatFromInt(same_words)) / @as(f64, @floatFromInt(pairs * dims)) });
    }
    // expert-major passes over 8192 rows: the plan of each pass alone, back to back
    const rows = max_rows;
    const pairs = rows * slots;
    std.debug.print("  expert-major passes over {d} rows (pass j: experts [j*512/P, (j+1)*512/P), the shared one in pass 0):\n", .{rows});
    for ([_]usize{ 1, 8, 16, 32 }) |passes| {
        const per = routed_n / passes;
        var calls: [32]PromptCall = undefined;
        var mems: [32]cuda.DeviceBuffer = undefined;
        var its: [32]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        defer for (mems[0..made], its[0..made]) |*m, *i| {
            m.free();
            i.free();
        };
        for (0..passes) |j| {
            var take: [table_n]bool = @splat(false);
            for (j * per..(j + 1) * per) |id| take[id] = true;
            take[512] = j == 0;
            pk_.build(picks[0..pairs], &take, &counts, &plan);
            mems[j] = try cuda.DeviceBuffer.fromHost(gpu.d, plan.memberBytes());
            its[j] = try cuda.DeviceBuffer.fromHost(gpu.d, plan.itemBytes());
            made += 1;
            calls[j] = .{ .k = &k, .s = s, .l = ld.res, .x = x.ptr, .picks = pkd.ptr, .n_members = plan.n_members, .n_items = plan.n_items, .dv = .{ .members = mems[j].ptr, .items = its[j].ptr, .scratch = scr.ptr }, .y = yk.ptr };
        }
        const ms = try t.best(s, 2, PromptPasses{ .calls = calls[0..passes] }, PromptPasses.go);
        std.debug.print("    {d:2} passes: {d:8.3} ms ({d:6.2} us/row)\n", .{ passes, ms, 1e3 * ms / @as(f64, @floatFromInt(rows)) });
    }
    check.pass("layer {d}: prefill experts kernel is runSlots' bytes, timed to {d} rows", .{ ld.layer, max_rows });
}
