//! exl3-prefill experts: the grouped, slot and expert-major masked passes timed on one layer.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../../check.zig");
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
