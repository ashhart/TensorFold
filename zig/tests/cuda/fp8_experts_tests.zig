//! Grouped block-FP8 experts against recorded fixture bytes (the Python engine's): packing, plan, up and down.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const ex_ = cuda.experts;
const fx8 = ex_.Fp8Experts;

fn param(fx: Fixture, name: []const u8) !usize {
    return @intCast(try fx.int(name));
}

fn arr(fx: Fixture, comptime fmt: []const u8, args: anytype) ![]u8 {
    var name: [32]u8 = undefined;
    return fx.bytes(try std.fmt.bufPrint(&name, fmt, args));
}

pub fn experts(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    const e = try param(fx, "experts");
    const d = try param(fx, "dims");
    const ni = try param(fx, "width");
    const slots = try param(fx, "slots");

    const gate = try fx.bytes("gate");
    defer a.free(gate);
    const up = try fx.bytes("up");
    defer a.free(up);
    const down = try fx.bytes("down");
    defer a.free(down);
    const pu = try a.alloc(u32, e * ni * d / 2);
    defer a.free(pu);
    const pd = try a.alloc(u32, e * ni * d / 4);
    defer a.free(pd);
    const one = ni * d;
    for (0..e) |x| {
        try fx8.packGateUp(a, pu[x * one / 2 ..][0 .. one / 2], gate[x * one ..][0..one], up[x * one ..][0..one], ni, d);
        fx8.packOne(pd[x * one / 4 ..][0 .. one / 4], down[x * one ..][0..one], d, ni);
    }
    inline for (.{ .{ "packed_up", pu }, .{ "packed_down", pd } }) |c| {
        const want = try fx.bytes(c[0]);
        defer a.free(want);
        try check.sameBytes(c[0], std.mem.sliceAsBytes(c[1]), want);
    }
    const sg = try fx.bytes("gate_scale");
    defer a.free(sg);
    const su = try fx.bytes("up_scale");
    defer a.free(su);
    const us = try a.alloc(u8, sg.len * 2); // [E][2][NI/128][D/128]: gate then up, per expert
    defer a.free(us);
    const per = sg.len / e;
    for (0..e) |x| {
        @memcpy(us[2 * x * per ..][0..per], sg[x * per ..][0..per]);
        @memcpy(us[(2 * x + 1) * per ..][0..per], su[x * per ..][0..per]);
    }
    const want_us = try fx.bytes("packed_up_scale");
    defer a.free(want_us);
    try check.sameBytes("packed_up_scale", us, want_us);
    const ds = try fx.bytes("down_scale");
    defer a.free(ds);
    check.pass("fp8 experts: {d} x [{d}, {d}] gate-up, down and scales packed as make() packs them", .{ e, ni, d });

    var dup = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pu));
    defer dup.free();
    var ddown = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pd));
    defer ddown.free();
    var dus = try cuda.DeviceBuffer.fromHost(gpu.d, us);
    defer dus.free();
    var dds = try cuda.DeviceBuffer.fromHost(gpu.d, ds);
    defer dds.free();
    const layer: ex_.Layer = .{ .format = .fp8g, .up = dup.ptr, .down = ddown.ptr, .up_scale = dus.ptr, .down_scale = dds.ptr, .width = ni, .dims = d, .experts = e };
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try fx8.load(gpu.d, sms);
    defer ex.unload();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();

    inline for (.{ .{ "decode", false }, .{ "prompt", true } }) |set| {
        var it = std.mem.tokenizeScalar(u8, try fx.string(set[0]), ',');
        while (it.next()) |tok| {
            const rows = try std.fmt.parseInt(usize, tok, 10);
            try oneCase(gpu, fx, router, .{ .fp8g = &ex }, stream, layer, rows, slots, set[1]);
        }
    }
}

fn oneCase(gpu: Gpu, fx: Fixture, router: grouped.Router, g: ex_.Grouped, s: cuda.Stream, l: ex_.Layer, rows: usize, slots: usize, prompt: bool) !void {
    const a = gpu.gpa;
    const pairs = rows * slots;
    const picks = try arr(fx, "picks{d}", .{rows});
    defer a.free(picks);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, picks);
    defer dpicks.free();
    const t: usize = if (prompt) ex_.promptTile(.fp8g) else 16;
    const wide = pairs > grouped.small;
    const n_max = grouped.maxItems(pairs, l.experts, 16);
    var bufs: [5]cuda.DeviceBuffer = undefined;
    const sizes = [5]usize{ pairs * 4, n_max * 12, 8, (if (wide) pairs else 1) * 4, (if (wide) (pairs + 1023) / 1024 * l.experts else 1) * 4 };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, &bufs) |n, *b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
        made += 1;
    }
    const plan: grouped.Plan = .{ .members = bufs[0].ptr, .items = bufs[1].ptr, .counts = bufs[2].ptr, .rank = bufs[3].ptr, .hist = bufs[4].ptr };
    try router.route(s, dpicks.ptr, pairs, l.experts, t, plan);
    try s.synchronize();

    const counts = try check.download(gpu, bufs[2]);
    defer a.free(counts);
    const want_counts = try arr(fx, "counts{d}", .{rows});
    defer a.free(want_counts);
    try check.sameBytes("plan counts", counts, want_counts);
    const n_items: usize = @intCast(std.mem.bytesToValue(i32, counts[0..4]));
    const items = try check.download(gpu, bufs[1]);
    defer a.free(items);
    const want_items = try arr(fx, "items{d}", .{rows});
    defer a.free(want_items);
    try check.sameBytes("plan items", items[0 .. n_items * 12], want_items[0 .. n_items * 12]);
    const members = try check.download(gpu, bufs[0]);
    defer a.free(members);
    const want_members = try arr(fx, "members{d}", .{rows});
    defer a.free(want_members);
    try check.sameBytes("plan members", members, want_members);

    const x = try arr(fx, "x{d}", .{rows});
    defer a.free(x);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
    defer dx.free();
    var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.width * 2);
    defer act.free();
    const bound = grouped.maxItems(pairs, l.experts, t);
    try act.fill8(0xff, s.handle); // a call that writes nothing fails rather than pass on old bytes
    const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = l.dims, .slots = slots };
    if (prompt) try g.prompt(s, .up, up_in, l, plan, act.ptr, bound, -1) else try g.decode(s, .up, up_in, l, plan, act.ptr, bound, -1);
    try s.synchronize();
    const got_act = try check.download(gpu, act);
    defer a.free(got_act);
    const want_act = try arr(fx, "act{d}", .{rows});
    defer a.free(want_act);
    try check.sameBytes("gate-up", got_act, want_act);

    const width: usize = if (prompt) 2 else 4;
    var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * width);
    defer y.free();
    try y.fill8(0xff, s.handle);
    const down_in: ex_.Rows = .{ .x = act.ptr, .stride = l.width };
    if (prompt) try g.prompt(s, .down_bf16, down_in, l, plan, y.ptr, bound, -1) else try g.decode(s, .down_f32, down_in, l, plan, y.ptr, bound, -1);
    try s.synchronize();
    const got_y = try check.download(gpu, y);
    defer a.free(got_y);
    const want_y = try arr(fx, "y{d}", .{rows});
    defer a.free(want_y);
    try check.sameBytes("down", got_y, want_y);
    check.pass("fp8 experts: {d} rows x {d} slots ({s}, {d} items) plan, up and down via experts.Grouped equal Python's bytes", .{ rows, slots, if (prompt) "prompt" else "decode", n_items });
}
