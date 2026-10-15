//! `ling-pack DIR LAYER [EXPERTS]`: a MoE layer's experts loaded and timed, bytes vs the host packers the loader used.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ling = @import("ling_cuda");
const check = @import("check.zig");
const lw = @import("ling_weights_tests.zig");

const W = ling.weights;
const X8 = cuda.experts.Fp8Experts;
const X4 = cuda.experts.Nvfp4Experts;

/// Host memory from /proc/self/status: the peak resident set since the last reset, and anonymous memory now (KiB).
const Host = struct {
    hwm: usize = 0,
    anon: usize = 0,

    fn read(io: std.Io) Host {
        var buf: [8192]u8 = undefined;
        const text = std.Io.Dir.cwd().readFile(io, "/proc/self/status", &buf) catch return .{};
        var h: Host = .{};
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            inline for (.{ .{ "VmHWM:", "hwm" }, .{ "RssAnon:", "anon" } }) |kv| if (std.mem.startsWith(u8, line, kv[0])) {
                var it = std.mem.tokenizeAny(u8, line[kv[0].len..], " \tkB");
                @field(h, kv[1]) = std.fmt.parseInt(usize, it.next() orelse "0", 10) catch 0;
            };
        }
        return h;
    }

    /// Resets VmHWM to the current resident set (proc(5) clear_refs 5).
    fn reset(io: std.Io) void {
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "/proc/self/clear_refs", .data = "5" }) catch {};
    }
};

fn secs(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(check.now(io) - t0)) / 1e9;
}

/// Every byte of layer i's routed experts 0..count read once from the mapping (the read alone, no device work).
fn touch(ck: *core.Checkpoint, i: u32, count: usize) !usize {
    var sum: u64 = 0;
    var bytes: usize = 0;
    var b: [64]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.experts.", .{i});
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (!std.mem.startsWith(u8, k, prefix)) continue;
        const rest = k[prefix.len..];
        const e = std.fmt.parseInt(usize, rest[0 .. std.mem.indexOfScalar(u8, rest, '.') orelse continue], 10) catch continue;
        if (e >= count) continue;
        const t = f.get(k).?;
        for (std.mem.bytesAsSlice(u64, @as([]align(1) const u8, t.bytes[0 .. t.bytes.len / 8 * 8]))) |w| sum +%= w;
        bytes += t.bytes.len;
    };
    std.mem.doNotOptimizeAway(sum);
    return bytes;
}

/// One expert's packed device words against the host packers' bytes from the same file tensors.
fn compare(gpu: check.Gpu, ck: *core.Checkpoint, i: u32, l: cuda.experts.Layer, a: std.mem.Allocator) !void {
    const ni = l.width;
    const d = l.dims;
    const nvfp4 = l.format == .nvfp4;
    const up_n = if (nvfp4) X4.bytes(1, ni, d, 2) / 4 else 2 * ni * d / 4;
    const down_n = if (nvfp4) X4.bytes(1, d, ni, 1) / 4 else d * ni / 4;
    const inv_n = (ni / 128) * (d / 128);
    const hu = try a.alloc(u32, up_n);
    const hd = try a.alloc(u32, down_n);
    const gu = try a.alloc(u32, up_n);
    const gd = try a.alloc(u32, down_n);
    const scale_up = try a.alloc(u8, l.experts * (if (nvfp4) 2 * 4 else 2 * inv_n * 4));
    const scale_down = try a.alloc(u8, l.experts * (if (nvfp4) 4 else inv_n * 4));
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = l.up_scale, .len = scale_up.len }).download(0, scale_up);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = l.down_scale, .len = scale_down.len }).download(0, scale_down);
    var b: [3][192]u8 = undefined;
    for (0..l.experts) |e| {
        var m: [3][]const u8 = undefined;
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| m[j] = try std.fmt.bufPrint(&b[j], "model.layers.{d}.mlp.experts.{d}.{s}", .{ i, e, p });
        if (nvfp4) {
            var x: [3]core.nvfp4.Linear = undefined;
            for (&x, m) |*y, name| y.* = try core.nvfp4.Linear.read(ck, name);
            X4.packGateUp(hu, .{ x[0].codes, x[0].scales }, .{ x[1].codes, x[1].scales }, ni, d);
            X4.packOne(hd, x[2].codes, x[2].scales, d, ni);
            const globals = [3]f32{ x[0].global, x[1].global, x[2].global };
            try check.sameBytes("nvfp4 gate and up globals", scale_up[8 * e ..][0..8], std.mem.sliceAsBytes(globals[0..2]));
            try check.sameBytes("nvfp4 down global", scale_down[4 * e ..][0..4], std.mem.sliceAsBytes(globals[2..3]));
        } else {
            var t: [3]core.checkpoint.Tensor = undefined;
            var s: [3]core.checkpoint.Tensor = undefined;
            for (&t, &s, m) |*w, *sc, name| {
                var nb: [200]u8 = undefined;
                w.* = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{name}));
                sc.* = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{name}));
            }
            try X8.packGateUp(a, hu, t[0].bytes, t[1].bytes, ni, d);
            X8.packOne(hd, t[2].bytes, d, ni);
            try check.sameBytes("fp8 gate scale_inv", scale_up[(2 * e) * inv_n * 4 ..][0 .. inv_n * 4], s[0].bytes);
            try check.sameBytes("fp8 up scale_inv", scale_up[(2 * e + 1) * inv_n * 4 ..][0 .. inv_n * 4], s[1].bytes);
            try check.sameBytes("fp8 down scale_inv", scale_down[e * inv_n * 4 ..][0 .. inv_n * 4], s[2].bytes);
        }
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = l.up, .len = l.experts * up_n * 4 }).download(e * up_n * 4, std.mem.sliceAsBytes(gu));
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = l.down, .len = l.experts * down_n * 4 }).download(e * down_n * 4, std.mem.sliceAsBytes(gd));
        var nb: [64]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&nb, "expert {d} gate-up words", .{e}), std.mem.sliceAsBytes(gu), std.mem.sliceAsBytes(hu));
        try check.sameBytes(try std.fmt.bufPrint(&nb, "expert {d} down words", .{e}), std.mem.sliceAsBytes(gd), std.mem.sliceAsBytes(hd));
    }
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const i = try std.fmt.parseInt(u32, rest[1], 10);
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const io = gpu.io;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var kit: lw.Kit = undefined;
    try kit.init(gpu);
    defer kit.deinit();
    var why: ling.config.Why = .{};
    var o = W.open(gpu.gpa, io, dir, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ dir, why.text() });
        return e;
    };
    defer o.ck.close();
    if (!o.c.isMoe(i)) return error.NotMoeLayer;
    const count = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else o.c.experts;
    const opts: W.Options = .{ .globals = false, .experts = count };
    // the read alone from a cold cache, then a load on hot pages (pack and upload), then a cold load (both)
    o.ck.release();
    var t0 = check.now(io);
    const bytes = try touch(&o.ck, i, count);
    const read_s = secs(io, t0);
    var hot_s: f64 = 0;
    var cold_s: f64 = 0;
    var peak: [2]usize = .{ 0, 0 };
    var host: [2]Host = undefined;
    for (0..2) |pass| {
        if (pass == 1) o.ck.release();
        Host.reset(io);
        const base = cuda.usage(true).device;
        t0 = check.now(io);
        var w = try W.load(gpu.gpa, gpu.d, s, &kit.pack, &o.ck, o.c, &.{i}, opts);
        defer w.deinit();
        if (pass == 0) hot_s = secs(io, t0) else cold_s = secs(io, t0);
        peak[pass] = cuda.usage(false).peak - base;
        host[pass] = Host.read(io);
        if (pass == 1) {
            const l = w.layer(i).?.ffn.moe.routed.grouped;
            try compare(gpu, &o.ck, i, l, arena.allocator());
            std.debug.print("RESULT layer {d}: {d} experts {t}, {d:.2} GiB of expert tensors\n", .{ i, count, l.format, @as(f64, @floatFromInt(bytes)) / (1 << 30) });
        }
    }
    std.debug.print("RESULT   read alone {d:.2} s ({d:.0} MiB/s); load on hot pages {d:.2} s; cold load {d:.2} s\n", .{ read_s, @as(f64, @floatFromInt(bytes >> 20)) / read_s, hot_s, cold_s });
    std.debug.print("RESULT   device peak above the start {d} MiB (hot), {d} MiB (cold); host VmHWM {d} MiB (hot), {d} MiB (cold), RssAnon {d} MiB\n", .{ peak[0] >> 20, peak[1] >> 20, host[0].hwm >> 10, host[1].hwm >> 10, host[1].anon >> 10 });
    check.pass("ling-pack layer {d}: every expert's packed words and scales equal the host packers' bytes", .{i});
}
