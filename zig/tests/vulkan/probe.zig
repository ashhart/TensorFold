//! Vulkan Q4/FP64 operator bring-up, not model qualification. Exit 0 pass, 1 fail, 2 usage, 124 fence timeout.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("vulkan_shaders");
const fx = @import("affine_fixture.zig");
const c = vk.c;
const dispatch = vk.dispatch;

const usage = "usage: tf-vulkan-probe [--vendor ID] [--timeout-ms MS]\n" ++
    "  --vendor ID      refuse a device from any other PCI vendor (Apple is 0x106b)\n" ++
    "  --timeout-ms MS  bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";
const repeats = 5;
const staging_bytes = 64 << 10;

/// Milliseconds on the awake clock since main started, printed on every milestone.
const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

pub fn main(init: std.process.Init) !u8 {
    const clock: Elapsed = .{ .io = init.io, .start = std.Io.Clock.awake.now(init.io) };
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var vendor: ?u32 = null;
    var timeout_ms: u64 = 10_000;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, args[i], "--vendor")) {
            vendor = std.fmt.parseInt(u32, value, 0) catch null;
            if (vendor == null) break;
        } else if (std.mem.eql(u8, args[i], "--timeout-ms")) {
            timeout_ms = std.fmt.parseInt(u64, value, 10) catch 0;
            if (timeout_ms == 0 or timeout_ms > 600_000) break;
        } else break;
    } else return if (try probe(init.gpa, clock, vendor, timeout_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout_ms: u64) !bool {
    const timeout = timeout_ms * std.time.ns_per_ms;
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{}); // refuses CPU (software) devices and Vulkan below 1.3
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    const v = pr.apiVersion;
    const driver = .{ std.mem.sliceTo(&dev.v12.driverName, 0), std.mem.sliceTo(&dev.v12.driverInfo, 0) };
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} " ++
        "api={d}.{d}.{d} driver=\"{s}\" info=\"{s}\" subgroup={d} fence-timeout={d}ms\n", .{
        clock.ms(),
        name,
        pr.vendorID,
        pr.deviceID,
        kind,
        v >> 22 & 0x7f,
        v >> 12 & 0x3ff,
        v & 0xfff,
    } ++ driver ++ .{ dev.v11.subgroupSize, timeout_ms });
    if (pr.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return false;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return false;
    };

    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, staging_bytes);
    defer staging.deinit();
    if (!try roundTrip(a, clock, &dev, &cmds, staging, timeout)) return false;

    const layout: dispatch.Layout = .{ .bindings = 5, .push_bytes = @sizeOf(fx.Params), .writes = 1 << 4 };
    var pipe = try dispatch.Pipeline.init(&dev, &shaders.qmm, layout);
    defer pipe.deinit();
    if (!std.mem.eql(u32, &pipe.local, &[_]u32{ fx.local_size_x, 1, 1 })) {
        std.debug.print("FAIL pipeline: qmm.spv LocalSize {any}, the fixture dispatches {d}x1x1\n", .{ pipe.local, fx.local_size_x });
        return false;
    }
    var passed: usize = 0;
    for (fx.cases, 0..) |case, n| {
        passed += @intFromBool(try affineCase(a, clock, &dev, &cmds, staging, &pipe, case, 0x766b + n, timeout));
    }
    const say = "RESULT vulkan-probe elapsed_ms={d} on \"{s}\": round trip equal; {d}/{d} qmm_f32 affine q4 g64 cases " ++
        "within the FP64 bound, {d} repeats each (operator bring-up, not a kernel-contract SUMMARY)\n";
    std.debug.print(say, .{ clock.ms(), name, passed, fx.cases.len, repeats });
    return passed == fx.cases.len;
}

/// Random bytes through a device-local buffer in several staging chunks and back.
fn roundTrip(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, timeout: u64) !bool {
    const sent = try a.alloc(u8, 3 * staging_bytes + 20);
    defer a.free(sent);
    const back = try a.alloc(u8, sent.len);
    defer a.free(back);
    var prng = std.Random.DefaultPrng.init(0x766b);
    prng.random().bytes(sent);
    @memset(back, 0);
    var buf = try vk.Buffer.init(dev, .device, sent.len);
    defer buf.deinit();
    try vk.orExit(vk.upload(cmds, buf, 0, staging, sent, timeout));
    try vk.orExit(vk.download(cmds, buf, 0, staging, back, timeout));
    if (std.mem.indexOfDiff(u8, sent, back)) |at| {
        std.debug.print("FAIL roundtrip: {d} bytes, first difference at byte {d}\n", .{ sent.len, at });
        return false;
    }
    std.debug.print("MILESTONE roundtrip-equal elapsed_ms={d} bytes={d} staging={d}\n", .{ clock.ms(), sent.len, staging_bytes });
    return true;
}

/// Five byte-identical FP64-bounded repeats, unchanged inputs/guards, and nonfinite-output negative controls.
fn affineCase(
    a: std.mem.Allocator,
    clock: Elapsed,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    pipe: *dispatch.Pipeline,
    case: fx.Case,
    seed: u64,
    timeout: u64,
) !bool {
    const f = try fx.Fixture.init(a, case, seed);
    defer f.deinit();
    const bytesOf = std.mem.sliceAsBytes;
    const data = [5][]const u8{ bytesOf(f.x), bytesOf(f.w), bytesOf(f.scales), bytesOf(f.biases), bytesOf(f.out) };
    var bufs: [5]vk.Buffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.deinit();
    var binds: [5]dispatch.Binding = undefined;
    for (data, &bufs, &binds) |bytes, *b, *bind| {
        b.* = try vk.Buffer.init(dev, .device, bytes.len);
        made += 1;
        bind.* = .{ .buffer = b, .len = bytes.len };
    }
    for (data[0..4], bufs[0..4]) |bytes, b| try vk.orExit(vk.upload(cmds, b, 0, staging, bytes, timeout));
    try pipe.bind(&binds);
    const p = f.params();
    const groups = f.groupCount();
    const got = try a.alloc(f32, f.out.len);
    defer a.free(got);
    const first = try a.alloc(u8, data[4].len);
    defer a.free(first);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, bufs[4], 0, staging, data[4], timeout));
        try vk.orExit(pipe.run(cmds, std.mem.asBytes(&p), .{ groups, 1, 1 }, timeout));
        try vk.orExit(vk.download(cmds, bufs[4], 0, staging, bytesOf(got), timeout));
        r = f.check(got) catch |e| {
            const m = f.measure(got);
            const out = m.row * p.matrix_n + m.col;
            const say = "FAIL affine-fp64 case=\"{s}\" repeat={d}: {s}; worst {d:.3} of bound at row {d} col {d}: " ++
                "got {e} want {e}\n";
            std.debug.print(say, .{ case.name, rep, @errorName(e), m.worst, m.row, m.col, got[p.output_offset + out], f.want[out] });
            return false;
        };
        if (rep == 0) {
            @memcpy(first, bytesOf(got));
        } else if (std.mem.indexOfDiff(u8, first, bytesOf(got))) |at| {
            std.debug.print("FAIL affine-fp64 case=\"{s}\" repeat={d}: byte {d} differs from the first run\n", .{ case.name, rep, at });
            return false;
        }
    }
    const back = try a.alloc(u8, @max(data[0].len, data[1].len, data[2].len, data[3].len));
    defer a.free(back);
    for (data[0..4], bufs[0..4], 0..) |bytes, b, n| {
        try vk.orExit(vk.download(cmds, b, 0, staging, back[0..bytes.len], timeout));
        if (std.mem.indexOfDiff(u8, bytes, back[0..bytes.len])) |at| {
            std.debug.print("FAIL affine-fp64 case=\"{s}\": binding {d} changed at byte {d}\n", .{ case.name, n, at });
            return false;
        }
    }
    const at = p.output_offset + r.row * p.matrix_n + r.col;
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |planted| {
        got[at] = planted;
        if (f.check(got)) |_| {
            std.debug.print("FAIL control case=\"{s}\": a planted {d} output passed the FP64 check\n", .{ case.name, planted });
            return false;
        } else |e| if (e != error.OutOfBound) {
            std.debug.print("FAIL control case=\"{s}\": planted {d} gave {s}, not OutOfBound\n", .{ case.name, planted, @errorName(e) });
            return false;
        }
    }
    const say = "MILESTONE affine-fp64 elapsed_ms={d} case=\"{s}\" groups={d} worst={d:.4} of bound at row {d} col {d}, " ++
        "max_abs={e} repeats={d} byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n";
    std.debug.print(say, .{ clock.ms(), case.name, groups, r.worst, r.row, r.col, r.max_abs, repeats });
    return true;
}
