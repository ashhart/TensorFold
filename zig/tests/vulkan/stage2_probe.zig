//! Vulkan stage 2 bring-up: BF16 residual add, SwiGLU and RMSNorm (Metal-oracle ASH_QWEN35
//! semantics) through the production NormMlp operator, checked on a real GPU.
//! Exit 0 pass, 1 fail, 2 usage, 124 fence timeout.
//! Operator bring-up only: no model, no performance claim, and no cross-backend bitexact claim —
//! the fixture states each gate's numerical acceptance.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage2_shaders");
const fx = @import("stage2_fixture.zig");
const contract = @import("contract");
const cx = contract.spec; // toBf16, fromBf16, the C constants

const usage = "usage: tf-stage2-gpu [--vendor ID] [--timeout-ms MS] | --help\n" ++
    "  --help          show this text and exit without opening a loader or device\n" ++
    "  --vendor ID     refuse a device whose Vulkan vendorID differs (decimal or 0x hex; no default)\n" ++
    "  --timeout-ms MS bound on every fence wait, 1 to 600000 (default 10000); a timeout ends the process\n";

/// Milliseconds on the awake clock since main started, printed on every milestone.
const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

const repeats = 3;
const planted_bits = [_]u16{ 0x7fc0, 0x7f80, 0xff80 };
const planted = [_][]const u8{ "nan", "+inf", "-inf" };

/// The first output whose reference is finite (u16 bits or f64): where a planted nonfinite must be refused.
fn firstFinite(want: anytype) ?usize {
    for (want, 0..) |w, i| {
        const v = if (@TypeOf(w) == u16) cx.fromBf16(w) else w;
        if (std.math.isFinite(v)) return i;
    }
    return null;
}

pub fn main(init: std.process.Init) !u8 {
    const clock: Elapsed = .{ .io = init.io, .start = std.Io.Clock.awake.now(init.io) };
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args[1..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print(usage, .{});
        return 0;
    };
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
    } else return if (try probe(init.gpa, clock, vendor, timeout_ms * std.time.ns_per_ms)) 0 else 1;
    std.debug.print(usage, .{});
    return 2;
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) !bool {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const types = [_][]const u8{ "other", "integrated", "discrete", "virtual", "cpu" };
    const device_kind = if (pr.deviceType < types.len) types[pr.deviceType] else "unknown";
    const v = pr.apiVersion;
    const driver = .{ std.mem.sliceTo(&dev.v12.driverName, 0), std.mem.sliceTo(&dev.v12.driverInfo, 0) };
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} " ++
        "api={d}.{d}.{d} driver=\"{s}\" info=\"{s}\" subgroup={d} storage16={} int16={} " ++
        "max-storage-range={d} max-buffer-size={d} fence-timeout={d}ms\n", .{
        clock.ms(),
        name,
        pr.vendorID,
        pr.deviceID,
        device_kind,
        v >> 22 & 0x7f,
        v >> 12 & 0x3ff,
        v & 0xfff,
    } ++ driver ++ .{ dev.v11.subgroupSize, dev.features.storage16, dev.features.int16, dev.properties.limits.maxStorageBufferRange, dev.v13.maxBufferSize, timeout / std.time.ns_per_ms });
    if (pr.deviceType == vk.c.VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std.debug.print("FAIL device: a CPU (software) Vulkan device is never a GPU result\n", .{});
        return false;
    }
    if (vendor) |want| if (pr.vendorID != want) {
        std.debug.print("FAIL device: vendor 0x{x:0>4} is not the requested 0x{x:0>4}\n", .{ pr.vendorID, want });
        return false;
    };

    var cmds = try vk.Commands.init(&dev);
    defer cmds.deinit();
    var staging = try vk.Buffer.init(&dev, .staging, 64 << 10);
    defer staging.deinit();
    // The compiled stage 2 closure: norm and swiglu carry the Ash8c Metal-oracle variant,
    // add the plain bf16 form; all three built with -DUSE_BF16 (see stage2-gpu-build).
    var m = try vk.norm_mlp.NormMlp.init(&dev, .{
        .norm_bf16 = &shaders.norm_bf16,
        .add_bf16 = &shaders.add_bf16,
        .swiglu_bf16 = &shaders.swiglu_bf16,
    });
    defer m.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    for (fx.add_cases) |case| {
        total += 1;
        passed += @intFromBool(try addGate(a, clock, &dev, &cmds, staging, &m, case, timeout));
    }
    for (fx.swiglu_cases) |case| {
        total += 1;
        passed += @intFromBool(try swigluGate(a, clock, &dev, &cmds, staging, &m, case, timeout));
    }
    for (fx.rms_cases) |case| {
        total += 1;
        passed += @intFromBool(try rmsGate(a, clock, &dev, &cmds, staging, &m, case, timeout));
    }
    std.debug.print("RESULT stage2 elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}

fn uploaded(_: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, bytes: []const u8, timeout: u64) !vk.Buffer {
    var b = try vk.Buffer.init(dev, .device, bytes.len);
    errdefer b.deinit();
    try vk.orExit(vk.upload(cmds, b, 0, staging, bytes, timeout));
    return b;
}

fn fetch(cmds: *vk.Commands, staging: vk.Buffer, b: vk.Buffer, out: []u8, timeout: u64) !void {
    try vk.orExit(vk.download(cmds, b, 0, staging, out, timeout));
}

fn bitsBuffer(a: std.mem.Allocator, shell: []const u16) ![]u16 {
    return a.alloc(u16, shell.len);
}

/// One bf16 binding's shell uploaded at its fixture offsets; the view is derived against the
/// caller-resident Bound16 (a stored View would point at boundShell's dead stack frame).
const Bound16 = struct {
    buf: vk.Buffer,
    byte_offset: u64,
    elements: u64,

    /// The checked window over this binding's own buffer, derived against caller storage
    /// (a stored View would point at boundShell's dead stack frame).
    fn view(b: *const Bound16) !vk.tensor.View {
        return vk.tensor.View.init(&b.buf, .bf16, b.byte_offset, b.elements);
    }
};

fn boundShell(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, at_elems: usize, elements: u64, timeout: u64) !Bound16 {
    var buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(shell), timeout);
    errdefer buf.deinit();
    return .{ .buf = buf, .byte_offset = at_elems * 2, .elements = elements };
}

fn checkUnchanged(a: std.mem.Allocator, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, buf: vk.Buffer, timeout: u64) !bool {
    const back = try a.alloc(u8, std.mem.sliceAsBytes(shell).len);
    defer a.free(back);
    try fetch(cmds, staging, buf, back, timeout);
    return std.mem.indexOfDiff(u8, std.mem.sliceAsBytes(shell), back) == null;
}

/// Out = a + b matching the shader's own pipeline bit for bit (host f32 add, one RNE bf16
/// store): three byte-identical repeats, planted nonfinite refused at a finite element, inputs
/// and poison guards unchanged.
fn addGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.norm_mlp.NormMlp, case: fx.AddCase, timeout: u64) !bool {
    const f = try fx.Add.init(a, case, 0x57a9);
    defer f.deinit();
    var in_a = try boundShell(a, dev, cmds, staging, f.a_bits, case.at.a, case.elems, timeout);
    defer in_a.buf.deinit();
    var in_b = try boundShell(a, dev, cmds, staging, f.b_bits, case.at.b, case.elems, timeout);
    defer in_b.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, case.elems, timeout);
    defer out.buf.deinit();
    const bits = try bitsBuffer(a, f.out);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(m.add(cmds, try out.view(), try in_a.view(), try in_b.view(), timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL residual-add case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else {
            first = try a.dupe(u16, bits);
        }
    }
    const mm = f.firstMismatch(bits) catch |e| {
        std.debug.print("FAIL residual-add case=\"{s}\": {s}\n", .{ case.name, @errorName(e) });
        return false;
    };
    if (mm) |bad| {
        std.debug.print("FAIL residual-add case=\"{s}\": {s} at element {d}: a=0x{x:0>4} ({s}) b=0x{x:0>4} ({s}) want=0x{x:0>4} ({s}) got=0x{x:0>4} ({s})\n", .{ case.name, @errorName(bad.kind), bad.index, bad.a, fx.classify(bad.a), bad.b, fx.classify(bad.b), bad.want, fx.classify(bad.want), bad.got, fx.classify(bad.got) });
        return false;
    }
    // A planted nonfinite output at a finite-want element must be refused by the same checker.
    const data_at = case.at.out + firstFinite(f.want).?;
    var mutated = try a.dupe(u16, bits);
    defer a.free(mutated);
    for (planted_bits, planted) |pb, pname| {
        mutated[data_at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL residual-add case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |_| {}
    }
    if (!try checkUnchanged(a, cmds, staging, f.a_bits, in_a.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.b_bits, in_b.buf, timeout)) {
        std.debug.print("FAIL residual-add case=\"{s}\": an input binding changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE residual-add elapsed_ms={d} case=\"{s}\" elements={d} matches f32-add + RNE-store pipeline bit for bit, repeats byte-identical, " ++
        "inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, case.elems });
    return true;
}

/// Out = silu(gate) * up within the stated bound: three byte-identical repeats, unchanged inputs
/// and guards, planted nonfinite refused.
fn swigluGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.norm_mlp.NormMlp, case: fx.SwigluCase, timeout: u64) !bool {
    const f = try fx.Swiglu.init(a, case, 0x57aa);
    defer f.deinit();
    var gate = try boundShell(a, dev, cmds, staging, f.gate, case.at.a, case.elems, timeout);
    defer gate.buf.deinit();
    var up = try boundShell(a, dev, cmds, staging, f.up, case.at.b, case.elems, timeout);
    defer up.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, case.elems, timeout);
    defer out.buf.deinit();
    const bits = try bitsBuffer(a, f.out);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(m.swiglu(cmds, try out.view(), try gate.view(), try up.view(), timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| {
            // No cached worst/position here: r belongs to the last PASSING repeat, and printing
            // it after a failure would misreport. The fixture error names the gate that failed.
            std.debug.print("FAIL swiglu case=\"{s}\" repeat={d}: {s}\n", .{ case.name, rep, @errorName(e) });
            return false;
        };
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL swiglu case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else {
            first = try a.dupe(u16, bits);
        }
    }
    var mutated = try a.dupe(u16, bits);
    defer a.free(mutated);
    const data_at = case.at.out + firstFinite(f.want).?;
    for (planted_bits, planted) |pb, pname| {
        mutated[data_at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL swiglu case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |_| {}
    }
    if (!try checkUnchanged(a, cmds, staging, f.gate, gate.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.up, up.buf, timeout)) {
        std.debug.print("FAIL swiglu case=\"{s}\": an input binding changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE swiglu elapsed_ms={d} case=\"{s}\" elements={d} worst={d:.4} of bound at element {d}, " ++
        "repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, case.elems, r.worst, r.col });
    return true;
}

/// RMSNorm through the production rmsNorm (which runs normPlan): contract bound with the named
/// f32 reduction budget, three byte-identical repeats, unchanged inputs and guards, planted
/// nonfinite refused, and the zero row's outputs exactly zero.
fn rmsGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.norm_mlp.NormMlp, case: fx.RmsCase, timeout: u64) !bool {
    const f = try fx.Rms.init(a, case, 0x57ab);
    defer f.deinit();
    var x = try boundShell(a, dev, cmds, staging, f.x, case.at.x, case.rows * case.width, timeout);
    defer x.buf.deinit();
    var gamma = try boundShell(a, dev, cmds, staging, f.gamma, case.at.gamma, case.width, timeout);
    defer gamma.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, case.rows * case.width, timeout);
    defer out.buf.deinit();
    const bits = try bitsBuffer(a, f.out);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |xx| a.free(xx);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(m.rmsNorm(cmds, try out.view(), try x.view(), try gamma.view(), case.rows, case.width, fx.rms_eps, timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| {
            // No cached worst/position here: r belongs to the last PASSING repeat, and printing
            // it after a failure would misreport. The fixture error names the gate that failed.
            std.debug.print("FAIL rms-norm case=\"{s}\" repeat={d}: {s}\n", .{ case.name, rep, @errorName(e) });
            return false;
        };
        if (first) |xx| {
            if (std.mem.indexOfDiff(u16, xx, bits)) |at| {
                std.debug.print("FAIL rms-norm case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else {
            first = try a.dupe(u16, bits);
        }
    }
    var mutated = try a.dupe(u16, bits);
    defer a.free(mutated);
    const data_at = case.at.out + r.row * case.width + r.col;
    for (planted_bits, planted) |pb, pname| {
        mutated[data_at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL rms-norm case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |_| {}
    }
    if (!try checkUnchanged(a, cmds, staging, f.x, x.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.gamma, gamma.buf, timeout)) {
        std.debug.print("FAIL rms-norm case=\"{s}\": an input binding changed\n", .{case.name});
        return false;
    }
    if (case.zero_row) {
        const zero_at = case.at.out + (case.rows - 1) * case.width;
        for (bits[zero_at .. zero_at + case.width]) |g| if (g != 0x0000 and g != 0x8000) {
            std.debug.print("FAIL rms-norm case=\"{s}\": the zero row produced 0x{x:0>4}\n", .{ case.name, g });
            return false;
        };
    }
    std.debug.print("MILESTONE rms-norm elapsed_ms={d} case=\"{s}\" rows={d} width={d} worst={d:.4} of bound at row {d} col {d}, " ++
        "repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused{s}\n", .{ clock.ms(), case.name, case.rows, case.width, r.worst, r.row, r.col, if (case.zero_row) ", zero row exactly zero" else "" });
    return true;
}

/// A host-only stand-in for a device buffer: View.init reads only `len`, and this regression
/// exists to prove the view derivation stays on caller storage, not to touch the driver.
fn hostBuffer(len: u64) vk.Buffer {
    return .{ .dev = undefined, .handle = null, .memory = null, .len = len, .kind = .device, .mapped = null, .coherent = false };
}

test "Bound16.view binds the caller-owned buffer, never a factory stack frame" {
    const b1: Bound16 = .{ .buf = hostBuffer(64), .byte_offset = 8, .elements = 28 };
    const b2: Bound16 = .{ .buf = hostBuffer(128), .byte_offset = 16, .elements = 56 };
    const v1 = try b1.view();
    const v2 = try b2.view();
    try std.testing.expect(v1.buffer == &b1.buf);
    try std.testing.expect(v2.buffer == &b2.buf);
    try std.testing.expect(v1.buffer != v2.buffer);
    try std.testing.expectEqual(@as(u64, 8), v1.byte_offset);
    try std.testing.expectEqual(@as(u64, 28), v1.elements);
    try std.testing.expectEqual(vk.tensor.Kind.bf16, v1.kind);
}

test "bound window keeps the offset, range and alignment checks" {
    const buf = hostBuffer(64);
    try std.testing.expectError(error.OutOfRange, vk.tensor.View.init(&buf, .bf16, 0, 0));
    try std.testing.expectError(error.Misaligned, vk.tensor.View.init(&buf, .bf16, 2, 4));
    try std.testing.expectError(error.OutOfRange, vk.tensor.View.init(&buf, .bf16, 8, 30)); // past the 64-byte end
    const b: Bound16 = .{ .buf = hostBuffer(64), .byte_offset = 8, .elements = 28 };
    _ = try b.view(); // the derived window is the checked one
}
