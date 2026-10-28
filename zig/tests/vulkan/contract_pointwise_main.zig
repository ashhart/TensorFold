//! tf-contract-pointwise: the pointwise shader contract on a real Vulkan GPU — the six
//! closures of stages 2-4 through their production pipeline shapes: RMSNorm, residual add
//! and SwiGLU through NormMlp; the f32-preserving residual add, the attention output gate
//! and the gated RMSNorm through the same dispatch model their wrappers use (identical
//! binding counts, push-block sizes and group math; the wrappers own the plans via
//! attention.gatePlan and gdn.gatedNormPlan).
//! Exit 0 pass, 1 fail, 2 usage, 124 fence timeout. Acceptance per gate is stated in the
//! fixtures; no model, no performance claim, no cross-backend bitexact claim.
const std = @import("std");
const fx = @import("contract_pointwise.zig");
const fx2 = @import("stage2_fixture.zig");
const contract = @import("contract");
const vk = @import("vulkan");
const shaders2 = @import("stage2_shaders");
const shaders3 = @import("stage3_shaders");
const shaders4 = @import("stage4_shaders");

const cx = contract.spec;

const usage = "usage: tf-contract-pointwise [--vendor ID] [--timeout-ms MS] | --help\n" ++
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

/// The first output whose reference is finite (u16 bits, u32 bits or f64): where a planted
/// nonfinite must be refused.
fn firstFinite(want: anytype) ?usize {
    for (want, 0..) |w, i| {
        const v: f64 = switch (@TypeOf(w)) {
            f64 => w,
            u16 => cx.fromBf16(w),
            u32 => @as(f32, @bitCast(w)),
            else => @compileError("want elements are f64, u16 or u32"),
        };
        if (std.math.isFinite(v)) return i;
    }
    return null;
}

/// attention_gate.comp's one-word push block (the wrapper's GateParams, local because the
/// wrapper keeps it private).
const GateParams = extern struct { words: u32 = 0 };
comptime {
    if (@sizeOf(GateParams) != 4) @compileError("attention_gate push block must stay one word");
    if (gated_push_bytes != @offsetOf(vk.params.Params, "beta") + 4) @compileError("fast_norm_gated push block drifted from Params");
}
/// fast_norm_gated.comp's push block ends at `beta`, exactly the wrapper's norm_push_bytes.
const gated_push_bytes: u32 = 68;

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
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} type={s} api={d}.{d}.{d} " ++
        "storage16={} int16={} fence-timeout={d}ms\n", .{ clock.ms(), name, pr.vendorID, pr.deviceID, device_kind, v >> 22 & 0x7f, v >> 12 & 0x3ff, v & 0xfff, dev.features.storage16, dev.features.int16, timeout / std.time.ns_per_ms });
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
    // Stage 2's three through the production operator; the preserve add compiled without
    // -DUSE_BF16 (f32 in, exact f32 sum out), the gate and gated norm in the same dispatch
    // shapes their wrappers use.
    var m = try vk.norm_mlp.NormMlp.init(&dev, .{
        .norm_bf16 = &shaders2.norm_bf16,
        .add_bf16 = &shaders2.add_bf16,
        .add_preserve_bf16 = &shaders2.add_preserve_bf16,
        .swiglu_bf16 = &shaders2.swiglu_bf16,
    });
    defer m.deinit();
    var preserve = try vk.dispatch.Pipeline.init(&dev, &shaders2.add_preserve_bf16, .{ .bindings = 3, .push_bytes = vk.params.cast_push_bytes, .writes = 1 << 2 });
    defer preserve.deinit();
    var gate = try vk.dispatch.Pipeline.init(&dev, &shaders3.gate_bf16, .{ .bindings = 3, .push_bytes = @sizeOf(GateParams), .writes = 1 << 2 });
    defer gate.deinit();
    var gated = try vk.dispatch.Pipeline.init(&dev, &shaders4.fast_norm_gated_bf16, .{ .bindings = 4, .push_bytes = gated_push_bytes, .writes = 1 << 3 });
    defer gated.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    for (fx2.add_cases) |case| {
        total += 1;
        passed += @intFromBool(try addGate(a, clock, &dev, &cmds, staging, &m, case, timeout));
    }
    for (fx2.swiglu_cases) |case| {
        total += 1;
        passed += @intFromBool(try swigluGate(a, clock, &dev, &cmds, staging, &m, case, timeout));
    }
    for (fx2.rms_cases) |case| {
        total += 1;
        passed += @intFromBool(try rmsGate(a, clock, &dev, &cmds, staging, &m, case, timeout));
    }
    for (fx.preserve_cases) |case| {
        total += 1;
        passed += @intFromBool(try preserveGate(a, clock, &dev, &cmds, staging, &preserve, case, timeout));
    }
    for (fx.gate_cases) |case| {
        total += 1;
        passed += @intFromBool(try outputGate(a, clock, &dev, &cmds, staging, &gate, case, timeout));
    }
    for (fx.gated_cases) |case| {
        total += 1;
        passed += @intFromBool(try gatedNormGate(a, clock, &dev, &cmds, staging, &gated, case, timeout));
    }
    std.debug.print("RESULT contract-pointwise elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
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

fn checkUnchanged(a: std.mem.Allocator, cmds: *vk.Commands, staging: vk.Buffer, shell: anytype, buf: vk.Buffer, timeout: u64) !bool {
    const bytes = std.mem.sliceAsBytes(shell);
    const back = try a.alloc(u8, bytes.len);
    defer a.free(back);
    try fetch(cmds, staging, buf, back, timeout);
    return std.mem.indexOfDiff(u8, bytes, back) == null;
}

/// One bf16 binding's shell uploaded at its fixture offsets; the view is derived against the
/// caller-resident Bound16 (a stored View would point at boundShell's dead stack frame).
const Bound16 = struct {
    buf: vk.Buffer,
    byte_offset: u64,
    elements: u64,

    fn view(b: *const Bound16) !vk.tensor.View {
        return vk.tensor.View.init(&b.buf, .bf16, b.byte_offset, b.elements);
    }
};

fn boundShell(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, at_elems: usize, elements: u64, timeout: u64) !Bound16 {
    var buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(shell), timeout);
    errdefer buf.deinit();
    return .{ .buf = buf, .byte_offset = at_elems * 2, .elements = elements };
}

/// One f32 binding's shell (the preserve add); same lifetime rule as Bound16.
const Bound32 = struct {
    buf: vk.Buffer,
    byte_offset: u64,
    elements: u64,

    fn view(b: *const Bound32) !vk.tensor.View {
        return vk.tensor.View.init(&b.buf, .f32, b.byte_offset, b.elements);
    }
};

fn boundShell32(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u32, at_elems: usize, elements: u64, timeout: u64) !Bound32 {
    var buf = try uploaded(a, dev, cmds, staging, std.mem.sliceAsBytes(shell), timeout);
    errdefer buf.deinit();
    return .{ .buf = buf, .byte_offset = at_elems * 4, .elements = elements };
}

fn reportFail(comptime what: []const u8, name: []const u8, e: anyerror) bool {
    std.debug.print("FAIL " ++ what ++ " case=\"" ++ "{s}" ++ "\": " ++ "{s}" ++ "\n", .{ name, @errorName(e) });
    return false;
}

/// Stage 2's three: identical to the stage 2 probe's gates (same fixtures, same production
/// operator), so the pointwise contract stands alone on a fresh device.
fn addGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.norm_mlp.NormMlp, case: fx2.AddCase, timeout: u64) !bool {
    const f = try fx2.Add.init(a, case, 0x59a9);
    defer f.deinit();
    var in_a = try boundShell(a, dev, cmds, staging, f.a_bits, case.at.a, case.elems, timeout);
    defer in_a.buf.deinit();
    var in_b = try boundShell(a, dev, cmds, staging, f.b_bits, case.at.b, case.elems, timeout);
    defer in_b.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, case.elems, timeout);
    defer out.buf.deinit();
    const bits = try a.alloc(u16, f.out.len);
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
        } else first = try a.dupe(u16, bits);
    }
    const mm = f.firstMismatch(bits) catch |e| return reportFail("residual-add", case.name, e);
    if (mm) |bad| {
        std.debug.print("FAIL residual-add case=\"{s}\": {s} at element {d}: a=0x{x:0>4} ({s}) b=0x{x:0>4} ({s}) want=0x{x:0>4} ({s}) got=0x{x:0>4} ({s})\n", .{ case.name, @errorName(bad.kind), bad.index, bad.a, fx.classify(bad.a), bad.b, fx.classify(bad.b), bad.want, fx.classify(bad.want), bad.got, fx.classify(bad.got) });
        return false;
    }
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
    std.debug.print("MILESTONE residual-add elapsed_ms={d} case=\"{s}\" elements={d} matches the f32-add + RNE-store pipeline bit for bit, repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, case.elems });
    return true;
}

fn swigluGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.norm_mlp.NormMlp, case: fx2.SwigluCase, timeout: u64) !bool {
    const f = try fx2.Swiglu.init(a, case, 0x59aa);
    defer f.deinit();
    var gate = try boundShell(a, dev, cmds, staging, f.gate, case.at.a, case.elems, timeout);
    defer gate.buf.deinit();
    var up = try boundShell(a, dev, cmds, staging, f.up, case.at.b, case.elems, timeout);
    defer up.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, case.elems, timeout);
    defer out.buf.deinit();
    const bits = try a.alloc(u16, f.out.len);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(m.swiglu(cmds, try out.view(), try gate.view(), try up.view(), timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| return reportFail("swiglu", case.name, e);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL swiglu case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
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
    std.debug.print("MILESTONE swiglu elapsed_ms={d} case=\"{s}\" elements={d} worst={d:.4} of bound at element {d}, repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, case.elems, r.worst, r.col });
    return true;
}

fn rmsGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.norm_mlp.NormMlp, case: fx2.RmsCase, timeout: u64) !bool {
    const f = try fx2.Rms.init(a, case, 0x59ab);
    defer f.deinit();
    var x = try boundShell(a, dev, cmds, staging, f.x, case.at.x, case.rows * case.width, timeout);
    defer x.buf.deinit();
    var gamma = try boundShell(a, dev, cmds, staging, f.gamma, case.at.gamma, case.width, timeout);
    defer gamma.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, case.rows * case.width, timeout);
    defer out.buf.deinit();
    const bits = try a.alloc(u16, f.out.len);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |xx| a.free(xx);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(m.rmsNorm(cmds, try out.view(), try x.view(), try gamma.view(), case.rows, case.width, fx2.rms_eps, timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| return reportFail("rms-norm", case.name, e);
        if (first) |xx| {
            if (std.mem.indexOfDiff(u16, xx, bits)) |at| {
                std.debug.print("FAIL rms-norm case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
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
    std.debug.print("MILESTONE rms-norm elapsed_ms={d} case=\"{s}\" rows={d} width={d} worst={d:.4} of bound at row {d} col {d}, repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused{s}\n", .{ clock.ms(), case.name, case.rows, case.width, r.worst, r.row, r.col, if (case.zero_row) ", zero row exactly zero" else "" });
    return true;
}

/// The preserved residual add: the exact f32 sum, bit for bit. A device that flushes fp32
/// denormals fails the classes rows HONESTLY — the fixture's reference is not fitted to the
/// flush, and this gate prints that finding as what it is.
fn preserveGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, p: *vk.dispatch.Pipeline, case: fx.PreserveCase, timeout: u64) !bool {
    const f = try fx.PreserveAdd.init(a, case, 0x59ac);
    defer f.deinit();
    var in_a = try boundShell32(a, dev, cmds, staging, f.a_bits, case.at.a, case.elems, timeout);
    defer in_a.buf.deinit();
    var in_b = try boundShell32(a, dev, cmds, staging, f.b_bits, case.at.b, case.elems, timeout);
    defer in_b.buf.deinit();
    var out = try boundShell32(a, dev, cmds, staging, f.out, case.at.out, case.elems, timeout);
    defer out.buf.deinit();
    const bits = try a.alloc(u32, f.out.len);
    defer a.free(bits);
    var first: ?[]u32 = null;
    defer if (first) |x| a.free(x);
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(p.bind(&.{ (try in_a.view()).bind(), (try in_b.view()).bind(), (try out.view()).bind() }));
        const push: vk.params.Params = .{ .count = try cast32(case.elems), .operation = 0, .lhs_size = try cast32(case.elems), .rhs_size = try cast32(case.elems) };
        try vk.orExit(p.run(cmds, std.mem.asBytes(&push)[0..vk.params.cast_push_bytes], try groups(case.elems, p.local[0], dev.properties.limits.maxComputeWorkGroupCount[0]), timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        if (first) |x| {
            if (std.mem.indexOfDiff(u32, x, bits)) |at| {
                std.debug.print("FAIL preserve-add case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u32, bits);
    }
    const mm = f.firstMismatch(bits) catch |e| return reportFail("preserve-add", case.name, e);
    if (mm) |bad| {
        std.debug.print("FAIL preserve-add case=\"{s}\": {s} at element {d}: a=0x{x:0>8} b=0x{x:0>8} want=0x{x:0>8} got=0x{x:0>8}\n", .{ case.name, @errorName(bad.kind), bad.index, bad.a, bad.b, bad.want, bad.got });
        if (case.fill == .classes and isDenormalBits(bad.want)) {
            std.debug.print("NOTE preserve-add: want is an fp32 subnormal and got differs — this device flushes fp32 denormals; the contract's denormal rows fail honestly by design\n", .{});
        }
        return false;
    }
    // A planted nonfinite output at a finite-want element must be refused by the same checker.
    const data_at = case.at.out + firstFinite(f.want).?;
    var mutated = try a.dupe(u32, bits);
    defer a.free(mutated);
    for (planted_bits, planted) |pb, pname| {
        mutated[data_at] = @as(u32, pb) << 16;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL preserve-add case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |_| {}
    }
    if (!try checkUnchanged(a, cmds, staging, f.a_bits, in_a.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.b_bits, in_b.buf, timeout)) {
        std.debug.print("FAIL preserve-add case=\"{s}\": an input binding changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE preserve-add elapsed_ms={d} case=\"{s}\" elements={d} matches the exact f32 sum bit for bit, repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, case.elems });
    return true;
}

fn isDenormalBits(bits: u32) bool {
    const v: f32 = @bitCast(bits);
    return std.math.isFinite(v) and v != 0 and @abs(v) < std.math.floatMin(f32);
}

/// The attention output gate: bf16(sigmoid(gate)) once, times ctx, one bf16 store.
fn outputGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, gate: *vk.dispatch.Pipeline, case: fx.GateCase, timeout: u64) !bool {
    const f = try fx.Gate.init(a, case, 0x59ad);
    defer f.deinit();
    const elems = case.rows * fx.gate_row_elems;
    var ctx = try boundShell(a, dev, cmds, staging, f.ctx, case.at.a, elems, timeout);
    defer ctx.buf.deinit();
    var qraw = try boundShell(a, dev, cmds, staging, f.qraw, case.at.b, case.rows * fx.gate_qraw_row_elems, timeout);
    defer qraw.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, elems, timeout);
    defer out.buf.deinit();
    const bits = try a.alloc(u16, f.out.len);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |x| a.free(x);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(gate.bind(&.{ (try ctx.view()).bind(), (try qraw.view()).bind(), (try out.view()).bind() }));
        const plan = try vk.attention.gatePlan(elems, elems, case.rows * fx.gate_qraw_row_elems, case.rows);
        const push: GateParams = .{ .words = plan.words };
        try vk.orExit(gate.run(cmds, std.mem.asBytes(&push), try groups(plan.words, gate.local[0], dev.properties.limits.maxComputeWorkGroupCount[0]), timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| return reportFail("output-gate", case.name, e);
        if (first) |x| {
            if (std.mem.indexOfDiff(u16, x, bits)) |at| {
                std.debug.print("FAIL output-gate case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
    }
    var mutated = try a.dupe(u16, bits);
    defer a.free(mutated);
    const data_at = case.at.out + firstFinite(f.want).?;
    for (planted_bits, planted) |pb, pname| {
        mutated[data_at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL output-gate case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |_| {}
    }
    if (!try checkUnchanged(a, cmds, staging, f.ctx, ctx.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.qraw, qraw.buf, timeout)) {
        std.debug.print("FAIL output-gate case=\"{s}\": an input binding changed\n", .{case.name});
        return false;
    }
    std.debug.print("MILESTONE output-gate elapsed_ms={d} case=\"{s}\" rows={d} worst={d:.4} of bound at row {d} col {d}, repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused\n", .{ clock.ms(), case.name, case.rows, r.worst, r.row, r.col });
    return true;
}

/// The gated RMSNorm: rows-family bound plus the stated mid budget.
fn gatedNormGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, g: *vk.dispatch.Pipeline, case: fx.GatedCase, timeout: u64) !bool {
    const f = try fx.GatedNorm.init(a, case, 0x59ae);
    defer f.deinit();
    const elems = case.rows * fx.gated_width;
    var x = try boundShell(a, dev, cmds, staging, f.x, case.at.x, elems, timeout);
    defer x.buf.deinit();
    var weight = try boundShell(a, dev, cmds, staging, f.weight, case.at.weight, fx.gated_width, timeout);
    defer weight.buf.deinit();
    var z = try boundShell(a, dev, cmds, staging, f.z, case.at.gate, elems, timeout);
    defer z.buf.deinit();
    var out = try boundShell(a, dev, cmds, staging, f.out, case.at.out, elems, timeout);
    defer out.buf.deinit();
    const bits = try a.alloc(u16, f.out.len);
    defer a.free(bits);
    var first: ?[]u16 = null;
    defer if (first) |xx| a.free(xx);
    var r: fx.Report = .{};
    for (0..repeats) |rep| {
        try vk.orExit(vk.upload(cmds, out.buf, 0, staging, std.mem.sliceAsBytes(f.out), timeout));
        try vk.orExit(g.bind(&.{ (try x.view()).bind(), (try weight.view()).bind(), (try z.view()).bind(), (try out.view()).bind() }));
        const plan = try vk.gdn.gatedNormPlan(elems, elems, elems, fx.gated_width, case.rows, fx.gated_eps);
        var push: vk.params.Params = .{
            .operation = 0, // gated mode
            .lhs_size = fx.gated_width, // per-column weight rows
            .reduce_size = fx.gated_width,
            .output_size = plan.rows,
        };
        push.alpha = fx.gated_eps;
        // One 256-thread workgroup per row, the wrapper's grid-stride row loop.
        try vk.orExit(g.run(cmds, std.mem.asBytes(&push)[0..gated_push_bytes], try groups(plan.rows, 1, dev.properties.limits.maxComputeWorkGroupCount[0]), timeout));
        try fetch(cmds, staging, out.buf, std.mem.sliceAsBytes(bits), timeout);
        r = f.check(bits) catch |e| return reportFail("gated-norm", case.name, e);
        if (first) |xx| {
            if (std.mem.indexOfDiff(u16, xx, bits)) |at| {
                std.debug.print("FAIL gated-norm case=\"{s}\": repeat {d} differs at element {d}\n", .{ case.name, rep, at });
                return false;
            }
        } else first = try a.dupe(u16, bits);
    }
    var mutated = try a.dupe(u16, bits);
    defer a.free(mutated);
    const data_at = case.at.out + r.row * fx.gated_width + r.col;
    for (planted_bits, planted) |pb, pname| {
        mutated[data_at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL gated-norm case=\"{s}\": a planted {s} output passed the check\n", .{ case.name, pname });
            return false;
        } else |_| {}
    }
    if (!try checkUnchanged(a, cmds, staging, f.x, x.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.weight, weight.buf, timeout) or !try checkUnchanged(a, cmds, staging, f.z, z.buf, timeout)) {
        std.debug.print("FAIL gated-norm case=\"{s}\": an input binding changed\n", .{case.name});
        return false;
    }
    if (case.zero_row) {
        const zero_at = case.at.out + (case.rows - 1) * fx.gated_width;
        for (bits[zero_at .. zero_at + fx.gated_width]) |gg| if (gg != 0x0000 and gg != 0x8000) {
            std.debug.print("FAIL gated-norm case=\"{s}\": the zero row produced 0x{x:0>4}\n", .{ case.name, gg });
            return false;
        };
    }
    std.debug.print("MILESTONE gated-norm elapsed_ms={d} case=\"{s}\" rows={d} width={d} worst={d:.4} of bound at row {d} col {d}, repeats byte-identical, inputs and guards unchanged, planted nan/+inf/-inf refused{s}\n", .{ clock.ms(), case.name, case.rows, fx.gated_width, r.worst, r.row, r.col, if (case.zero_row) ", zero row exactly zero" else "" });
    return true;
}

fn cast32(v: u64) !u32 {
    return std.math.cast(u32, v) orelse error.TooLarge;
}

fn groups(count: u64, local: u32, max_x: u32) ![3]u32 {
    const n = try cast32((count + local - 1) / local);
    return .{ @min(n, max_x), 1, 1 };
}

test {
    std.testing.refAllDecls(@This());
}
