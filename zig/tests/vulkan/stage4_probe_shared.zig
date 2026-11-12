// Stage 4 probe: the clock, the buffer helpers, the planted refusals and the input dumps the gates share.

const std = @import("std");
const vk = @import("vulkan");
const fx = @import("stage4_fixture.zig");

pub const repeats = 3;
pub const state_elems: u64 = fx.Model.state_elements;

pub const Elapsed = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    pub fn ms(e: Elapsed) i64 {
        return e.start.durationTo(std.Io.Clock.awake.now(e.io)).toMilliseconds();
    }
};

pub fn uploadShell(dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, timeout: u64) !vk.Buffer {
    var b = try vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(shell).len);
    errdefer b.deinit();
    try vk.orExit(vk.upload(cmds, b, 0, staging, std.mem.sliceAsBytes(shell), timeout));
    return b;
}

pub fn fetch(cmds: *vk.Commands, staging: vk.Buffer, b: vk.Buffer, out: []u8, timeout: u64) !void {
    try vk.orExit(vk.download(cmds, b, 0, staging, out, timeout));
}

pub fn unchanged(a: std.mem.Allocator, cmds: *vk.Commands, staging: vk.Buffer, shell: []const u16, buf: vk.Buffer, timeout: u64) !bool {
    const back = try a.alloc(u16, shell.len);
    defer a.free(back);
    try fetch(cmds, staging, buf, std.mem.sliceAsBytes(back), timeout);
    return std.mem.indexOfDiff(u16, shell, back) == null;
}

/// Planted refusals on a downloaded GOOD output.
pub fn plantRefuse(f: anytype, good: []const u16, name: []const u8, gate: []const u8) !bool {
    const mutated = try f.a.dupe(u16, good);
    defer f.a.free(mutated);
    var first_finite: ?usize = null;
    for (good, 0..) |bits, i| {
        if (std.math.isFinite(fx.c.fromBf16(bits))) {
            first_finite = i;
            break;
        }
    }
    const at = first_finite orelse return error.NoFiniteReference;
    mutated[at] = 0x4f00; // 2^31: an absolutely wrong finite the budgets must refuse
    if (f.check(mutated)) |_| {
        std.debug.print("FAIL {s} case=\"{s}\": a planted wrong finite output passed\n", .{ gate, name });
        return false;
    } else |_| {}
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, [_][]const u8{ "nan", "+inf", "-inf" }) |pb, pname| {
        mutated[at] = pb;
        if (f.check(mutated)) |_| {
            std.debug.print("FAIL {s} case=\"{s}\": a planted {s} output passed\n", .{ gate, name, pname });
            return false;
        } else |_| {}
    }
    return true;
}

// Planted refusals on a downloaded GOOD output; `extra` is passed through to two-slice checkers (state bits beside y).
pub fn plantRefuse2(f: anytype, good: []const u16, extra: []const u16, name: []const u8, gate: []const u8) !bool {
    const mutated = try f.a.dupe(u16, good);
    defer f.a.free(mutated);
    const extra_mut = try f.a.dupe(u16, extra);
    defer f.a.free(extra_mut);
    var first_finite: ?usize = null;
    for (good, 0..) |bits, i| {
        if (std.math.isFinite(fx.c.fromBf16(bits))) {
            first_finite = i;
            break;
        }
    }
    const at = first_finite orelse return error.NoFiniteReference;
    mutated[at] = 0x4f00; // 2^31: an absolutely wrong finite the budgets must refuse
    if (f.check(mutated, extra_mut, null)) |_| {
        std.debug.print("FAIL {s} case=\"{s}\": a planted wrong finite output passed\n", .{ gate, name });
        return false;
    } else |_| {}
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |pb| {
        mutated[at] = pb;
        if (f.check(mutated, extra_mut, null)) |_| {
            std.debug.print("FAIL {s} case=\"{s}\": a planted nonfinite output passed\n", .{ gate, name });
            return false;
        } else |_| {}
    }
    return true;
}

/// Uploaded host inputs; GPU softplus/exp intermediates are not stored.
pub fn printGateInputs(a_bits: []const u16, b_bits: []const u16, a_log_pairs: []const u16, dt_bits: []const u16, gate_index: usize) void {
    const hv = gate_index % 16;
    const alog_word = @as(u32, a_log_pairs[2 * hv]) | (@as(u32, a_log_pairs[2 * hv + 1]) << 16);
    const alog: f32 = @bitCast(alog_word);
    const ref = fx.gateRefOracle(a_bits[gate_index], b_bits[gate_index], alog, dt_bits[hv]);
    std.debug.print("  HOST gate token={d} head={d} a=0x{x:0>4} ({e}) b=0x{x:0>4} ({e}) dt=0x{x:0>4} ({e}) A_log=0x{x:0>8} ({e}) oracle_g={e} oracle_beta={e}\n", .{
        gate_index / 16, hv, a_bits[gate_index], fx.c.fromBf16(a_bits[gate_index]), b_bits[gate_index], fx.c.fromBf16(b_bits[gate_index]), dt_bits[hv], fx.c.fromBf16(dt_bits[hv]), alog_word, alog, ref.g, ref.beta,
    });
}

pub fn printGateInputsOne(f: fx.SplitOne, v: fx.Violation) void {
    std.debug.print("  DIAG output={s}\n", .{@tagName(v.output)});
    if (v.output == .g or v.output == .beta) {
        printGateInputs(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, v.index);
    } else {
        std.debug.print("  HOST soa index={d} conv=0x{x:0>4} ({e})\n", .{ v.index, f.conv_out_bits[v.index], fx.c.fromBf16(f.conv_out_bits[v.index]) });
    }
}

pub fn printSplitGateInputs(f: fx.Split, v: fx.Violation) void {
    std.debug.print("  DIAG output={s}\n", .{@tagName(v.output)});
    if (v.output == .g or v.output == .beta) {
        printGateInputs(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, v.index);
    } else {
        const section: usize = @intCast(fx.tokens * fx.Model.linear_dim);
        const kind = v.index / section;
        const within = v.index % section;
        const token = within / 2048;
        const head = (within % 2048) / 128;
        const col = within % 128;
        const source = token * 6144 + kind * 2048 + head * 128 + col;
        std.debug.print("  HOST soa token={d} section={d} head={d} col={d} conv=0x{x:0>4} ({e})\n", .{ token, kind, head, col, f.conv_out_bits[source], fx.c.fromBf16(f.conv_out_bits[source]) });
    }
}

pub fn printRecurInputs(f: fx.Split, v: fx.Violation, final_token: u64) void {
    std.debug.print("  DIAG output={s}\n", .{@tagName(v.output)});
    const head = if (v.output == .y) (v.index / 128) % 16 else v.index / (128 * 128);
    const token: usize = if (v.output == .y) v.index / 2048 else @intCast(final_token);
    printGateInputs(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, token * 16 + head);
}
