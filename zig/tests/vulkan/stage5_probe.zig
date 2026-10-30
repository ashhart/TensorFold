// Greedy.select vs CPU max (248320 BF16; low-index ties, ±0 equal); NonFinite first-bad index; exit codes 0/1/2/124.
const std = @import("std");
const vk = @import("vulkan");
const shaders = @import("stage5_shaders");
const fx = @import("stage5_fixture.zig");

const usage = "usage: tf-stage5-gpu [--vendor ID] [--timeout-ms MS] | --help\n" ++
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

fn shaderSet() vk.greedy.Shaders {
    return .{
        .finite_logits = &shaders.finite_logits,
        .argreduce_chunk = &shaders.argreduce_chunk,
        .argreduce_combine = &shaders.argreduce_combine,
    };
}

fn probe(a: std.mem.Allocator, clock: Elapsed, vendor: ?u32, timeout: u64) !bool {
    var rt = try vk.Runtime.open();
    defer rt.close();
    var dev = try vk.Device.open(&rt, .{ .features = .{ .storage16 = true, .int16 = true } });
    defer dev.close();
    const pr = dev.properties;
    const name = std.mem.sliceTo(&pr.deviceName, 0);
    const v = pr.apiVersion;
    std.debug.print("MILESTONE device-found elapsed_ms={d} name=\"{s}\" vendor=0x{x:0>4} device=0x{x:0>4} " ++
        "api={d}.{d}.{d} storage16={} int16={} subgroups={d} shuffle-relative={} subgroup-size={d} fence-timeout={d}ms\n", .{
        clock.ms(),                                                                               name,
        pr.vendorID,                                                                              pr.deviceID,
        v >> 22 & 0x7f,                                                                           v >> 12 & 0x3ff,
        v & 0xfff,                                                                                dev.features.storage16,
        dev.features.int16,                                                                       dev.v11.subgroupSupportedStages,
        dev.v11.subgroupSupportedOperations & vk.c.VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT != 0, dev.v11.subgroupSize,
        timeout / std.time.ns_per_ms,
    });
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
    // A full 248320-element bf16 row rides staging: 485 KiB.
    var staging = try vk.Buffer.init(&dev, .staging, 4 << 20);
    defer staging.deinit();

    var passed: usize = 0;
    var total: usize = 0;
    total += 1;
    passed += @intFromBool(try argmaxGate(a, clock, &dev, &cmds, staging, .vocab, timeout));
    total += 1;
    passed += @intFromBool(try argmaxGate(a, clock, &dev, &cmds, staging, .ties, timeout));
    total += 1;
    passed += @intFromBool(try argmaxGate(a, clock, &dev, &cmds, staging, .negative, timeout));
    total += 1;
    passed += @intFromBool(try zeroGate(a, clock, &dev, &cmds, staging, timeout));
    total += 1;
    passed += @intFromBool(try nonFiniteGate(a, clock, &dev, &cmds, staging, timeout));
    total += 1;
    passed += @intFromBool(try minCountGate(a, clock, &dev, &cmds, staging, timeout));
    total += 1;
    passed += @intFromBool(try oddCountGate(a, clock, &dev, &cmds, staging, timeout));
    std.debug.print("RESULT stage5 elapsed_ms={d} on \"{s}\": passed={d} total={d}\n", .{ clock.ms(), name, passed, total });
    return passed == total;
}

fn gpuFail(gate: []const u8, e: anyerror) bool {
    std.debug.print("FAIL greedy gate=\"{s}\": {s}\n", .{ gate, @errorName(e) });
    return false;
}

fn refFail(gate: []const u8, e: anyerror) bool {
    std.debug.print("FAIL greedy gate=\"{s}\": the reference refused a finite row: {s}\n", .{ gate, @errorName(e) });
    return false;
}

// Upload the full poisoned row; caller owns the buffer and creates its address-stable view.
fn uploadRow(gate: []const u8, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, row: fx.Row, timeout: u64) ?vk.Buffer {
    var b = vk.Buffer.init(dev, .device, std.mem.sliceAsBytes(row.bits).len) catch |e| {
        _ = gpuFail(gate, e);
        return null;
    };
    vk.orExit(vk.upload(cmds, b, 0, staging, std.mem.sliceAsBytes(row.bits), timeout)) catch |e| {
        b.deinit();
        _ = gpuFail(gate, e);
        return null;
    };
    return b;
}

// Run 3 selects via orExit; tokens must match, mismatches print FAIL, and GPU errors propagate raw.
fn selectRepeat(gate: []const u8, g: *vk.Greedy, cmds: *vk.Commands, staging: vk.Buffer, view: vk.tensor.View, timeout: u64) !u32 {
    var first: ?u32 = null;
    for (0..repeats) |rep| {
        const tok = try vk.orExit(g.select(cmds, staging, view, timeout));
        if (first) |f| {
            if (tok != f) {
                std.debug.print("FAIL greedy gate=\"{s}\": repeat {d} selected {d}, first repeat selected {d}\n", .{ gate, rep, tok, f });
                return error.RepeatMismatch;
            }
        } else first = tok;
    }
    return first.?;
}

fn repeatFail(gate: []const u8, e: anyerror) bool {
    if (e != error.RepeatMismatch) return gpuFail(gate, e);
    return false;
}

// Download the full shell and require bit identity: poisoned guards and logits remain unchanged.
fn shellChecked(gate: []const u8, a: std.mem.Allocator, cmds: *vk.Commands, staging: vk.Buffer, row: fx.Row, buf: vk.Buffer, timeout: u64) bool {
    const back = a.alloc(u16, row.bits.len) catch |e| return gpuFail(gate, e);
    defer a.free(back);
    vk.orExit(vk.download(cmds, buf, 0, staging, std.mem.sliceAsBytes(back), timeout)) catch |e| return gpuFail(gate, e);
    row.checkShell(back) catch |e| {
        std.debug.print("FAIL greedy gate=\"{s}\": the logits shell is not intact ({s})\n", .{ gate, @errorName(e) });
        return false;
    };
    return true;
}

fn greedyFor(gate: []const u8, dev: *const vk.Device, count: u64) ?vk.Greedy {
    return vk.Greedy.init(dev, shaderSet(), count) catch |e| {
        _ = gpuFail(gate, e);
        return null;
    };
}

const RowKind = enum { vocab, ties, negative };

/// One finite row: repeat-stable token equal to the reference, shell intact.
fn singleRowGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, gate: []const u8, row: fx.Row, timeout: u64) !bool {
    var buf = uploadRow(gate, dev, cmds, staging, row, timeout) orelse return false;
    defer buf.deinit();
    const view = vk.tensor.View.init(&buf, .bf16, fx.at * 2, row.count) catch |e| return gpuFail(gate, e);
    var g = greedyFor(gate, dev, row.count) orelse return false;
    defer g.deinit();
    const tok = selectRepeat(gate, &g, cmds, staging, view, timeout) catch |e| return repeatFail(gate, e);
    const want = row.reference() catch |e| return refFail(gate, e);
    if (tok != want) {
        std.debug.print("FAIL greedy gate=\"{s}\": selected {d}, the reference says {d}\n", .{ gate, tok, want });
        return false;
    }
    if (!shellChecked(gate, a, cmds, staging, row, buf, timeout)) return false;
    std.debug.print("MILESTONE greedy elapsed_ms={d} gate=\"{s}\" count={d} token={d} reference={d} repeats={d} identical inputs-guards-unchanged\n", .{ clock.ms(), gate, row.count, tok, want, repeats });
    return true;
}

fn argmaxGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, kind: RowKind, timeout: u64) !bool {
    const gate = switch (kind) {
        .vocab => "vocab-248320",
        .ties => "ties-across-chunks",
        .negative => "all-negative",
    };
    var row = switch (kind) {
        .vocab => fx.random(a, fx.vocab, 0x57a5) catch |e| return gpuFail(gate, e),
        .ties => fx.ties(a, fx.vocab, 0x57a6) catch |e| return gpuFail(gate, e),
        .negative => fx.negative(a, fx.vocab, 0x57a7) catch |e| return gpuFail(gate, e),
    };
    defer row.deinit();
    return singleRowGate(a, clock, dev, cmds, staging, gate, row, timeout);
}

/// +0/-0 ties in both orders: element 5 wins either way, against one Greedy.
fn zeroGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, timeout: u64) !bool {
    const gate = "signed-zero-both-orders";
    var rows: [2]fx.Row = .{
        fx.zero(a, fx.vocab, 0x57a8, .plus_first) catch |e| return gpuFail(gate, e),
        fx.zero(a, fx.vocab, 0x57a8 + 1, .minus_first) catch |e| return gpuFail(gate, e),
    };
    defer for (&rows) |*r| r.deinit();
    var g = greedyFor(gate, dev, fx.vocab) orelse return false;
    defer g.deinit();
    var tokens: [2]u32 = undefined;
    for (&rows, 0..) |*row, i| {
        var buf = uploadRow(gate, dev, cmds, staging, row.*, timeout) orelse return false;
        defer buf.deinit();
        const view = vk.tensor.View.init(&buf, .bf16, fx.at * 2, row.count) catch |e| return gpuFail(gate, e);
        tokens[i] = selectRepeat(gate, &g, cmds, staging, view, timeout) catch |e| return repeatFail(gate, e);
        const want = row.reference() catch |e| return refFail(gate, e);
        if (tokens[i] != want) {
            std.debug.print("FAIL greedy gate=\"{s}\": order {d} selected {d}, the reference says {d}\n", .{ gate, i, tokens[i], want });
            return false;
        }
        if (!shellChecked(gate, a, cmds, staging, row.*, buf, timeout)) return false;
    }
    std.debug.print("MILESTONE greedy elapsed_ms={d} gate=\"{s}\" count={d} plus-first-token={d} minus-first-token={d} reference=lowest-index repeats={d} identical inputs-guards-unchanged\n", .{ clock.ms(), gate, fx.vocab, tokens[0], tokens[1], repeats });
    return true;
}

// Non-winning -inf/+inf/NaN (even at winner) must refuse; decode() reports first-bad index, not GPU failure.
fn nonFiniteGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, timeout: u64) !bool {
    const gate = "nonfinite-refused";
    const cases = [_]struct { kind: fx.NonFinite, bad: usize, label: []const u8 }{
        .{ .kind = .nan, .bad = fx.vocab / 3, .label = "nan" },
        .{ .kind = .pinf, .bad = fx.vocab / 2, .label = "+inf" },
        .{ .kind = .ninf, .bad = 2 * fx.vocab / 3, .label = "-inf" },
    };
    var g = greedyFor(gate, dev, fx.vocab) orelse return false;
    defer g.deinit();
    var bad: [cases.len]usize = undefined;
    for (cases, 0..) |cs, i| {
        var row = fx.nonFinite(a, fx.vocab, 0x57a9 + i, cs.kind, cs.bad) catch |e| return gpuFail(gate, e);
        defer row.deinit();
        if (row.reference()) |_| {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": the fixture reference accepted a non-finite row\n", .{ gate, cs.label });
            return false;
        } else |e| {
            if (e != error.NonFinite) return refFail(gate, e);
        }
        if (row.firstBad() != cs.bad) {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": the fixture names first bad {d}, planted {d}\n", .{ gate, cs.label, row.firstBad(), cs.bad });
            return false;
        }
        var buf = uploadRow(gate, dev, cmds, staging, row, timeout) orelse return false;
        defer buf.deinit();
        const view = vk.tensor.View.init(&buf, .bf16, fx.at * 2, row.count) catch |e| return gpuFail(gate, e);
        if (vk.orExit(g.select(cmds, staging, view, timeout))) |tok| {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": a non-finite row selected token {d}\n", .{ gate, cs.label, tok });
            return false;
        } else |e| {
            if (e != error.NonFinite) {
                std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": expected NonFinite, got {s}\n", .{ gate, cs.label, @errorName(e) });
                return false;
            }
        }
        // Refusal is not GPU failure: read production Result and require the planted first-offender index.
        var ev: vk.greedy.Result = undefined;
        vk.orExit(vk.download(cmds, g.results, 0, staging, std.mem.asBytes(&ev), timeout)) catch |e2| {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": evidence download: {s}\n", .{ gate, cs.label, @errorName(e2) });
            return false;
        };
        if (ev.finite_flag == 0) {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": the evidence claims a finite verdict\n", .{ gate, cs.label });
            return false;
        }
        if (ev.first_bad != cs.bad) {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": first-bad evidence {d} is not the planted {d}\n", .{ gate, cs.label, ev.first_bad, cs.bad });
            return false;
        }
        if (vk.greedy.decode(@intCast(row.count), ev)) |_| {
            std.debug.print("FAIL greedy gate=\"{s}\" kind=\"{s}\": the evidence decodes to a token, not a refusal\n", .{ gate, cs.label });
            return false;
        } else |e2| {
            if (e2 != error.NonFinite) return gpuFail(gate, e2);
        }
        if (!shellChecked(gate, a, cmds, staging, row, buf, timeout)) return false;
        bad[i] = ev.first_bad;
    }
    std.debug.print("MILESTONE greedy elapsed_ms={d} gate=\"{s}\" count={d} nan-first-bad={d} pinf-first-bad={d} ninf-first-bad={d} verdict=NonFinite decode-refuses inputs-guards-unchanged\n", .{ clock.ms(), gate, fx.vocab, bad[0], bad[1], bad[2] });
    return true;
}

fn minCountGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, timeout: u64) !bool {
    const gate = "min-count-2";
    var row = fx.minTwo(a) catch |e| return gpuFail(gate, e);
    defer row.deinit();
    return singleRowGate(a, clock, dev, cmds, staging, gate, row, timeout);
}

// Odd bf16 count uploads but View refuses with Misaligned before dispatch.
fn oddCountGate(a: std.mem.Allocator, clock: Elapsed, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, timeout: u64) !bool {
    const gate = "odd-count-refused";
    var row = fx.random(a, 7, 0x57ad) catch |e| return gpuFail(gate, e);
    defer row.deinit();
    var buf = uploadRow(gate, dev, cmds, staging, row, timeout) orelse return false;
    defer buf.deinit();
    if (vk.tensor.View.init(&buf, .bf16, fx.at * 2, row.count)) |_| {
        std.debug.print("FAIL greedy gate=\"{s}\": an odd bf16 count produced a usable view\n", .{gate});
        return false;
    } else |e| {
        if (e != error.Misaligned) return gpuFail(gate, e);
    }
    std.debug.print("MILESTONE greedy elapsed_ms={d} gate=\"{s}\" count={d} refused=Misaligned before-any-dispatch\n", .{ clock.ms(), gate, row.count });
    return true;
}
