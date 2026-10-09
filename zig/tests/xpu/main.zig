//! GPU test runner for the Intel GPU runtime and kernels: `tf-xpu-test <command> [--device N]`; exit 1 on failure.

const std = @import("std");
const rt = @import("rt.zig");
const xpu = @import("xpu");
const fixture = @import("fixture.zig");

pub const panic = std.debug.FullPanic(xpu.rt.panicDrain);

const Test = struct { name: []const u8, run: *const fn () anyerror!void };

const tests = [_]Test{
    .{ .name = "smoke", .run = @import("smoke.zig").run },
    .{ .name = "basic", .run = @import("basic.zig").run },
    .{ .name = "qmv", .run = @import("qmv.zig").run },
    .{ .name = "moe", .run = @import("moe.zig").run },
    .{ .name = "mamba", .run = @import("mamba.zig").run },
    .{ .name = "attn", .run = @import("attn.zig").run },
};

const usage =
    \\usage: tf-xpu-test <command> [--device N]
    \\  info      every Level Zero device and the one picked
    \\  smoke     a SPIR-V vector add: copies, launch, read back
    \\  basic     RMSNorm and the 4-bit embedding lookup against the reference
    \\  qmv       the 4-bit matvec on 256 real rows
    \\  moe       router, top-6 routing, expert MLPs, combine and the chain
    \\  mamba     Mamba mixer ops and a decode sequence
    \\  attn      attention block, split-K attention, lm_head rows, argmax
    \\  all       smoke, basic, qmv, moe, mamba, attn
    \\fixtures: TF_FIXTURES_DIR (default $TF_FIXTURES_DIR)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    xpu.stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--device") and i + 1 < args.len) {
            rt.ordinal = try std.fmt.parseInt(u32, args[i + 1], 10);
            i += 1;
        } else {
            std.debug.print("unknown option {s}\n{s}", .{ args[i], usage });
            return 2;
        }
    }
    const dir = init.environ_map.get("TF_FIXTURES_DIR") orelse blk: {
        const home = init.environ_map.get("HOME") orelse return error.NoHome;
        break :blk try std.fs.path.join(init.arena.allocator(), &.{ home, "tensorfold-fixtures" });
    };
    fixture.init(init.io, dir);
    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "info")) {
        @import("info.zig").run() catch |e| return fail(cmd, e);
        return 0;
    }
    var failed: usize = 0;
    var ran: usize = 0;
    for (tests) |t| {
        if (!std.mem.eql(u8, cmd, "all") and !std.mem.eql(u8, cmd, t.name)) continue;
        if (xpu.stop.requested()) {
            std.debug.print("stopped before {s}\n", .{t.name});
            break;
        }
        ran += 1;
        if (t.run()) |_| {
            std.debug.print("PASS {s}\n", .{t.name});
        } else |e| {
            if (e == error.FileNotFound) {
                std.debug.print("SKIP {s}: fixtures not found in {s}; generate them with tools/zig/xpu_fixtures_*.py\n", .{ t.name, dir });
                continue;
            }
            failed += 1;
            _ = fail(t.name, e);
        }
    }
    if (ran == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    return if (failed == 0) 0 else 1;
}

fn fail(name: []const u8, e: anyerror) u8 {
    std.debug.print("FAIL {s}: {t}\n", .{ name, e });
    return 1;
}
