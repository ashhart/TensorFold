//! FZ_CALL_LOG: a prompt pass's calls (rows, parts, wall and GPU milliseconds), logged in one line after its first token.
const std = @import("std");

const N = 16;

pub const CallLog = struct {
    rows: [N]u32 = undefined,
    parts: [N]u8 = undefined,
    wall: [N]f32 = undefined,
    gpu: [N]f32 = undefined,
    n: usize = 0,
    total: usize = 0,

    /// A call that began at mach time `t0` with the model's GPU seconds at `g0`; `g1` now.
    pub fn note(c: *CallLog, rows: usize, parts: usize, t0: u64, g0: f64, g1: f64) void {
        c.total += 1;
        if (c.n == N) return;
        c.rows[c.n] = @intCast(rows);
        c.parts[c.n] = @intCast(parts);
        c.wall[c.n] = @floatCast(@as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) * 125 / 3 / 1e6); // mach ticks: 125/3 ns
        c.gpu[c.n] = @floatCast((g1 - g0) * 1e3);
        c.n += 1;
    }

    pub fn log(c: *const CallLog, rank: u32) void {
        var buf: [1024]u8 = undefined;
        var len: usize = 0;
        for (0..c.n) |i| {
            const s = std.fmt.bufPrint(buf[len..], " {d}x{d} {d:.1}/{d:.1}", .{ c.rows[i], c.parts[i], c.wall[i], c.gpu[i] }) catch break;
            len += s.len;
        }
        std.log.info("prompt calls on rank {d} (rows x parts, wall/GPU ms):{s}{s}", .{ rank, buf[0..len], if (c.total > c.n) " ..." else "" });
    }
};
