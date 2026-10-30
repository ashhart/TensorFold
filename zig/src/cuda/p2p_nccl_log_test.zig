//! NCCL log capture and parsing: NCCL's own line formats (transport/p2p.cc, shm.cc, net.cc) and the private file.
const std = @import("std");
const builtin = @import("builtin");
const log = @import("p2p_nccl_log.zig");

fn render(s: *const log.Summary) ![]const u8 {
    const T = struct {
        var buf: [1024]u8 = undefined;
    };
    var w = std.Io.Writer.fixed(&T.buf);
    try log.render(&w, s);
    return w.buffered();
}

test "nccl log: P2P read and P2P write are told apart by NCCL's /read suffix" {
    const text =
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/1 : 0[0] -> 2[2] via P2P/IPC
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 02/0 : 0[0] -> 3[3] via P2P/direct pointer/read
    ;
    const s = log.parse(text, 0);
    try std.testing.expectEqualStrings("rank 1 P2P read (2 connections); rank 2 P2P write (1 connection); rank 3 P2P read (1 connection)", try render(&s));
}

test "nccl log: SHM in the older and current spellings" {
    const text =
        \\[2026-10-27 09:00:00] gpu-a:4101:4120 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via SHM/direct/direct
        \\gpu-a:4101:4120 [1] NCCL INFO Channel 01 : 1[1] -> 0[0] via SHM/direct
    ;
    const s = log.parse(text, 1);
    try std.testing.expectEqualStrings("rank 0 SHM (2 connections)", try render(&s));
}

test "nccl log: NET over IB and over sockets, sends and receives, by the peer's rank" {
    const text =
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/IB/0/GDRDMA
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 01/0 : 2[0] -> 0[0] [receive] via NET/Socket/0
    ;
    const s = log.parse(text, 0);
    try std.testing.expectEqualStrings("rank 1 NET/IB (2 connections); rank 2 NET/Socket (1 connection)", try render(&s));
}

test "nccl log: nothing reported when NCCL logged rings, versions and other ranks' links only" {
    const text =
        \\gpu-a:4101:4101 [0] NCCL INFO NCCL version 2.30.7+cuda13.0
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/02 :    0   1
        \\gpu-a:4101:4120 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/CUMEM/read
        \\gpu-a:4101:4120 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via NVLS/unknown
    ;
    const s = log.parse(text, 0);
    try std.testing.expectEqual(@as(usize, 0), s.count);
    try std.testing.expectEqualStrings("", try render(&s));
}

const FakeEnv = struct {
    const slots = 8;
    var names: [slots][64:0]u8 = undefined;
    var values: [slots][256:0]u8 = undefined;
    var used: [slots]bool = @splat(false);
    pub var sets: usize = 0;
    pub var unsets: usize = 0;

    pub fn reset() void {
        used = @splat(false);
        sets = 0;
        unsets = 0;
    }

    fn find(n: []const u8) ?usize {
        for (used, 0..) |u, i| if (u and std.mem.eql(u8, std.mem.sliceTo(&names[i], 0), n)) return i;
        return null;
    }

    pub fn put(n: []const u8, v: []const u8) void {
        const i = find(n) orelse std.mem.indexOfScalar(bool, &used, false).?;
        names[i] = @splat(0);
        values[i] = @splat(0);
        @memcpy(names[i][0..n.len], n);
        @memcpy(values[i][0..v.len], v);
        used[i] = true;
    }

    pub fn get(n: [*:0]const u8) ?[*:0]const u8 {
        return &values[find(std.mem.span(n)) orelse return null];
    }

    fn set(n: [*:0]const u8, v: [*:0]const u8) void {
        sets += 1;
        put(std.mem.span(n), std.mem.span(v));
    }

    fn unset(n: [*:0]const u8) void {
        unsets += 1;
        if (find(std.mem.span(n))) |i| used[i] = false;
    }

    pub const env: log.Env = .{ .get = get, .set = set, .unset = unset };
};

pub const fake_env = FakeEnv;

fn exists(path: [*:0]const u8) bool {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

/// Writes `text` where the capture told NCCL to log, as NCCL would.
pub fn writeLog(text: []const u8) !void {
    const p = FakeEnv.get("NCCL_DEBUG_FILE") orelse return error.NoCapture;
    const fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.Open;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, text.ptr, text.len) != @as(isize, @intCast(text.len))) return error.Write;
}

test "nccl log capture: a private file NCCL logs into, the environment handed back, the file removed" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    FakeEnv.reset();
    var c = log.Capture.begin(FakeEnv.env);
    try std.testing.expectEqual(log.State.on, c.state);
    try std.testing.expectEqualStrings("INFO", std.mem.span(FakeEnv.get("NCCL_DEBUG").?));
    try std.testing.expectEqualStrings(log.subsystems, std.mem.span(FakeEnv.get("NCCL_DEBUG_SUBSYS").?));
    const file = std.mem.span(FakeEnv.get("NCCL_DEBUG_FILE").?);
    try std.testing.expect(std.mem.indexOf(u8, file, "/tf-p2p-") != null and std.mem.indexOfScalar(u8, file, '%') == null);
    try writeLog("gpu-a:1:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read\n");
    c.restore(FakeEnv.env);
    try std.testing.expectEqual(@as(usize, 3), FakeEnv.unsets);
    try std.testing.expect(FakeEnv.get("NCCL_DEBUG") == null and FakeEnv.get("NCCL_DEBUG_FILE") == null);
    var buf: [4096]u8 = undefined;
    const s = log.parse(c.read(&buf), 0);
    try std.testing.expectEqual(@as(usize, 1), s.count);
    c.remove();
    try std.testing.expect(!exists(&c.dir));
}

test "nccl log capture: any NCCL_DEBUG* the user set is left alone and nothing is captured" {
    for (log.vars) |v| {
        FakeEnv.reset();
        FakeEnv.put(v, "WARN");
        var c = log.Capture.begin(FakeEnv.env);
        try std.testing.expectEqual(log.State.user, c.state);
        c.restore(FakeEnv.env);
        try std.testing.expectEqual(@as(usize, 0), FakeEnv.sets + FakeEnv.unsets);
        try std.testing.expectEqualStrings("WARN", std.mem.span(FakeEnv.get(v).?));
    }
}

test "nccl log capture: a temporary directory NCCL would expand (%) is not used" {
    FakeEnv.reset();
    FakeEnv.put("TMPDIR", "/tmp/%h");
    const c = log.Capture.begin(FakeEnv.env);
    try std.testing.expectEqual(log.State.unavailable, c.state);
    try std.testing.expectEqual(@as(usize, 0), FakeEnv.sets);
}
