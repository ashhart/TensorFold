//! Opt-in raw-logit snapshots on the forward's stream; no additional kernels or work on unscored requests.
const std = @import("std");
const cuda = @import("cuda");
const pipe = @import("logprob_pipe.zig");
const Forward = @import("forward.zig").Forward;
const Row = @import("lanes").logprob.Row;

pub const op: i64 = 68;

pub const State = struct {
    captured: pipe.Capture = .{},
    buffers: [2]?cuda.DeviceBuffer = @splat(null),
    d: ?*const cuda.Driver = null,
    stream: ?cuda.Stream = null,

    pub fn deinit(s: *State, a: std.mem.Allocator) void {
        s.captured.deinit(a);
        for (&s.buffers) |*b| if (b.*) |*v| v.free();
    }

    fn scratch(p: *anyopaque, slot: u32, bytes: usize) anyerror!u64 {
        const s: *State = @ptrCast(@alignCast(p));
        const b = &s.buffers[slot];
        if (b.* == null or b.*.?.len < bytes) {
            if (b.*) |*v| v.free();
            b.* = null;
            b.* = try cuda.DeviceBuffer.alloc(s.d.?, bytes);
        }
        return b.*.?.ptr;
    }

    fn pad(p: *anyopaque, src: u64, cols: usize, stride: usize, dst: u64) anyerror!void {
        const s: *State = @ptrCast(@alignCast(p));
        try s.d.?.check(s.d.?.api.cuMemsetD32Async(dst, @bitCast(-std.math.inf(f32)), stride, s.stream.?.handle), "cuMemsetD32Async");
        try s.d.?.check(s.d.?.api.cuMemcpyDtoDAsync_v2(dst, src, 4 * cols, s.stream.?.handle), "cuMemcpyDtoDAsync");
    }

    fn download(p: *anyopaque, src: u64, dst: []u8) anyerror!void {
        const s: *State = @ptrCast(@alignCast(p));
        try s.stream.?.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = s.d.?, .ptr = src, .len = dst.len }, 0, dst);
    }

    fn streamOf(p: *anyopaque) cuda.abi.Stream {
        const s: *State = @ptrCast(@alignCast(p));
        return s.stream.?.handle;
    }
};

pub fn capture(f: *Forward, row0: u32, n: usize) !void {
    const bound = @max(f.last_rows, f.runner.rows);
    if (n == 0 or row0 > bound or n > bound - row0) return error.LogprobRow;
    if (f.leads()) try f.link.?.send(&.{ op, row0, @intCast(n) });
    const s = &f.logprobs;
    s.d = f.runner.d;
    s.stream = f.runner.stream;
    const own = try @import("partition.zig").share(f.cfg.vocab, f.comm.rank(), f.comm.world());
    const ptr = (f.runner.addressOf("w.logits") orelse return error.Unbound) + 4 * @as(u64, own.len) * row0;
    try s.captured.gather(f.gpa, f.comm, .{ .ptr = s, .vtable = &.{ .scratch = State.scratch, .pad = State.pad, .download = State.download, .stream = State.streamOf } }, ptr, n, f.cfg.vocab);
}

pub fn rows(f: *const Forward, start: usize, picks: []const u32, k: u8, out: []Row) !void {
    if (out.len != picks.len) return error.LogprobRow;
    for (picks, out, start..) |pick, *row, r| row.* = try f.logprobs.captured.row(r, pick, k);
}
