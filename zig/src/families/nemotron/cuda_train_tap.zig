//! A pocket lesson's prompts in a conversation, read at any length: the engine's prompt chunks with the pass's taps.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const tfw = @import("cuda_train_fwd.zig");
const lesson_route = @import("living").lesson_route;
const Trainer = @import("cuda_train.zig").Trainer;
const pack_rows = @import("cuda_train.zig").pack_rows;

/// A sequence of its own for the conversations, and the Mamba states after the last one run (its KV rows stay put).
pub const Tap = struct {
    seq: state.Seq,
    cap: usize, // cache rows: the longest prompt it reads, at least a pass's
    planes: [2]cuda.DeviceBuffer, // the SSM and conv states after `talk`
    talk: u64 = 0, // that conversation's tokens as one number (0: none kept)
    rows: usize = 0, // its length
    runs: usize = 0, // conversations run
    reused: usize = 0, // prompts that found theirs kept

    pub fn deinit(tp: *Tap) void {
        for (&tp.planes) |*b| b.free();
        tp.seq.deinit();
    }
};

/// MoE inputs of `ids` at `want` and `layers` into `out` [want, layers, D] as a keys pass reads them; `context` shared.
pub fn rows(t: *Trainer, ids: []const u32, context: usize, want: []const u32, layers: []const usize, out: []u16) !void {
    const D = t.c.hidden;
    if (ids.len == 0 or context > ids.len or out.len != want.len * layers.len * D) return error.BadTap;
    for (want, 0..) |r, i| if (r >= ids.len or (i > 0 and r <= want[i - 1])) return error.BadTap;
    for (layers) |l| if (l >= t.c.layers or t.c.kinds[l] != .moe) return error.BadTap;
    if (t.tap) |*tp| if (tp.cap < ids.len) {
        tp.deinit();
        t.tap = null;
    };
    if (t.tap == null) t.tap = try init(t, @max(4 * pack_rows, std.mem.alignForward(usize, ids.len, pack_rows)));
    const tp = &t.tap.?;
    const talk = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids[0..context]));
    const m = state.mambaPlanes(t.c);
    var r: usize = 0;
    if (context > 0 and tp.talk == talk and tp.rows == context and (want.len == 0 or want[0] >= context)) {
        // the conversation as it ended: its Mamba states back, its keys and values still in the cache
        for (tp.planes, [_]u64{ tp.seq.ptr[2], tp.seq.ptr[3] }, m) |src, dst, n| try t.e.ops().copy(dst, src.ptr, n);
        tp.reused += 1;
    } else {
        try tp.seq.arena.?.buf.fill8(0, t.e.stream.handle);
        tp.talk = 0;
        try run(t, tp, ids, 0, context, want, &r, layers, out);
        if (context > 0) {
            for (tp.planes, [_]u64{ tp.seq.ptr[2], tp.seq.ptr[3] }, m) |dst, src, n| try t.e.ops().copy(dst.ptr, src, n);
            tp.talk = talk;
            tp.rows = context;
            tp.runs += 1;
        }
    }
    try run(t, tp, ids, context, ids.len, want, &r, layers, out);
    if (r != want.len) return error.BadTap;
}

/// ids[from..to] in prompt chunks of at most a pass's rows, states kept; MoE inputs of want[r.*..] read on the way.
fn run(t: *Trainer, tp: *Tap, ids: []const u32, from: usize, to: usize, want: []const u32, r: *usize, layers: []const usize, out: []u16) !void {
    const c = t.c;
    const e = t.e;
    const o = e.ops();
    const D: usize = c.hidden;
    var b = t.buf;
    inline for (state.seq_fields, 0..) |name, i| @field(b, name) = tp.seq.ptr[i];
    var f = fwd.Forward.init(c, &e.w, &b, o, tp.cap, e.nch, false);
    const scale: tfw.Scale = .{ .rows = t.host.dev + 4 * pack_rows * 4, .from = &t.from };
    var a = from;
    while (a < to) {
        const n = @min(to - a, pack_rows);
        try e.stream.synchronize(); // the host rows below are host memory the last piece's kernels may still read
        try o.upload(b.p_ids, std.mem.sliceAsBytes(ids[a..][0..n]));
        @memset(t.host.slice(f32, 5 * pack_rows)[4 * pack_rows ..][0..n], 1);
        lesson_route.flag(t.sites.latch.slice(u32, lesson_route.max_slots * lesson_route.max_blocks), lesson_route.trainer_slot, lesson_route.Acting.empty);
        @memset(t.score(), 0);
        f.route = lesson_route.trained(n, .{ .decide = 0, .whole = true });
        var w: fwd.Walk = .{ .rows = n, .pos = a };
        try f.chunkBegin(&w);
        for (0..c.layers) |l| {
            try f.chunkPre(&w, l);
            try f.chunkMixer(&w, l);
            if (c.kinds[l] == .moe) {
                try o.copy(t.inputs.dev() + l * pack_rows * D * 2, b.y, n * D * 2);
                try tfw.experts(&f, e.w.blocks[l].moe, n, scale, l);
                w.delta = .moe;
            } else try f.chunkPost(&w, l);
        }
        t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + n };
        var k = r.*;
        while (k < want.len and want[k] < a + n) : (k += 1) {}
        if (k > r.*) try e.stream.synchronize();
        for (want[r.*..k], r.*..) |row, i| for (layers, 0..) |l, j| {
            @memcpy(out[(i * layers.len + j) * D ..][0..D], t.normed(l, row - a));
        };
        r.* = k;
        a += n;
    }
}

fn init(t: *Trainer, cap: usize) !Tap {
    const e = t.e;
    var seq = try state.Seq.init(e.ctx.d, t.c, cap, &.{}, e.stream, null);
    errdefer seq.deinit();
    const m = state.mambaPlanes(t.c);
    var ssm = try cuda.DeviceBuffer.alloc(e.ctx.d, m[0]);
    errdefer ssm.free();
    const conv = try cuda.DeviceBuffer.alloc(e.ctx.d, m[1]);
    std.log.info("slide: prompts in a conversation are read through the engine's prompt chunks, up to {d} tokens", .{cap});
    return .{ .seq = seq, .cap = cap, .planes = .{ ssm, conv } };
}
