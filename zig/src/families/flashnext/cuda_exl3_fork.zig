//! A multi-part EXL3 projection's parts on parallel streams (a DeltaNet layer's q|k|v, z, b, a; attention's
//! q, k, v, indexer): each EXL3 part after the first runs on its own side stream with its own Dense scratch (rotated
//! input, split partials, counters, stitch buffer), forked from and joined back into the pass's stream with events,
//! so a captured graph holds them as parallel branches. Every part is the same kernels on the same inputs writing
//! its own columns: the same bits (tf-cuda-test flashnext-exl3-graph eq). A decode projection launches 80-192
//! programs that leave an SM wave partly empty; the next part fills it.
//! TF_FLASHNEXT_EXL3_FORK=0 runs the parts one after another (upstream order).

const std = @import("std");
const cuda = @import("cuda");
const load = @import("cuda_exl3_load.zig");

pub const branches = 3;
/// Rows a forked projection takes at most (decode rounds and verify windows; prompts keep one stream).
pub const max_rows: usize = 16;

pub const Fork = struct {
    side: [branches]cuda.Stream,
    dense: [branches]cuda.exl3.Dense,
    tmp: [branches]cuda.DeviceBuffer,
    ev_fork: cuda.Event,
    ev_join: [branches]cuda.Event,
    tmp_bytes: usize,
};

var forks: std.AutoHashMapUnmanaged(*load.Ops, *Fork) = .empty;

pub fn enabled() bool {
    if (force) |f| return f;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_FORK") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// A test's override of TF_FLASHNEXT_EXL3_FORK.
pub var force: ?bool = null;

/// Ops.reserve's hook: the branches' streams, kernels and scratch for `o`'s widest projection, before any pass.
pub fn reserve(o: *load.Ops) !void {
    if (o.max_k == 0 or forks.contains(o)) return;
    const f = try o.gpa.create(Fork);
    errdefer o.gpa.destroy(f);
    f.tmp_bytes = max_rows * @max(o.max_part_n, 128) * 2;
    var made: usize = 0;
    errdefer for (0..made) |i| {
        f.dense[i].unload();
        f.side[i].deinit();
        f.tmp[i].free();
        f.ev_join[i].deinit();
    };
    f.ev_fork = try cuda.Event.init(o.d, false);
    errdefer f.ev_fork.deinit();
    for (0..branches) |i| {
        f.side[i] = try cuda.Stream.init(o.d, true);
        f.dense[i] = try cuda.exl3.Dense.load(o.d);
        f.tmp[i] = try cuda.DeviceBuffer.alloc(o.d, f.tmp_bytes);
        f.ev_join[i] = try cuda.Event.init(o.d, false);
        made += 1;
        const de = &f.dense[i];
        de.xh = try cuda.DeviceBuffer.alloc(o.d, 128 * o.max_k * 2);
        if (o.max_z > 0) de.z = try cuda.DeviceBuffer.alloc(o.d, o.max_z);
        de.counters = try cuda.DeviceBuffer.alloc(o.d, o.max_counters);
    }
    try forks.put(o.gpa, o, f);
}

pub fn release(o: *load.Ops) void {
    const kv = forks.fetchRemove(o) orelse return;
    const f = kv.value;
    for (0..branches) |i| {
        f.side[i].synchronize() catch {};
        f.dense[i].unload();
        f.side[i].deinit();
        f.tmp[i].free();
        f.ev_join[i].deinit();
    }
    f.ev_fork.deinit();
    o.gpa.destroy(f);
}

fn copy2d(s: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try s.d.check(s.d.api.cuMemcpy2DAsync_v2(&m, s.handle), "cuMemcpy2DAsync");
}

/// Pass.lane()'s parts loop (bf16 out, sk 1) with the EXL3 parts after the first on side streams; false when the
/// projection does not fork (one EXL3 part, too many rows, no branches reserved), and the caller runs its loop.
/// `run(part, y)` runs a part on the pass's stream as the caller's loop would.
pub fn lane(p: anytype, x: u64, e: *const load.Proj, out: u64, m: usize, comptime runMain: anytype) !bool {
    if (!enabled() or m > max_rows) return false;
    var n_exl3: usize = 0;
    for (e.parts) |pt| n_exl3 += @intFromBool(pt.kind == .exl3);
    if (n_exl3 < 2) return false;
    const f = forks.get(e.ops) orelse return false;
    for (e.parts) |pt| if (pt.kind == .exl3 and m * pt.n * 2 > f.tmp_bytes and pt.n != e.n and m != 1) return false;
    const s = p.stream;
    try f.ev_fork.record(s);
    var b: usize = 0;
    var first = true;
    for (e.parts) |pt| {
        const w = switch (pt.kind) {
            .exl3 => |w| w,
            .native => {
                try runMain(p, e, pt, x, m, out);
                continue;
            },
        };
        if (first or b == branches) {
            first = false;
            try runMain(p, e, pt, x, m, out);
            continue;
        }
        const st = f.side[b];
        try st.wait(f.ev_fork);
        const de = &f.dense[b];
        if (pt.n == e.n or m == 1) {
            try de.decode(st, w, x, .bf16, m, out + pt.col * 2, .bf16);
        } else {
            const tmp = f.tmp[b].ptr;
            try de.decode(st, w, x, .bf16, m, tmp, .bf16);
            try copy2d(st, out + pt.col * 2, e.n * 2, tmp, pt.n * 2, pt.n * 2, m);
        }
        try f.ev_join[b].record(st);
        b += 1;
    }
    for (f.ev_join[0..b]) |ev| try s.wait(ev);
    return true;
}
