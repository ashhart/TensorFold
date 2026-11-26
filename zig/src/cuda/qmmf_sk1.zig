//! One physical CTA keeps both original NVFP4 K-slice chains and their final FP32 addition.
const std = @import("std");
const q = @import("qmmf.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch_ = @import("launch.zig");

pub const Set = [2][2]Function;

pub fn on() !bool {
    return @import("sys").environment.read(bool, "TF_QWEN_NVFP4_SK1", false);
}

pub fn symbol(comptime bm: u32, comptime out: q.Out) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi0ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ bm, @intFromBool(out == .f32) });
}

pub fn load(l: *const q.Lane) !Set {
    var set: Set = undefined;
    inline for (.{ q.Out.bf16, q.Out.f32 }, 0..) |out, oi| inline for (.{ 16, 32 }, 0..) |bm, bi| {
        const f = try l.mod.function(symbol(bm, out));
        try f.allowDynamicShared(q.sharedBytes(.fp4, bm));
        set[oi][bi] = f;
    };
    return set;
}

/// Only Qwen's ModelOpt loader calls this; an unset switch loads no new functions and changes no lane state.
pub fn apply(l: *const q.Lane, capability: u32) !?Set {
    if (!try on()) return null;
    if (capability != 120 and capability != 121) return error.QwenNvfp4Sk1NeedsSm12x;
    if (l.half_n != null or try @import("sys").environment.read(bool, "TF_Q35_GROUP_PROJ", false))
        return error.QwenNvfp4Sk1ScheduleConflict;
    return try load(l);
}

pub fn eligible(m: usize, w: q.Weight) bool {
    return m > 0 and m < q.fused_rows and w.mode == .fp4 and w.n == 6144 and w.k == 5120 and
        w.npad == 6144 and q.slices(w) == 2;
}

/// The same tile, stage, arguments and epilogue as main, with FUSE retaining the original logical SK=2.
pub fn matmul(loaded: ?Set, l: q.Lane, s: Stream, x: u64, ldx: usize, m: usize, w: q.Weight, y: u64, out: q.Out) !bool {
    const set = loaded orelse return false;
    if (!eligible(m, w)) return false;
    const bm = q.bucket(m);
    const f = if (bm == 64) (if (out == .f32) l.f32_fns else l.fns)[0][6] else set[@backingInt(out)][@intFromBool(bm == 32)];
    const rows_t = (m + bm - 1) / bm;
    const band = @max(1, @min(rows_t, (12 << 20) / (bm * w.k * 2)));
    var a: launch_.Args = .{};
    a.add(x);
    a.add(w.codes);
    a.add(w.scales);
    a.add(w.scale);
    a.add(y);
    a.add(@as(u64, 0));
    a.add(@as(i32, @intCast(m)));
    a.add(@as(i32, @intCast(w.n)));
    a.add(@as(i32, @intCast(w.k)));
    a.add(@as(i32, 2));
    a.add(@as(i32, @intCast(w.npad)));
    a.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
    a.add(@as(i32, @intCast(band)));
    try launch_.launch(f, .{ .grid = .{ .x = @intCast(rows_t * (w.n / 64)) }, .block = .{ .x = 128 }, .shared = q.sharedBytes(.fp4, bm), .pdl = l.pdl }, s, &a);
    return true;
}
