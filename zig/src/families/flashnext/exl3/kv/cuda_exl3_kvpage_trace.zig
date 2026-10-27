//! The KV selection trace a test records from the attention pass (what each row's selection read).

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");

const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const att_rows = parts.att_rows;

/// Records: u32 layer (attention index), u32 first position, u32 rows, u32 kind (0 prompt, 1 decode round).
pub const Trace = struct {
    gpa: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,
    /// Record prompt rows too (else only decode rounds).
    prompt: bool = true,
    rows: usize = 0,
    ids: []i32 = &.{},
    nk: []i32 = &.{},
    sp: []i32 = &.{},

    pub fn init(gpa: std.mem.Allocator, prompt: bool) !Trace {
        return .{ .gpa = gpa, .prompt = prompt, .ids = try gpa.alloc(i32, att_rows * idw), .nk = try gpa.alloc(i32, att_rows), .sp = try gpa.alloc(i32, att_rows) };
    }

    pub fn deinit(t: *Trace) void {
        t.out.deinit(t.gpa);
        t.gpa.free(t.ids);
        t.gpa.free(t.nk);
        t.gpa.free(t.sp);
    }

    fn word(t: *Trace, v: u32) !void {
        try t.out.appendSlice(t.gpa, std.mem.asBytes(&v));
    }

    /// The `m` rows at `pos` the select kernel just listed in s.qids / qnk / qspr (one sync and three small copies).
    pub fn record(t: *Trace, stream: cuda.Stream, ai: usize, pos: usize, m: usize, decode: bool, qids: u64, qnk: u64, qspr: u64) !void {
        if (!decode and !t.prompt) return;
        try stream.synchronize();
        const d = stream.d;
        try d.check(d.api.cuMemcpyDtoH_v2(t.ids.ptr, qids, m * idw * 4), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(t.nk.ptr, qnk, m * 4), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(t.sp.ptr, qspr, m * 4), "cuMemcpyDtoH");
        try t.word(@intCast(ai));
        try t.word(@intCast(pos));
        try t.word(@intCast(m));
        try t.word(@intFromBool(decode));
        for (0..m) |r| {
            if (t.sp[r] == 0) {
                try t.out.appendSlice(t.gpa, &[2]u8{ 0, 0 });
                continue;
            }
            const cnt: u16 = @intCast(top);
            try t.out.appendSlice(t.gpa, std.mem.asBytes(&cnt));
            const row = t.ids[r * idw ..][0 .. top * ratio];
            for (0..top) |j| {
                const b: u16 = @intCast(@divTrunc(row[j * ratio], @as(i32, ratio)));
                try t.out.appendSlice(t.gpa, std.mem.asBytes(&b));
            }
        }
        t.rows += m;
    }
};
