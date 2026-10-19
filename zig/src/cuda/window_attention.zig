//! window_attention.cu's sliding-window attention for every CUDA family: decode rows over a ring of keys.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// A row's heads and keys (pos - win, pos] (win 0: all) at cache row slot * ring + key % ring; p32: wide P in halves.
pub const Window = struct { h: usize, hk: usize, d: usize, win: usize, ring: usize, scale: f32, p32: bool = false };

/// Head dims window_rows.cu serves, with any query heads a key head up to 16.
pub fn wide(d: usize) bool {
    return d == 256;
}

/// window_rows.cu's decode chunk: keys [c * chunk_keys, (c + 1) * chunk_keys), merged in chunk order.
pub const chunk_keys = 256;
/// Prompt rows' dynamic shared bytes at 256: 4 slots of 32 keys, then the warps' partial scores.
const rows_smem = [_]u32{4 * 32 * 32 * 16 + 8 * 16 * 32 * 4};

/// The prompt kernel's instances: head dim 128, 8 warps, 4 or 8 query heads a block (`hpc`), 8 staging slots.
const prompt_name = "_ZN16tf_window_prompt12wattn_kernelILi128ELi8ELi{d}ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifii";
const prompt_smem: u32 = 8 * 32 * 16 * 16; // NS * HALF * (D / 8) uint4
/// tf_window_decode_tc: four warps' K and V slots of 32 keys (64 KiB), reused for the warps' partials.
const tc_smem: u32 = 4 * 2 * 32 * 16 * 16;

pub const WindowAttention = struct {
    mod: Module,
    part_fn: Function,
    merge_fn: Function,
    tc_fn: Function,
    prompt_fns: [2]Function, // 4, then 8 query heads a block
    rows: Module, // window_rows.cu
    wide_fns: [rows_smem.len]Wide,

    /// window_rows.cu's kernels at one head dim: decode partials and merge, prompt rows (bf16 P, then P32).
    pub const Wide = struct { part: Function, merge: Function, prompt: [2]Function };

    pub fn load(d: *const Driver) !WindowAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.window_attention);
        errdefer mod.unload();
        var p: [2]Function = undefined;
        inline for (.{ 4, 8 }, 0..) |hpc, i| {
            p[i] = try mod.function(std.fmt.comptimePrint(prompt_name, .{hpc}));
            try p[i].allowDynamicShared(prompt_smem);
        }
        var rows = try Module.load(d, kernels.window_rows);
        errdefer rows.unload();
        var wf: [rows_smem.len]Wide = undefined;
        inline for (.{256}, &wf, rows_smem) |dim, *f, smem| {
            f.* = .{ .part = try rows.function(std.fmt.comptimePrint("tf_wrows_part_{d}", .{dim})), .merge = try rows.function(std.fmt.comptimePrint("tf_wrows_merge_{d}", .{dim})), .prompt = .{ try rows.function(std.fmt.comptimePrint("tf_wrows_prompt_{d}", .{dim})), try rows.function(std.fmt.comptimePrint("tf_wrows_prompt_{d}_p32", .{dim})) } };
            for (f.prompt) |pf| try pf.allowDynamicShared(smem);
        }
        return .{ .mod = mod, .part_fn = try mod.function("tf_window_part"), .merge_fn = try mod.function("tf_window_merge"), .tc_fn = tc: {
            const f = try mod.function("tf_window_decode_tc");
            try f.allowDynamicShared(tc_smem);
            break :tc f;
        }, .prompt_fns = p, .rows = rows, .wide_fns = wf };
    }

    pub fn unload(g: *WindowAttention) void {
        g.rows.unload();
        g.mod.unload();
    }

    /// The dynamic shared bytes window_rows.cu's prompt rows take at head dim d.
    pub fn promptShared(d: usize) u32 {
        _ = d;
        return rows_smem[0];
    }

    pub fn wideFns(g: *const WindowAttention, d: usize) *const Wide {
        _ = d;
        return &g.wide_fns[0];
    }

    /// Chunks a wide decode row's keys can touch: the window's, or (win 0) the ring's or linear cache's positions.
    pub fn chunks(w: Window) usize {
        return if (w.win > 0) (w.win + chunk_keys - 2) / chunk_keys + 1 else (w.ring + chunk_keys - 1) / chunk_keys;
    }

    /// Key tiles a decode row's window can touch (its partials a key head).
    pub fn tiles(w: Window) usize {
        return if (w.win > 0) (w.win + 62) / 64 + 1 else (w.ring + 63) / 64 + 1;
    }

    /// The fp32 partials `decode` writes for up to `rows` rows: o, then m and l.
    pub fn scratchBytes(w: Window, rows: usize) usize {
        if (wide(w.d)) {
            const parts = rows * w.hk * chunks(w) * (w.h / w.hk);
            return parts * w.d * 4 + 2 * parts * 4;
        }
        if (w.d == 128) return 16; // the tensor-core form keeps its partials in shared memory
        const parts = rows * w.hk * tiles(w) * (w.h / w.hk);
        return parts * w.d * 4 + 2 * parts * 4;
    }

    /// n rows q [n, h, d] at pos/slot (int32 [n]) -> out [n, h, d]; `scratch`: scratchBytes(w, n) (d != 128).
    pub fn decode(g: *const WindowAttention, s: Stream, q: u64, kc: u64, vc: u64, out: u64, pos: u64, slot: ?u64, w: Window, scratch: u64, n: usize) !void {
        if (wide(w.d)) return decodeWith(g.wideFns(w.d), s, q, kc, vc, out, pos, slot, w, scratch, n);
        if (w.hk == 0 or w.h % w.hk != 0 or w.h / w.hk > 16 or w.d > 128 or w.ring == 0) return error.WindowShape;
        if (w.win == 0 and w.ring > 4096) return error.WindowShape; // an unwindowed row's tiles come from the ring
        if (n == 0) return;
        if (w.d == 128) { // tensor cores; the heads of a row are one mma tile
            var c: launch_.Args = .{};
            for ([_]u64{ q, kc, vc, out, pos, slot orelse 0 }) |v| c.add(v);
            c.add(@as(i32, @intCast(w.h)));
            c.add(@as(i32, @intCast(w.hk)));
            c.add(@as(i32, @intCast(w.win)));
            c.add(@as(i64, @intCast(w.ring)));
            c.add(w.scale);
            return launch_.launch(g.tc_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = 1 }, .block = .{ .x = 128 }, .shared = tc_smem }, s, &c);
        }
        const t = tiles(w);
        const parts = n * w.hk * t * (w.h / w.hk);
        const po = scratch;
        const pm = po + parts * w.d * 4;
        const pl = pm + parts * 4;
        var a: launch_.Args = .{};
        for ([_]u64{ q, kc, vc, pos, slot orelse 0, po, pm, pl }) |v| a.add(v);
        a.add(@as(i32, @intCast(w.h)));
        a.add(@as(i32, @intCast(w.hk)));
        a.add(@as(i32, @intCast(w.d)));
        a.add(@as(i32, @intCast(w.win)));
        a.add(@as(i64, @intCast(w.ring)));
        a.add(w.scale);
        try launch_.launch(g.part_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = @intCast(t) }, .block = .{ .x = 256 } }, s, &a);
        var b: launch_.Args = .{};
        for ([_]u64{ po, pm, pl, out }) |v| b.add(v);
        b.add(@as(i32, @intCast(w.h)));
        b.add(@as(i32, @intCast(w.hk)));
        b.add(@as(i32, @intCast(w.d)));
        b.add(@as(i32, @intCast(t)));
        try launch_.launch(g.merge_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = 1 }, .block = .{ .x = 256 } }, s, &b);
    }

    /// Wide decode rows by `f` (wideFns, or the contract's controls): chunk partials (scratchBytes), their merge.
    pub fn decodeWith(f: *const Wide, s: Stream, q: u64, kc: u64, vc: u64, out: u64, pos: u64, slot: ?u64, w: Window, scratch: u64, n: usize) !void {
        if (!wide(w.d) or w.hk == 0 or w.h % w.hk != 0 or w.h / w.hk > 16 or w.ring == 0 or w.ring > std.math.maxInt(i32)) return error.WindowShape;
        if (n == 0) return;
        const t = chunks(w);
        const parts = n * w.hk * t * (w.h / w.hk);
        const po = scratch;
        const pm = po + parts * w.d * 4;
        const pl = pm + parts * 4;
        var a: launch_.Args = .{};
        for ([_]u64{ q, kc, vc, pos, slot orelse 0, po, pm, pl }) |v| a.add(v);
        a.add(@as(i32, @intCast(w.h)));
        a.add(@as(i32, @intCast(w.hk)));
        a.add(@as(i32, @intCast(w.win)));
        a.add(@as(i64, @intCast(w.ring)));
        a.add(w.scale);
        try launch_.launch(f.part, .{ .grid = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = @intCast(t) }, .block = .{ .x = 256 } }, s, &a);
        var b: launch_.Args = .{};
        for ([_]u64{ po, pm, pl, out }) |v| b.add(v);
        b.add(@as(i32, @intCast(w.h)));
        b.add(@as(i32, @intCast(w.hk)));
        b.add(@as(i32, @intCast(t)));
        for ([_]u64{ q, kc, vc, pos, slot orelse 0 }) |v| b.add(v);
        b.add(@as(i32, @intCast(w.win)));
        b.add(@as(i64, @intCast(w.ring)));
        b.add(w.scale);
        try launch_.launch(f.merge, .{ .grid = .{ .x = @intCast(n), .y = @intCast(w.hk), .z = @intCast(w.h / w.hk * w.d / 256) }, .block = .{ .x = 256 } }, s, &b);
    }

    /// n prompt rows of one stream at p0, p0 + 1, ... (d 128); kc/vc that stream's ring, key p at row p % ring.
    pub fn prompt(g: *const WindowAttention, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, n: usize, w: Window) !void {
        if (wide(w.d)) return g.promptRows(s, q, kc, vc, out, p0, n, w, null);
        const grp = w.h / w.hk;
        if (w.d != 128 or w.hk == 0 or w.h % w.hk != 0 or grp % 4 != 0) return error.WindowShape;
        if (n == 0) return;
        const hpc: usize = if (grp % 8 == 0) 8 else 4;
        var a: launch_.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, n, w.h, w.hk, grp }) |v| a.add(@as(i32, @intCast(v)));
        a.add(w.scale);
        a.add(@as(i32, @intCast(w.win)));
        a.add(@as(i32, @intCast(w.ring)));
        const rows = 16 * (8 / hpc);
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast((n + rows - 1) / rows), .y = @intCast(w.hk * (grp / hpc)), .z = 1 }, .block = .{ .x = 256 }, .shared = prompt_smem };
        try launch_.launch(g.prompt_fns[if (hpc == 8) 1 else 0], cfg, s, &a);
    }

    /// Wide prompt rows, G up to 16; row i's last key hi[i] (int32, in [p0 + i, p0 + n)) or its position (null).
    pub fn promptRows(g: *const WindowAttention, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, n: usize, w: Window, hi: ?u64) !void {
        if (!wide(w.d)) return error.WindowShape;
        return promptWith(g.wideFns(w.d), s, q, kc, vc, out, p0, n, w, hi);
    }

    /// promptRows by `f` (wideFns, or the contract's controls).
    pub fn promptWith(f: *const Wide, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, n: usize, w: Window, hi: ?u64) !void {
        const grp = if (w.hk > 0) w.h / w.hk else 0;
        if (!wide(w.d) or w.hk == 0 or w.h % w.hk != 0 or grp > 16 or p0 + n > std.math.maxInt(i32)) return error.WindowShape;
        if (n == 0) return;
        const ng = 1024 / w.d; // 16-row groups a block: 8 warps, a 128-dim slice each
        const hb = if (grp % ng == 0) ng else 1;
        const rows = 16 * (ng / hb);
        var a: launch_.Args = .{};
        for ([_]u64{ q, kc, vc, out, hi orelse 0 }) |v| a.add(v);
        for ([_]usize{ p0, n, w.h, w.hk, grp }) |v| a.add(@as(i32, @intCast(v)));
        a.add(w.scale);
        a.add(@as(i32, @intCast(w.win)));
        a.add(@as(i32, @intCast(w.ring)));
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast((n + rows - 1) / rows), .y = @intCast(w.hk * (grp / hb)), .z = 1 }, .block = .{ .x = 256 }, .shared = rows_smem[0] };
        try launch_.launch(f.prompt[@intFromBool(w.p32)], cfg, s, &a);
    }
};
