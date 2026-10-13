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

/// The heads and the keys a row sees: (pos - win, pos] (win 0: all), cache row slot * ring + key % ring.
pub const Window = struct { h: usize, hk: usize, d: usize, win: usize, ring: usize, scale: f32 };

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

    pub fn load(d: *const Driver) !WindowAttention {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.window_attention);
        errdefer mod.unload();
        var p: [2]Function = undefined;
        inline for (.{ 4, 8 }, 0..) |hpc, i| {
            p[i] = try mod.function(std.fmt.comptimePrint(prompt_name, .{hpc}));
            try p[i].allowDynamicShared(prompt_smem);
        }
        return .{ .mod = mod, .part_fn = try mod.function("tf_window_part"), .merge_fn = try mod.function("tf_window_merge"), .tc_fn = tc: {
            const f = try mod.function("tf_window_decode_tc");
            try f.allowDynamicShared(tc_smem);
            break :tc f;
        }, .prompt_fns = p };
    }

    pub fn unload(g: *WindowAttention) void {
        g.mod.unload();
    }

    /// Key tiles a decode row's window can touch (its partials a key head).
    pub fn tiles(w: Window) usize {
        return if (w.win > 0) (w.win + 62) / 64 + 1 else (w.ring + 63) / 64 + 1;
    }

    /// The fp32 partials `decode` writes for up to `rows` rows: o, then m and l.
    pub fn scratchBytes(w: Window, rows: usize) usize {
        if (w.d == 128) return 16; // the tensor-core form keeps its partials in shared memory
        const parts = rows * w.hk * tiles(w) * (w.h / w.hk);
        return parts * w.d * 4 + 2 * parts * 4;
    }

    /// n rows q [n, h, d] at pos/slot (int32 [n]) -> out [n, h, d]; `scratch`: scratchBytes(w, n) (d != 128).
    pub fn decode(g: *const WindowAttention, s: Stream, q: u64, kc: u64, vc: u64, out: u64, pos: u64, slot: ?u64, w: Window, scratch: u64, n: usize) !void {
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

    /// n prompt rows of one stream at p0, p0 + 1, ... (d 128); kc/vc that stream's ring, key p at row p % ring.
    pub fn prompt(g: *const WindowAttention, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, n: usize, w: Window) !void {
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
};
