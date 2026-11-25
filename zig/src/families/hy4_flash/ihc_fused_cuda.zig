//! Fused iHC boundary launchers (`hy4_ihc_fused`): 1-launch ticket/coop (`R <= 64`) and prefill (`R > 64`).
const std = @import("std");
const cuda = @import("cuda");
pub const hc = @import("hc_cuda.zig");
const hc_cuda = hc;

const Stream = cuda.Stream;
const Function = cuda.Function;
const DeviceBuffer = cuda.DeviceBuffer;

pub const hidden = 3072;
const nch = 24;
const nu = 48;
const kv = 9;
const max_r = 16;
const max_ticket_r = 64;

pub const Mode = enum(u3) { boundary = 0, entry = 1, head = 2, post_only = 3, head_only = 4 };

/// Exact mirror of `hy4ihc::Args` in `ihc_fused.cuh` (passed by value as `__grid_constant__`).
pub const Args = extern struct {
    g: u64 = 0, // fp32 [world][R][H]
    gs: i64 = 0, // rank stride in floats
    world: i32 = 1,
    r: i32 = 0,
    x: u64 = 0, // bf16 [R][4][H], updated in place
    post_prev: u64 = 0, // fp32 [R][4]
    post_nw: u64 = 0, // bf16 [H]
    fn_: u64 = 0, // fp32 [M][12288]
    base: u64 = 0, // fp32 [M]
    scale: u64 = 0, // fp32 [2] or [1]
    next_nw: u64 = 0, // bf16 [H]
    a: u64 = 0, // bf16 [R][H]
    post_next: u64 = 0, // fp32 [R][4] (may alias post_prev)
    wp: u64 = 0,
    cnt: u64 = 0,
    ssy: u64 = 0,
    sso: u64 = 0,
    yb: u64 = 0,
    ob: u64 = 0,
    dbg_tot: u64 = 0,
    rpg: i32 = 1,
    eps: f32 = 1e-5,
    hc_eps: f32 = 1e-6,
    mag: f32 = 2.0,
    round_in: i32 = 1,
    pad: i32 = 0,

    comptime {
        std.debug.assert(@sizeOf(Args) == 176);
    }
};

pub fn ticketSmem(upc: u32, rpg: u32, m: u32) u32 {
    const pre: u32 = if (m > 0) hidden * 2 + 64 else 0;
    const main_b: u32 = (upc * m * 256 + rpg * 4 * upc * 64 + rpg * upc * 64) * 4 + rpg * 4 * upc * 64 * 2 + upc * 64 * 2 + rpg * 16;
    const fin_b: u32 = if (m > 0) 3840 + rpg * hidden * 2 else 0;
    return pre + @max(main_b, fin_b);
}

pub fn preTiledSmem(rb: u32) u32 {
    return 3 * rb * 512 * 2 + 8 * rb * kv * 4 + rb * 9 * 4 + rb * 4 * 4 + 64 * 4;
}

fn modeM(mode: Mode) u32 {
    return switch (mode) {
        .head, .head_only => 4,
        .post_only => 0,
        .boundary, .entry => 8,
    };
}

pub const Ihc = struct {
    mod: cuda.Module,
    ticket: [5]Function, // boundary, entry, head, post_only, head_only
    coop: [4]Function, // boundary, entry, head, head_only
    post_vec: Function,
    pre_rb2: Function,
    pre_rb4: Function,
    head_tiled: Function,
    fused_prefill_m8: ?Function = null,
    fused_prefill_m4: ?Function = null,
    wp: DeviceBuffer, // fp32 [64][9][24][8]
    cnt: DeviceBuffer, // i32 [16], zero at rest, self-cleaning
    ssy: DeviceBuffer, // fp32 [16][8]
    sso: DeviceBuffer, // fp32 [16][8]
    yb: DeviceBuffer, // fp32 [16][3072]
    ob: DeviceBuffer, // bf16 [16][3072]

    pub fn load(d: *const cuda.Driver) !Ihc {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_ihc_fused);
        errdefer mod.unload();
        var ihc: Ihc = undefined;
        ihc.mod = mod;
        const maxb = ticketSmem(4, 4, 8);
        const tnames = [5][:0]const u8{ "hy4_ihc_fused_boundary", "hy4_ihc_entry", "hy4_ihc_epilogue_head", "hy4_ihc_post_only", "hy4_ihc_head_only" };
        for (tnames, 0..) |nm, i| {
            ihc.ticket[i] = try mod.function(nm);
            try ihc.ticket[i].allowDynamicShared(maxb);
        }
        const cnames = [4][:0]const u8{ "hy4_ihc_fused_boundary_coop", "hy4_ihc_entry_coop", "hy4_ihc_epilogue_head_coop", "hy4_ihc_head_only_coop" };
        for (cnames, 0..) |nm, i| ihc.coop[i] = try mod.function(nm);
        ihc.post_vec = try mod.function("hy4_ihc_post_vec");
        ihc.pre_rb2 = try mod.function("hy4_ihc_pre_tiled_rb2");
        try ihc.pre_rb2.allowDynamicShared(preTiledSmem(2));
        ihc.pre_rb4 = try mod.function("hy4_ihc_pre_tiled_rb4");
        try ihc.pre_rb4.allowDynamicShared(preTiledSmem(4));
        ihc.head_tiled = try mod.function("hy4_ihc_head_tiled");
        try ihc.head_tiled.allowDynamicShared(preTiledSmem(8));
        ihc.fused_prefill_m8 = mod.function("hy4_ihc_fused_prefill_m8") catch null;
        ihc.fused_prefill_m4 = mod.function("hy4_ihc_fused_prefill_m4") catch null;

        ihc.wp = try DeviceBuffer.alloc(d, max_ticket_r * kv * nch * 8 * 4);
        errdefer ihc.wp.free();
        const zeros: [16 * 4]u8 = @splat(0);
        ihc.cnt = try DeviceBuffer.fromHost(d, &zeros);
        errdefer ihc.cnt.free();
        ihc.ssy = try DeviceBuffer.alloc(d, max_r * 8 * 4);
        errdefer ihc.ssy.free();
        ihc.sso = try DeviceBuffer.alloc(d, max_r * 8 * 4);
        errdefer ihc.sso.free();
        ihc.yb = try DeviceBuffer.alloc(d, max_r * hidden * 4);
        errdefer ihc.yb.free();
        ihc.ob = try DeviceBuffer.alloc(d, max_r * hidden * 2);
        return ihc;
    }

    pub fn unload(ihc: *Ihc) void {
        ihc.ob.free();
        ihc.yb.free();
        ihc.sso.free();
        ihc.ssy.free();
        ihc.cnt.free();
        ihc.wp.free();
        ihc.mod.unload();
        ihc.* = undefined;
    }

    pub const Params = struct {
        mode: Mode,
        rows: usize,
        g: u64 = 0, // summed partial fp32 [rows, 3072] (world = 1)
        x: u64, // bf16 [rows, 4, 3072], updated in place
        post_prev: u64 = 0,
        post_nw: u64 = 0, // bf16 [3072]
        mix: ?hc_cuda.Mix = null,
        next_nw: u64 = 0, // bf16 [3072]
        a: u64 = 0, // bf16 [rows, 3072]
        post_next: u64 = 0,
        eps: f32,
        hc_eps: f32,
        mag: f32,
        round_in: bool = true,
    };

    pub fn run(ihc: *const Ihc, s: Stream, p: Params) !void {
        if (p.rows == 0) return;
        const r_u32: u32 = @intCast(p.rows);
        var a: Args = .{
            .g = p.g,
            .gs = @intCast(p.rows * hidden),
            .world = 1,
            .r = @intCast(p.rows),
            .x = p.x,
            .post_prev = p.post_prev,
            .post_nw = p.post_nw,
            .fn_ = if (p.mix) |m| m.fn_ else 0,
            .base = if (p.mix) |m| m.base else 0,
            .scale = if (p.mix) |m| m.scale else 0,
            .next_nw = p.next_nw,
            .a = p.a,
            .post_next = p.post_next,
            .wp = ihc.wp.ptr,
            .cnt = ihc.cnt.ptr,
            .ssy = ihc.ssy.ptr,
            .sso = ihc.sso.ptr,
            .yb = ihc.yb.ptr,
            .ob = ihc.ob.ptr,
            .rpg = 1,
            .eps = p.eps,
            .hc_eps = p.hc_eps,
            .mag = p.mag,
            .round_in = @intFromBool(p.round_in),
        };
        if (p.rows <= 2) {
            const upc: u32 = if (p.rows == 1) 2 else 4;
            a.rpg = 1;
            var la: cuda.Args = .{};
            la.add(a);
            try cuda.launch.launch(ihc.ticket[@backingInt(p.mode)], .{
                .grid = .{ .x = nu / upc, .y = r_u32, .z = 1 },
                .block = .{ .x = 256 },
                .shared = ticketSmem(upc, 1, modeM(p.mode)),
            }, s, &la);
            return;
        }
        if (p.rows <= max_r) {
            const ci: usize = switch (p.mode) {
                .boundary => 0,
                .entry => 1,
                .head => 2,
                .head_only => 3,
                .post_only => {
                    var la: cuda.Args = .{};
                    la.add(a);
                    try cuda.launch.launch(ihc.post_vec, .{ .grid = .{ .x = r_u32 }, .block = .{ .x = 256 } }, s, &la);
                    return;
                },
            };
            var la: cuda.Args = .{};
            la.add(a);
            try cuda.launch.launch(ihc.coop[ci], .{
                .grid = .{ .x = nu },
                .block = .{ .x = 256 },
                .cooperative = true,
            }, s, &la);
            return;
        }
        if (p.rows <= max_ticket_r) {
            if (p.mode == .post_only) {
                var la: cuda.Args = .{};
                la.add(a);
                try cuda.launch.launch(ihc.post_vec, .{ .grid = .{ .x = r_u32 }, .block = .{ .x = 256 } }, s, &la);
                return;
            }
            // 1-launch fused ticket boundary for R = 17..64 (rpg = 4, upc = 2, cnt[0..15]).
            const upc: u32 = 2;
            const rpg: u32 = 4;
            a.rpg = @intCast(rpg);
            var la: cuda.Args = .{};
            la.add(a);
            try cuda.launch.launch(ihc.ticket[@backingInt(p.mode)], .{
                .grid = .{ .x = nu / upc, .y = (r_u32 + rpg - 1) / rpg, .z = 1 },
                .block = .{ .x = 256 },
                .shared = ticketSmem(upc, rpg, modeM(p.mode)),
            }, s, &la);
            return;
        }
        // Prefill (R > 64): 1-pass fused post+pre/head when loaded (0-bit equal, skips X' DRAM re-read).
        if (p.mode == .boundary) {
            if (ihc.fused_prefill_m8) |f| {
                var la: cuda.Args = .{};
                la.add(a);
                try cuda.launch.launch(f, .{ .grid = .{ .x = r_u32 }, .block = .{ .x = 256 } }, s, &la);
                return;
            }
        } else if (p.mode == .head) {
            if (ihc.fused_prefill_m4) |f| {
                var la: cuda.Args = .{};
                la.add(a);
                try cuda.launch.launch(f, .{ .grid = .{ .x = r_u32 }, .block = .{ .x = 256 } }, s, &la);
                return;
            }
        }
        // Fallback / entry / post_only / head_only: post_vec + pre_tiled_rb2/rb4 (or head_tiled).
        if (p.mode == .boundary or p.mode == .post_only or p.mode == .head) {
            var la: cuda.Args = .{};
            la.add(a);
            try cuda.launch.launch(ihc.post_vec, .{ .grid = .{ .x = r_u32 }, .block = .{ .x = 256 } }, s, &la);
        }
        if (p.mode == .boundary or p.mode == .entry) {
            var la: cuda.Args = .{};
            la.add(a);
            if (p.rows <= 96) {
                try cuda.launch.launch(ihc.pre_rb2, .{
                    .grid = .{ .x = (r_u32 + 1) / 2 },
                    .block = .{ .x = 256 },
                    .shared = preTiledSmem(2),
                }, s, &la);
            } else {
                try cuda.launch.launch(ihc.pre_rb4, .{
                    .grid = .{ .x = (r_u32 + 3) / 4 },
                    .block = .{ .x = 256 },
                    .shared = preTiledSmem(4),
                }, s, &la);
            }
        } else if (p.mode == .head or p.mode == .head_only) {
            var la: cuda.Args = .{};
            la.add(a);
            try cuda.launch.launch(ihc.head_tiled, .{
                .grid = .{ .x = (r_u32 + 7) / 8 },
                .block = .{ .x = 256 },
                .shared = preTiledSmem(8),
            }, s, &la);
        }
    }
};

test "ihc_fused_cuda: Args layout and dynamic shared byte formulas match contract records" {
    try std.testing.expectEqual(@as(usize, 176), @sizeOf(Args));
    try std.testing.expectEqual(@as(u32, 46672), ticketSmem(4, 1, 8));
    try std.testing.expectEqual(@as(u32, 30288), ticketSmem(4, 1, 4));
    try std.testing.expectEqual(@as(u32, 34624), ticketSmem(2, 4, 4));
    try std.testing.expectEqual(@as(u32, 7696), ticketSmem(4, 1, 0));
    try std.testing.expectEqual(@as(u32, 7080), preTiledSmem(2));
    try std.testing.expectEqual(@as(u32, 13904), preTiledSmem(4));
    try std.testing.expectEqual(@as(u32, 27552), preTiledSmem(8));
}
