//! Multi-GPU Level Zero P2P collectives (TP=1, 2, 4) on Intel Xe2 using cross-device events and a 4-slot P2P ring.
const std = @import("std");
const ze = @import("ze");

pub const MAX_TP = 4;
pub const RING_SLOTS = 4;
pub const MAX_PIPE_STEPS = 256; // 64 layers * 2 collectives = 128 steps per forward; 256 leaves ample headroom
/// Threshold above which 2-shot P2P ReduceScatter + DMA AllGather replaces 1-shot P2P AllReduce.
pub const RS_THRESHOLD_BYTES: usize = 512 * 1024;

pub const Kernels = struct {
    ar2_f16: ze.H,
    ar4_f16: ze.H,
    rs2_f16: ze.H,
    rs4_f16: ze.H,
    ar2_bf16: ze.H,
    ar4_bf16: ze.H,
    ar2_f32: ze.H,
    ar4_f32: ze.H,
    rs4_f32: ze.H,
};

pub const TpComm = struct {
    tp: usize,
    max_bytes: usize,
    send_buf: [MAX_TP][RING_SLOTS]*anyopaque = undefined,
    rs_buf: [MAX_TP][RING_SLOTS]*anyopaque = undefined,
    k: [MAX_TP]Kernels = undefined,
    // [step * 2 * MAX_TP + kind * MAX_TP + rank], kind 0 = send ready, kind 1 = rs ready
    events: [MAX_PIPE_STEPS * 2 * MAX_TP]ze.H = @splat(null),
    step: usize = 0,

    pub fn init(gpus: []ze.Gpu, tp_bin: []const u8, max_bytes: usize) !TpComm {
        if (gpus.len != 1 and gpus.len != 2 and gpus.len != 4) return error.InvalidTpDegree;
        var c: TpComm = .{
            .tp = gpus.len,
            .max_bytes = (max_bytes + 4095) / 4096 * 4096,
        };
        if (c.tp == 1) return c;

        const rs_cap = (c.max_bytes / c.tp + 4095) / 4096 * 4096;
        for (gpus, 0..) |*g, r| {
            for (0..RING_SLOTS) |s| {
                c.send_buf[r][s] = try g.deviceAlloc(c.max_bytes);
                c.rs_buf[r][s] = try g.deviceAlloc(rs_cap);
            }
            const m = try g.module(tp_bin);
            c.k[r] = .{
                .ar2_f16 = try loadK(g, m, "tp2_allreduce_f16"),
                .ar4_f16 = try loadK(g, m, "tp4_allreduce_f16"),
                .rs2_f16 = try loadK(g, m, "tp2_rs_f16"),
                .rs4_f16 = try loadK(g, m, "tp4_rs_f16"),
                .ar2_bf16 = try loadK(g, m, "tp2_allreduce_bf16"),
                .ar4_bf16 = try loadK(g, m, "tp4_allreduce_bf16"),
                .ar2_f32 = try loadK(g, m, "tp2_allreduce_f32"),
                .ar4_f32 = try loadK(g, m, "tp4_allreduce_f32"),
                .rs4_f32 = try loadK(g, m, "tp4_rs_f32"),
            };
        }
        // Make all ring buffers P2P-resident on every peer device in the shared context.
        for (gpus) |*g_dst| {
            for (0..c.tp) |r_src| {
                for (0..RING_SLOTS) |s| {
                    try g_dst.makeResident(c.send_buf[r_src][s], c.max_bytes);
                    try g_dst.makeResident(c.rs_buf[r_src][s], rs_cap);
                }
            }
        }
        try ze.Gpu.sharedEvents(gpus, c.events[0 .. MAX_PIPE_STEPS * 2 * c.tp]);
        return c;
    }

    fn loadK(g: *ze.Gpu, m: ze.H, name: [*:0]const u8) !ze.H {
        const k = try g.kernel(m, name);
        try ze.check(g.api.zeKernelSetGroupSize(k, 256, 1, 1));
        try g.indirect(k);
        return k;
    }

    /// Rank `r`'s current P2P send buffer so the preceding GEMM can write directly into it without an extra copy.
    pub fn currentSendBuf(c: *const TpComm, r: usize) *anyopaque {
        return c.send_buf[r][c.step % RING_SLOTS];
    }

    fn readyEv(c: *TpComm, s: usize, r: usize) ze.H {
        return c.events[(s * 2 + 0) * c.tp + r];
    }

    fn rsEv(c: *TpComm, s: usize, r: usize) ze.H {
        return c.events[(s * 2 + 1) * c.tp + r];
    }

    /// Synchronizes all `gpus` command lists and resets all consumed pipeline events on the host.
    pub fn syncAll(c: *TpComm, gpus: []ze.Gpu, lists: []const ze.H) !void {
        for (gpus, lists) |*g, l| {
            try ze.check(g.api.zeCommandListHostSynchronize(l, std.math.maxInt(u64)));
        }
        if (c.tp > 1 and c.step > 0) {
            const total_evs = c.step * 2 * c.tp;
            for (c.events[0..total_evs]) |ev| {
                try ze.check(gpus[0].api.zeEventHostReset(ev));
            }
            c.step = 0;
        }
    }

    /// Queues an async cross-GPU FP16 AllReduce from `src_opt` (or `currentSendBuf(r)` if null) into `dst[r]`.
    pub fn allReduceF16(c: *TpComm, gpus: []ze.Gpu, lists: []const ze.H, src_opt: ?[]const *anyopaque, dst: []const *anyopaque, count_f16: usize) !void {
        if (c.tp == 1) {
            if (src_opt) |src| if (src[0] != dst[0]) {
                try ze.check(gpus[0].api.zeCommandListAppendMemoryCopy(lists[0], dst[0], src[0], count_f16 * 2, null, 0, null));
            };
            return;
        }
        if (c.step >= MAX_PIPE_STEPS) try c.syncAll(gpus, lists);
        const bytes = count_f16 * 2;
        if (bytes > c.max_bytes or count_f16 % 4 != 0) return error.InvalidAllReduceSize;

        const s = c.step;
        const slot = s % RING_SLOTS;
        c.step += 1;

        var waits: [MAX_TP]ze.H = undefined;
        for (0..c.tp) |r| {
            const sb = c.send_buf[r][slot];
            if (src_opt) |src| if (src[r] != sb) {
                try ze.check(gpus[r].api.zeCommandListAppendMemoryCopy(lists[r], sb, src[r], bytes, null, 0, null));
            };
            const ev = c.readyEv(s, r);
            try ze.check(gpus[r].api.zeCommandListAppendSignalEvent(lists[r], ev));
            waits[r] = ev;
        }

        const n4: u32 = @intCast(count_f16 / 4);
        const use_rs = bytes > RS_THRESHOLD_BYTES and (n4 % @as(u32, @intCast(c.tp)) == 0);

        if (!use_rs) {
            const groups: ze.GroupCount = .{ .x = (n4 + 255) / 256 };
            for (0..c.tp) |r| {
                const g = &gpus[r];
                if (c.tp == 2) {
                    const k = c.k[r].ar2_f16;
                    try g.arg(k, 0, c.send_buf[0][slot]);
                    try g.arg(k, 1, c.send_buf[1][slot]);
                    try g.arg(k, 2, dst[r]);
                    try g.arg(k, 3, @as(i32, @intCast(n4)));
                    try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, null, 2, &waits));
                } else {
                    const k = c.k[r].ar4_f16;
                    try g.arg(k, 0, c.send_buf[0][slot]);
                    try g.arg(k, 1, c.send_buf[1][slot]);
                    try g.arg(k, 2, c.send_buf[2][slot]);
                    try g.arg(k, 3, c.send_buf[3][slot]);
                    try g.arg(k, 4, dst[r]);
                    try g.arg(k, 5, @as(i32, @intCast(n4)));
                    try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, null, 4, &waits));
                }
            }
        } else {
            const slice_n4: u32 = n4 / @as(u32, @intCast(c.tp));
            const slice_bytes: usize = @as(usize, slice_n4) * 8;
            const groups: ze.GroupCount = .{ .x = (slice_n4 + 255) / 256 };
            var rs_waits: [MAX_TP]ze.H = undefined;
            for (0..c.tp) |r| {
                const g = &gpus[r];
                const off4: i32 = @intCast(@as(u32, @intCast(r)) * slice_n4);
                const rev = c.rsEv(s, r);
                rs_waits[r] = rev;
                if (c.tp == 2) {
                    const k = c.k[r].rs2_f16;
                    try g.arg(k, 0, c.send_buf[0][slot]);
                    try g.arg(k, 1, c.send_buf[1][slot]);
                    try g.arg(k, 2, c.rs_buf[r][slot]);
                    try g.arg(k, 3, off4);
                    try g.arg(k, 4, @as(i32, @intCast(slice_n4)));
                    try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, rev, 2, &waits));
                } else {
                    const k = c.k[r].rs4_f16;
                    try g.arg(k, 0, c.send_buf[0][slot]);
                    try g.arg(k, 1, c.send_buf[1][slot]);
                    try g.arg(k, 2, c.send_buf[2][slot]);
                    try g.arg(k, 3, c.send_buf[3][slot]);
                    try g.arg(k, 4, c.rs_buf[r][slot]);
                    try g.arg(k, 5, off4);
                    try g.arg(k, 6, @as(i32, @intCast(slice_n4)));
                    try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, rev, 4, &waits));
                }
            }
            for (0..c.tp) |r| {
                try ze.check(gpus[r].api.zeCommandListAppendWaitOnEvents(lists[r], @intCast(c.tp), &rs_waits));
                for (0..c.tp) |src_r| {
                    const dst_slice: *anyopaque = @ptrFromInt(@intFromPtr(dst[r]) + src_r * slice_bytes);
                    try ze.check(gpus[r].api.zeCommandListAppendMemoryCopy(lists[r], dst_slice, c.rs_buf[src_r][slot], slice_bytes, null, 0, null));
                }
            }
        }
    }

    /// Queues an asynchronous cross-GPU deterministic FP32 AllReduce (used by FlashNext / FP32 partials).
    pub fn allReduceF32(c: *TpComm, gpus: []ze.Gpu, lists: []const ze.H, src_opt: ?[]const *anyopaque, dst: []const *anyopaque, count_f32: usize) !void {
        if (c.tp == 1) {
            if (src_opt) |src| if (src[0] != dst[0]) {
                try ze.check(gpus[0].api.zeCommandListAppendMemoryCopy(lists[0], dst[0], src[0], count_f32 * 4, null, 0, null));
            };
            return;
        }
        if (c.step >= MAX_PIPE_STEPS) try c.syncAll(gpus, lists);
        const bytes = count_f32 * 4;
        if (bytes > c.max_bytes or count_f32 % 4 != 0) return error.InvalidAllReduceSize;

        const s = c.step;
        const slot = s % RING_SLOTS;
        c.step += 1;

        var waits: [MAX_TP]ze.H = undefined;
        for (0..c.tp) |r| {
            const sb = c.send_buf[r][slot];
            if (src_opt) |src| if (src[r] != sb) {
                try ze.check(gpus[r].api.zeCommandListAppendMemoryCopy(lists[r], sb, src[r], bytes, null, 0, null));
            };
            const ev = c.readyEv(s, r);
            try ze.check(gpus[r].api.zeCommandListAppendSignalEvent(lists[r], ev));
            waits[r] = ev;
        }

        const n4: u32 = @intCast(count_f32 / 4);
        const use_rs = c.tp == 4 and bytes > RS_THRESHOLD_BYTES and (n4 % 4 == 0);
        if (!use_rs) {
            const groups: ze.GroupCount = .{ .x = (n4 + 255) / 256 };
            for (0..c.tp) |r| {
                const g = &gpus[r];
                if (c.tp == 2) {
                    const k = c.k[r].ar2_f32;
                    try g.arg(k, 0, c.send_buf[0][slot]);
                    try g.arg(k, 1, c.send_buf[1][slot]);
                    try g.arg(k, 2, dst[r]);
                    try g.arg(k, 3, @as(i32, @intCast(n4)));
                    try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, null, 2, &waits));
                } else {
                    const k = c.k[r].ar4_f32;
                    try g.arg(k, 0, c.send_buf[0][slot]);
                    try g.arg(k, 1, c.send_buf[1][slot]);
                    try g.arg(k, 2, c.send_buf[2][slot]);
                    try g.arg(k, 3, c.send_buf[3][slot]);
                    try g.arg(k, 4, dst[r]);
                    try g.arg(k, 5, @as(i32, @intCast(n4)));
                    try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, null, 4, &waits));
                }
            }
        } else {
            const slice_n4: u32 = n4 / 4;
            const slice_bytes: usize = @as(usize, slice_n4) * 16;
            const groups: ze.GroupCount = .{ .x = (slice_n4 + 255) / 256 };
            var rs_waits: [MAX_TP]ze.H = undefined;
            for (0..4) |r| {
                const g = &gpus[r];
                const off4: i32 = @intCast(@as(u32, @intCast(r)) * slice_n4);
                const rev = c.rsEv(s, r);
                rs_waits[r] = rev;
                const k = c.k[r].rs4_f32;
                try g.arg(k, 0, c.send_buf[0][slot]);
                try g.arg(k, 1, c.send_buf[1][slot]);
                try g.arg(k, 2, c.send_buf[2][slot]);
                try g.arg(k, 3, c.send_buf[3][slot]);
                try g.arg(k, 4, c.rs_buf[r][slot]);
                try g.arg(k, 5, off4);
                try g.arg(k, 6, @as(i32, @intCast(slice_n4)));
                try ze.check(g.api.zeCommandListAppendLaunchKernel(lists[r], k, &groups, rev, 4, &waits));
            }
            for (0..4) |r| {
                try ze.check(gpus[r].api.zeCommandListAppendWaitOnEvents(lists[r], 4, &rs_waits));
                for (0..4) |src_r| {
                    const dst_slice: *anyopaque = @ptrFromInt(@intFromPtr(dst[r]) + src_r * slice_bytes);
                    try ze.check(gpus[r].api.zeCommandListAppendMemoryCopy(lists[r], dst_slice, c.rs_buf[src_r][slot], slice_bytes, null, 0, null));
                }
            }
        }
    }
};
