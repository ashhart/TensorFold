//! The LiLiCorr tap-ring absorb: ring keep/trim, the re-projections a trim forces, the tap/surgery dumps.

const std = @import("std");
const head = @import("cuda_lilicorr_head.zig");
const lc = @import("cuda_lilicorr.zig");

const Absorb = lc.Absorb;
const run = lc.run;
const roundq = @import("cuda_lilicorr_round.zig");
const norm = roundq.norm;
const projDense = roundq.projDense;
const projSums = roundq.projSums;
const ringK = lc.ringK;
const ringV = lc.ringV;
const surgeryOn = lc.surgeryOn;
const stageOn = lc.stageOn;
const stageDump = lc.stageDump;
const staging_rows = lc.staging_rows;
const anchor_at = lc.anchor_at;
const max_rows = lc.max_rows;
const Ctx = lc.Ctx;
const Draft = lc.Draft;

/// qwen35_draft.cu:47-55's by-value row list (int n; int rows[64]): the Zig type matches its capacity and stride.
pub const kernel_row_cap = 64;

pub const TapRows = extern struct {
    n: c_int,
    rows: [kernel_row_cap]c_int,
};

/// The span gate DERIVED (A5): a context's span opens at >=2 accepted rows past prompt+absorb, any prompt length.
pub fn spanOpen(ctx: *const lc.Ctx) bool {
    const base = ctx.span_base orelse return false;
    return ctx.end >= base + 2;
}

pub fn absorb(d: *Draft, items: []const Absorb) !void {
    for (items) |it| {
        const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
        if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
        if (it.rows) |rs| {
            try absorbRows(d, ctx, it.features.buffer, it.features.offset, rs.len, rs, it.prompt);
        } else try absorbRows(d, ctx, it.features.buffer, it.features.offset, it.features.rows, null, it.prompt);
    }
}

pub fn absorbRows(d: *Draft, ctx: *Ctx, tap_ring: u64, start_: u64, n_: usize, rows_: ?[]const u32, prompt: bool) !void {
    const c = d.c;
    d.projection_role = if (prompt) .prompt else .live;
    var start = start_;
    var n = n_;
    if (rows_) |rs| if (rs.len != n or n > max_rows) return error.BadDraftRows;
    if (start < ctx.end) ctx.end = start;
    if (start > ctx.end) ctx.begin = start;
    const keep = @min(c.window, d.tap_rows - max_rows);
    if (n > keep) {
        start += n - keep;
        n = keep;
        ctx.begin = start;
    }
    ctx.end = start;
    var done: usize = 0;
    var last_rows: usize = 0;
    while (n > 0) {
        // The dense staging chunk runs at staging_rows; an explicit row list runs at the

        // kernel's own by-value TapRows capacity, and each chunk takes THE CHUNK'S slice

        // rs[done..done+rows] (never the list's head again — per-chunk restart was the sparse/permuted defect).
        const rows = if (rows_ != null) @min(n, kernel_row_cap) else @min(n, staging_rows);
        last_rows = rows;
        if (rows_) |rs| {
            var tr: TapRows = .{ .n = @intCast(rows), .rows = @splat(0) };
            for (rs[done..][0..rows], 0..) |r, i| tr.rows[i] = @intCast(r);
            try run(d.e, d.fns.gather_rows, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, @intCast(d.tap_rows)), @as(c_longlong, @intCast(start)), tr, @as(c_int, @intCast(c.n_taps * c.hidden)), d.taps_in });
        } else try run(d.e, d.fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap_ring, @as(c_int, @intCast(d.tap_rows)), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.n_taps * c.hidden)), d.taps_in });

        // Lazy TF_TAP_DEBUG gate check
        const tap_debug = if (lc.tap_debug_env) |val| val else blk: {
            const val = (@import("sys").environment.read(bool, "TF_TAP_DEBUG", false) catch false);
            lc.tap_debug_env = val;
            break :blk val;
        };

        // Print tap-gather event if debug is enabled
        if (tap_debug) {
            // DtoH the gathered bytes and hash them FNV-1a 64
            const bytes_len = rows * c.n_taps * c.hidden * 2;
            const prompt_ = prompt;

            if (bytes_len > 0) {
                // Allocate host buffer to hold the bytes
                const host_buffer = try d.gpa.alloc(u8, bytes_len);
                defer d.gpa.free(host_buffer);

                // Issue DtoH copy - must use try for the check call as per the law
                try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(host_buffer.ptr, d.taps_in, bytes_len, d.e.stream.handle), "cuMemcpyDtoHAsync_v2 tapshash");
                try d.e.stream.synchronize();
                if (prompt and start == 0 and !lc.tap_dumped) {
                    lc.tap_dumped = true;
                    const f = try std.Io.Dir.cwd().createFile(d.e.io, "/tmp/our_taps_in.bin", .{});
                    defer f.close(d.e.io);
                    try std.Io.File.writePositionalAll(f, d.e.io, host_buffer, 0);
                    std.debug.print("[oracle-ours] wrote {d} bytes of taps_in rows={d}\n", .{ host_buffer.len, rows });
                }
                if (!prompt) {
                    var pbuf: [64]u8 = undefined;
                    const pth = std.fmt.bufPrint(&pbuf, "/tmp/our_dec_gather_{d}.bin", .{start_}) catch unreachable;
                    const fd = try std.Io.Dir.cwd().createFile(d.e.io, pth, .{});
                    defer fd.close(d.e.io);
                    try std.Io.File.writePositionalAll(fd, d.e.io, host_buffer, 0);
                    std.debug.print("[oracle-ours] dec write rows={d} start={d}\n", .{ rows, start_ });
                }

                // Calculate FNV-1a 64-bit hash
                var hash: u64 = 0xcbf29ce484222325;
                for (host_buffer) |byte| {
                    hash = hash ^ @as(u64, byte);
                    hash = hash * 0x100000001b3;
                }

                std.debug.print("[tap-gather] start={d} n={d} prompt={any} tapshash={x:0>16} begin={d} end={d}\n", .{ start_, rows, prompt_, hash, ctx.begin, ctx.end });
            } else {
                std.debug.print("[tap-gather] start={d} n={d} prompt={any} tapshash=0000000000000000 begin={d} end={d}\n", .{ start_, rows, prompt_, ctx.begin, ctx.end });
            }
        }

        try projSums(d, d.w.fc, d.taps_in, d.th, rows);
        try norm(d, d.th, d.w.hidden_norm, d.thn, rows);
        if (tap_debug and prompt and !lc.th_dumped) {
            lc.th_dumped = true;
            const tb = rows * c.hidden * 2;
            const th_host = try d.gpa.alloc(u8, tb);
            defer d.gpa.free(th_host);
            try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(th_host.ptr, d.th, tb, d.e.stream.handle), "cuMemcpyDtoHAsync thdump");
            try d.e.stream.synchronize();
            const f2 = try std.Io.Dir.cwd().createFile(d.e.io, "/tmp/our_th.bin", .{});
            defer f2.close(d.e.io);
            try std.Io.File.writePositionalAll(f2, d.e.io, th_host, 0);
            std.debug.print("[oracle-ours] wrote th rows={d}\n", .{rows});
        }

        // CONTEXT PATH: DIRECT PROJECTION (per the vLLM reference, verified 11/24 from the

        // nightly sources): precompute_and_store_context_kv NEVER runs the backbone — the

        // context K/V are per-layer k/v projections of the ONE normed combined vector

        // (rms_norm(combined, hidden_norm)), then per-layer k-norm + rope, then cache write.

        // The backbone runs ONLY in the decode forward over the frame's embeds.
        for (d.w.layers, 0..) |l, i| {
            try projDense(d, l.k, d.thn, d.kabs, rows);
            try run(d.e, d.fns.norm_rope, .{ rows * c.kv_heads, 1, 1 }, .{ 32, 1, 1 }, .{ d.kabs, l.k_norm, d.kn, @as(c_int, @intCast(c.kv_heads)), @as(c_int, @intCast(c.kv_heads * c.head_dim)), @as(c_longlong, @intCast(start)), c.eps, c.theta });
            try projDense(d, l.v, d.thn, d.vabs, rows);
            try run(d.e, d.fns.ring_put, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ d.kn, d.vabs, ringK(d, ctx, i), ringV(d, ctx, i), @as(c_int, @intCast(c.kvWidth())), @as(c_longlong, @intCast(start)), @as(c_int, @intCast(c.ringRows())) });

            // Surgery dumps for context path - point at d.thn instead of d.h
            if (surgeryOn()) {
                const bytes_len = rows * c.hidden * 2;
                const host_buffer = try d.gpa.alloc(u8, bytes_len);
                defer d.gpa.free(host_buffer);
                try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(host_buffer.ptr, d.thn, bytes_len, d.e.stream.handle), "cuMemcpyDtoHAsync_v2 ctx");
                try d.e.stream.synchronize();

                // Calculate stats
                var maxabs: f32 = 0.0;
                var nan_count: usize = 0;
                var inf_count: usize = 0;
                var hash: u64 = 0xcbf29ce484222325;
                for (host_buffer) |byte| {
                    hash = hash ^ @as(u64, byte);
                    hash = hash * 0x100000001b3;
                }
                for (0..host_buffer.len / 2) |ix| {
                    const bits = std.mem.readInt(u16, host_buffer[ix * 2 ..][0..2], .little);
                    const f = @as(f32, @bitCast(@as(u32, bits) << 16));
                    maxabs = @max(maxabs, @abs(f));
                    if (std.math.isNan(f)) nan_count += 1;
                    if (std.math.isInf(f)) inf_count += 1;
                }
                std.debug.print("[surgery] layer={d} stage=ctx maxabs={d:.3} nan={d} inf={d} hash={x:0>16}\n", .{ i, maxabs, nan_count, inf_count, hash });

                // File dump for first prompt absorb
                if (prompt and start_ == 0 and !lc.surgery_dumped) {
                    const f = try std.Io.Dir.cwd().createFile(d.e.io, try std.fmt.allocPrint(d.gpa, "/tmp/our_surgery_l{d}_ctx.bin", .{i}), .{});
                    defer f.close(d.e.io);
                    try std.Io.File.writePositionalAll(f, d.e.io, host_buffer, 0);
                }
            }
        }
        if (surgeryOn() and prompt and start_ == 0 and done > 0) lc.surgery_dumped = true;
        if (tap_debug) {
            const th_bytes = rows * c.hidden * 2;
            const th_host = try d.gpa.alloc(u8, th_bytes);
            defer d.gpa.free(th_host);
            try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(th_host.ptr, d.thn, th_bytes, d.e.stream.handle), "cuMemcpyDtoHAsync thn");
            try d.e.stream.synchronize();
            var thh: u64 = 0xcbf29ce484222325;
            for (th_host) |byte| {
                thh = thh ^ @as(u64, byte);
                thh = thh * 0x100000001b3;
            }
            const rk_bytes = c.ringRows() * c.kvWidth() * 2;
            const rk_host = try d.gpa.alloc(u8, rk_bytes);
            defer d.gpa.free(rk_host);
            try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(rk_host.ptr, ringK(d, ctx, 0), rk_bytes, d.e.stream.handle), "cuMemcpyDtoHAsync ringK0");
            try d.e.stream.synchronize();
            var rkh: u64 = 0xcbf29ce484222325;
            for (rk_host) |byte| {
                rkh = rkh ^ @as(u64, byte);
                rkh = rkh * 0x100000001b3;
            }
            std.debug.print("[tap-proj] n={d} thnhash={x:0>16} ringK0hash={x:0>16}\n", .{ rows, thh, rkh });
        }
        start += rows;
        n -= rows;
        done += rows;
        ctx.end = start;
    }
    if (ctx.end > c.window and ctx.begin < ctx.end - c.window) ctx.begin = ctx.end - c.window;
    // The owner A5 span base: every prompt-flagged absorb moves the base to the phase's END

    // (prefill chunks and the prompt's last-row first-token absorb all carry prompt=true),

    // so after the prompt phase ctx.span_base = prompt+absorb row count, whatever the prompt

    // length; accepted live rows (prompt=false) never move it. A decode-phase re-ingest that

    // arrives prompt-flagged re-arms the base — the span then re-opens only after 2 fresh

    // rows past it (conservative, no stale attention window).
    if (prompt) ctx.span_base = ctx.end;
    // the anchor: THE BACKBONE'S FINAL-NORM OUTPUT at the newest committed row (the reference's

    // context-forward last_hidden semantics — the old source was the raw deep-tap residual

    // through hidden_norm, a completely different vector; the start scores = anchor_out . in_vec,

    // so a wrong anchor flattens the head's ranking). Fills on EVERY absorb (prompt + decode).

    // Anchor is the last row of d.th (rms with d.w.hidden_norm over it): d.th holds only

    // the FINAL chunk's rows, so the index is (last chunk rows - 1)*hidden*2 — the former

    // done-1 accumulated across chunks and read past the staging buffer on any multi-chunk absorb.
    if (done > 0) {
        const last = d.th + (@as(u64, @intCast(last_rows - 1)) * c.hidden * 2);
        try d.e.k.norms.addRmsTail(d.e.stream, last, null, d.w.hidden_norm, c.hidden, d.anchor, null, 1, c.hidden, c.eps, .{ .round = true });
        try d.e.ctx.d.check(d.e.ctx.d.api.cuMemcpyDtoHAsync_v2(d.staged.bytes.ptr + anchor_at, d.anchor, c.hidden * 2, d.e.stream.handle), "cuMemcpyDtoHAsync anchor");
        if (stageOn() and lc.stage_fwd_count < 3) try stageDump(d, d.anchor, c.hidden * 2, "ours_anchor");
        ctx.anchor_valid = true;
    } else if (!prompt) ctx.anchor_valid = false;
}
