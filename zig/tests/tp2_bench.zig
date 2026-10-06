//! TP=2's exchange as the GPU sees it, between two Macs over MCDMA. Each layer the GPU writes its partial rows into the registered window and posts the layer's sequence; the host sends them to the peer as one write-and-signal, waits for the peer's, and serves the sequence; a one-thread GPU wait spins until then, and both ranks sum the two partials in rank order. Arms: tp2 (the exchange), local (the host serves at once: the GPU-host handoff alone), none (no handoff). tf-tp2-bench SETTINGS.json   (rank, library, links, layers, rounds, rows)
const std = @import("std");
const mtl = @import("metal");
const fabric = @import("fabric");

const D = 2560;
const ROWS_MAX = 16;
const PART = ROWS_MAX * D * 4; // one partial: fp32 rows
const OUT = 0; // my partial: the GPU writes it, the host sends it
const IN = PART; // the peer's partial lands here
const FLAG = 2 * PART; // the peer's signal: the last sequence whose partial has landed
const SYNC = FLAG + 16384; // GPU-posted and host-served sequences, 4 KiB apart (K3's round words)
const WINDOW = SYNC + 4 * 4096;
const SYNC_HOST = 0;
const SYNC_GPU = 1024;
const SYNC_GAVE_UP = 2048;

const Settings = struct { rank: u32, library: []const u8, links: []const fabric.mcdma.Link, layers: u32 = 48, rounds: u32 = 50, rows: []const u32 = &.{ 1, 4, 16 } };
const Arm = enum { none, local, tp2 };

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void tp_partial(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& salt [[buffer(2)]],
    \\                       uint i [[thread_position_in_grid]]) {
    \\  if (i < n) out[i] = float((i * 2654435761u + salt * 40503u) & 1023u) * 0.125f;
    \\}
    \\kernel void tp_post(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]]) {
    \\  atomic_store_explicit(&sync[1024], seq, memory_order_relaxed);
    \\}
    \\kernel void tp_wait(device atomic_uint* sync [[buffer(0)]], constant uint& seq [[buffer(1)]]) {
    \\  uint polls = 0;
    \\  while (int(atomic_load_explicit(&sync[0], memory_order_relaxed) - seq) < 0) {
    \\    if (++polls > 400000000u) { atomic_fetch_add_explicit(&sync[2048], 1u, memory_order_relaxed); return; }
    \\  }
    \\}
    \\kernel void tp_sum(device const float* p0 [[buffer(0)]], device const float* p1 [[buffer(1)]], device float* out [[buffer(2)]],
    \\                   constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    \\  if (i < n) out[i] = p0[i] + p1[i];
    \\}
;

fn value(i: u32, salt: u32) f32 {
    return @as(f32, @floatFromInt((i *% 2654435761 +% salt *% 40503) & 1023)) * 0.125;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.SettingsRequired;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 20));
    const s = try std.json.parseFromSliceLeaky(Settings, a, text, .{ .allocate = .alloc_always });
    if (s.rank > 1 or s.links.len != 1) return error.TwoRanksOneLink;
    const peer: u32 = 1 - s.rank;
    const lib_path = try a.dupeSentinel(u8, s.library, 0);
    const ep = try fabric.mcdma.Endpoint.create(init.gpa, lib_path, .{ .rank = s.rank, .ranks = 2, .window_bytes = WINDOW, .staging_bytes = 4 << 20, .links = s.links, .timeout_ns = 30 * std.time.ns_per_s, .connect_timeout_ns = 120 * std.time.ns_per_s });
    defer ep.deinit();
    const rd = ep.rdma();
    const win = rd.window();
    @memset(win, 0);

    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const wbuf = try device.bufferNoCopy(win.ptr, win.len, opts);
    const result = try device.buffer(PART, opts);
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    const p_partial = try mtl.Pipeline.init(device, lib, "tp_partial", false);
    const p_post = try mtl.Pipeline.init(device, lib, "tp_post", false);
    const p_wait = try mtl.Pipeline.init(device, lib, "tp_wait", false);
    const p_sum = try mtl.Pipeline.init(device, lib, "tp_sum", false);
    const sync: [*]u32 = @ptrCast(@alignCast(win.ptr + SYNC));
    const flag: *u64 = @ptrCast(@alignCast(win.ptr + FLAG));

    // both ranks ready before any timing: a signal each way
    try rd.signal(peer, FLAG + 8, 1);
    while (@atomicLoad(u64, @as(*u64, @ptrCast(@alignCast(win.ptr + FLAG + 8))), .acquire) < 1) std.atomic.spinLoopHint();

    var seq: u32 = 0;
    for (s.rows) |rows| for ([_]Arm{ .none, .local, .tp2 }) |arm| {
        const n: u32 = rows * D;
        var best: f64 = 1e30;
        var total: f64 = 0;
        var wrong: usize = 0;
        for (0..s.rounds) |round| {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            const first = seq + 1;
            for (0..s.layers) |layer| {
                const salt: u32 = @intCast(layer * 131 + round * 7 + s.rank * 1000003);
                enc.setPipeline(p_partial);
                enc.setBuffer(wbuf, OUT, 0);
                enc.setBytes(std.mem.asBytes(&n), 1);
                enc.setBytes(std.mem.asBytes(&salt), 2);
                enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
                if (arm != .none) {
                    seq += 1;
                    enc.setPipeline(p_post);
                    enc.setBuffer(wbuf, SYNC, 0);
                    enc.setBytes(std.mem.asBytes(&seq), 1);
                    enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
                    enc.setPipeline(p_wait);
                    enc.setBuffer(wbuf, SYNC, 0);
                    enc.setBytes(std.mem.asBytes(&seq), 1);
                    enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
                }
                enc.setPipeline(p_sum);
                enc.setBuffer(wbuf, if (s.rank == 0) OUT else IN, 0);
                enc.setBuffer(wbuf, if (s.rank == 0) IN else OUT, 1);
                enc.setBuffer(result, 0, 2);
                enc.setBytes(std.mem.asBytes(&n), 3);
                enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
            }
            enc.end();
            const t0 = mtl.clock.seconds();
            cb.commit();
            if (arm != .none) {
                var k = first;
                while (k <= seq) : (k += 1) {
                    while (@atomicLoad(u32, &sync[SYNC_GPU], .acquire) < k) std.atomic.spinLoopHint();
                    if (arm == .tp2) {
                        try rd.write2Signal(peer, IN, win[OUT..][0 .. n * 4], &.{}, FLAG, k);
                        while (@atomicLoad(u64, flag, .acquire) < k) std.atomic.spinLoopHint();
                    }
                    @atomicStore(u32, &sync[SYNC_HOST], k, .release);
                }
            }
            cb.wait();
            const dt = mtl.clock.seconds() - t0;
            if (cb.failure()) |msg| {
                std.log.err("command buffer failed: {s}", .{msg});
                return error.GpuFailed;
            }
            best = @min(best, dt);
            total += dt;
            if (arm == .tp2) { // the last layer's sums, in rank order
                const got = result.slice(f32, n);
                const salt_me: u32 = @intCast((s.layers - 1) * 131 + round * 7 + s.rank * 1000003);
                const salt_peer: u32 = @intCast((s.layers - 1) * 131 + round * 7 + peer * 1000003);
                var i: u32 = 0;
                while (i < n) : (i += 97) {
                    const mine = value(i, salt_me);
                    const theirs = value(i, salt_peer);
                    const want = if (s.rank == 0) mine + theirs else theirs + mine;
                    if (got[i] != want) wrong += 1;
                }
            }
        }
        const per_layer_us = best / @as(f64, @floatFromInt(s.layers)) * 1e6;
        std.debug.print("TP2 rank{d} rows={d} arm={s} round_best_ms={d:.3} round_mean_ms={d:.3} per_layer_us={d:.1} wrong={d} gave_up={d}\n", .{ s.rank, rows, @tagName(arm), best * 1e3, total / @as(f64, @floatFromInt(s.rounds)) * 1e3, per_layer_us, wrong, sync[SYNC_GAVE_UP] });
    };
    try rd.flush();
    std.debug.print("TP2 rank{d} DONE\n", .{s.rank});
}
