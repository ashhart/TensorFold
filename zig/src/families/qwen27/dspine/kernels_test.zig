//! DSpine's Metal contract: the bridge and table kernels on the dspine_mlx fixture, against it and the host reference.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const host = @import("host.zig");
const Kernels = @import("kernels.zig").Kernels;
const Ref = @import("kernels.zig").Ref;
const t = std.testing;

const blob = @embedFile("testdata/bridge-fixture.safetensors");
const H = 128;
const C = 64;
const R = 64;
const V = 96;
const T = 6;
const K = 4;
const opts = mtl.ResourceOptions.shared;

fn tensor(header: core.safetensors.Header, data: []const u8, name: []const u8) ![]const u8 {
    const e = header.get(name) orelse return error.MissingTensor;
    return data[e.begin..e.end];
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: Kernels,
    ops: core.draft_ops.Ops,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    fn buffer(g: *Gpu, bytes: usize) !Ref {
        const b = try g.device.buffer(@max(bytes, 16), opts);
        try g.owned.append(t.allocator, b);
        @memset(b.contents()[0..@max(bytes, 16)], 0);
        return .{ .buf = b };
    }
    fn upload(g: *Gpu, bytes: []const u8) !Ref {
        const r = try g.buffer(bytes.len);
        @memcpy(r.buf.contents()[0..bytes.len], bytes);
        return r;
    }
    fn linear(g: *Gpu, e: mtl.ComputeEncoder, x: Ref, w: Ref, rows: u32, n: u32, k: u32) !Ref {
        const y = try g.buffer(rows * n * 2);
        try g.ops.linear(e, x, w, y, .{ .rows = rows, .n = n, .k = k, .x_stride = k, .w_stride = k, .y_stride = n });
        e.barrier();
        return y;
    }
    fn deinit(g: *Gpu) void {
        for (g.owned.items) |b| b.deinit();
        g.owned.deinit(t.allocator);
        g.ops.deinit();
        g.kernels.deinit();
        g.queue.deinit();
        g.device.deinit();
    }
};

fn bf16s(r: Ref, n: usize) []const u16 {
    return r.buf.slice(u16, n);
}

fn worstUlps(want: []align(1) const u16, got: []const u16) u32 {
    var worst: u32 = 0;
    for (want, got) |a, b| worst = @max(worst, if ((a ^ b) & 0x8000 != 0 and (a | b) & 0x7fff != 0) 1 << 16 else @abs(@as(i32, a) - @as(i32, b)));
    return worst;
}

const Run = struct { res: Ref, xn: Ref, rms: Ref, packed_rows: Ref };

/// The last write on the GPU, as forward.zig encodes it; `write` its feature write (zeros for the control).
fn bridge(g: *Gpu, header: core.safetensors.Header, data: []const u8, write: Ref) !Run {
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    const raw = try g.upload(try tensor(header, data, "x1"));
    var ids: [T]u32 = @splat(3);
    ids[0] = std.mem.readInt(u32, (try tensor(header, data, "anchor"))[0..4], .little);
    const id_ref = try g.upload(std.mem.sliceAsBytes(&ids));
    var packed_w: [3 * R * C * 2]u8 = undefined;
    for ([_][]const u8{ "gate", "pred", "prev" }, 0..) |name, i| @memcpy(packed_w[i * R * C * 2 ..][0 .. R * C * 2], try tensor(header, data, name));
    const xn = try g.buffer(T * H * 2);
    const rms = try g.buffer(T * 4);
    g.kernels.read(e, raw, xn, rms, T, H, 1e-6);
    e.barrier();
    const projected = try g.linear(e, xn, try g.upload(try tensor(header, data, "read_last")), T, C, H);
    const codes = try g.buffer(T * C * 2);
    g.kernels.code(e, projected, try g.upload(try tensor(header, data, "codebook")), id_ref, codes, T, C, T, 1e-6);
    e.barrier();
    const packed_rows = try g.linear(e, codes, try g.upload(&packed_w), T, 3 * R, C);
    const msg = try g.buffer(T * R * 2);
    g.kernels.message(e, packed_rows, try g.upload(try tensor(header, data, "gate_bias")), msg, T, R, T);
    e.barrier();
    const delta = try g.linear(e, msg, write, T, H, R);
    const run: Run = .{ .res = try g.buffer(T * H * 2), .xn = try g.buffer(T * H * 2), .rms = rms, .packed_rows = packed_rows };
    g.kernels.write(e, raw, delta, rms, try g.upload(try tensor(header, data, "norm1")), run.res, run.xn, T, H, T, 1e-6);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg_text| {
        std.debug.print("{s}\n", .{msg_text});
        return error.DraftGpuFailure;
    }
    return run;
}

fn open(arena: std.mem.Allocator) !struct { core.safetensors.Header, []const u8 } {
    const n = std.mem.readInt(u64, blob[0..8], .little);
    const data = blob[8 + n ..];
    return .{ try core.safetensors.parseHeader(arena, blob[8..][0..n], data.len), data };
}

fn gpu() !Gpu {
    const device = try mtl.Device.init();
    errdefer device.deinit();
    const queue = try device.queue();
    errdefer queue.deinit();
    const kernels = try Kernels.init(device);
    errdefer kernels.deinit();
    return .{ .device = device, .queue = queue, .kernels = kernels, .ops = try core.draft_ops.Ops.init(device) };
}

test "DSpine Metal contract: the bridge kernels write as dspine_mlx, the table chains its drafts; no message, no match" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const header, const data = try open(arena.allocator());
    var g = try gpu();
    defer g.deinit();
    const write_all = try tensor(header, data, "write");
    const run = try bridge(&g, header, data, try g.upload(write_all[H * R * 2 ..][0 .. H * R * 2]));
    const ulps_of = struct {
        fn f(hd: core.safetensors.Header, d: []const u8, name: []const u8, got: Ref, n: usize) !u32 {
            return worstUlps(std.mem.bytesAsSlice(u16, try tensor(hd, d, name)), bf16s(got, n));
        }
    }.f;
    try t.expect(try ulps_of(header, data, "out1", run.res, T * H) <= 2);
    try t.expect(try ulps_of(header, data, "normed1", run.xn, T * H) <= 2);
    try t.expect(try ulps_of(header, data, "stash_packed", run.packed_rows, T * 3 * R) <= 2);
    for (std.mem.bytesAsSlice(f32, try tensor(header, data, "stash_rms")), run.rms.buf.slice(f32, T)) |a, b| try t.expectApproxEqRel(a, b, 1e-5);

    // the table over the fixture's candidates; the head affine-4 as the target's, its base rows dequantized
    var cand: [(T - 1) * K]u32 = undefined;
    for (&cand, std.mem.bytesAsSlice(u32, try tensor(header, data, "candidates"))) |*c, v| c.* = v;
    var rows: [T - 1]core.bf16_topk.Row = @splat(.{});
    for (&rows, 0..) |*row, i| {
        row.count = K;
        for (0..K) |j| row.entries[j].token = cand[i * K + j];
    }
    const head_bf16 = try t.allocator.alloc(u16, V * H);
    defer t.allocator.free(head_bf16);
    @memcpy(head_bf16, std.mem.bytesAsSlice(u16, try tensor(header, data, "head")));
    var q4 = try core.affine4.prepare(t.allocator, head_bf16, V, H);
    defer q4.deinit();
    const head_q4: @import("kernels.zig").Head = .{ .words = try g.upload(std.mem.sliceAsBytes(q4.words)), .scales = try g.upload(std.mem.sliceAsBytes(q4.scales)), .biases = try g.upload(std.mem.sliceAsBytes(q4.biases)) };
    const top = try g.upload(std.mem.sliceAsBytes(&rows));
    const row_words: u32 = @sizeOf(core.bf16_topk.Row) / 4;
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    const anchor = std.mem.readInt(u32, (try tensor(header, data, "anchor"))[0..4], .little);
    const senders = try g.buffer((T - 1) * K * C * 2);
    const sids = try g.buffer((T - 1) * K * 4);
    g.kernels.senders(e, top, try g.upload(try tensor(header, data, "codebook")), senders, sids, .{ .rows = T - 1, .k = K, .code = C, .row_words = row_words, .anchor = anchor, .scale = 8 });
    e.barrier();
    var pv: [2 * R * C * 2]u8 = undefined;
    @memcpy(pv[0 .. R * C * 2], try tensor(header, data, "pred"));
    @memcpy(pv[R * C * 2 ..], try tensor(header, data, "prev"));
    const tsv = try g.linear(e, senders, try g.upload(&pv), (T - 1) * K, 2 * R, C);
    const raw_ref = try g.upload(try tensor(header, data, "x1"));
    const norm_ref = try g.upload(try tensor(header, data, "norm1"));
    const base = try g.buffer((T - 1) * K * 4);
    g.kernels.base(e, raw_ref, norm_ref, head_q4, top, base, .{ .rows = T - 1, .k = K, .hidden = H, .row_words = row_words, .raw_first = 1 });
    e.barrier();
    const table = try g.buffer((T - 1) * K * K * 4);
    g.kernels.table(e, run.packed_rows, tsv, try g.upload(try tensor(header, data, "gate_bias")), try g.upload(try tensor(header, data, "tg")), top, base, run.rms, table, .{ .rows = T - 1, .k = K, .rank = R, .row_words = row_words, .rms_first = 1 });
    e.end();
    cb.commit();
    cb.wait();
    try t.expect(cb.failure() == null);
    for (sids.buf.slice(u32, (T - 1) * K), 0..) |sid, p| try t.expectEqual(if (p < K) anchor else cand[p - K], sid);

    // the host table on the same stash and dequantized head rows: the kernel's scores and its chain
    var raw: [T * H]u16 = undefined;
    var packed_rows: [T * 3 * R]u16 = undefined;
    var rms: [T]f32 = undefined;
    @memcpy(&raw, std.mem.bytesAsSlice(u16, try tensor(header, data, "x1")));
    @memcpy(&packed_rows, bf16s(run.packed_rows, T * 3 * R));
    @memcpy(&rms, run.rms.buf.slice(f32, T));
    var head_rows: [(T - 1) * K * H]u16 = undefined;
    for (cand, 0..) |id, j| for (0..H) |x| {
        const g64 = id * (H / 64) + x / 64;
        const code = (q4.words[id * (H / 8) + x / 8] >> @intCast(4 * (x % 8))) & 15;
        head_rows[j * H + x] = host.round(host.float(q4.scales[g64]) * @as(f32, @floatFromInt(code)) + host.float(q4.biases[g64]));
    };
    var want: [(T - 1) * K * K]f32 = undefined;
    try host.table(.{ .dims = .{ .hidden = H, .code = C, .rank = R, .vocab = V, .eps = 1e-6 }, .topk = K, .stash = .{ .raw = &raw, .rms = &rms, .packed_rows = &packed_rows }, .anchor = anchor, .candidates = &cand, .head_rows = &head_rows, .norm_w = std.mem.bytesAsSlice(u16, try tensor(header, data, "norm1")), .bias = std.mem.bytesAsSlice(u16, try tensor(header, data, "gate_bias")), .tg = std.mem.bytesAsSlice(u16, try tensor(header, data, "tg")), .tsv = std.mem.bytesAsSlice(u16, try tensor(header, data, "tsv")) }, &want);
    for (want, table.buf.slice(f32, want.len)) |a, b| try t.expect(@abs(a - b) <= 1e-3 * @abs(a) + 1e-2);
    var host_drafts: [T - 1]u32 = undefined;
    var drafts: [T - 1]u32 = undefined;
    core.dspine.chain(&cand, &want, K, &host_drafts);
    core.dspine.chain(&cand, table.buf.slice(f32, (T - 1) * K * K), K, &drafts);
    try t.expectEqualSlices(u32, &host_drafts, &drafts);

    // the planted control: the message dropped (a zero write), the written rows leave dspine_mlx's
    const control = try bridge(&g, header, data, try g.buffer(H * R * 2));
    try t.expect(try ulps_of(header, data, "out1", control.res, T * H) > 2);
}
