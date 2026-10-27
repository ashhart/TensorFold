//! GpuPass on the device: passes started from the window's pick, graphed passes, the pass's glue steps.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");
const ph = @import("phases.zig");
const gpu = @import("dspark_gpu.zig");
const GpuPass = gpu.GpuPass;
const checkCandidates = gpu.checkCandidates;

/// The device path is on and past its checks (a pass launched outside `propose` / `begin` then counts no check).
pub fn devReady(x: *const GpuPass) bool {
    const d = x.dev orelse return false;
    return d.on() and d.stats.passes >= dsd.check_passes;
}

/// after_pick, every rank: the window's taps into the rings, the statics on device, the pass, the event.
pub fn afterPick(ctx: *anyopaque, pick: fwd.Forward.PickDevice) anyerror!void {
    const x: *GpuPass = @ptrCast(@alignCast(ctx));
    const a = x.armed orelse return;
    if (a.sampled) return; // the sampler's choice starts it (afterSample)
    x.armed = null;
    if (!pick.chain or pick.n != a.n or pick.start != a.start or pick.host == null or x.begun != null) return;
    if (!x.devReady() or a.n > x.ingest_rows) return;
    try x.launchFrom(a.start, a.n, pick.out, pick.n);
    if (x.f.comm.rank() == 0 or x.f.link == null) x.dl = .{ .start = a.start, .n = a.n, .host = pick.host.?, .rows = a.n };
}

/// after_sample, every rank: the window's accept on the device (ds_accept_rows), then the pass as afterPick.
pub fn afterSample(ctx: *anyopaque, row0: u32, n: u32, chosen: u64) anyerror!void {
    const x: *GpuPass = @ptrCast(@alignCast(ctx));
    const a = x.armed orelse return;
    if (!a.sampled) return;
    x.armed = null;
    const p = x.f.slot.pending orelse return;
    if (row0 != 0 or n != a.n or p.start != a.start or p.n != a.n or x.begun != null) return;
    if (!x.devReady() or a.n > x.ingest_rows or x.f.ids.len < n) return;
    const r = x.f.runner;
    // [ids n | segment 3 | acc 3] int64
    const h: []i64 = @alignCast(std.mem.bytesAsSlice(i64, x.shost.?.bytes[0 .. 8 * 128]));
    for (x.f.ids[0..n], 0..) |t, i| h[i] = t;
    h[64..67].* = .{ 0, n, @intCast(a.start) };
    const dev = x.sscr.?.ptr;
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev, .len = 8 * 67 }, 0, std.mem.sliceAsBytes(h[0..67]), r.stream.handle);
    try r.kernels.others(r.stream).dsAcceptRows(chosen, dev, dev + 8 * 64, 1, dev + 8 * 68);
    try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.shost.?.bytes.ptr + 8 * 68, dev + 8 * 68, 8 * 3, r.stream.handle), "cuMemcpyDtoHAsync");
    try x.staged.?.record(r.stream);
    // acc [accepted, bonus, next position] reads as a pick of 0 rows: ds_stage takes pick[1], pick[2]
    try x.launchFrom(a.start, a.n, dev + 8 * 68, 0);
    if (x.f.comm.rank() == 0 or x.f.link == null) x.dl = .{ .start = a.start, .n = 0, .host = @ptrCast(h[68..].ptr), .rows = a.n, .wait = true };
}

/// The device start's launches, every rank: taps, statics (ds_stage from `pick`), the pass, the event.
pub fn launchFrom(x: *GpuPass, start: u64, n: u32, pick: u64, nw: u32) !void {
    const a = .{ .start = start, .n = n };
    const d = x.dev.?;
    const r = x.f.runner;
    // every row of the window: rows past the commit are overwritten before any later pass reads them
    try x.ingestAsync(a.start, 0, a.n);
    // the statics as fillStatics stages them (placeholders), then every position-dependent one from the pick
    const st = x.st;
    const sv = try d.statics(0, @intCast(st.len64()), @intCast(st.len32()));
    x.fillStatics(0, a.start, sv[0], sv[1]);
    const s64 = r.addressOf("w.ds.i64") orelse return error.Unbound;
    const s32 = r.addressOf("w.ds.i32") orelse return error.Unbound;
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s64, .len = 8 * sv[0].len }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s32, .len = 4 * sv[1].len }, 0, std.mem.sliceAsBytes(sv[1]), r.stream.handle);
    try r.kernels.others(r.stream).dsStage(pick, nw, s64, s32, @intCast(st.n), @intCast(st.window), @intCast(emit.ringRows(x.f.cfg)), x.valid, x.shape.noise, stageOffsets(st));
    r.glue = .{ .ctx = x, .run = glueFn };
    const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
    try d.issueReplay(0, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = passBody });
    try d.mark();
    try d.settle(); // counted here on every rank (rank 0's collect does not count it again)
}

/// ds_stage's offsets: positions, tokens, counts, lo, hi, anchor (ids and start at 0).
pub fn stageOffsets(st: emit.Statics) [6]i64 {
    return .{ st.positions(), st.tokens(), st.counts(), st.lo(), st.hi(), st.anchor() };
}

/// A device-launched pass nobody took: its results dropped; the next pass follows it on the stream.
pub fn dropDevice(x: *GpuPass) void {
    if (x.dl != null) x.stats.device_dropped += 1;
    x.dl = null;
    x.armed = null;
}

pub fn devOn(x: *const GpuPass) bool {
    const d = x.dev orelse return false;
    return d.on();
}

/// The device path: the statics from pinned staging (no sync), the pass (its graph once checks passed), the event.
pub fn devLaunch(x: *GpuPass, anchor: u32, start: u64) !void {
    const d = x.dev.?;
    const st = x.st;
    const sv = try d.statics(0, @intCast(st.len64()), @intCast(st.len32()));
    x.fillStatics(anchor, start, sv[0], sv[1]);
    const r = x.f.runner;
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * sv[0].len }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * sv[1].len }, 0, std.mem.sliceAsBytes(sv[1]), r.stream.handle);
    r.glue = .{ .ctx = x, .run = glueFn };
    const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
    try d.issue(0, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = passBody });
    try d.mark();
}

pub fn passBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
    const x: *GpuPass = @ptrCast(@alignCast(ctx));
    _ = stream; // the runner's stream
    try x.f.runner.window(x.pass_calls);
}

/// The device pass's results on the host (`unpack`: rank 0's chain reads them), a check's comparison, the verdict.
pub fn devCollect(x: *GpuPass, unpack: bool) !void {
    const d = x.dev.?;
    {
        const tw = ph.start(x.f.phases, .@"draft.wait");
        defer tw.stop();
        try d.wait();
    }
    if (d.checked < dsd.check_passes) try d.check(x.last_logits, x.last_v, @intCast(x.st.n), 0);
    if (unpack) try d.unpack(@intCast(x.st.n), 0, x.f.cfg.vocab, x.cand, x.cval, if (x.conf != null) x.hid else null);
    try d.settle();
}

/// A device-launched pass's results (rank 0): waited for and unpacked; it was counted at its launch.
pub fn devResults(x: *GpuPass) !void {
    const d = x.dev.?;
    {
        const tw = ph.start(x.f.phases, .@"draft.wait");
        defer tw.stop();
        try d.wait();
    }
    try d.unpack(@intCast(x.st.n), 0, x.f.cfg.vocab, x.cand, x.cval, if (x.conf != null) x.hid else null);
}

pub fn device(x: *GpuPass, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
    return @import("grow_device.zig").grow(slot, x.f.runner.d, bytes);
}

pub fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |d| n *= @intCast(d);
    return n;
}

pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const x: *GpuPass = @ptrCast(@alignCast(ctx));
    const step = c.name["glue.".len..];
    const s = r.stream.handle;
    const k = r.kernels.others(r.stream);
    const comm = x.f.comm;
    const D: usize = x.f.cfg.hidden;
    const Wd: usize = comm.world();
    if (std.mem.eql(u8, step, "ds_embed")) {
        // the block's ids (staged) through this rank's vocabulary rows, their all-sum, x hc_mult into the streams
        const n: usize = @intCast(x.st.n);
        const vocab_part = try @import("partition.zig").share(x.f.cfg.vocab, comm.rank(), Wd);
        const V: usize = vocab_part.len;
        const embed = r.weights.get("embed") orelse return error.MissingWeight;
        const rows = try x.device(&x.send, 2 * n * D);
        try k.embedSend(try r.tensorAddr(c.args[2].arg.t), n, embed.ptr, @intCast(vocab_part.lo), V, D, rows);
        const recv = try r.tensorAddr(c.args[1].arg.t);
        try comm.allGather(rows, recv, n * D, .bf16, s);
        return k.embedSum(recv, Wd, n, D, try r.tensorAddr(c.args[0].arg.t));
    }
    if (std.mem.eql(u8, step, "exchange_f32")) {
        const part = c.args[0].arg.t;
        const n = numel(part);
        const half = try x.device(&x.send, 2 * n);
        try k.castBf16(try r.tensorAddr(part), half, n);
        return comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
    }
    if (std.mem.eql(u8, step, "widen")) // q.to(fp32): exact
        return k.widenF32(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
    if (std.mem.eql(u8, step, "ds_out")) {
        if (x.dev) |dv| if (dv.on()) {
            x.last_logits = try r.tensorAddr(c.args[0].arg.t);
            x.last_v = @intCast(c.args[0].arg.t.shape[1]);
            return dv.launch(r, c.args[0].arg.t, c.args[1].arg.t, @intCast(x.st.n), 0);
        };
        return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
    }
    return error.GlueNotBuilt;
}

/// The host's step after the head: this rank's best k columns, every rank's lists gathered, the head hidden.
pub fn hostStep(x: *GpuPass, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor) !void {
    const th = ph.start(x.f.phases, .@"draft.host");
    defer th.stop();
    const comm = x.f.comm;
    const n: usize = @intCast(x.st.n);
    const V: usize = @intCast(logits_t.shape[1]);
    const D: usize = x.f.cfg.hidden;
    const k: usize = x.k;
    const Wd: usize = comm.world();
    const gpa = x.gpa;
    const logits = try gpa.alloc(f32, n * V);
    defer gpa.free(logits);
    const hid16 = try gpa.alloc(u16, n * D);
    defer gpa.free(hid16);
    try r.stream.synchronize();
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(logits_t), .len = 4 * logits.len }, 0, std.mem.sliceAsBytes(logits));
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(hid_t), .len = 2 * hid16.len }, 0, std.mem.sliceAsBytes(hid16));
    for (hid16, x.hid) |b, *h| h.* = dspark.bf16(b);
    // packed [n][2k] fp32: values, then ids as bits
    const packed_ = try gpa.alloc(f32, n * 2 * k);
    defer gpa.free(packed_);
    const ids = try gpa.alloc(i32, k);
    defer gpa.free(ids);
    const id0: u32 = (try @import("partition.zig").share(x.f.cfg.vocab, comm.rank(), Wd)).lo;
    for (0..n) |i| {
        const row = packed_[i * 2 * k ..][0 .. 2 * k];
        try dspark.candidates(gpa, logits[i * V ..][0..V], id0, ids, row[0..k]);
        for (ids, row[k..]) |id, *o| o.* = @bitCast(id);
    }
    const bytes = 4 * packed_.len;
    const src = try x.device(&x.send, bytes);
    const dst = try x.device(&x.gathered, bytes * Wd);
    try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_), r.stream.handle);
    try comm.allGather(src, dst, packed_.len, .f32, r.stream.handle); // bits only (ids as fp32 bit patterns)
    try r.stream.synchronize();
    const all = try gpa.alloc(f32, packed_.len * Wd);
    defer gpa.free(all);
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = dst, .len = 4 * all.len }, 0, std.mem.sliceAsBytes(all));
    // [W][n][2k] -> per row: rank 0's k, rank 1's k, ...
    const c = Wd * k;
    for (0..Wd) |w| for (0..n) |i| {
        const src_row = all[(w * n + i) * 2 * k ..][0 .. 2 * k];
        @memcpy(x.cval[i * c + w * k ..][0..k], src_row[0..k]);
        for (src_row[k..], x.cand[i * c + w * k ..][0..k]) |v, *o| o.* = @bitCast(v);
    };
    try checkCandidates(x.cand, x.f.cfg.vocab);
}
