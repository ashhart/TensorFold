//! The prefill's own workspace (TF_DSV41_PF_4K / _PF_TBO): plan, agreement, acquire, release.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const image_opts = @import("image_opts.zig");
const fwd = @import("forward.zig");
const graphs = @import("graphs.zig");
const serve_knobs = @import("serve_knobs.zig");
const Forward = fwd.Forward;
const pf = @import("forward_prefill.zig");
const encoderLayers = pf.encoderLayers;
const keepsShared = pf.keepsShared;
const lastStartWith = pf.lastStartWith;
const log = pf.log;
const planPrograms = pf.planPrograms;

/// Moves workspace roles that `forward` holds at least as large into `keep`, so the 4K program names the forward's.
pub fn k4Trim(gpa: std.mem.Allocator, wp: *buffers.Plan, forward: *const buffers.Plan, keep: *block_prefill.Keep) !void {
    var i: usize = 0;
    while (i < wp.sizes.count()) {
        const k = wp.sizes.keys()[i];
        const v = wp.sizes.values()[i];
        const at = std.mem.indexOf(u8, k, K4.tag) orelse {
            i += 1;
            continue;
        };
        const bare = try std.mem.concat(gpa, u8, &.{ k[0..at], k[at + K4.tag.len ..] });
        const have = forward.sizes.get(bare) orelse 0;
        if (have < v or keep.contains(bare)) {
            gpa.free(bare);
            if (have >= v) wp.sizes.orderedRemoveAt(i) else i += 1;
            continue;
        }
        try keep.put(gpa, bare, {});
        wp.sizes.orderedRemoveAt(i);
    }
}

/// The workspace programs: TF_DSV41_PF_4K's 4,096-row encoder segment and TF_DSV41_PF_TBO's pair, at 0 and last start.
pub fn planWorkspace(f: *Forward, wp: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const replay = pf.state(f).mode == .replay;
    if (pf.state(f).k4) {
        const enc = try encoderLayers(f, layers);
        const o4 = k4Options(image_opts.planOptions(f));
        const top4 = try lastStartWith(f, o4, enc, K4.rows, .encoder);
        for ([_]i64{ 0, top4 }) |start| {
            var ka = std.heap.ArenaAllocator.init(f.gpa);
            defer ka.deinit();
            const cs = try block_prefill.emitReplay(ka.allocator(), f.cfg, f.widths, o4, enc, K4.rows, start, .encoder);
            try wp.add(try block_prefill.ownCalls(ka.allocator(), cs, K4.tag));
        }
    }
    if (replay and f.opts.pf_tbo and rows >= 2 * block_prefill.tbo_min) {
        const enc = try encoderLayers(f, layers);
        const k4 = pf.state(f).k4;
        const op = if (k4) k4Options(image_opts.planOptions(f)) else image_opts.planOptions(f);
        const seg: u32 = if (k4) K4.rows else rows;
        const top = @max(0, try lastStartWith(f, op, enc, seg, .encoder) - seg); // B at the last start a segment takes
        for ([_]i64{ 0, top }) |start| {
            var ta = std.heap.ArenaAllocator.init(f.gpa);
            defer ta.deinit();
            const cs = try block_prefill.emitTbo(ta.allocator(), f.cfg, f.widths, op, enc, seg, seg, start);
            try wp.add(try wsPair(ta.allocator(), cs, k4));
        }
    }
}

/// The bytes the workspace allocates at a long prompt; 0 when neither TF_DSV41_PF_4K nor TF_DSV41_PF_TBO is on.
pub fn workspaceBytes(f: *Forward, rows: u32) !u64 {
    if (!(pf.state(f).k4 or (pf.state(f).mode == .replay and f.opts.pf_tbo))) return 0;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    var wp: buffers.Plan = .{ .a = la.allocator() };
    const layers = try f.backbone(la.allocator());
    try planWorkspace(f, &wp, rows, layers);
    if (keepsShared(f)) {
        // the same trim `plan` makes: the roles the forward's prefill programs hold already
        var fp: buffers.Plan = .{ .a = la.allocator() };
        try planPrograms(f, &fp, rows, layers);
        var keep: block_prefill.Keep = .empty;
        try k4Trim(la.allocator(), &wp, &fp, &keep);
    }
    return planBytes(&wp);
}

pub fn planBytes(wp: *const buffers.Plan) u64 {
    var total: u64 = 0;
    for (wp.sizes.keys(), wp.sizes.values()) |k, v| {
        if (std.mem.indexOf(u8, k, K4.tag) == null) continue;
        total += std.mem.alignForward(u64, v, 256);
    }
    return total;
}

/// TF_DSV41_PF_4K's workspace: a 4K segment's roles in their own namespace, allocated per long prompt, then released.
pub const K4 = struct {
    pub const rows: u32 = serve_knobs.pf4k_rows;
    pub const tag = "k4~";
    plan: ?buffers.Plan = null,
    /// the bare roles a 4K segment shares with the forward (k4Trim under TF_DSV41_PF_WS_SHARE=1; else none)
    keep: block_prefill.Keep = .empty,
    buf: ?cuda.DeviceBuffer = null,
    ready: bool = false,
    bytes: u64 = 0,
    allocs: u64 = 0,
    releases: u64 = 0,
    /// the exchange send buffer and Engram rows while a 4K segment runs; the forward's own are never regrown
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
};

/// The rows a prompt segment holds: 4,096 under TF_DSV41_PF_4K, else the forward's (the round planner's rows too).
pub fn segmentRows(f: *const Forward) usize {
    const st = pf.state(f);
    return if (st.k4 and st.ws_ok) K4.rows else @intCast(f.opts.prefill_rows);
}

/// Whether this prompt call may use the workspace, agreed by every rank; no: 2,048-row segments, same bits, logged.
pub fn wsAgreed(f: *Forward) !bool {
    const st = pf.state(f);
    const ws = &st.k4ws;
    if (ws.buf != null) return true;
    const pl = ws.plan orelse return false;
    const bytes = planBytes(&pl);
    const hard = wsFloorGiB();
    const avail = graphs.memAvailableGiB();
    const left = if (avail) |v| v - @as(f64, @floatFromInt(bytes)) / (1 << 30) else hard;
    if (st.agreement == null) st.agreement = try graphs.Agreement.init(f.runner.d, f.comm, f.runner.stream);
    const ok = try st.agreement.?.agree().all(st.agreement.?.agree().ctx, left >= hard);
    if (!ok) {
        st.ws_fallbacks += 1;
        if (st.ws_fallbacks <= 4 or st.ws_fallbacks % 64 == 0)
            if (!quiet) log.warn("prefill workspace: {d} MiB does not fit on every rank (rank {d}: {d:.2} GiB MemAvailable leaves {d:.2}, floor {d:.1} GiB, TF_DSV41_PF_WS_FLOOR_GIB): this prompt piece runs 2,048-row segments (fallback {d})", .{ bytes >> 20, f.comm.rank(), avail orelse -1, left, hard, st.ws_fallbacks });
    }
    return ok;
}

/// The options a 4,096-row segment emits with: the forward's at 4,096 rows with x3gm one block a segment.
pub fn k4Options(o: block.Options) block.Options {
    var x = o;
    x.prefill_rows = K4.rows;
    x.pf4k = true;
    return x;
}

/// A TF_DSV41_PF_TBO pair's calls with private roles in the workspace (B's always, A's too at 4K): no forward bytes.
pub fn wsPair(a: std.mem.Allocator, cs: []const calls.Call, all: bool) ![]calls.Call {
    if (all) return block_prefill.ownCalls(a, cs, K4.tag);
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *x| {
        x.* = c;
        const args = try a.alloc(calls.Named, c.args.len);
        for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try wsArg(a, y.arg) };
        x.args = args;
    }
    return out;
}

pub fn wsArg(a: std.mem.Allocator, x: calls.Arg) !calls.Arg {
    switch (x) {
        .t, .opaque_table => |t| {
            if (t.role != .buf or std.mem.indexOf(u8, t.role.buf, "b~") == null) return x;
            var u = t;
            u.role = .{ .buf = try block_prefill.ownRole(a, t.role.buf, K4.tag) };
            return if (x == .t) .{ .t = u } else .{ .opaque_table = u };
        },
        .list => |l| {
            const o = try a.alloc(calls.Arg, l.len);
            for (l, o) |y, *z| z.* = try wsArg(a, y);
            return .{ .list = o };
        },
        else => return x,
    }
}

/// The workspace's roles bound to one fresh zeroed allocation (the first 4K segment of a prompt).
pub fn k4Acquire(f: *Forward) !void {
    const ws = &pf.state(f).k4ws;
    if (ws.ready) return;
    const pl = ws.plan orelse return error.Unbound;
    const total = planBytes(&pl);
    // the room was agreed first (wsAgreed): every rank fits the workspace above the hard floor
    const t0 = nowMs();
    if (ws.buf != null) try f.runner.stream.synchronize();
    if (ws.buf == null) {
        ws.buf = try cuda.DeviceBuffer.alloc(f.runner.d, @max(total, 256));
        ws.bytes = total;
        ws.allocs += 1;
    }
    const b = ws.buf.?;
    errdefer for (pl.sizes.keys()) |k| {
        if (std.mem.indexOf(u8, k, K4.tag) != null) f.runner.forget(k);
    };
    try b.fill8(0, null);
    var at: u64 = 0;
    for (pl.sizes.keys(), pl.sizes.values()) |k, v| {
        if (std.mem.indexOf(u8, k, K4.tag) == null) continue;
        try f.runner.external(k, b.ptr + at);
        at += std.mem.alignForward(u64, v, 256);
    }
    try f.runner.stream.synchronize();
    ws.ready = true;
    const ms = nowMs() - t0;
    if (ws.allocs <= 4 or ws.allocs % 64 == 0) log.info("prefill 4K: rank {d}: workspace {d} MiB allocated and zeroed in {d:.1} ms (prompt {d}), held while the prompt prefills", .{ f.comm.rank(), total >> 20, ms, ws.allocs });
}

pub fn nowMs() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}

/// TF_DSV41_PF_WS_FLOOR_GIB: MemAvailable the workspace must leave on every rank (default TF_DSV41_FLOOR_HARD_GIB).
pub fn wsFloorGiB() f64 {
    const v = std.c.getenv("TF_DSV41_PF_WS_FLOOR_GIB") orelse return hardFloorGiB();
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch hardFloorGiB();
}

/// TF_DSV41_FLOOR_HARD_GIB (4, the boot check's).
pub fn hardFloorGiB() f64 {
    const v = std.c.getenv("TF_DSV41_FLOOR_HARD_GIB") orelse return 4.0;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch 4.0;
}

/// The workspace back (a prompt's CED finish: no 4K segment runs until the next prompt's first).
pub fn k4Release(f: *Forward) !void {
    const ws = &pf.state(f).k4ws;
    var b = ws.buf orelse return;
    try f.runner.stream.synchronize();
    const before = graphs.memAvailableGiB();
    if (ws.plan) |pl| for (pl.sizes.keys()) |k| {
        if (std.mem.indexOf(u8, k, K4.tag) != null) f.runner.forget(k);
    };
    b.free();
    ws.buf = null;
    ws.ready = false;
    // the prefill -> decode handoff: the row graphs' memory floor right after the free (first prompts, then every 64th)
    ws.releases += 1;
    if (ws.releases <= 4 or ws.releases % 64 == 0)
        log.info("prefill 4K: rank {d}: workspace {d} MiB freed at the prompt's end (release {d}); MemAvailable {d:.2} -> {d:.2} GiB", .{ f.comm.rank(), ws.bytes >> 20, ws.releases, before orelse -1, graphs.memAvailableGiB() orelse -1 });
    if (ws.send) |*x| x.free();
    if (ws.gathered) |*x| x.free();
    ws.send = null;
    ws.gathered = null;
}
