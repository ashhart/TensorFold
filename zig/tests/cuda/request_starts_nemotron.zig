//! Named-start snapshots through Nemotron's native prompt path, with real CUDA state and logits.
const std = @import("std");
const n = @import("nemotron");
const check = @import("check.zig");
const common = @import("request_start_checks.zig");
const reuse = n.reuse;

const Saver = struct {
    target: *reuse.Target,
    saved: [32]*anyopaque = undefined,
    at: [32]u32 = undefined,
    count: usize = 0,
    err: ?anyerror = null,

    fn call(ctx: *anyopaque, at: u32) void {
        const s: *Saver = @ptrCast(@alignCast(ctx));
        if (s.err != null) return;
        if (s.count == s.saved.len) {
            s.err = error.TooManySnapshots;
            return;
        }
        const saved = reuse.save(s.target, null, at) catch |err| {
            s.err = err;
            return;
        };
        s.saved[s.count] = saved;
        s.at[s.count] = at;
        s.count += 1;
    }
    fn drop(s: *Saver) void {
        for (s.saved[0..s.count]) |saved| reuse.drop(s.target, saved);
    }
};

fn prefill(e: *n.Engine, ids: []const u32, from: usize, h: *n.Head, saver: ?*Saver, starts: []const u32) !u32 {
    const keep: ?n.engine.Keep = if (saver) |s| .{ .at = &.{}, .ctx = s, .call = Saver.call } else null;
    if (comptime @typeInfo(@TypeOf(n.Engine.prefillFrom)).@"fn".param_types.len == 7) {
        return e.prefillFrom(ids, from, null, h, null, keep);
    }
    return e.prefillFrom(ids, from, null, h, null, keep, starts);
}

fn digest(gpu: check.Gpu, e: *n.Engine, h: *n.Head, token: u32) !common.Digest {
    try e.stream.synchronize();
    var state = common.Sha.init(.{});
    const row = n.state.kvRowBytes(e.c);
    for ([2]u64{ e.b.k_cache, e.b.v_cache }) |p| for (0..e.c.count(.attention)) |layer| {
        try common.add(gpu, &state, p + layer * e.max_len * row, e.pos * row);
    };
    const m = n.state.mambaPlanes(e.c);
    try common.add(gpu, &state, e.b.ssm, m[0]);
    try common.add(gpu, &state, e.b.conv_base, m[1]);
    var head = common.Sha.init(.{});
    for ([2]u64{ h.k_cache, h.v_cache }) |p| try common.add(gpu, &head, p, (e.pos - 1) * row);
    var logits = common.Sha.init(.{});
    try common.add(gpu, &logits, e.b.p_logits, e.c.vocab * 2);
    var hidden = common.Sha.init(.{});
    try common.add(gpu, &hidden, e.b.p_hidden + (e.pos - 1 - e.last_chunk) * e.c.hidden * 2, e.c.hidden * 2);
    return .{ .state = state.finalResult(), .head = head.finalResult(), .logits = logits.finalResult(), .hidden = hidden.finalResult(), .token = token };
}

pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8, n_text: []const u8, patterns: []const u8, segments_text: []const u8) !void {
    const ids = try common.readIds(gpu, ids_path, n_text);
    defer gpu.gpa.free(ids);
    const other = try gpu.gpa.dupe(u32, ids);
    defer gpu.gpa.free(other);
    const segments = try std.fmt.parseInt(usize, segments_text, 10);
    if (segments < 1 or segments > 4) return error.BadSegments;
    const e = try n.Engine.init(gpu.gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = false, .sampling = null, .segments = segments });
    defer e.deinit();
    try common.validateIds(ids, e.c.vocab);
    common.poison(other, e.c.vocab);
    const h = try n.Head.init(e);
    defer h.deinit();
    var target: reuse.Target = .{ .e = e, .head = h };
    const fresh = try digest(gpu, e, h, try prefill(e, ids, 0, h, null, &.{}));
    common.show("nemotron cold", fresh);
    if (std.mem.eql(u8, patterns, "-")) return;
    _ = try reuse.slots(&target, 32);
    var it = std.mem.splitScalar(u8, patterns, ';');
    while (it.next()) |pattern| {
        var buf: [32]u32 = undefined;
        const starts = try common.points(pattern, ids.len, &buf);
        const cache_off = try digest(gpu, e, h, try prefill(e, ids, 0, h, null, starts));
        try common.equal("nemotron named cache-off", cache_off, fresh);
        var saver: Saver = .{ .target = &target };
        defer saver.drop();
        const kept = try digest(gpu, e, h, try prefill(e, ids, 0, h, &saver, starts));
        if (saver.err) |err| return err;
        try check.expect(std.mem.eql(u32, starts, saver.at[0..saver.count]), "Nemotron named starts {s}: saved {any}, expected {any}", .{ pattern, saver.at[0..saver.count], starts });
        try common.equal("nemotron named hooked", kept, fresh);
        for (saver.at[0..saver.count], saver.saved[0..saver.count]) |at, saved| {
            _ = try prefill(e, other, 0, h, null, &.{});
            try reuse.restore(&target, null, saved);
            const got = try digest(gpu, e, h, try prefill(e, ids, at, h, null, starts));
            std.debug.print("RESULT nemotron restored-at={d} segments={d}\n", .{ at, segments });
            try common.equal("nemotron resumed", got, fresh);
        }
    }
    check.pass("nemotron-request-starts: device state, MTP rows, hidden, logits and first token equal", .{});
}
