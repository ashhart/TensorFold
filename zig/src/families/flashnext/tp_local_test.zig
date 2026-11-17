//! A paired window's actual encoding on a recording device: decode merges every layer's MoE locally on both Macs.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const tp_local = @import("tp_local.zig");
const tp_seq = @import("tp_seq.zig");
const Handoff = @import("tp_handoff.zig").Handoff;
const Job = @import("tp_job.zig").Job;
const testing = std.testing;

fn buffers(v: anytype, b: fz.Buf) void {
    const T = @TypeOf(v.*);
    if (T == fz.Buf) {
        v.* = b;
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| buffers(&@field(v.*, name), b),
        .array => for (v) |*x| buffers(x, b),
        else => {},
    }
}

fn site(r: *fz.Run, role: []const u8) !void {
    for (@import("roles_gen.zig").entries) |entry| {
        if (!std.mem.startsWith(u8, entry.site, role) or entry.site[role.len] != '|') continue;
        const text = for (@import("kernel_sources").flashnext_gen.sources) |source| {
            if (std.mem.eql(u8, source.name, entry.file)) break source.text;
        } else return error.NoSource;
        const v = try r.arena.create(fz.Variant);
        v.* = .{ .inputs = try r.arena.dupe([]const u8, entry.inputs), .outputs = try r.arena.dupe([]const u8, entry.outputs), .meta = try r.arena.dupe([]const u8, entry.meta), .name = entry.function, .text = text, .pipe = .{ .name = entry.function, .text = text } };
        try r.roles.put(r.arena, entry.site, .{ .v = v, .grid = mtl.Size.of(entry.grid[0], entry.grid[1], entry.grid[2]), .tg = mtl.Size.of(entry.tg[0], entry.tg[1], entry.tg[2]) });
    }
}

const Window = struct {
    jobs: []const Job,
    ops: []const mtl.HandoffOp,
    fn count(w: Window, pipeline: []const u8) usize {
        var n: usize = 0;
        for (w.ops) |op| n += @intFromBool(op.kind == .dispatch and std.mem.eql(u8, op.pipeline, pipeline));
        return n;
    }
    fn kinds(w: Window, kind: @FieldType(Job, "kind")) usize {
        var n: usize = 0;
        for (w.jobs) |j| n += @intFromBool(j.kind == kind);
        return n;
    }
};

/// One window of `rows` encoded on `rank` from a fresh pair state, decode mode or not; jobs and ops are copied out.
fn encode(a: std.mem.Allocator, rank: u32, rows: usize, decode: bool) !Window {
    mtl.allocator = a;
    mtl.resetHandoffs();
    var r: fz.Run = .{ .arena = a, .device = .{}, .queue = .{}, .serial = true, .xnew = true, .fused_xsum = true, .skip = 0xff & ~@as(u32, 1 << 2) };
    try site(&r, "qa_embed_rows@embed");
    r.xgu_pipe = .{ .name = "fz_xgu" };
    r.xdown_pipe = .{ .name = "fz_xdown" };
    const dummy: fz.Buf = .{ .b = try r.device.buffer(1 << 20, fz.opts) };
    var m: fz.Model = undefined;
    @memset(std.mem.asBytes(&m), 0);
    buffers(&m, dummy);
    m.r = &r;
    m.cap = 128;
    m.ple.sizes = @splat(1);
    const lane: fz.Lane = .{ .six = .{ .wq = dummy, .sbt = dummy } };
    m.head = lane;
    m.ple.kv = lane;
    for (&m.layers, 0..) |*l, i| {
        l.linear = i % 4 != 3;
        l.proj = lane;
        l.out = lane;
    }
    inline for (@typeInfo(fz.Tmp).@"struct".field_names, @typeInfo(fz.Tmp).@"struct".field_types) |name, T| @field(m.t, name) = if (T == fz.Buf) .{ .b = try r.device.buffer(512 << 10, fz.opts) } else .{ dummy, dummy };
    var tp: fz.Tp2 = undefined;
    tp.rank = rank;
    tp.seq = 0;
    tp.seqs = .{};
    tp.queued = .init(0);
    tp.failed = .init(false);
    tp.stop = .init(false);
    tp.last_seq = 0;
    tp.last_x = .{ .flag = 0, .slot = 0 };
    tp.jobs = try a.alloc(Job, 1024);
    tp.wbuf = try r.device.buffer(64, fz.opts);
    tp.one = try r.device.buffer(16, fz.opts);
    tp.pick_tmp = try r.device.buffer(16, fz.opts);
    tp.partial = try r.device.buffer(64, fz.opts);
    tp.branches = try r.device.buffer(tp_local.BYTES, fz.opts);
    tp_local.arm(tp.branches);
    inline for (.{ "copy", "branch", "argmax", "pick", "merge", "plain", "ids" }, .{ "tp_copy", "tp_branch", "tp_argmax_part", "tp_pick", "tp_merge", "tp_plain", "tp_ids" }) |field, name|
        @field(tp, field ++ "_pipe") = .{ .name = name };
    tp.handoff = try Handoff.init(.{}, &tp.failed, &tp.stop, rank);
    defer tp.handoff.deinit();
    r.tp = &tp;
    r.tp_decode = decode;
    r.enc = (mtl.CommandBuffer{ .id = 1 }).compute(.serial);
    mtl.record_handoffs = true;
    defer mtl.record_handoffs = false;
    m.windowMeta(rows);
    try m.windowEncode(rows, m.t.ids8);
    try testing.expect(!r.tp_layer);
    return .{ .jobs = try a.dupe(Job, tp.jobs[1 .. tp.seq + 1]), .ops = try a.dupe(mtl.HandoffOp, mtl.handoff_ops[0..mtl.handoff_op_count]) };
}

fn own(op: mtl.HandoffOp, at: usize) [2]u32 {
    return std.mem.bytesToValue([2]u32, op.args[at].bytes[0..8]);
}

test "pair decode: every layer's MoE merges locally with no exchange job or seam, and outside decode it exchanges" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 1, 2, 3, 5, 16 }) |rows| for (0..2) |rank| for ([_]bool{ false, true }) |decode| {
        const w = try encode(a, @intCast(rank), rows, decode);
        const mine = tp_seq.ownRows(@intCast(rank), rows);
        try testing.expectEqual(@as(usize, 36), w.kinds(.reduce));
        try testing.expectEqual(@as(usize, 1), w.kinds(.pick));
        try testing.expectEqual(@as(usize, if (decode) 0 else 48), w.kinds(.exchange));
        var ends: usize = 0;
        for (w.ops) |op| ends += @intFromBool(op.kind == .end);
        try testing.expectEqual(w.jobs.len, ends); // one seam a job, and none for a local merge
        const ranges: usize = if (decode) (if (rows > 1) 2 else 1) else @intFromBool(mine[1] > mine[0]);
        try testing.expectEqual(48 * ranges, w.count("tp_branch"));
        try testing.expectEqual(@as(usize, 48), w.count("tp_merge"));
        var seen: usize = 0;
        for (w.ops, 0..) |op, i| if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_merge")) {
            // a decode merge follows its own layer's branches directly, with no handoff between them
            if (decode) try testing.expectEqualStrings("tp_branch", w.ops[i - 1].pipeline);
            seen += 1;
        };
        try testing.expectEqual(@as(usize, 48), seen);
    };
}

test "pair decode: both Macs run every row's experts and number the remaining messages alike" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 1, 2, 7, 16 }) |rows| {
        var w: [2]Window = undefined;
        for (0..2) |rank| {
            w[rank] = try encode(a, @intCast(rank), rows, true);
            try testing.expectEqual(@as(usize, 48), w[rank].count("fz_xgu"));
            try testing.expectEqual(@as(usize, 48), w[rank].count("fz_xdown"));
            for (w[rank].ops) |op| if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "fz_xgu")) {
                try testing.expectEqual([2]u32{ 0, @intCast(rows) }, own(op, 17));
                try testing.expectEqual(rows * 11, op.grid.depth);
            };
        }
        try testing.expectEqual(w[0].jobs.len, w[1].jobs.len);
        for (w[0].jobs, w[1].jobs, 1..) |j0, j1, x| {
            try testing.expectEqual(j0.kind, j1.kind);
            try testing.expectEqual(@as(u64, x), j0.value); // flags rise by one a message on both Macs
            try testing.expectEqual(j0.value, j1.value);
            try testing.expectEqual(j0.remote, j1.remote);
        }
    }
}

test "pair decode: a one-row window's experts run on rank 1 too, which outside decode leaves them to rank 0" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |decode| {
        const w = try encode(a, 1, 1, decode);
        try testing.expectEqual(@as(usize, if (decode) 48 else 0), w.count("fz_xgu"));
        try testing.expectEqual(@as(usize, if (decode) 48 else 0), w.count("fz_xdown"));
        try testing.expectEqual(@as(usize, if (decode) 48 else 0), w.count("tp_branch"));
        for (w.ops) |op| if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_branch")) {
            try testing.expectEqual([2]u32{ 0, 1 }, own(op, 4));
        };
    }
}
