//! A paired window's actual encoding on a recording device: decode makes every split stage's halves on both Macs.
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
    partial: usize, // the pair's partial and local buffer ids
    branches: usize,
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

const LANE = 1 << 1;
const GDN = 1 << 4;
const HEAD = 1 << 7;
const roles = [_][]const u8{ "qa_embed_rows@embed", "lane_qmm_bytes_grouped@gdn.in", "q4_gdn@gdn", "lane_qmm_bytes_grouped@gdn.out", "lane_qmm_bytes_grouped@head", "lane_qmm_bytes_grouped@att.proj", "lane_qmm_bytes_grouped@att.o", "lane_qmm_bytes_grouped@ple.kv", "mtp:qa_embed_rows@embed", "mtp:q4_rms_rows@mtp.enorm", "mtp:q4_rms_rows@mtp.hnorm" };
/// The MTP head's draft vocabulary in the test: 546 columns from the 1 MiB weight, 500 of them scored.
const DRAFT_IDS = 500;

/// One window of `rows` on `rank` from a fresh pair state, decode mode or not; jobs and ops are copied out.
fn encode(a: std.mem.Allocator, rank: u32, rows: usize, decode: bool) !Window {
    return encodeAs(a, rank, rows, decode, false);
}

/// The same, or with `draft` one MTP head step on one row in place of the window.
fn encodeAs(a: std.mem.Allocator, rank: u32, rows: usize, decode: bool, draft: bool) !Window {
    mtl.allocator = a;
    mtl.resetHandoffs();
    var r: fz.Run = .{ .arena = a, .device = .{}, .queue = .{}, .serial = true, .xnew = true, .fused_xsum = true, .skip = if (draft) 0xff else 0xff & ~@as(u32, 1 << 2 | LANE | GDN | HEAD), .dense = draft };
    for (roles) |role| try site(&r, role);
    inline for (.{ "add", "kv", "dense16", "dense8" }) |name| @field(r, name ++ "_pipe") = .{ .name = name };
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
    inline for (.{ "proj", "out", "fce", "fch", "draft" }) |name| @field(m.mtp, name) = lane;
    m.mtp.ids_n = DRAFT_IDS;
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
    if (draft) try m.mtpEncode(0, 1, m.t.ids8, dummy, m.mtp.pick) else {
        m.windowMeta(rows);
        try m.windowEncode(rows, m.t.ids8);
    }
    try testing.expect(!r.tp_layer);
    return .{ .jobs = try a.dupe(Job, tp.jobs[1 .. tp.seq + 1]), .ops = try a.dupe(mtl.HandoffOp, mtl.handoff_ops[0..mtl.handoff_op_count]), .partial = tp.partial.id, .branches = tp.branches.id };
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
        try testing.expectEqual(@as(usize, if (decode) 0 else 36), w.kinds(.reduce));
        try testing.expectEqual(@as(usize, if (decode) 0 else 1), w.kinds(.pick));
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

test "pair decode: both Macs run every row's experts and queue no message" {
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
        for (w) |x| try testing.expectEqual(@as(usize, 0), x.jobs.len);
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

fn has(op: mtl.HandoffOp, needle: []const u8) bool {
    return op.kind == .dispatch and std.mem.indexOf(u8, op.text, needle) != null;
}

fn word(op: mtl.HandoffOp, at: usize, comptime T: type) T {
    return std.mem.bytesToValue(T, op.args[at].bytes[0..@sizeOf(T)]);
}

/// A window's DeltaNet launches by kind (in-projection, step, out-projection, stream update), as op indices in order.
const Layers = struct {
    at: [4][72]usize = undefined,
    n: [4]usize = @splat(0),
    fn of(w: Window) Layers {
        var l: Layers = .{};
        for (w.ops, 0..) |op, i| {
            const kind: ?usize = if (has(op, " + (sg * 96) / SK;")) 2 else if (has(op, "if (j < 259) return 512 + j - 256;")) 0 else if (op.kind == .dispatch and std.mem.indexOf(u8, op.pipeline, "q4_gdn") != null) 1 else if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_plain")) 3 else null;
            if (kind) |k| {
                l.at[k][l.n[k]] = i;
                l.n[k] += 1;
            }
        }
        return l;
    }
};

test "pair decode: each DeltaNet layer runs both ranks' launches on each Mac and sums both partials in rank order" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 1, 2, 5, 8, 16 }) |rows| {
        var ref: [2]Window = undefined; // each rank's own launches outside decode, exchanged
        var refl: [2]Layers = undefined;
        for (0..2) |k| {
            ref[k] = try encode(a, @intCast(k), rows, false);
            refl[k] = Layers.of(ref[k]);
            const l = refl[k];
            try testing.expectEqual([4]usize{ 36, 36, 36, 36 }, l.n);
            try testing.expectEqual(@as(usize, 36), ref[k].count("tp_copy"));
            const q = try std.fmt.allocPrint(a, "if (j < 32) return {d} + j - 0;", .{32 * k});
            try testing.expect(has(ref[k].ops[l.at[0][0]], q));
            try testing.expectEqual(k == 1, has(ref[k].ops[l.at[1][0]], "threadgroup_position_in_grid.x) + 24;"));
            try testing.expect(has(ref[k].ops[l.at[2][0]], if (k == 0) "g_begin = 0 + (sg * 96)" else "g_begin = 96 + (sg * 96)"));
        }
        for (0..2) |rank| {
            const w = try encode(a, @intCast(rank), rows, true);
            const l = Layers.of(w);
            try testing.expectEqual([4]usize{ 72, 72, 72, 36 }, l.n);
            try testing.expectEqual(@as(usize, 0), w.jobs.len);
            try testing.expectEqual(@as(usize, 0), w.count("tp_copy"));
            for (w.ops) |op| try testing.expect(op.kind == .compute or op.kind == .dispatch); // no seam, signal or wait
            for (0..36) |layer| {
                const p = w.ops[l.at[3][layer]];
                var parts: [2]mtl.Arg = undefined; // by rank
                for (0..2) |j| {
                    const k = if (j == 0) rank else 1 - rank; // this rank's launches first, then the peer's
                    const x = 2 * layer + j;
                    for (0..3) |kind| { // the owning rank's own pipeline and grid, before this layer's sum
                        const got = w.ops[l.at[kind][x]];
                        const want = ref[k].ops[refl[k].at[kind][layer]];
                        try testing.expectEqualStrings(want.pipeline, got.pipeline);
                        try testing.expectEqualStrings(want.text, got.text);
                        try testing.expectEqual(want.grid, got.grid);
                        try testing.expect(l.at[kind][x] < l.at[3][layer] and (layer == 0 or l.at[kind][x] > l.at[3][layer - 1]));
                    }
                    parts[k] = w.ops[l.at[2][x]].args[5];
                    try testing.expectEqual(if (k == rank) w.partial else w.branches, parts[k].id);
                    try testing.expectEqual(@as(usize, if (k == rank) 0 else tp_local.PARTIAL_AT), parts[k].off);
                }
                const exchanged = ref[rank].ops[refl[rank].at[3][layer]];
                try testing.expectEqual(exchanged.grid, p.grid);
                try testing.expectEqual(word(exchanged, 5, u32), word(p, 5, u32));
                try testing.expectEqual(w.branches, p.args[0].id);
                try testing.expectEqual(tp_local.PASS, word(p, 1, u64));
                try testing.expectEqual(parts[rank], p.args[3]);
                try testing.expectEqual(parts[1 - rank], p.args[4]);
                // tp_plain adds MINE + PEER on rank 0 and PEER + MINE on rank 1: rank 0's partial first on both Macs
                try testing.expectEqual(parts[0], if (word(p, 5, u32) == 0) p.args[3] else p.args[4]);
            }
        }
    }
}

test "pair decode: the head takes both vocabulary halves and both argmax parts here, then one local pick" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const half = fz.VOCAB / 32 / 2;
    for ([_]usize{ 1, 3, 16 }) |rows| {
        var ref: [2]Window = undefined;
        for (0..2) |k| ref[k] = try encode(a, @intCast(k), rows, false);
        for (0..2) |rank| {
            const w = try encode(a, @intCast(rank), rows, true);
            var heads: [2]mtl.HandoffOp = undefined;
            var n: usize = 0;
            var after: usize = 0;
            for (w.ops, 0..) |op, i| if (has(op, try std.fmt.allocPrint(a, "if (j < {d}) return ", .{half}))) {
                heads[n] = op;
                n += 1;
                after = i;
            };
            try testing.expectEqual(@as(usize, 2), n);
            const tail = w.ops[after + 1 ..];
            try testing.expectEqual(@as(usize, 3), tail.len);
            for (0..2) |j| {
                const k = if (j == 0) rank else 1 - rank;
                try testing.expect(has(heads[j], try std.fmt.allocPrint(a, "if (j < {d}) return {d} + j - 0;", .{ half, half * k })));
                const own_head = for (ref[k].ops) |op| {
                    if (has(op, try std.fmt.allocPrint(a, "if (j < {d}) return ", .{half}))) break op;
                } else return error.NoHead;
                try testing.expectEqualStrings(own_head.text, heads[j].text);
                try testing.expectEqual(own_head.grid, heads[j].grid);
                const part = tail[j];
                const want = for (ref[k].ops) |op| {
                    if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_argmax_part")) break op;
                } else return error.NoPart;
                try testing.expectEqualStrings("tp_argmax_part", part.pipeline);
                try testing.expectEqual(want.grid, part.grid);
                try testing.expectEqual(word(want, 2, [4]u32), word(part, 2, [4]u32));
                try testing.expectEqual(want.args[0].off, part.args[0].off);
                try testing.expectEqual(mtl.Arg{ .id = w.branches, .off = tp_local.PICKS_AT + j * tp_local.PICK }, part.args[1]);
                try testing.expectEqual(w.branches, part.args[3].id);
                try testing.expectEqual(tp_local.PASS, word(part, 4, u64));
            }
            const pick = tail[2];
            try testing.expectEqualStrings("tp_pick", pick.pipeline);
            try testing.expectEqual(w.branches, pick.args[0].id);
            try testing.expectEqual(tp_local.PASS, word(pick, 1, u64));
            try testing.expectEqual(tail[0].args[1], pick.args[3]); // mine: this rank's half, as the swap binds it
            try testing.expectEqual(tail[1].args[1], pick.args[4]);
        }
    }
}

test "pair decode: an MTP draft takes both halves of its vocabulary, picks here and maps the id with no job" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ref: [2]Window = undefined;
    for (0..2) |k| {
        ref[k] = try encodeAs(a, @intCast(k), 1, false, true);
        try testing.expectEqual(@as(usize, 1), ref[k].kinds(.pick));
    }
    const n = (1 << 20) * 4 / (3 * 2560);
    const half = (n / 32 + 1) / 2;
    const want_parts = [2][2]u32{ .{ 0, half * 32 }, .{ half * 32, DRAFT_IDS - half * 32 } };
    for (0..2) |rank| {
        const w = try encodeAs(a, @intCast(rank), 1, true, true);
        try testing.expectEqual(@as(usize, 0), w.jobs.len);
        for (w.ops) |op| try testing.expect(op.kind == .compute or op.kind == .dispatch);
        const first = for (w.ops, 0..) |op, i| {
            if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_argmax_part")) break i;
        } else return error.NoPart;
        const ops = w.ops[first - 2 ..];
        try testing.expectEqual(@as(usize, 6), ops.len);
        for (0..2) |k| { // both tile halves with the shared dense kernel, rank 0's first on both Macs
            try testing.expectEqualStrings("dense16", ops[k].pipeline);
            try testing.expectEqual(@as(u32, @intCast(half * k)), word(ops[k], 4, [4]u32)[3]);
            try testing.expectEqual(32 * 16 * @as(usize, if (k == 0) half else n / 32 - half), ops[k].grid.width);
            const exchanged = for (ref[k].ops, 0..) |op, i| {
                if (op.kind == .dispatch and std.mem.eql(u8, op.pipeline, "tp_argmax_part")) break ref[k].ops[i - 1];
            } else return error.NoPart;
            try testing.expectEqual(word(exchanged, 4, [4]u32), word(ops[k], 4, [4]u32));
            try testing.expectEqual(exchanged.grid, ops[k].grid);
        }
        for (0..2) |j| {
            const k = if (j == 0) rank else 1 - rank;
            const part = ops[2 + j];
            try testing.expectEqual([4]u32{ @intCast(n), want_parts[k][0], want_parts[k][1], 0 }, word(part, 2, [4]u32));
            try testing.expectEqual(mtl.Arg{ .id = w.branches, .off = tp_local.PICKS_AT + j * tp_local.PICK }, part.args[1]);
            try testing.expectEqual(tp_local.PASS, word(part, 4, u64));
        }
        try testing.expectEqualStrings("tp_pick", ops[4].pipeline);
        try testing.expectEqual(ops[2].args[1], ops[4].args[3]);
        try testing.expectEqual(ops[3].args[1], ops[4].args[4]);
        try testing.expectEqualStrings("tp_ids", ops[5].pipeline);
        try testing.expectEqual(ops[4].args[5], ops[5].args[0]); // the merged pick, then its token id
        try testing.expectEqual(w.branches, ops[5].args[3].id);
        try testing.expectEqual(tp_local.PASS, word(ops[5], 4, u64));
    }
}
