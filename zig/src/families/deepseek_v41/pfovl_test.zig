//! Host tests of the prefill exchange overlap (TF_DSV41_PF_OVERLAP): the reader, and each exchange in `k` row pieces.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const branches = @import("branches.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_PF_OVERLAP: unset / 0 / 1 off, 2-8 row pieces, anything else refused" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expectEqual(@as(i64, 0), try pk.pfOverlap(&Fake.get));
    for ([_][2][]const u8{ .{ "0", "0" }, .{ "1", "0" }, .{ "2", "2" }, .{ "4", "4" }, .{ "8", "8" } }) |c| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP", c[0] }};
        try testing.expectEqual(try std.fmt.parseInt(i64, c[1], 10), try pk.pfOverlap(&Fake.get));
    }
    for ([_][]const u8{ "9", "-1", "on" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP", v }};
        try testing.expectError(error.BadKnob, pk.pfOverlap(&Fake.get));
    }
}

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

fn roleOf(x: calls.Arg) []const u8 {
    return switch (x.t.role) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

/// A call the overlap re-emits per piece: wo_a / wo_b / shared-expert projections, their glue, and the exchanges.
fn isPieced(c: calls.Call) bool {
    if (std.mem.startsWith(u8, c.name, "glue.exchange")) return true;
    if (std.mem.eql(u8, c.name, "glue.swiglu") or std.mem.eql(u8, c.name, "glue.moe_sum")) return true;
    // a projection's rot_in, unpack and GEMM read the producer's weights or its words ("s.<w>.lanes")
    for (c.args) |x| if (x.arg == .t and x.arg.t.role != .empty) {
        const r = roleOf(x.arg);
        for ([_][]const u8{ ".attn.wo_a.", ".attn.wo_b", ".shared.0." }) |w| if (std.mem.indexOf(u8, r, w) != null) return true;
    };
    return false;
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

const Mode = struct { n: i64, o: block.Options };

test "TF_DSV41_PF_OVERLAP on the prefill emitter: k pieces an exchange behind their producers, disjoint rows across the streams, every other call unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const D: i64 = cfg.hidden;
    const base: block.Options = .{ .gm_v2 = .one, .branches = true, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var big = base;
    big.pf4k = true;
    big.prefill_rows = 4096;
    for ([_]Mode{ .{ .n = 2048, .o = base }, .{ .n = 1024, .o = base }, .{ .n = 4096, .o = big } }) |md| {
        for ([_]i64{ 2, 4 }) |k| {
            var on_o = md.o;
            on_o.pf_overlap = k;
            const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
            const on = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, md.n, 0, false);
            // every call outside the pieced producers and exchanges: the same, in order
            var i: usize = 0;
            var j: usize = 0;
            while (true) {
                while (i < off.len and isPieced(off[i])) i += 1;
                while (j < on.len and isPieced(on[j])) j += 1;
                if (i == off.len or j == on.len) break;
                try testing.expectEqualStrings(off[i].name, on[j].name);
                try testing.expectEqual(off[i].args.len, on[j].args.len);
                i += 1;
                j += 1;
            }
            // the pieced projections' rot_in: k per projection in place of one
            const xs = count(off, "glue.exchange");
            try testing.expectEqual(@as(usize, 2 * layers.len), xs);
            try testing.expectEqual(@as(usize, 0), count(on, "glue.exchange"));
            try testing.expectEqual(xs * @as(usize, @intCast(k)), count(on, "glue.exchange_rows"));
            try testing.expectEqual(count(off, "tensorfold_exl3_linear_v4.rot_in") + (@as(usize, @intCast(k)) - 1) * layers.len * (4 + 1 + 3), count(on, "tensorfold_exl3_linear_v4.rot_in"));
            try testing.expectEqual(@as(usize, @intCast(k)) * count(off, "glue.moe_sum"), count(on, "glue.moe_sum"));
            // the streams: only the partials are shared inside a fork region
            const sh = try branches.shared(aa, on);
            for (sh) |r| try testing.expect(std.mem.eql(u8, r, "L.part") or std.mem.eql(u8, r, "L.moe16"));
            try testing.expect(sh.len == 2);
            try testing.expectEqual(@as(usize, 0), (try branches.shared(aa, off)).len);
            try pieces(on, md.n, k, D);
        }
        // fewer than 256 rows a piece, or no side stream: the knob-off program
        for ([_]block.Options{ blk: {
            var o = md.o;
            o.pf_overlap = 8;
            break :blk o;
        }, blk: {
            var o = md.o;
            o.pf_overlap = 4;
            o.branches = false;
            break :blk o;
        } }) |o| {
            if (o.pf_overlap == 8 and md.n >= 8 * 256) continue;
            const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
            const same = try bp.emitPrefill(aa, &cfg, &w, o, &layers, md.n, 0, false);
            try testing.expectEqual(off.len, same.len);
            for (off, same) |x, y| try testing.expectEqualStrings(x.name, y.name);
        }
    }
}

/// Each exchange's pieces tile [0, n) on the 16-row grid: k - 1 forked, the last joined, each after its producer.
fn pieces(cs: []const calls.Call, n: i64, k: i64, D: i64) !void {
    var next: i64 = 0;
    var seen: i64 = 0;
    for (cs, 0..) |c, idx| {
        if (!std.mem.eql(u8, c.name, "glue.exchange_rows")) continue;
        const send = c.args[0].arg.t;
        const into = c.args[1].arg.t;
        const r0 = c.args[2].arg.i;
        const m = send.shape[0];
        try testing.expectEqual(next, r0);
        try testing.expectEqual(@as(i64, 0), @mod(r0, 16));
        try testing.expectEqual(r0 * D * 2, send.offset);
        try testing.expectEqualSlices(i64, &.{ 2, n, D }, into.shape);
        // the piece's all-gather scratch (no NCCL p2p): [world, m, D] of its own role
        const tmp = c.args[3].arg.t;
        try testing.expectEqualStrings("L.xrows", roleOf(c.args[3].arg));
        try testing.expectEqualSlices(i64, &.{ 2, m, D }, tmp.shape);
        seen += 1;
        const last = @mod(seen, k) == 0;
        try testing.expectEqual(!last, c.side);
        try testing.expectEqual(!last, c.fork);
        try testing.expectEqual(last, c.join);
        // its producer: the call before it wrote these rows of the same role
        const prev = cs[idx - 1];
        const out = for (prev.args) |x| {
            if (x.arg == .t and x.arg.t.role == .buf and std.mem.eql(u8, x.arg.t.role.buf, roleOf(c.args[0].arg))) break x.arg.t;
        } else return error.TestUnexpectedResult;
        try testing.expectEqual(send.offset, out.offset);
        try testing.expectEqual(m, out.shape[0]);
        next = if (last) 0 else r0 + m;
        if (last) try testing.expectEqual(n, r0 + m);
    }
}

test "TF_DSV41_PF_OVERLAP_SITE: unset / 0 off, 1 on, anything else refused" {
    Fake.pairs = &.{};
    try testing.expect(!try pk.pfOverlapSite(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP_SITE", "1" }};
    try testing.expect(try pk.pfOverlapSite(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP_SITE", "0" }};
    try testing.expect(!try pk.pfOverlapSite(&Fake.get));
    for ([_][]const u8{ "2", "on", "-1" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_PF_OVERLAP_SITE", v }};
        try testing.expectError(error.BadKnob, pk.pfOverlapSite(&Fake.get));
    }
    Fake.pairs = &.{};
}

fn isSite(c: calls.Call) bool {
    return std.mem.eql(u8, c.name, "tf_dsv41_mhc_pf_v1.run") or std.mem.eql(u8, c.name, "_finish_k");
}

test "TF_DSV41_PF_OVERLAP_SITE on the prefill emitter: every exchange piece on the side with its mark, the next boundary site in the same row pieces after its own piece, every other call unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const D: i64 = cfg.hidden;
    const base: block.Options = .{ .gm_v2 = .one, .branches = true, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var big = base;
    big.pf4k = true;
    big.prefill_rows = 4096;
    for ([_]Mode{ .{ .n = 2048, .o = base }, .{ .n = 4096, .o = big } }) |md| {
        for ([_]i64{ 2, 4 }) |k| {
            var ov_o = md.o;
            ov_o.pf_overlap = k;
            var on_o = ov_o;
            on_o.pf_overlap_site = true;
            const ov = try bp.emitPrefill(aa, &cfg, &w, ov_o, &layers, md.n, 0, false);
            const on = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, md.n, 0, false);
            // outside the sites and the exchanges' stream marks: the PF_OVERLAP program, call for call
            var i: usize = 0;
            var j: usize = 0;
            while (true) {
                while (i < ov.len and isSite(ov[i])) i += 1;
                while (j < on.len and isSite(on[j])) j += 1;
                if (i == ov.len or j == on.len) break;
                try testing.expectEqualStrings(ov[i].name, on[j].name);
                try testing.expectEqual(ov[i].args.len, on[j].args.len);
                i += 1;
                j += 1;
            }
            try testing.expectEqual(ov.len - i, on.len - j);
            // every boundary site after an exchange (2 a layer but the first block's attention site) in k pieces
            const sites: usize = 2 * layers.len - 1;
            const kk: usize = @intCast(k);
            try testing.expectEqual(count(ov, "tf_dsv41_mhc_pf_v1.run") + (kk - 1) * sites, count(on, "tf_dsv41_mhc_pf_v1.run"));
            try testing.expectEqual(count(ov, "_finish_k") + (kk - 1) * sites, count(on, "_finish_k"));
            try sitePieces(on, md.n, k, D);
            // inside a fork region the streams share only the partials and the gathered buffers, at disjoint rows
            const sh = try branches.shared(aa, on);
            for (sh) |r| {
                const okr = std.mem.eql(u8, r, "L.part") or std.mem.eql(u8, r, "L.moe16") or std.mem.eql(u8, r, "L.recv.attn") or std.mem.eql(u8, r, "L.recv.moe");
                if (!okr) std.debug.print("shared role {s}\n", .{r});
                try testing.expect(okr);
            }
            try testing.expectEqual(@as(usize, 4), sh.len);
        }
    }
    // the knob needs the pieces: without PF_OVERLAP (or the side stream) the program is the knob-off one
    var o = base;
    o.pf_overlap_site = true;
    const off = try bp.emitPrefill(aa, &cfg, &w, base, &layers, 2048, 0, false);
    const same = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 2048, 0, false);
    try testing.expectEqual(off.len, same.len);
    for (off, same) |x, y| {
        try testing.expectEqualStrings(x.name, y.name);
        try testing.expect(!y.side and y.mark == 0 and y.wait == 0);
    }
}

/// Under TF_DSV41_PF_OVERLAP_SITE: piece p forks with mark p + 1 and the next site's k pairs wait on those marks.
fn sitePieces(cs: []const calls.Call, n: i64, k: i64, D: i64) !void {
    var xp: i64 = 0; // exchange pieces seen in the current exchange
    var sp: i64 = 0; // site pieces seen in the current site
    var rows: [8][2]i64 = undefined;
    var sites: usize = 0;
    for (cs, 0..) |c, idx| {
        if (std.mem.eql(u8, c.name, "glue.exchange_rows")) {
            try testing.expectEqual(@as(i64, 0), sp);
            const r0 = c.args[2].arg.i;
            const m = c.args[0].arg.t.shape[0];
            rows[@intCast(xp)] = .{ r0, m };
            const tail = idx + 1 == cs.len;
            if (tail and xp + 1 == k) {
                try testing.expect(!c.side and !c.fork and c.join and c.mark == 0);
            } else {
                try testing.expect(c.side and c.fork and !c.join);
                try testing.expectEqual(@as(u8, @intCast(xp + 1)), c.mark);
            }
            xp += 1;
            if (xp == k) xp = 0;
            continue;
        }
        try testing.expectEqual(@as(u8, 0), c.mark);
        if (std.mem.eql(u8, c.name, "tf_dsv41_mhc_pf_v1.run")) {
            const R = c.args[10].arg.i;
            if (R == n) continue; // a whole site (the first block's)
            const r = rows[@intCast(sp)];
            try testing.expectEqual(r[1], R);
            try testing.expectEqual(r[0] * 4 * D * 2, c.args[0].arg.t.offset);
            const g = c.args[2].arg.t;
            try testing.expectEqualSlices(i64, &.{ 2, r[1], D }, g.shape);
            try testing.expectEqualSlices(i64, &.{ n * D, D, 1 }, g.stride);
            try testing.expectEqual(r[0] * D * 2, g.offset);
            const last = sp + 1 == k;
            try testing.expectEqual(last, c.join);
            try testing.expectEqual(@as(u8, if (last) 0 else @intCast(sp + 1)), c.wait);
            const fin = cs[idx + 1];
            try testing.expectEqualStrings("_finish_k", fin.name);
            try testing.expectEqual(r[1], fin.grid[0]);
            for (fin.args) |x| if (std.mem.eql(u8, x.name, "OUT")) try testing.expectEqual(r[0] * D * 2, x.arg.t.offset);
            sp += 1;
            if (sp == k) {
                sp = 0;
                sites += 1;
            }
            continue;
        }
        try testing.expect(c.wait == 0);
    }
    try testing.expectEqual(@as(i64, 0), sp);
    try testing.expect(sites > 0);
}

test "TF_DSV41_MHC_SITE_ROWS: the reader, and every boundary site in row pieces on the main stream, every other call unchanged" {
    Fake.pairs = &.{};
    try testing.expectEqual(@as(i64, 0), try pk.mhcSiteRows(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_MHC_SITE_ROWS", "512" }};
    try testing.expectEqual(@as(i64, 512), try pk.mhcSiteRows(&Fake.get));
    for ([_][]const u8{ "128", "520", "8192", "x" }) |v| {
        Fake.pairs = &.{.{ "TF_DSV41_MHC_SITE_ROWS", v }};
        try testing.expectError(error.BadKnob, pk.mhcSiteRows(&Fake.get));
    }
    Fake.pairs = &.{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    const D: i64 = cfg.hidden;
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048 };
    var big = base;
    big.pf4k = true;
    big.prefill_rows = 4096;
    for ([_]Mode{ .{ .n = 2048, .o = base }, .{ .n = 4096, .o = big } }) |md| {
        var on_o = md.o;
        on_o.mhc_site_rows = 512;
        const off = try bp.emitPrefill(aa, &cfg, &w, md.o, &layers, md.n, 0, false);
        const on = try bp.emitPrefill(aa, &cfg, &w, on_o, &layers, md.n, 0, false);
        var i: usize = 0;
        var j: usize = 0;
        while (true) {
            while (i < off.len and isSite(off[i])) i += 1;
            while (j < on.len and isSite(on[j])) j += 1;
            if (i == off.len or j == on.len) break;
            try testing.expectEqualStrings(off[i].name, on[j].name);
            i += 1;
            j += 1;
        }
        try testing.expectEqual(off.len - i, on.len - j);
        const k: usize = @intCast(@divExact(md.n, 512));
        const sites: usize = 2 * layers.len - 1; // the boundaries: every site but the first block's attention site
        try testing.expectEqual(count(off, "tf_dsv41_mhc_pf_v1.run") + (k - 1) * sites, count(on, "tf_dsv41_mhc_pf_v1.run"));
        try testing.expectEqual(count(off, "_finish_k") + (k - 1) * sites, count(on, "_finish_k"));
        // the pieces tile each site's rows in order: its streams' rows, its gathered rows, its "w.out" rows
        var next: i64 = 0;
        for (on, 0..) |c, idx| {
            try testing.expect(!c.side and !c.fork and c.wait == 0 and c.mark == 0);
            if (!std.mem.eql(u8, c.name, "tf_dsv41_mhc_pf_v1.run")) continue;
            const R = c.args[10].arg.i;
            if (R == md.n) continue;
            try testing.expectEqual(@as(i64, 512), R);
            try testing.expectEqual(next * 4 * D * 2, c.args[0].arg.t.offset);
            try testing.expectEqual(next * D * 2, c.args[2].arg.t.offset);
            for (on[idx + 1].args) |x| if (std.mem.eql(u8, x.name, "OUT")) try testing.expectEqual(next * D * 2, x.arg.t.offset);
            next += R;
            if (next == md.n) next = 0;
        }
        try testing.expectEqual(@as(i64, 0), next);
    }
}
