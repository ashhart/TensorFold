//! Host tests of row mode on the real emitter: the backbone window rewritten by rowmode.transform reads the row table.

const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const check = @import("calls_check.zig");
const graphs = @import("graphs.zig");
const rowtab = @import("rowtab.zig");
const rowmode = @import("rowmode.zig");
const Config = @import("config.zig").Config;

test {
    _ = @import("branches_test.zig"); // TF_DSV41_BRANCHES / MHC_DEFER on the same emitter
    _ = @import("kx_early_test.zig"); // TF_DSV41_KV_SPLIT_EARLY on the same emitter
    _ = rowtab;
    _ = rowmode;
    _ = @import("aot_needs.zig");
    _ = @import("batch.zig"); // TF_DSV41_WIN_PROF's timer
}

const S: u32 = 4;
const rmax: u32 = 16;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/expert-k2.txt"));
    return w;
}

/// One slot's pool at `limit` positions (kv_state.zig's view: every page + null; split: half + null + discard).
fn poolOf(limit: i64, split: bool) block.Pool {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .comp_pages = if (split) @divExact(pages, 2) + 2 else pages + 1, .ik_pages = pages + 1, .pts = pages, .split = split };
}

fn named(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

/// The row table's column `c` as the rewritten calls must pass it.
fn isCol(x: calls.Arg, c: rowtab.Col, R: i64) bool {
    const t = switch (x) {
        .t => |t| t,
        else => return false,
    };
    const role = switch (t.role) {
        .buf => |b| b,
        else => return false,
    };
    return std.mem.eql(u8, role, rowmode.table_role) and t.dt == .i64 and t.shape.len == 1 and t.shape[0] == R and t.offset == rowtab.colOffset(c, rmax);
}

fn roleOf(x: calls.Arg) ?[]const u8 {
    return switch (x) {
        .t => |t| switch (t.role) {
            .buf => |b| b,
            else => null,
        },
        else => null,
    };
}

/// Rows [a, a + n) of the table's column `c` (a row block of backend._blocked: RowWin.sub).
fn isColAt(x: calls.Arg, c: rowtab.Col, b: Rows) bool {
    const t = switch (x) {
        .t => |t| t,
        else => return false,
    };
    const role = switch (t.role) {
        .buf => |r| r,
        else => return false,
    };
    return std.mem.eql(u8, role, rowmode.table_role) and t.dt == .i64 and t.shape.len == 1 and t.shape[0] == b.n and t.offset == rowtab.colOffset(c, rmax) + 8 * @as(usize, @intCast(b.a));
}

const Rows = struct { a: i64, n: i64 };

const Counts = struct {
    kv_store: usize = 0,
    pool_norm: usize = 0,
    index_k: usize = 0,
    scores: usize = 0,
    rope: usize = 0,
    attn: usize = 0,
    topk: usize = 0,
    dense: usize = 0,
    // the long-context selections
    dtopk: usize = 0,
    keys: usize = 0,
    block_keys: usize = 0,
    counts: usize = 0,
    blocks: usize = 0,
};

fn checkProgram(one: []const calls.Call, rows: []const calls.Call, R: i64, pts: i64) !Counts {
    var n: Counts = .{};
    try testing.expectEqual(one.len, rows.len);
    // backend._blocked's row blocks (glue "block_pos" [role, start + a, a, rows]) by their position role
    var blocks: std.StringHashMapUnmanaged(Rows) = .empty;
    defer blocks.deinit(testing.allocator);
    for (one, rows) |o, c| {
        if (c.glue and std.mem.eql(u8, c.name, "glue.block_pos")) {
            try blocks.put(testing.allocator, roleOf(c.args[0].arg).?, .{ .a = c.args[2].arg.i, .n = c.args[3].arg.i });
            n.blocks += 1;
        }
        if (c.glue and std.mem.eql(u8, c.name, "glue.counts")) {
            // backend.visible_counts: each row's own position
            try testing.expect(isColAt(c.args[2].arg, .pos, .{ .a = 0, .n = R }));
            n.counts += 1;
        }
        const b: Rows = if (named(o, "POS")) |pv| (if (roleOf(pv)) |r| (blocks.get(r) orelse Rows{ .a = 0, .n = R }) else Rows{ .a = 0, .n = R }) else .{ .a = 0, .n = R };
        // the same launch: name, grid, argument count
        try testing.expectEqualStrings(o.name, c.name);
        try testing.expectEqual(o.grid, c.grid);
        try testing.expectEqual(o.args.len, c.args.len);
        const nm = c.name;
        if (c.triton) {
            const writes = std.mem.eql(u8, nm, "_kv_store") or std.mem.eql(u8, nm, "_pool_norm") or std.mem.eql(u8, nm, "_index_k");
            const reads = std.mem.eql(u8, nm, "_scores");
            if (std.mem.eql(u8, nm, "_rope")) {
                try testing.expect(isCol(named(c, "POS").?, .pos, R));
                n.rope += 1;
            }
            if (std.mem.eql(u8, nm, "_dtopk") or std.mem.eql(u8, nm, "_block_keys")) {
                // dtopk.select / blocks, index.candidate_blocks with rows=True: q = POS[r]
                try testing.expect(isColAt(named(c, "POS").?, .pos, b));
                try testing.expect(named(c, "ROWS").?.b);
                try testing.expectEqual(b.n, c.grid[0]);
                if (std.mem.eql(u8, nm, "_dtopk")) n.dtopk += 1 else n.block_keys += 1;
            }
            if (std.mem.eql(u8, nm, "_keys")) {
                n.keys += 1;
                for (o.args, c.args) |x, y| try testing.expectEqualDeep(x.arg, y.arg);
            }
            if (writes or reads) {
                try testing.expect(isColAt(named(c, "POS").?, .pos, if (reads) b else Rows{ .a = 0, .n = R }));
                try testing.expect(isColAt(named(c, "SL").?, if (writes) .wslot else .rslot, if (reads) b else Rows{ .a = 0, .n = R }));
                if (reads) try testing.expectEqual(b.n, c.grid[0]);
                try testing.expect(named(c, "ROWS").?.b);
                // PTS where the kernel pages (compress._row_paging): the stacked tables' row stride with a table
                if (named(c, "PTS")) |pv| {
                    const pt = named(c, "PT").?;
                    if (roleOf(pt)) |role| {
                        try testing.expect(rowmode.isTable(role));
                        try testing.expectEqual(pts, pv.i);
                        try testing.expectEqualSlices(i64, &.{ S, pts }, pt.t.shape);
                    } else try testing.expectEqual(@as(i64, 0), pv.i);
                }
            }
            if (std.mem.eql(u8, nm, "_kv_store")) {
                n.kv_store += 1;
                if (named(c, "RATIO").?.i == 0) {
                    // the SWA ring: S rings stacked
                    const v = named(c, "V").?.t;
                    try testing.expect(rowmode.isRing(roleOf(named(c, "V").?).?));
                    try testing.expectEqual(@as(i64, S) * named(o, "V").?.t.shape[0], v.shape[0]);
                }
            }
            if (std.mem.eql(u8, nm, "_pool_norm")) {
                n.pool_norm += 1;
                try testing.expect(isCol(named(c, "PREV").?, .prev, R));
                const split = named(c, "SPLIT").?.b;
                try testing.expectEqual(@as(i64, if (split) S else 0), named(c, "OFF").?.i);
                if (split) try testing.expectEqualSlices(i64, &.{ S, named(o, "CARRY").?.t.shape[1] }, named(c, "CARRY").?.t.shape);
            }
            if (std.mem.eql(u8, nm, "_index_k")) n.index_k += 1;
            if (std.mem.eql(u8, nm, "_scores")) n.scores += 1;
        } else if (std.mem.eql(u8, nm, "tf_dsv41_attn_cuda_v1.attn")) {
            n.attn += 1;
            try testing.expect(isCol(c.args[9].arg, .pos, R));
            try testing.expect(isCol(c.args[10].arg, .rslot, R));
            try testing.expect(rowmode.isRing(roleOf(c.args[5].arg).?));
            try testing.expectEqual(@as(i64, S) * o.args[5].arg.t.shape[0], c.args[5].arg.t.shape[0]);
            const pt_on = roleOf(c.args[11].arg) != null and c.args[11].arg.t.role != .empty;
            try testing.expectEqual(if (pt_on) pts else 0, c.args[21].arg.i);
        } else if (std.mem.eql(u8, nm, "tf_dsv41_attn_cuda_v1.topk")) {
            n.topk += 1;
            try testing.expect(isCol(c.args[17].arg, .pos, R));
        } else if (std.mem.eql(u8, nm, "glue.kx_dense")) {
            n.dense += 1;
            try testing.expectEqualSlices(i64, &.{ S, pts }, c.args[2].arg.t.shape);
        }
        // the one-slot position tensor is read by nobody but the positions glue
        if (!c.glue) for (c.args) |x| if (roleOf(x.arg)) |role| try testing.expect(!std.mem.eql(u8, role, "w.pos"));
    }
    return n;
}

test "row mode: the whole backbone window, paged pool, every CSA2 step on the row table" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    for ([_]bool{ false, true }) |r1| {
        var o: block.Options = .{ .limit = 8192, .r1 = r1, .expert_topp = 0.85 };
        o.pool = poolOf(o.limit, false);
        const pts = o.pool.?.pts;
        for ([_]u32{ 1, 4, 12, 16 }) |R| {
            const s: graphs.Settings = .{};
            const ctx = graphs.bucketOf(5000, s.bucket, s.grow);
            const start = graphs.emitStart(R, ctx, s, @intCast(o.limit));
            const one = try block.emit(a, &cfg, &w, o, &backbone, R, start, true);
            const rows = try rowmode.transform(a, one, R, .{ .slots = S, .rmax = rmax, .pts = pts });
            const n = try checkProgram(one, rows, R, pts);
            // every backbone layer stores its SWA row; every full layer pools, keys and scores; every layer attends
            try testing.expect(n.kv_store >= 40 and n.attn == 40 and n.pool_norm > 0 and n.pool_norm == n.index_k and n.scores > 0 and n.topk == n.scores);
            try testing.expect(r1 or n.rope >= 40);
            // the indexer scores the context bucket's keys: NK = (bucket end + 1) / ratio
            for (rows) |c| if (std.mem.eql(u8, c.name, "_scores") and !named(c, "GATHER").?.b) {
                const ratio = named(c, "RATIO").?.i;
                const end: i64 = @intCast(graphs.bucketEnd(ctx, s.bucket, @intCast(o.limit), s.grow) + 1);
                try testing.expectEqual(@divFloor(end, ratio), named(c, "NK").?.i);
            };
            // the plan: S rings, S carries, S page tables; every other role as one slot's
            var p1: buffers.Plan = .{ .a = a };
            try p1.add(one);
            var pr: buffers.Plan = .{ .a = a };
            try pr.add(rows);
            var stacked: usize = 0;
            for (p1.sizes.keys(), p1.sizes.values()) |role, v| {
                const got = pr.sizes.get(role).?;
                if (rowmode.isRing(role) or rowmode.isCarry(role) or rowmode.isTable(role)) {
                    try testing.expectEqual(@as(u64, S) * v, got);
                    stacked += 1;
                } else try testing.expectEqual(v, got);
            }
            try testing.expect(stacked > 40);
            try testing.expect(pr.sizes.get(rowmode.table_role).? <= rowtab.tableBytes(rmax));
        }
    }
}

test "row mode: split KV's dense exchange reads the stacked split table; a union is refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    var o: block.Options = .{ .limit = 8192 };
    o.pool = poolOf(o.limit, true);
    const pts = o.pool.?.pts;
    const one = try block.emit(a, &cfg, &w, o, &backbone, 8, 2040, true);
    const rows = try rowmode.transform(a, one, 8, .{ .slots = S, .rmax = rmax, .pts = pts });
    const n = try checkProgram(one, rows, 8, pts);
    try testing.expect(n.dense > 0);
    const big = try block.emit(a, &cfg, &w, o, &backbone, 24, 2040, true);
    try testing.expectError(error.UnionInRows, rowmode.transform(a, big, 24, .{ .slots = S, .rmax = 64, .pts = pts }));
    // a row window past 16 rows (4 slots' verify windows, TF_DSV41_ROWS_CAP 64) exchanges dense
    var ro = o;
    ro.rows = true;
    for ([_]u32{ 17, 24, 64 }) |R| {
        const wide = try block.emit(a, &cfg, &w, ro, &backbone, R, 2040, true);
        const wrows = try rowmode.transform(a, wide, R, .{ .slots = S, .rmax = 64, .pts = pts });
        var dense: usize = 0;
        for (wrows) |c| {
            try testing.expect(!std.mem.eql(u8, c.name, "glue.kx_union"));
            dense += @intFromBool(std.mem.eql(u8, c.name, "glue.kx_dense"));
        }
        try testing.expect(dense > 0);
    }
    // the contiguous slot has no row mode
    try testing.expectError(error.NoPool, rowmode.transform(a, one, 8, .{ .slots = S, .rmax = rmax, .pts = 0 }));
}

// -- 1M context, 4 slots, the served knobs ----

/// The reference attention plan's entries a thread (512 threads, MAX_CL 8, 4096 a CTA); past 31 dtopk runs instead.
fn pyEpt(nk: i64) i64 {
    const cl = @max(1, @min(8, std.math.divCeil(i64, nk, 4096) catch unreachable));
    var ept = @max(1, std.math.divCeil(i64, nk, cl * 512) catch unreachable);
    ept += 1 - @mod(ept, 2);
    return ept;
}

/// The selection launches the reference backend makes for one R-row window over `end` positions.
fn pyExpect(cfg: *const Config, R: i64, end: i64, budget: i64) Counts {
    var x: Counts = .{};
    for (0..cfg.layers) |l| {
        const L: u32 = @intCast(l);
        const md = cfg.mode(L);
        if (md != .full and md != .reindex) continue;
        x.scores += 1;
        if (md == .reindex) {
            x.topk += 1; // attn_cuda.gather_cand over 2,048 x 8 keys
            continue;
        }
        const cand = cfg.isCandidateSource(L);
        const nvis = @max(@divFloor(end, cfg.compressRatio(L)), 1);
        if (R * nvis * 12 > budget) {
            // _blocked: _row_blocks' steps, each _scores + _keys (+ _block_keys), then visible_counts
            const st = if (R > 16) @max(16, @divFloor(budget, nvis * 12)) else R;
            const nb: usize = @intCast(std.math.divCeil(i64, R, st) catch unreachable);
            x.scores += nb - 1;
            x.keys += nb;
            if (cand) x.block_keys += nb;
            x.counts += 1;
            if (nb > 1) x.blocks += nb;
        } else if (pyEpt(nvis) <= 31) {
            x.topk += 1; // attn_cuda.select (select_blocks: both jobs in one launch)
        } else {
            x.dtopk += 1; // dtopk.select
            if (cand) {
                if (pyEpt(std.math.divCeil(i64, nvis, cfg.candidate_block_size) catch unreachable) <= 31) x.topk += 1 else x.dtopk += 1;
            }
        }
    }
    return x;
}

fn servedOptions(limit: i64) block.Options {
    var o: block.Options = .{ .limit = limit, .r1 = true, .expert_topp = 0.85, .index_budget = 64 << 20, .rows = true, .rope_rows = limit + 2048 };
    o.pool = poolOf(limit, false);
    return o;
}

fn expectSelections(got: Counts, want: Counts) !void {
    inline for (.{ "scores", "keys", "block_keys", "dtopk", "topk", "counts", "blocks" }) |f| {
        if (@field(got, f) != @field(want, f)) {
            std.debug.print("selection count {s}: got {d}, Python's backend {d}\n", .{ f, @field(got, f), @field(want, f) });
            return error.TestExpectedEqual;
        }
    }
}

test "row mode at 1M, 4 slots, the served knobs: every bucket and context bucket rewrites, the selections are Python's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    const o = servedOptions(limit);
    const pts = o.pool.?.pts;
    const s: graphs.Settings = .{};
    var seen_dtopk = false;
    var seen_blocked = false;
    var seen_multi = false;
    for (rowtab.buckets) |R| {
        if (R > rmax) break;
        for ([_]u64{ 0, 5000, 130_000, 300_000, 700_000, @intCast(limit - 1) }) |last| {
            const ctx = graphs.bucketOf(last, s.bucket, s.grow);
            const start = graphs.emitStart(R, ctx, s, @intCast(limit));
            if (start < 0) continue;
            const end: i64 = start + R;
            const one = try block.emit(a, &cfg, &w, o, &backbone, R, start, true);
            const rows = try rowmode.transform(a, one, R, .{ .slots = S, .rmax = rmax, .pts = pts });
            const n = try checkProgram(one, rows, R, pts);
            try expectSelections(n, pyExpect(&cfg, R, end, o.index_budget));
            seen_dtopk = seen_dtopk or n.dtopk > 0;
            seen_blocked = seen_blocked or n.keys > 0;
            seen_multi = seen_multi or n.blocks > 0;
            // a blocked row-mode window scores the bucket's keys in every block (RowWin.sub keeps cap)
            for (rows) |c| if (std.mem.eql(u8, c.name, "_scores") and !named(c, "GATHER").?.b)
                try testing.expectEqual(@divFloor(end, named(c, "RATIO").?.i), named(c, "NK").?.i);
        }
    }
    // the plan's windows (Batch.plan: every bucket up to the cap at the limit's end) and the roles they size
    var pr: buffers.Plan = .{ .a = a };
    for (rowtab.buckets) |R| {
        if (R > rmax) break;
        const one = try block.emit(a, &cfg, &w, o, &backbone, R, limit - R, true);
        const rows = try rowmode.transform(a, one, R, .{ .slots = S, .rmax = rmax, .pts = pts });
        const n = try checkProgram(one, rows, R, pts);
        try expectSelections(n, pyExpect(&cfg, R, limit, o.index_budget));
        try pr.add(rows);
    }
    // at the limit every full layer is past attn_cuda's plan; 16 rows x 1M keys pass the 64 MiB budget (_blocked)
    try testing.expectEqual(@as(u64, 16 * (1 << 20) * 8), pr.sizes.get("L.ix.keys").?);
    try testing.expectEqual(@as(u64, 16 * (1 << 20) * 4), pr.sizes.get("L.scores").?);
    try testing.expectEqual(@as(u64, 16 * (1 << 17) * 8), pr.sizes.get("L.ix.bkeys").?);
    try testing.expect(pr.sizes.get(rowmode.table_role).? <= rowtab.tableBytes(rmax));
    try testing.expect(seen_dtopk and seen_blocked and !seen_multi);
}

test "row mode at 1M: buckets past 16 rows run backend._blocked's 16-row blocks on their rows of the table" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    const o = servedOptions(limit);
    const pts = o.pool.?.pts;
    // these programs' table holds R rows (rmax = R, countsOf)
    for ([_]u32{ 24, 32, 48, 64 }) |R| {
        const one = try block.emit(a, &cfg, &w, o, &backbone, R, limit - R, true);
        // the blocks: 16 rows each, every one scoring the limit's keys, POS = the block's rows
        var blocks: usize = 0;
        for (one) |c| if (c.glue and std.mem.eql(u8, c.name, "glue.block_pos")) {
            // backend._row_blocks: 16 rows from each multiple of 16, the last block the rest
            try testing.expectEqual(@as(i64, 0), @mod(c.args[2].arg.i, 16));
            try testing.expectEqual(@min(16, @as(i64, R) - c.args[2].arg.i), c.args[3].arg.i);
            blocks += 1;
        };
        try testing.expect(blocks > 0);
        try expectSelections(try countsOf(a, one, R, pts), pyExpect(&cfg, R, limit, o.index_budget));
    }
}

/// The selections of a program rewritten with rmax = R (buckets past the shared tests' 16-row table).
fn countsOf(a: std.mem.Allocator, one: []const calls.Call, R: u32, pts: i64) !Counts {
    const rows = try rowmode.transform(a, one, R, .{ .slots = S, .rmax = R, .pts = pts });
    var n: Counts = .{};
    var blocks: std.StringHashMapUnmanaged(Rows) = .empty;
    defer blocks.deinit(testing.allocator);
    for (one, rows) |o, c| {
        if (c.glue and std.mem.eql(u8, c.name, "glue.block_pos")) {
            try blocks.put(testing.allocator, roleOf(c.args[0].arg).?, .{ .a = c.args[2].arg.i, .n = c.args[3].arg.i });
            n.blocks += 1;
        }
        if (c.glue and std.mem.eql(u8, c.name, "glue.counts")) n.counts += 1;
        if (!c.triton) {
            if (std.mem.eql(u8, c.name, "tf_dsv41_attn_cuda_v1.topk")) n.topk += 1;
            continue;
        }
        const b: Rows = if (named(o, "POS")) |pv| (if (roleOf(pv)) |r| (blocks.get(r) orelse Rows{ .a = 0, .n = R }) else Rows{ .a = 0, .n = R }) else .{ .a = 0, .n = R };
        const col = struct {
            fn at(x: calls.Arg, c_: rowtab.Col, bb: Rows, rm: u32) bool {
                const t = x.t;
                return std.mem.eql(u8, t.role.buf, rowmode.table_role) and t.shape[0] == bb.n and t.offset == rowtab.colOffset(c_, rm) + 8 * @as(usize, @intCast(bb.a));
            }
        }.at;
        if (std.mem.eql(u8, c.name, "_scores")) {
            n.scores += 1;
            try testing.expect(col(named(c, "POS").?, .pos, b, R) and col(named(c, "SL").?, .rslot, b, R));
        }
        if (std.mem.eql(u8, c.name, "_keys")) n.keys += 1;
        if (std.mem.eql(u8, c.name, "_block_keys")) {
            n.block_keys += 1;
            try testing.expect(col(named(c, "POS").?, .pos, b, R) and named(c, "ROWS").?.b);
        }
        if (std.mem.eql(u8, c.name, "_dtopk")) n.dtopk += 1;
    }
    return n;
}

test "one slot at 1M with the served budget: verify windows past 5 rows select through _keys as Python's backend does" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    const limit: i64 = 1 << 20;
    var o = servedOptions(limit);
    o.rows = false;
    // Forward.plan's windows (1..16 rows at the limit's end) and a window at 400K
    for (1..17) |r| {
        const R: i64 = @intCast(r);
        for ([_]i64{ limit - R, 400_000 }) |start| {
            const one = try block.emit(a, &cfg, &w, o, &backbone, R, start, true);
            var n: Counts = .{};
            for (one) |c| {
                if (c.triton and std.mem.eql(u8, c.name, "_scores")) n.scores += 1;
                if (c.triton and std.mem.eql(u8, c.name, "_keys")) n.keys += 1;
                if (c.triton and std.mem.eql(u8, c.name, "_block_keys")) n.block_keys += 1;
                if (c.triton and std.mem.eql(u8, c.name, "_dtopk")) {
                    n.dtopk += 1;
                    try testing.expect(!named(c, "ROWS").?.b);
                }
                if (std.mem.eql(u8, c.name, "tf_dsv41_attn_cuda_v1.topk")) n.topk += 1;
                if (std.mem.eql(u8, c.name, "glue.counts")) n.counts += 1;
            }
            try expectSelections(n, pyExpect(&cfg, R, start + R, o.index_budget));
        }
    }
    // a one-slot window over 16 rows past the stream threshold would stream (backend._stream_select): refused
    try testing.expectError(error.Unsupported, block.emit(a, &cfg, &w, o, &backbone, 24, 400_000, true));
}

test "rowtab at 1M: four slots' windows at the limit's end, padded, within the limit" {
    const limit: u64 = 1 << 20;
    const mix = [_]rowtab.Seg{
        .{ .slot = 3, .start = limit - 6, .rows = 6 },
        .{ .slot = 0, .start = 900_000, .rows = 1 },
        .{ .slot = 2, .start = 131_071, .rows = 4 },
        .{ .slot = 1, .start = 0, .rows = 2 },
    };
    try rowtab.check(&mix, S, limit);
    try testing.expectError(error.PastLimit, rowtab.check(&.{.{ .slot = 0, .start = limit - 1, .rows = 2 }}, S, limit));
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    const R = rowtab.bucketRows(rowtab.totalRows(&mix), rmax).?;
    try testing.expectEqual(@as(u32, 16), R);
    var t: [rowtab.ncol * rmax]i64 = undefined;
    var r32: [rmax]i32 = undefined;
    try rowtab.plan(&mix, &ids, R, rmax, S, &t, &r32);
    const pos = t[@backingInt(rowtab.Col.pos) * rmax ..][0..R];
    try testing.expectEqualSlices(i64, &.{ limit - 6, limit - 5, limit - 4, limit - 3, limit - 2, limit - 1, 900_000, 131_071, 131_072, 131_073, 131_074, 0, 1, 1, 1, 1 }, pos);
    const ws = t[@backingInt(rowtab.Col.wslot) * rmax ..][0..R];
    try testing.expectEqualSlices(i64, &.{ 3, 3, 3, 3, 3, 3, 0, 2, 2, 2, 2, 1, 1, -1, -1, -1 }, ws);
    // the window's context bucket: the limit's last (its end is the limit)
    const s: graphs.Settings = .{};
    const ctx = graphs.bucketOf(rowtab.lastPos(&mix), s.bucket, s.grow);
    try testing.expectEqual(limit - 1, graphs.bucketEnd(ctx, s.bucket, limit, s.grow));
}

test "TF_DSV41_ROUND_GRAPH=split on the served row window: head + tail = the window, no side or deferred fork open at the boundary, the head ~ its layers" {
    const round_graph = @import("round_graph.zig");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const w = try widths(a);
    const cfg: Config = .{};
    var backbone: [40]u32 = undefined;
    for (&backbone, 0..) |*l, i| l.* = @intCast(i);
    for ([_]bool{ false, true }) |r1| for ([_]bool{ false, true }) |br| for ([_]u32{ 1, 5, 16 }) |R| {
        var o: block.Options = .{ .limit = 8192, .r1 = r1, .expert_topp = 0.85, .branches = br, .mhc_defer = br and r1 };
        o.pool = poolOf(o.limit, false);
        const pts = o.pool.?.pts;
        const s: graphs.Settings = .{};
        const ctx = graphs.bucketOf(5000, s.bucket, s.grow);
        const one = try block.emit(a, &cfg, &w, o, &backbone, R, graphs.emitStart(R, ctx, s, @intCast(o.limit)), true);
        const rows = try rowmode.transform(a, one, R, .{ .slots = S, .rmax = rmax, .pts = pts });
        var layer_at: [41]usize = undefined; // each layer's first call (layer 0: the window's)
        var nl: usize = 1;
        layer_at[0] = 0;
        for (rows, 0..) |c, i| if (c.begin == .layer) {
            layer_at[nl] = i;
            nl += 1;
        };
        try testing.expectEqual(@as(usize, 40), nl);
        for ([_]u32{ 1, 2, 7 }) |head| {
            const at = round_graph.splitAt(rows, head) orelse return error.NoSplit;
            // past the head's layers' calls, before the next layer's end: a few calls into layer `head` at most
            try testing.expect(at >= layer_at[head] and at < layer_at[head + 1]);
            // the runner's fork state before `at` (runner.issue): neither the side nor the deferred stream open
            var side = false;
            var dfr = false;
            var sides: usize = 0;
            for (rows[0..at]) |c| {
                if (c.join) side = false;
                if (c.fork) side = true;
                if (c.defer_join) dfr = false;
                if (c.defer_side) dfr = true;
                if (c.side) sides += 1;
            }
            try testing.expect(!side and !dfr);
            // the tail's first side launch comes after a fork of its own (runner.issue refuses a side call unforked)
            for (rows[at..]) |c| {
                if (c.fork) break;
                try testing.expect(!c.side);
            }
            if (br) try testing.expect(sides > 0);
        }
        try testing.expectEqual(@as(?usize, null), round_graph.splitAt(rows, 40));
    };
}

test {
    _ = graphs; // graphs.zig's own tests (the room rule, the bucket ends)
    _ = @import("forms_inplace.zig"); // TF_DSV41_FORMS_INPLACE's binding
    _ = @import("ex_stacks.zig"); // TF_DSV41_EX_STACKS's pairs and binding
}
