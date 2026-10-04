//! A fixed schedule of mixed rounds (prompt chunks and drafted windows of three streams) run through one layer.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const synth = @import("synth.zig");

/// One stream's rows in a round: a prompt chunk (`commit`), or a drafted window keeping its first `keep` rows.
pub const Seg = struct { stream: u8, rows: u32, commit: bool, keep: u32 = 0 };

pub const schedule = [_][]const Seg{
    &.{ .{ .stream = 0, .rows = 5, .commit = true }, .{ .stream = 1, .rows = 3, .commit = true } },
    &.{ .{ .stream = 0, .rows = 4, .commit = false, .keep = 2 }, .{ .stream = 1, .rows = 1, .commit = false, .keep = 1 }, .{ .stream = 2, .rows = 6, .commit = true } },
    &.{ .{ .stream = 2, .rows = 3, .commit = false, .keep = 3 }, .{ .stream = 0, .rows = 3, .commit = false, .keep = 1 }, .{ .stream = 1, .rows = 2, .commit = false, .keep = 0 } },
    &.{ .{ .stream = 1, .rows = 16, .commit = false, .keep = 7 }, .{ .stream = 0, .rows = 1, .commit = false, .keep = 1 }, .{ .stream = 2, .rows = 40, .commit = true } },
    &.{.{ .stream = 0, .rows = 64, .commit = true }},
    &.{ .{ .stream = 2, .rows = 16, .commit = false, .keep = 9 }, .{ .stream = 0, .rows = 120, .commit = true }, .{ .stream = 1, .rows = 120, .commit = true } },
    &.{ .{ .stream = 0, .rows = 40, .commit = true }, .{ .stream = 1, .rows = 72, .commit = true }, .{ .stream = 2, .rows = 16, .commit = false, .keep = 4 } },
    &.{ .{ .stream = 0, .rows = 16, .commit = false, .keep = 16 }, .{ .stream = 1, .rows = 16, .commit = false, .keep = 3 } },
};

/// How the schedule's rows are grouped into rounds: as written, a segment a round, or a kept row a round.
pub const Comp = enum { mixed, split, serial };

pub const Key = struct { stream: u8, round: u8, row: u16 };

/// Each row's layer outputs as bf16 words: the prefix after attention, the MLP's normed input, the MLP's output.
pub const Results = struct {
    gpa: std.mem.Allocator,
    rows: std.AutoArrayHashMapUnmanaged(Key, []u16) = .empty,

    pub fn deinit(r: *Results) void {
        for (r.rows.values()) |v| r.gpa.free(v);
        r.rows.deinit(r.gpa);
    }
};

pub const Run = struct {
    gpa: std.mem.Allocator,
    queue: mtl.Queue,
    ctx: k3.layer.Ctx,
    w: *const k3.weights.Layer,
    layer: u32,
    state: *k3.state.State,
};

const Row = struct { key: Key, seg: u32 };

/// The checker's name for a row input: "in.<stream>.<round>.<row>.<what>[<block>]".
pub fn inputName(buf: *[96]u8, k: Key, what: []const u8, block: ?usize) []const u8 {
    if (block) |b| return std.fmt.bufPrint(buf, "in.{d}.{d}.{d}.{s}{d}", .{ k.stream, k.round, k.row, what, b }) catch unreachable;
    return std.fmt.bufPrint(buf, "in.{d}.{d}.{d}.{s}", .{ k.stream, k.round, k.row, what }) catch unreachable;
}

fn inputs(run: Run, rows: []const Row) void {
    const c = run.ctx.c;
    const H = c.hidden;
    const sc = run.ctx.sc;
    const R = rows.len;
    var buf: [96]u8 = undefined;
    for (rows, 0..) |row, r| {
        synth.rowValues(sc.prefix.slice(u16, (r + 1) * H)[r * H ..], inputName(&buf, row.key, "prefix", null));
        synth.rowValues(sc.delta.slice(u16, (r + 1) * H)[r * H ..], inputName(&buf, row.key, "delta", null));
        for (0..c.blocksBefore(run.layer)) |e| {
            const plane = sc.blocks.slice(u16, (e + 1) * R * H)[e * R * H ..];
            synth.rowValues(plane[r * H .. (r + 1) * H], inputName(&buf, row.key, "block", e));
        }
    }
}

/// One round on the GPU: lay out its segments, write the rows' inputs, encode the layer, wait, record outputs.
fn round(run: Run, segs: []const k3.round.Segment, rows: []const Row, out: *Results) !void {
    const sc = run.ctx.sc;
    var tokens: [256]u32 = @splat(0);
    try sc.load(run.state, segs, tokens[0..rows.len]);
    inputs(run, rows);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = run.queue.commandBuffer();
    const e = cb.compute(.serial);
    try k3.layer.encode(run.ctx, e, run.layer, run.w);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.debug.print("command buffer failed: {s}\n", .{text});
        return error.GpuFailed;
    }
    const H = run.ctx.c.hidden;
    for (rows, 0..) |row, r| {
        const v = try run.gpa.alloc(u16, 3 * H);
        @memcpy(v[0..H], sc.prefix.slice(u16, (r + 1) * H)[r * H ..]);
        @memcpy(v[H .. 2 * H], sc.xin.slice(u16, (r + 1) * H)[r * H ..]);
        @memcpy(v[2 * H ..], sc.delta.slice(u16, (r + 1) * H)[r * H ..]);
        const gop = try out.rows.getOrPut(run.gpa, row.key);
        if (gop.found_existing) return error.RowTwice;
        gop.value_ptr.* = v;
    }
}

/// The schedule through layer `run.layer` in composition `comp`, streams on slots base..base+2 (fresh).
pub fn all(run: Run, comp: Comp, base: u32, out: *Results) !void {
    for (0..3) |s| run.state.reset(base + @as(u32, @intCast(s)));
    for (schedule, 0..) |segs_in, ri| {
        var segs: std.ArrayList(k3.round.Segment) = .empty;
        defer segs.deinit(run.gpa);
        var rows: std.ArrayList(Row) = .empty;
        defer rows.deinit(run.gpa);
        for (segs_in) |g| {
            const slot = base + g.stream;
            if (comp == .serial) {
                for (0..(if (g.commit) g.rows else g.keep)) |j| {
                    const one = [_]k3.round.Segment{.{ .slot = slot, .rows = 1, .commit = true }};
                    const row = [_]Row{.{ .key = .{ .stream = g.stream, .round = @intCast(ri), .row = @intCast(j) }, .seg = 0 }};
                    try round(run, &one, &row, out);
                    run.state.advance(slot, 1, true, 0);
                }
                continue;
            }
            try segs.append(run.gpa, .{ .slot = slot, .rows = g.rows, .commit = g.commit });
            for (0..g.rows) |j| try rows.append(run.gpa, .{ .key = .{ .stream = g.stream, .round = @intCast(ri), .row = @intCast(j) }, .seg = @intCast(segs.items.len - 1) });
            if (comp == .split) {
                try round(run, segs.items, rows.items, out);
                run.state.advance(slot, g.rows, g.commit, g.keep);
                segs.clearRetainingCapacity();
                rows.clearRetainingCapacity();
            }
        }
        if (comp == .mixed) {
            try round(run, segs.items, rows.items, out);
            for (segs_in) |g| run.state.advance(base + g.stream, g.rows, g.commit, g.keep);
        }
    }
}

/// Rows of `b` against the same rows of `a`, bit for bit; the count of rows compared.
pub fn same(a: *const Results, b: *const Results, what: []const u8) !usize {
    var n: usize = 0;
    var it = b.rows.iterator();
    while (it.next()) |kv| {
        const other = a.rows.get(kv.key_ptr.*) orelse return error.MissingRow;
        if (!std.mem.eql(u16, other, kv.value_ptr.*)) {
            const k = kv.key_ptr.*;
            var first: usize = 0;
            while (other[first] == kv.value_ptr.*[first]) first += 1;
            std.debug.print("{s}: stream {d} round {d} row {d} differs (word {d}: 0x{x:0>4} vs 0x{x:0>4})\n", .{ what, k.stream, k.round, k.row, first, other[first], kv.value_ptr.*[first] });
            return error.RowsDiffer;
        }
        n += 1;
    }
    return n;
}
