//! Row mode's program: a decode window's CSA2 steps rewritten to read positions and slots from the row table.

const std = @import("std");
const calls = @import("block_calls.zig");
const rowtab = @import("rowtab.zig");
const Arg = calls.Arg;
const Tensor = calls.Tensor;

pub const Error = error{ UnknownRowsCall, NoPool, UnionInRows, BadCall };

pub const table_role = "s.rows.tab";

pub const Spec = struct {
    /// stacked slots
    slots: u32,
    /// the row table's columns hold this many rows
    rmax: u32,
    /// pages a slot's table holds (the stacked tables' row stride)
    pts: i64,
};

/// A row block of the program: rows [a, a + n).
const Block = struct { a: i64, n: i64 };

pub const Rewriter = struct {
    a: std.mem.Allocator,
    s: Spec,
    /// rows of the program (the bucket)
    R: i64,
    /// backend._blocked's row blocks by their position role ("L.ix.pos<j>", from the glue "block_pos")
    blocks: std.StringHashMapUnmanaged(Block) = .empty,

    fn col(w: *const Rewriter, c: rowtab.Col) Arg {
        return w.colOf(c, .{ .a = 0, .n = w.R });
    }

    /// Rows [b.a, b.a + b.n) of column `c`.
    fn colOf(w: *const Rewriter, c: rowtab.Col, b: Block) Arg {
        return .{ .t = .{ .role = .{ .buf = table_role }, .dt = .i64, .shape = w.dup(&.{b.n}), .stride = w.dup(&.{1}), .offset = @intCast(rowtab.colOffset(c, w.s.rmax) + 8 * @as(usize, @intCast(b.a))) } };
    }

    /// The rows a call's POS names: a row block's, else the program's.
    fn rowsOf(w: *const Rewriter, args: []const calls.Named) Block {
        const p = find(args, "POS") orelse return .{ .a = 0, .n = w.R };
        const role = switch (args[p].arg) {
            .t => |t| switch (t.role) {
                .buf => |b| b,
                else => null,
            },
            else => null,
        };
        return if (role) |r| (w.blocks.get(r) orelse .{ .a = 0, .n = w.R }) else .{ .a = 0, .n = w.R };
    }

    fn dup(w: *const Rewriter, xs: []const i64) []const i64 {
        return w.a.dupe(i64, xs) catch @panic("rowmode: out of memory");
    }

    /// A stacked role's tensor as the row program sees it: S x the rows (rings), [S, cols] (carries, page tables).
    fn stacked(w: *const Rewriter, t: Tensor) !Tensor {
        const b = switch (t.role) {
            .buf => |b| b,
            else => return t,
        };
        const S: i64 = w.s.slots;
        var x = t;
        if (isRing(b) or isCarry(b)) {
            if (t.offset != 0 or t.shape.len != 2) return error.BadCall;
            if (isRing(b)) {
                x.shape = w.dup(&.{ S * t.shape[0], t.shape[1] });
                x.stride = t.stride;
            } else {
                x.shape = w.dup(&.{ S, t.shape[1] });
                x.stride = w.dup(&.{ t.shape[1], 1 });
            }
            return x;
        }
        if (isTable(b)) {
            if (t.offset != 0 or t.shape.len != 1 or t.shape[0] != w.s.pts) return error.BadCall;
            x.shape = w.dup(&.{ S, w.s.pts });
            x.stride = w.dup(&.{ w.s.pts, 1 });
            return x;
        }
        return t;
    }

    fn arg(w: *const Rewriter, x: Arg) !Arg {
        return switch (x) {
            .t => |t| .{ .t = try w.stacked(t) },
            else => x,
        };
    }

    /// One call in row mode (a fresh Call; the input is not changed).
    pub fn call(w: *Rewriter, c: calls.Call) !calls.Call {
        var out = c;
        const args = try w.a.alloc(calls.Named, c.args.len);
        for (c.args, args) |x, *y| y.* = .{ .name = x.name, .arg = try w.arg(x.arg) };
        out.args = args;
        if (c.glue) {
            if (eq(c.name, "glue.kx_union")) return error.UnionInRows;
            if (eq(c.name, "glue.block_pos")) {
                // [position role, start + a, a, rows]: the block's rows of the table from here on
                if (args.len != 4 or args[0].arg != .t or args[0].arg.t.role != .buf) return error.BadCall;
                const b: Block = .{ .a = args[2].arg.i, .n = args[3].arg.i };
                if (b.a < 0 or b.n < 1 or b.a + b.n > w.R) return error.BadCall;
                try w.blocks.put(w.a, args[0].arg.t.role.buf, b);
            }
            if (eq(c.name, "glue.counts")) {
                // [sel, counts, positions int64 [R], ratio]: each row's own position
                if (args.len != 4) return error.BadCall;
                args[2].arg = w.col(.pos);
            }
            return out;
        }
        if (c.triton) return w.triton(out);
        if (eq(c.name, "tf_dsv41_kv_glue_v1.norm_store")) {
            if (args.len != 15) return error.BadCall;
            args[5].arg = w.col(.pos);
            args[6].arg = w.col(.wslot);
            args[14].arg = .{ .b = true };
            return out;
        }
        if (eq(c.name, "tf_dsv41_attn_cuda_v1.attn")) {
            if (args.len < 23) return error.BadCall;
            args[9].arg = w.col(.pos);
            args[10].arg = w.col(.rslot);
            const pt_on = isTensor(args[11].arg);
            args[21].arg = .{ .i = if (pt_on) w.s.pts else 0 };
            return out;
        }
        if (eq(c.name, "tf_dsv41_attn_cuda_v1.topk") or eq(c.name, "tf_dsv41_topk_b_v1.topk")) {
            if (args.len < 22) return error.BadCall;
            args[17].arg = w.col(.pos);
            return out;
        }
        return out;
    }

    fn triton(w: *const Rewriter, c: calls.Call) !calls.Call {
        const args = @constCast(c.args);
        const writes = eq(c.name, "_kv_store") or eq(c.name, "_pool_norm") or eq(c.name, "_index_k");
        const reads = eq(c.name, "_scores") or eq(c.name, "_scores_b");
        if (eq(c.name, "_rope")) {
            const p = find(args, "POS") orelse return error.BadCall;
            args[p].arg = w.col(.pos);
            return c;
        }
        const b = w.rowsOf(args);
        if (eq(c.name, "_dtopk") or eq(c.name, "_dtopk_b") or eq(c.name, "_block_keys")) {
            // dtopk select / blocks and candidate blocks with rows on: q = POS[r] (the scores are the row's already)
            const p = find(args, "POS") orelse return error.BadCall;
            const r = find(args, "ROWS") orelse return error.BadCall;
            if (find(args, "SL") != null) return error.UnknownRowsCall;
            args[p].arg = w.colOf(.pos, b);
            args[r].arg = .{ .b = true };
            return c;
        }
        if (!writes and !reads) {
            // a call with row knobs that is not one of the above: refused rather than run with one slot's view
            for (args) |x| if (eq(x.name, "ROWS") or eq(x.name, "SL")) return error.UnknownRowsCall;
            return c;
        }
        for (args) |*x| {
            if (eq(x.name, "POS")) x.arg = w.colOf(.pos, b);
            if (eq(x.name, "SL")) x.arg = w.colOf(if (writes) .wslot else .rslot, b);
            if (eq(x.name, "ROWS")) x.arg = .{ .b = true };
        }
        if (find(args, "PTS")) |i| {
            const pt = find(args, "PT") orelse return error.BadCall;
            args[i].arg = .{ .i = if (isTensor(args[pt].arg)) w.s.pts else 0 };
        }
        if (eq(c.name, "_pool_norm")) {
            const split = if (find(args, "SPLIT")) |i| (args[i].arg == .b and args[i].arg.b) else false;
            const prev = find(args, "PREV") orelse return error.BadCall;
            args[prev].arg = w.col(.prev);
            const off = find(args, "OFF") orelse return error.BadCall;
            args[off].arg = .{ .i = if (split) @as(i64, w.s.slots) else 0 };
        }
        return c;
    }
};

/// The row-mode program of `cs` (block.emit's window of R rows); `cs` is not changed.
pub fn transform(a: std.mem.Allocator, cs: []const calls.Call, R: u32, s: Spec) ![]const calls.Call {
    if (s.pts <= 0) return error.NoPool;
    var w: Rewriter = .{ .a = a, .s = s, .R = R };
    defer w.blocks.deinit(a);
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *o| o.* = try w.call(c);
    return out;
}

fn eq(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn find(args: []const calls.Named, name: []const u8) ?usize {
    for (args, 0..) |x, i| if (eq(x.name, name)) return i;
    return null;
}

fn isTensor(x: Arg) bool {
    return switch (x) {
        .t => |t| t.role != .empty,
        else => false,
    };
}

/// "s.L<i>.swa.v" / ".s": a layer's SWA ring
pub fn isRing(b: []const u8) bool {
    return std.mem.startsWith(u8, b, "s.L") and (std.mem.endsWith(u8, b, ".swa.v") or std.mem.endsWith(u8, b, ".swa.s"));
}

/// "s.L<i>.carry": a ratio-2 kv source's carry
pub fn isCarry(b: []const u8) bool {
    return std.mem.startsWith(u8, b, "s.L") and std.mem.endsWith(u8, b, ".carry");
}

/// "s.kv.pt" / "s.kv.ct": a slot's page table / split table
pub fn isTable(b: []const u8) bool {
    return eq(b, "s.kv.pt") or eq(b, "s.kv.ct");
}

test "rowmode: stacked roles by name" {
    try std.testing.expect(isRing("s.L7.swa.v") and isRing("s.L7.swa.s") and !isRing("s.L7.comp.v"));
    try std.testing.expect(isCarry("s.L2.carry") and !isCarry("w.L2.comp"));
    try std.testing.expect(isTable("s.kv.pt") and isTable("s.kv.ct") and !isTable("s.kv.comp.L2"));
}
