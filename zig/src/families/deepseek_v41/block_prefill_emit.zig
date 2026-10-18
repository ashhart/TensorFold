//! Prefill programs as calls: single and multi-segment runs, CED replay passes, TBO micro-batches.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const config = @import("config.zig");
const ced = @import("ced.zig");
const prefill = @import("block_prefill.zig");
const Config = config.Config;
const E = prefill.E;
const Arg = calls.Arg;
const split_rows = prefill.split_rows;
const router_chunk = prefill.router_chunk;
const Seg = prefill.Seg;
const Span = prefill.Span;
const Part = prefill.Part;

/// The most rows a prefill program takes: one router CHUNK, 4,096 under TF_DSV41_PF_4K.
fn rowsCap(o: block.Options) i64 {
    return if (o.pf4k) 2 * router_chunk else router_chunk;
}

/// Several slots' prompt segments or CED decoder replays in one run; each segment's CSA2 steps on its own slot.
pub fn emitMulti(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, spans_in: []const Span, part: Part) ![]calls.Call {
    if (spans_in.len == 0 or o.image_rows != 0) return error.Unsupported;
    if (o.pool) |pl| if (pl.split) return error.Unsupported; // split KV's exchanges: one segment a run
    const spans = try a.dupe(Span, spans_in);
    var total: i64 = 0;
    // decoder replay rows past the split mHC programs' 32; a prompt piece's past the short segment's 16
    const floor: i64 = if (part == .decoder) split_rows else 16;
    var cnt: i64 = 0;
    for (spans) |*sp| {
        if (sp.n <= floor) return error.Unsupported;
        sp.row0 = total;
        sp.cnt0 = cnt;
        total += sp.n;
        cnt = std.mem.alignForward(i64, cnt + sp.n, 4);
    }
    if (total > o.prefill_rows or total > rowsCap(o)) return error.Unsupported;
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = total, .start = spans[0].start };
    var s: Seg = .{ .e = &e, .staged = false, .ring = 0, .part = part, .multi = spans };
    s.setRing();
    try s.run(layers, false);
    return e.out.items;
}

/// The multi-segment runs' programs at their largest, for the buffer plan; a run its options refuse adds nothing.
pub fn planRuns(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, enc: []const u32, dec: ?[]const u32, rows: i64, part: Part) !std.ArrayList([]const calls.Call) {
    var out: std.ArrayList([]const calls.Call) = .empty;
    if (rows > 2 * 17) {
        const sp = [_]Span{ .{ .slot = 0, .start = 0, .n = rows - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
        if (emitMulti(a, cfg, w, o, enc, &sp, part)) |cs| try out.append(a, cs) else |e| if (e != error.Unsupported) return e;
    }
    // the counts' widest layout: 15 segments of 17 rows (each padded 3 to the next multiple of 4) and the rest
    if (rows > 16 * 17) {
        var sp: [16]Span = undefined;
        for (sp[0..15], 0..) |*x, i| x.* = .{ .slot = @intCast(i), .start = 0, .n = 17 };
        sp[15] = .{ .slot = 15, .start = 0, .n = rows - 15 * 17 };
        if (emitMulti(a, cfg, w, o, enc, &sp, part)) |cs| try out.append(a, cs) else |e| if (e != error.Unsupported) return e;
    }
    if (dec) |d| {
        var tails: [16]Span = undefined;
        const k: usize = @intCast(@min(16, @divFloor(rows, ced.keep)));
        if (k >= 2) {
            for (tails[0..k], 0..) |*t, i| t.* = .{ .slot = @intCast(i), .start = 0, .n = ced.keep };
            if (emitMulti(a, cfg, w, o, d, tails[0..k], .decoder)) |cs| try out.append(a, cs) else |e| if (e != error.Unsupported) return e;
        }
    }
    return out;
}

/// CED replay's programs (TF_DSV41_PREFILL=replay), `n` rows from `start`: `.encoder` or `.decoder` (see ced.zig).
pub fn emitReplay(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, part: Part) ![]calls.Call {
    if (layers.len == 0) return error.Unsupported;
    return emitPart(a, cfg, w, o, layers, n, start, part == .whole, part);
}

/// The calls of a prefill segment of `n` rows from `start` over `layers`, plus final norm + head when `head`.
pub fn emitPrefill(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool) ![]calls.Call {
    return emitPart(a, cfg, w, o, layers, n, start, head, .whole);
}

fn emitPart(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool, part: Part) ![]calls.Call {
    return emitMarked(a, cfg, w, o, layers, n, start, head, part, null);
}

pub const tbo_min: i64 = 512; // TF_DSV41_PF_TBO: micro-batches of at least this many rows (smaller run alone)

/// Persistent roles used as scratch: a program in its own namespace (`ownRole`) takes its own copy of each.
pub const own_scratch = [_][]const u8{ "s.ex.", "s.pf.xh", "s.pf.w", "s.router.", "s.stage.", "s.attn." };

/// A role in namespace `tag`: "w.x" -> "w.<tag>x", "L.x" -> "L.<tag>x", scratch "s.x" -> "s.<tag>x"; state keeps its.
pub fn ownRole(a: std.mem.Allocator, r: []const u8, comptime tag: []const u8) ![]const u8 {
    if (std.mem.startsWith(u8, r, "s.")) {
        // a role already in a namespace ("s.b~ex.z") is scratch by its bare name
        var bare = r[2..];
        while (std.mem.indexOf(u8, bare, "~")) |i| {
            if (std.mem.indexOfScalar(u8, bare[0..i], '.') != null) break;
            bare = bare[i + 1 ..];
        }
        for (own_scratch) |p| if (std.mem.startsWith(u8, bare, p[2..])) return std.fmt.allocPrint(a, "s." ++ tag ++ "{s}", .{r[2..]});
        return r;
    }
    const dot = std.mem.indexOfScalar(u8, r, '.') orelse return std.fmt.allocPrint(a, tag ++ "{s}", .{r});
    return std.fmt.allocPrint(a, "{s}" ++ tag ++ "{s}", .{ r[0 .. dot + 1], r[dot + 1 ..] });
}

/// Micro-batch B's name of a role (TF_DSV41_PF_TBO).
pub fn tboRole(a: std.mem.Allocator, r: []const u8) ![]const u8 {
    return ownRole(a, r, "b~");
}

/// The roles a namespace leaves as they are (ownCallsKeep): the forward's, which hold them already.
pub const Keep = std.StringHashMapUnmanaged(void);

fn ownArg(a: std.mem.Allocator, x: Arg, comptime tag: []const u8, keep: ?*const Keep) !Arg {
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            if (t.role != .buf) break :blk x;
            if (keep) |k| if (k.contains(t.role.buf)) break :blk x;
            var u = t;
            u.role = .{ .buf = try ownRole(a, t.role.buf, tag) };
            break :blk if (x == .t) .{ .t = u } else .{ .opaque_table = u };
        },
        .list => |l| blk: {
            const out = try a.alloc(Arg, l.len);
            for (l, out) |y, *z| z.* = try ownArg(a, y, tag, keep);
            break :blk .{ .list = out };
        },
        else => x,
    };
}

fn tboArg(a: std.mem.Allocator, x: Arg) !Arg {
    return ownArg(a, x, "b~", null);
}

/// `cs` with every role in namespace `tag` (ownRole), the calls otherwise as they are.
pub fn ownCalls(a: std.mem.Allocator, cs: []const calls.Call, comptime tag: []const u8) ![]calls.Call {
    return ownCallsKeep(a, cs, tag, null);
}

/// `ownCalls` but for the roles in `keep`, which stay the forward's.
pub fn ownCallsKeep(a: std.mem.Allocator, cs: []const calls.Call, comptime tag: []const u8, keep: ?*const Keep) ![]calls.Call {
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *x| {
        x.* = c;
        const args = try a.alloc(calls.Named, c.args.len);
        for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try ownArg(a, y.arg, tag, keep) };
        x.args = args;
    }
    return out;
}

/// TF_DSV41_PF_TBO: two consecutive CED encoder segments as one program on two streams, B half a layer behind A.
pub fn emitTbo(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, na: i64, nb: i64, start: i64) ![]calls.Call {
    if (!o.pf_tbo or !o.branches or na < tbo_min or nb < tbo_min or o.image_rows != 0) return error.Unsupported;
    // B's first row starts a ratio-2 pair: its compressed rows begin after A's last
    if (@mod(start, 2) != 0 or @mod(na, 2) != 0) return error.Unsupported;
    if (o.pool) |pl| if (pl.split) return error.Unsupported;
    var ma: std.ArrayList(usize) = .empty;
    var mb: std.ArrayList(usize) = .empty;
    const pa = try emitMarked(a, cfg, w, o, layers, na, start, false, .encoder, &ma);
    const pb = try emitMarked(a, cfg, w, o, layers, nb, start + na, false, .encoder, &mb);
    if (ma.items.len != mb.items.len) return error.Unsupported;
    // the phases: alternating attention / MoE parts, the last the encoder's end (D's site + compressor)
    const nph = ma.items.len + 1;
    var out: std.ArrayList(calls.Call) = .empty;
    var started = false; // B issued any call since the last join (its next part forks)
    var b_engram = false; // B issued an Engram step since the last join (A's next Engram step joins it first)
    for (0..nph) |k| {
        const a0 = if (k == 0) 0 else ma.items[k - 1];
        const a1 = if (k < ma.items.len) ma.items[k] else pa.len;
        const b0 = if (k == 0) 0 else mb.items[k - 1];
        const b1 = if (k < mb.items.len) mb.items[k] else pb.len;
        const attn = k % 2 == 0;
        const L = layers[k / 2];
        // A's part on the main stream
        const first_a = out.items.len;
        for (pa[a0..a1]) |c| {
            if (std.mem.eql(u8, c.name, "glue.ced_stash")) continue;
            try out.append(a, c);
        }
        if (b_engram and attn and out.items.len > first_a) {
            for (pa[a0..a1]) |c| if (std.mem.eql(u8, c.name, "glue.engram_rows")) {
                out.items[first_a].join = true;
                started = false;
                b_engram = false;
                break;
            };
        }
        if (attn and cfg.isKvSource(L) and cfg.compressRatio(L) == 2) {
            // the slot's ratio-2 carry: A's last projection row, as A's commit (forward_prefill.commitAt) would copy it
            const row: i64 = 2 * @as(i64, cfg.head_dim);
            const src: Arg = .{ .t = .{ .role = .{ .buf = try std.fmt.allocPrint(a, "w.L{d}.proj", .{L}) }, .dt = .f32, .shape = try a.dupe(i64, &.{ 1, row }), .stride = try a.dupe(i64, &.{ row, 1 }), .offset = (na - 1) * row * 4 } };
            const dst: Arg = .{ .t = .{ .role = .{ .buf = try std.fmt.allocPrint(a, "s.L{d}.carry", .{L}) }, .dt = .f32, .shape = try a.dupe(i64, &.{ 1, row }), .stride = try a.dupe(i64, &.{ row, 1 }) } };
            const args = try a.alloc(calls.Named, 2);
            args[0] = .{ .arg = src };
            args[1] = .{ .arg = dst };
            try out.append(a, .{ .triton = false, .glue = true, .name = "glue.copy_rows", .args = args });
        }
        // B's part on the side stream, its attention parts forked behind A's
        for (pb[b0..b1], 0..) |c, j| {
            var x = c;
            const args = try a.alloc(calls.Named, c.args.len);
            for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try tboArg(a, y.arg) };
            x.args = args;
            x.side = true;
            x.fork = j == 0 and (attn or !started);
            x.join = false;
            x.begin = .none;
            try out.append(a, x);
            started = true;
            if (std.mem.eql(u8, c.name, "glue.engram_rows")) b_engram = true;
        }
    }
    try out.append(a, .{ .triton = false, .glue = true, .name = "glue.tbo_join", .args = &.{}, .join = true });
    return out.items;
}

fn emitMarked(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool, part: Part, marks: ?*std.ArrayList(usize)) ![]calls.Call {
    if (n < 1 or n > o.prefill_rows or n > rowsCap(o)) return error.Unsupported;
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = start, .marks = marks };
    const W: i64 = cfg.window;
    // forward.staging: the slot's ring holds 2 x window rows; a longer segment goes through the staging ring
    const staged = n + W - 1 > 2 * W;
    const stage: i64 = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(o.prefill_rows + W - 1)) catch return error.Unsupported);
    const ring = if (staged) stage else 2 * W;
    if (ring < n + W - 1) return error.Unsupported; // attn_pf: the ring must hold the rows and their windows
    var s: Seg = .{ .e = &e, .staged = staged, .ring = ring, .short = n <= 16, .part = part };
    try s.run(layers, head);
    if (o.holdsDeferred(n)) calls.holdDeferred(e.out.items);
    return e.out.items;
}
