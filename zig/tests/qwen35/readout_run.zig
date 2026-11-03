//! `readout <model dir> [--len N] [--cut N] [--streams K]`: gate 2a's rows through a fake readout that copies them.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");

const Engine = qwen35.engine.Engine;
const Hip = qwen35.hip_lanes.Hip;
const Readout = qwen35.hip_lanes.Readout;

/// One stream's copy of every final-norm row it consumed, and the pieces they came in as (first row, count).
const Copy = struct {
    rows: []u8,
    pieces: std.ArrayList([2]u32) = .empty,
    closed: bool = false,
};

/// The test-only readout: each piece's rows downloaded into the stream's copy, constant logits at the finish.
const Copier = struct {
    gpa: std.mem.Allocator,
    driver: *hip.Runtime,
    width: usize,
    len: usize,
    copies: std.ArrayList(*Copy) = .empty,

    const vtable: Readout.VTable = .{ .open = open, .consume = consume, .finish = finish, .close = close };

    fn readout(c: *Copier) Readout {
        return .{ .ctx = c, .info = .{ .max_rows = @intCast(c.len), .max_questions = 4, .max_options = 8 }, .vtable = &vtable };
    }

    fn of(ctx: *anyopaque) *Copier {
        return @ptrCast(@alignCast(ctx));
    }

    fn copyOf(st: *Readout.State) *Copy {
        return @ptrCast(@alignCast(st));
    }

    fn open(ctx: *anyopaque, _: *const lanes.readout.Spec) anyerror!*Readout.State {
        const c = of(ctx);
        const copy = try c.gpa.create(Copy);
        errdefer c.gpa.destroy(copy);
        copy.* = .{ .rows = try c.gpa.alloc(u8, c.len * c.width) };
        errdefer c.gpa.free(copy.rows);
        @memset(copy.rows, 0xff);
        try c.copies.append(c.gpa, copy);
        return @ptrCast(copy);
    }

    fn consume(ctx: *anyopaque, st: *Readout.State, rows: hip.ops.Tensor, first_row: usize, count: usize, stream: hip.Stream) anyerror!void {
        const c = of(ctx);
        const copy = copyOf(st);
        try stream.synchronize();
        try hip.runtime.check(c.driver.api.hipMemcpyDtoH(copy.rows[first_row * c.width ..].ptr, @ptrFromInt(rows.ptr), count * c.width));
        try copy.pieces.append(c.gpa, .{ @intCast(first_row), @intCast(count) });
    }

    fn finish(_: *anyopaque, _: *Readout.State, logits: []u16) anyerror!void {
        @memset(logits, 0x3f80);
    }

    fn close(_: *anyopaque, st: *Readout.State) void {
        copyOf(st).closed = true; // the run compares it after the lane is gone
    }

    fn deinit(c: *Copier) void {
        for (c.copies.items) |copy| {
            copy.pieces.deinit(c.gpa);
            c.gpa.free(copy.rows);
            c.gpa.destroy(copy);
        }
        c.copies.deinit(c.gpa);
    }
};

/// Two questions over the prompt's head; the fake readout reads none of it, the lane core's checks pass it.
fn specFor(logits: []u16) lanes.readout.Spec {
    return .{ .question_spans = &.{ .{ 0, 2 }, .{ 4, 5 } }, .question_types = &.{ 1, 0 }, .option_spans = &.{ .{ 2, 3 }, .{ 3, 4 }, .{ 5, 7 } }, .option_question = &.{ 0, 0, 1 }, .logits = logits };
}

/// `n` readouts of `ids` together: one piece of each in turn until every prompt is in, each finished and released.
fn together(gpa: std.mem.Allocator, h: *Hip, ids: []const u32, n: usize, abandon: bool) !void {
    const b = Hip.backend(h);
    var logits: [8][3]u16 = undefined;
    var specs: [8]lanes.readout.Spec = undefined;
    var streams: [8]lanes.Stream = undefined;
    var done: [8]bool = @splat(false);
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    // on any error every lane not yet released goes first, its readout state closed while the copier lives
    defer for (streams[0..made], done[0..made]) |*s, d| if (!d) b.vtable.release(b.ptr, s);
    for (0..n) |i| {
        specs[i] = specFor(&logits[i]);
        streams[i] = try lanes.Stream.init(gpa, .{ .id = "readout", .prompt = ids, .max_new = 0, .drafts = false, .readout = &specs[i] });
        made += 1;
    }
    var left = n;
    while (left > 0) {
        for (streams[0..n], done[0..n]) |*s, *d| {
            if (d.*) continue;
            if (abandon) { // one piece in, then released with no finish: a cancel between pieces
                _ = try b.vtable.prefill_step.?(b.ptr, s, 0);
                b.vtable.release(b.ptr, s);
                d.* = true;
                left -= 1;
                continue;
            }
            if (!try b.vtable.prefill_step.?(b.ptr, s, 0)) continue;
            try b.vtable.readout_finish.?(b.ptr, s, s.readout.?.logits);
            b.vtable.release(b.ptr, s);
            d.* = true;
            left -= 1;
        }
    }
}

/// Whether `got` holds the reference's rows and pieces exactly.
fn same(ref: *const Copy, got: *const Copy) bool {
    return got.closed and std.mem.eql(u8, ref.rows, got.rows) and std.mem.eql([2]u32, ref.pieces.items, got.pieces.items);
}

/// Whether `copy` consumed the first `count` pieces of a `len`-row prompt at `cut`, from row 0 with no gap.
fn covers(copy: *const Copy, len: usize, cut: usize, count: usize) bool {
    if (copy.pieces.items.len != count) return false;
    for (copy.pieces.items, 0..) |p, k| if (p[0] != k * cut or p[1] != @min(cut, len - k * cut)) return false;
    return true;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 1) return error.MissingArgument;
    var len: usize = 3000;
    var cut: usize = qwen35.prefix.chunk;
    var n: usize = 4;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        const value = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--len")) len = value else if (std.mem.eql(u8, args[i], "--cut")) cut = value else if (std.mem.eql(u8, args[i], "--streams")) n = @min(value, 8) else return error.UnknownOption;
    }
    if (i != args.len or cut == 0 or len == 0) return error.UnknownOption; // the gate's pieces need a fixed cut
    const ids = try ids_file.synthetic(gpa, len);
    defer gpa.free(ids);
    const policy = (try @import("group.zig").resolve("", 0)).policy;
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = len + 64, .streams = n, .batch_rows = 16 * n, .cut_rows = cut, .policy = policy });
    defer e.deinit();
    // the copier outlives the backend: `h.deinit` closes any lane's readout state it still holds
    var copier: Copier = .{ .gpa = gpa, .driver = &e.driver, .width = e.model().spec.hidden * e.model().act.size(), .len = len };
    defer copier.deinit();
    var cuts: std.ArrayList(u32) = .empty;
    defer cuts.deinit(gpa);
    const h = try Hip.init(gpa, e);
    defer h.deinit();
    h.cuts = &cuts;
    try h.useReadout(copier.readout());
    try together(gpa, h, ids, 1, false); // copy 0: the reference
    try together(gpa, h, ids, n, false); // copies 1 ..= n
    try together(gpa, h, ids, 1, true); // abandoned after its first piece
    try together(gpa, h, ids, 1, false); // the readout after it
    const copies = copier.copies.items;
    const ref = copies[0];
    std.debug.print("readout gate 2a: {d} rows x {d} bytes, cut {d}, pieces {any}\n", .{ len, copier.width, cut, ref.pieces.items });
    var failed = false;
    if (copies.len != n + 3) return error.ReadoutsNotOpened;
    const pieces = (len + cut - 1) / cut;
    if (!covers(ref, len, cut, pieces) or !covers(copies[n + 1], len, cut, 1)) {
        std.debug.print("FAIL the pieces consumed do not cover the prompt from row 0 at cut {d}\n", .{cut});
        failed = true;
    }
    for (copies[1 .. n + 1], 1..) |c, k| if (!same(ref, c)) {
        std.debug.print("FAIL readout {d} of {d} together differs from the one alone\n", .{ k, n });
        failed = true;
    };
    if (!same(ref, copies[n + 2])) {
        std.debug.print("FAIL the readout after a cancelled one differs from the one alone\n", .{});
        failed = true;
    }
    const hash = std.hash.Wyhash.hash(0, ref.rows);
    if (failed) return error.RowsDiffer;
    std.debug.print("PASS 1 vs {d} together and after a cancel: rows byte-identical (wyhash {x}), {d} cuts recorded\n", .{ n, hash, cuts.items.len });
}
