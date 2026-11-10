//! `tf-qwen35-cuda <model> readout <ids file> [streams]`: gate 2a's rows through the CUDA lanes with a test readout.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const q = @import("qwen35");

const Readout = q.native.Readout;

/// One stream's copy of every final-norm row it consumed, the pieces they came in as, and its closes.
const Copy = struct {
    rows: []u8,
    pieces: std.ArrayList([2]u32) = .empty,
    closes: usize = 0,
};

/// The test-only readout: each piece's rows downloaded into the stream's copy; `fail_at` fails that piece's consume.
const Copier = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    width: usize,
    len: usize,
    fail_at: ?usize = null,
    failed: usize = 0, // planted failures that fired
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

    fn open(ctx: *anyopaque, _: *const lanes.readout.Spec, _: []const u32) anyerror!*Readout.State {
        const c = of(ctx);
        const copy = try c.gpa.create(Copy);
        errdefer c.gpa.destroy(copy);
        copy.* = .{ .rows = try c.gpa.alloc(u8, c.len * c.width) };
        errdefer c.gpa.free(copy.rows);
        @memset(copy.rows, 0xff);
        try c.copies.append(c.gpa, copy);
        return @ptrCast(copy);
    }

    fn consume(ctx: *anyopaque, st: *Readout.State, rows: u64, first_row: usize, count: usize, stream: cuda.Stream) anyerror!void {
        const c = of(ctx);
        const copy = copyOf(st);
        if (c.fail_at) |k| if (copy.pieces.items.len == k) {
            c.failed += 1;
            return error.PlantedConsumeFailure;
        };
        try stream.synchronize();
        try c.d.check(c.d.api.cuMemcpyDtoH_v2(copy.rows[first_row * c.width ..].ptr, rows, count * c.width), "cuMemcpyDtoH");
        try copy.pieces.append(c.gpa, .{ @intCast(first_row), @intCast(count) });
    }

    fn finish(_: *anyopaque, _: *Readout.State, logits: []u16) anyerror!void {
        @memset(logits, 0x3f80);
    }

    fn close(_: *anyopaque, st: *Readout.State) void {
        copyOf(st).closes += 1; // compared after the lane is gone
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

/// Two questions over the prompt's head; the test readout reads none of it.
fn specFor(logits: []u16) lanes.readout.Spec {
    return .{ .question_spans = &.{ .{ 0, 2 }, .{ 4, 5 } }, .question_types = &.{ 1, 0 }, .option_spans = &.{ .{ 2, 3 }, .{ 3, 4 }, .{ 5, 7 } }, .option_question = &.{ 0, 0, 1 }, .logits = logits };
}

const Mode = enum { finish, abandon, fail };

/// `n` readouts together a piece each in turn (finished, abandoned or failing); `text` fills between pieces: true.
fn together(gpa: std.mem.Allocator, b: lanes.backend.Backend, ids: []const u32, n: usize, mode: Mode, text: ?*lanes.Stream) !bool {
    var between = false;
    var logits: [8][3]u16 = undefined;
    var specs: [8]lanes.readout.Spec = undefined;
    var streams: [8]lanes.Stream = undefined;
    var done: [8]bool = @splat(false);
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    defer for (streams[0..made], done[0..made]) |*s, d| if (!d) b.vtable.release(b.ptr, s);
    for (0..n) |i| {
        specs[i] = specFor(&logits[i]);
        streams[i] = try lanes.Stream.init(gpa, .{ .id = "readout", .prompt = ids, .max_new = 0, .drafts = false, .readout = &specs[i] });
        made += 1;
    }
    var left = n;
    var round: usize = 0;
    while (left > 0) : (round += 1) for (streams[0..n], done[0..n], 0..) |*s, *d, i| {
        if (text) |t| if (round == 1 and i == 0 and left == n) {
            while (!try b.vtable.prefill_step.?(b.ptr, t, 0)) {}
            between = true;
        };
        if (d.*) continue;
        const filled = b.vtable.prefill_step.?(b.ptr, s, 0) catch |err| {
            if (mode != .fail or err != error.PlantedConsumeFailure) return err;
            b.vtable.release(b.ptr, s); // as the lane host drops a failed fill
            d.* = true;
            left -= 1;
            continue;
        };
        if (mode == .abandon or filled) {
            if (mode == .finish) try b.vtable.readout_finish.?(b.ptr, s, s.readout.?.logits);
            b.vtable.release(b.ptr, s);
            d.* = true;
            left -= 1;
        }
    };
    return between;
}

fn same(ref: *const Copy, got: *const Copy) bool {
    return got.closes == 1 and std.mem.eql(u8, ref.rows, got.rows) and std.mem.eql([2]u32, ref.pieces.items, got.pieces.items);
}

/// Whether `copy` consumed `count` pieces of `piece` rows of a `len`-row prompt from row 0 with no gap.
fn covers(copy: *const Copy, len: usize, piece: usize, count: usize) bool {
    if (copy.pieces.items.len != count) return false;
    for (copy.pieces.items, 0..) |p, k| if (p[0] != k * piece or p[1] != @min(piece, len - k * piece)) return false;
    return true;
}

pub fn run(init: std.process.Init, a: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len != 4 and args.len != 5) return 2;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(64 << 20));
    var list: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try list.append(a, try std.fmt.parseInt(u32, t, 10));
    const ids = list.items;
    const n = if (args.len == 5) @min(try std.fmt.parseInt(usize, args[4], 10), 8) else 4;
    const gpa = init.gpa;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const loaded = try q.native.openLanes(gpa, init.io, &ctx, args[1], null, .{ .context = ids.len + 64, .drafts = false });
    // the copier outlives the lanes: their deinit closes any readout state still open
    var copier: Copier = .{ .gpa = gpa, .d = &driver, .width = 0, .len = ids.len };
    defer copier.deinit();
    defer loaded.deinit(loaded.ctx);
    const l: *q.native.Lanes = @ptrCast(@alignCast(loaded.ctx));
    copier.width = l.e.g.hidden * 2;
    const b = l.useReadout(copier.readout());
    const piece: usize = q.native.readout_piece_rows;
    if (ids.len <= piece) return error.PromptFitsOnePiece; // the cancel and the failure need a second piece
    _ = try together(gpa, b, ids, 1, .finish, null); // copy 0: the reference
    // copies 1 ..= n together, beside a text stream that prefills between their pieces
    var talker = try lanes.Stream.init(gpa, .{ .id = "talker", .prompt = ids[0..@min(64, ids.len)], .max_new = 4 });
    defer talker.deinit(gpa);
    const between = try together(gpa, b, ids, n, .finish, &talker);
    b.vtable.release(b.ptr, &talker);
    _ = try together(gpa, b, ids, 1, .abandon, null); // released after its first piece: a cancel between pieces
    copier.fail_at = 1;
    _ = try together(gpa, b, ids, 1, .fail, null); // its second consume fails
    copier.fail_at = null;
    _ = try together(gpa, b, ids, 1, .finish, null); // the readout after both
    const copies = copier.copies.items;
    if (copies.len != n + 4) return error.ReadoutsNotOpened;
    const ref = copies[0];
    var failed = false;
    const pieces = (ids.len + piece - 1) / piece;
    if (!covers(ref, ids.len, piece, pieces) or !covers(copies[n + 1], ids.len, piece, 1) or !covers(copies[n + 2], ids.len, piece, 1)) {
        std.debug.print("FAIL the pieces consumed do not cover the prompt from row 0 at piece {d}\n", .{piece});
        failed = true;
    }
    for (copies[1 .. n + 1], 1..) |c, k| if (!same(ref, c)) {
        std.debug.print("FAIL readout {d} of {d} together beside a text stream differs from the one alone\n", .{ k, n });
        failed = true;
    };
    if (!between or copier.failed != 1) {
        std.debug.print("FAIL the text stream stepped between pieces: {}; planted failures fired: {d}\n", .{ between, copier.failed });
        failed = true;
    }
    if (copies[n + 1].closes != 1 or copies[n + 2].closes != 1) {
        std.debug.print("FAIL a cancelled or failed readout's state closed {d} and {d} times\n", .{ copies[n + 1].closes, copies[n + 2].closes });
        failed = true;
    }
    if (!same(ref, copies[n + 3])) {
        std.debug.print("FAIL the readout after a cancelled and a failed one differs from the one alone\n", .{});
        failed = true;
    }
    if (failed) return 1;
    std.debug.print("PASS 1 vs {d} together with a text prompt between their pieces, after a cancel and a failed consume: {d} rows in {d} pieces of {d}, byte-identical (wyhash {x}); each state closed once\n", .{ n, ids.len, pieces, piece, std.hash.Wyhash.hash(0, ref.rows) });
    return 0;
}
