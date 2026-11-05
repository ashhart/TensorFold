//! Host tests for Disk.validate: the invariants nothing else maintains, and the checks that stay.
const std = @import("std");
const t = std.testing;
const codec = @import("session_codec.zig");
const sg = @import("sessions_gpu.zig");
const ced = @import("ced.zig");
const eh = @import("engram_host.zig");
const fwd = @import("forward.zig");
const config = @import("config.zig");
const cuda_kv = @import("cuda_kv.zig");
const Blob = @import("kv").sessions.disk.Blob;
const Measure = @import("kv").sess.Measure;

const vocab: u32 = 1000;
const capacity: u64 = 1 << 24;
const window: u64 = 128;
const role = "swa_values";

// One arena per image, so the leak checker only ever reports a real leak.
const Image = struct {
    arena: *std.heap.ArenaAllocator,
    slot: []u8,
    piece: []u8,
    blobs: []Blob,
    pieces: []sg.Piece,

    fn write(self: *Image, off: usize, v: u64) void {
        std.mem.writeInt(u64, self.slot[off..][0..8], v, .little);
    }

    fn writeId(self: *Image, off: usize, v: u32) void {
        std.mem.writeInt(u32, self.slot[off..][0..4], v, .little);
    }
};

// The stash's ids ride at ced_at + 16, indexed the way the restore reads them back.
fn idOff(i: usize) usize {
    return sg.Sessions.ced_at + 16 + 4 * i;
}

// Only the six fields `validate` reads are assigned; the rest is never touched, so no GPU is needed.
fn sessions(im: *Image, m: Measure, ds_rings: bool) !sg.Sessions {
    const a = im.arena.allocator();
    const cfg = try a.create(config.Config);
    cfg.* = undefined;
    cfg.vocab = vocab;
    const f = try a.create(fwd.Forward);
    f.* = undefined;
    f.cfg = cfg;
    const kx = try a.create(cuda_kv.Kv);
    kx.* = undefined;
    kx.opts.capacity = capacity;
    var s: sg.Sessions = undefined;
    s.f = f;
    s.kx = kx;
    s.pieces = im.pieces;
    s.measure = m;
    s.ds_rings = ds_rings;
    return s;
}

fn meas(replay: bool, ds_blocks: u64) Measure {
    return .{ .rings = 1, .window = window, .carries = 0, .head_dim = 1, .hidden = 1, .hc = 1, .ds_blocks = ds_blocks, .replay = replay };
}

fn image(pos: u64, tail_len: u64, from: u64, hi: u64, ds: u64) !Image {
    const ar = try t.allocator.create(std.heap.ArenaAllocator);
    errdefer t.allocator.destroy(ar);
    ar.* = std.heap.ArenaAllocator.init(t.allocator);
    const a = ar.allocator();
    var im = Image{
        .arena = ar,
        .slot = try a.alloc(u8, sg.Sessions.slot_bytes),
        .piece = try a.alloc(u8, 4),
        .blobs = try a.alloc(Blob, 2),
        .pieces = try a.alloc(sg.Piece, 1),
    };
    errdefer free(&im);
    im.pieces[0] = .{ .role = role, .len = 4 };
    @memset(im.slot, 0);
    @memset(im.piece, 0);
    im.write(0, pos);
    im.write(8, tail_len);
    for (0..eh.max_ngram) |i| im.writeId(16 + 4 * i, 7); // valid tail ids: only stash/ds can fail below
    im.write(sg.Sessions.ced_at, from);
    im.write(sg.Sessions.ced_at + 8, hi);
    im.write(sg.Sessions.ds_at, ds);
    im.blobs[0] = .{ .name = "slot", .bytes = im.slot };
    im.blobs[1] = .{ .name = role, .bytes = im.piece };
    return im;
}

fn free(im: *Image) void {
    im.arena.deinit();
    t.allocator.destroy(im.arena);
}

// A prompt ingested as one segment: from = 0, hi = the whole prompt, a span far past the 128-slot ids ring.
test "session_codec: a replay stash spanning far more than the ring is a valid image" {
    var im = try image(59776, 3, 0, 59776, 128);
    defer free(&im);
    const s = try sessions(&im, meas(true, 1), false);
    const m = try codec.validate(&s, im.blobs);
    try t.expectEqual(@as(u64, 59776), m.pos);
    try t.expectEqual(@as(u64, 0), m.stash.from);
    try t.expectEqual(@as(u64, 59776), m.stash.hi);
    try t.expectEqual(@as(u64, 128), m.ds);
}

// Sessions.init gives rings only under `!replay and ds_blocks > 0`; snapshot parks ds for any ds_blocks > 0.
test "session_codec: a parked drafter offset without drafter rings is a valid image" {
    var im = try image(9000, 3, 0, 9000, window);
    defer free(&im);
    const s = try sessions(&im, meas(true, 1), false);
    try t.expect(!s.ds_rings);
    const m = try codec.validate(&s, im.blobs);
    try t.expectEqual(window, m.ds);
}

// A false invariant removed must not take a real one with it: the last held position is still checked.
test "session_codec: an out-of-vocabulary id inside the kept window still fails" {
    var im = try image(59776, 3, 0, 59776, 128);
    defer free(&im);
    im.writeId(idOff((59776 - 1) % ced.ring), vocab); // the last held position
    const s = try sessions(&im, meas(true, 1), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

// Positions below lo() = @max(from, hi -| keep) are recycled rows the window no longer owns, so unvalidated.
test "session_codec: an id below the held window is not validated" {
    var im = try image(59776, 3, 0, 59776, 128);
    defer free(&im);
    const below: usize = (59776 - ced.keep - 1) % ced.ring; // one row before lo()
    im.writeId(idOff(below), 0xFFFF_FFFF);
    const s = try sessions(&im, meas(true, 1), false);
    const m = try codec.validate(&s, im.blobs);
    try t.expectEqual(@as(u64, 59776), m.stash.hi);
}

test "session_codec: a blob count that is not pieces + 1 fails" {
    var im = try image(9000, 3, 0, 9000, 0);
    defer free(&im);
    const s = try sessions(&im, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs[0..1]));
}

test "session_codec: a role the pieces do not declare fails" {
    var im = try image(9000, 3, 0, 9000, 0);
    defer free(&im);
    im.blobs[1].name = "not_a_role";
    const s = try sessions(&im, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

test "session_codec: a piece of the wrong length fails" {
    var im = try image(9000, 3, 0, 9000, 0);
    defer free(&im);
    im.piece = try im.arena.allocator().alloc(u8, 3); // the piece declares 4
    @memset(im.piece, 0);
    im.blobs[1].bytes = im.piece;
    const s = try sessions(&im, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

test "session_codec: a slot of the wrong size fails" {
    var im = try image(9000, 3, 0, 9000, 0);
    defer free(&im);
    const short = try im.arena.allocator().alloc(u8, sg.Sessions.slot_bytes - 1);
    @memcpy(short, im.slot[0..short.len]);
    im.blobs[0].bytes = short;
    const s = try sessions(&im, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

test "session_codec: a position past the pool's capacity fails" {
    var im = try image(capacity + 1, 3, 0, capacity + 1, 0);
    defer free(&im);
    const s = try sessions(&im, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

test "session_codec: from ahead of hi, and hi ahead of pos, both fail" {
    var a = try image(9000, 3, 9001, 9000, 0);
    defer free(&a);
    var b = try image(9000, 3, 0, 9001, 0);
    defer free(&b);
    const s = try sessions(&a, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, a.blobs));
    try t.expectError(error.BadBlob, codec.validate(&s, b.blobs));
}

test "session_codec: a tail id past the vocabulary fails" {
    var im = try image(9000, 3, 0, 9000, 0);
    defer free(&im);
    im.writeId(16, vocab); // tail position 0, inside tail_len
    const s = try sessions(&im, meas(true, 0), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

test "session_codec: a drafter offset past the window fails" {
    var im = try image(9000, 3, 0, 9000, window + 1);
    defer free(&im);
    const s = try sessions(&im, meas(true, 1), false);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}

// The surviving replay guard: dropping the span bound must not have swept this one away with it.
test "session_codec: a stash outside replay still fails" {
    var im = try image(9000, 3, 0, 9000, 0);
    defer free(&im);
    const s = try sessions(&im, meas(false, 0), true);
    try t.expectError(error.BadBlob, codec.validate(&s, im.blobs));
}
