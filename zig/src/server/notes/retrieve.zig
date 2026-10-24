//! The lookup on a store's current index: the notes a question finds, rendered, and fit to the cap by the tokenizer.
const std = @import("std");
const model_text = @import("../model_text.zig");
const clock = @import("../clock.zig");
const index = @import("index.zig");
const lookup = @import("lookup.zig");
const Storage = @import("storage.zig").Storage;
const Allocator = std.mem.Allocator;

/// How one lookup is made: the hop (null: the snapshot's).
pub const Way = struct { hop: ?index.Hop = null };

pub const Failure = error{ OutOfMemory, Tokenizer };

/// What retrieve() found.
pub const Found = struct {
    ids: []const u32,
    block: []const u8, // "" when nothing is retrieved: then no system message goes in
    tokens: usize, // the block's tokens by the model's tokenizer (fit()'s count)
    pack_tokens: u32, // pack()'s sum of the parts
    dropped: usize, // notes fit() took out
    candidates: u32,
    long: u32 = 0, // sa with longer matches on: the n-grams over n_max weighted
    lookup_ns: i96,
    fit_ns: i96,
    in_store: usize = 0, // notes in the index it read
};

/// The store's kept scratch, made again when the index has more notes than it has room for.
fn scratchFor(s: *Storage, idx: *const index.Index) Allocator.Error!*lookup.Scratch {
    var room: usize = 0; // a scratch for fewer notes comes back with room for more: writes add them one by one
    if (s.scratch) |*k| {
        if (k.slot.len >= idx.notes() and (idx.kind != .sa or k.seen.len >= idx.notes())) return k;
        room = @max(1024, idx.notes() / 8);
        k.deinit();
        s.scratch = null;
    }
    s.scratch = try lookup.Scratch.initRoom(s.zeroing.allocator(), idx, room);
    return &s.scratch.?;
}

/// NoteStore.retrieve(owner, question, cap, past, exclude) from the question's ids; the caller holds the store's lock.
pub fn retrieve(s: *Storage, io: std.Io, a: Allocator, text: model_text.Text, qids: []const u32, past: bool, cap: u32, exclude: []const u32, way: Way) Failure!Found {
    const current = s.idx.?;
    var view: index.Index = undefined; // a shallow copy: the same notes and maps, read only
    const idx = if (way.hop) |h| blk: {
        view = current.*;
        view.settings.hop = h;
        break :blk &view;
    } else current;
    const sc = try scratchFor(s, idx);
    const t0 = clock.nowNs(io);
    const packed_notes = try lookup.lookup(idx, sc, qids, past, cap, exclude, .{}); // longer matches off, as LLWZ's tests
    const chosen = try a.dupe(u32, packed_notes.chosen);
    var shown = try lookup.render(idx, sc, chosen);
    const t1 = clock.nowNs(io);
    // fit(): the whole block counted by the model's tokenizer; the lowest ranked note goes while it is over the cap
    var n = chosen.len;
    var tokens: usize = 0;
    while (n > 0) {
        // Oversized retry encodes must not accumulate in the request arena.
        var fit_arena = std.heap.ArenaAllocator.init(s.zeroing.allocator());
        defer fit_arena.deinit();
        tokens = (text.encode(fit_arena.allocator(), shown.block, false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Tokenizer,
        }).len;
        if (tokens <= cap) break;
        n -= 1;
        tokens = 0;
        shown = try lookup.render(idx, sc, chosen[0..n]);
    }
    const t2 = clock.nowNs(io);
    return .{
        .ids = try a.dupe(u32, shown.ids),
        .block = try a.dupe(u8, shown.block),
        .tokens = tokens,
        .pack_tokens = packed_notes.tokens,
        .dropped = chosen.len - n,
        .candidates = packed_notes.candidates,
        .long = packed_notes.long,
        .lookup_ns = t1 - t0,
        .fit_ns = t2 - t1,
        .in_store = idx.notes(),
    };
}
