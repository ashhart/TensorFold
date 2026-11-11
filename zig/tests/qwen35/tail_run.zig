//! The prompt tail on a device: a chunk's last row through its tail, and a chunk that stops at its keys, against whole.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;
const fwd = qwen35.forward;

/// One tile and less (the whole chunk runs), a 64-row tile's edges, a chunk's edges, and past a 2,048-row step.
pub const lengths = [_]usize{ 1, 63, 64, 65, 1023, 1024, 1025, 2049, 4096 };

/// Rows run after the chunk on its caches, whose attention reads every key the chunk wrote.
pub const after = 16;

/// Positions the longest case takes.
pub const need = lengths[lengths.len - 1] + after;

/// What first differs from the whole chunk, and at which byte.
pub const Diff = struct { what: []const u8, byte: usize };

/// One run's bytes: the chunk's last final row and its logits, then the `after` rows' final rows.
const Bytes = struct {
    row: []u8,
    logits: []u8,
    next: []u8,

    fn init(a: std.mem.Allocator, e: *const Engine) !Bytes {
        const m = e.model();
        const width = m.spec.hidden * m.act.size();
        return .{ .row = try a.alloc(u8, width), .logits = try a.alloc(u8, m.head.n * m.act.size()), .next = try a.alloc(u8, after * width) };
    }
};

fn copy(e: *Engine, to: []u8, from: u64) !void {
    try hip.runtime.check(e.driver.api.hipMemcpyDtoH(to.ptr, @ptrFromInt(from), to.len));
}

/// `ids[0..len]` as one chunk read as `reads`, then `after` rows on its caches read whole, into `out`.
fn run(gpa: std.mem.Allocator, e: *Engine, ids: hip.DeviceBuffer, len: usize, reads: fwd.Reads, out: Bytes) !void {
    const m = e.model();
    var caches = try e.newCaches(len + after);
    defer {
        e.drain();
        caches.deinit(gpa);
    }
    e.prompts.reset();
    const o = e.ops(&e.prompts);
    const kept = try fwd.spanFor(o, m, &caches, ids.base(), len, 0, null, reads);
    if (kept.from < len) {
        const row = kept.row(len - 1, m.spec.hidden);
        const logits = try o.project(row, m.head, 1, false);
        try e.stream.synchronize();
        try copy(e, out.row, row.ptr);
        try copy(e, out.logits, logits.ptr);
    }
    e.prompts.reset();
    const next = try fwd.span(o, m, &caches, ids.base() + len * 4, after, len, null);
    try e.stream.synchronize();
    try copy(e, out.next, next.ptr);
}

/// A chunk of `len` rows whole, through the tail (its last row read) and keys only (nothing read): where they differ.
pub fn runCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, len: usize) !?Diff {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try Bytes.init(a, e);
    const tail = try Bytes.init(a, e);
    const keys = try Bytes.init(a, e);
    var dev = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(ids[0 .. len + after]));
    defer dev.free();
    try run(gpa, e, dev, len, Engine.passes.readout, whole);
    try run(gpa, e, dev, len, Engine.passes.prefill, tail);
    try run(gpa, e, dev, len, Engine.passes.advance, keys);
    const pairs = [_]struct { []const u8, []const u8, []const u8 }{
        .{ "the tail's last final row", whole.row, tail.row },
        .{ "the tail's last row logits", whole.logits, tail.logits },
        .{ "the rows after the tail's chunk", whole.next, tail.next },
        .{ "the rows after a keys-only chunk", whole.next, keys.next },
    };
    for (pairs) |p| if (std.mem.indexOfDiff(u8, p[1], p[2])) |at| return .{ .what = p[0], .byte = at };
    return null;
}
