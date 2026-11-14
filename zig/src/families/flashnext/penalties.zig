//! Presence and frequency penalties on shared rounds: each row's entries taken from its logits before the argmax.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const be = lanes.backend;
const Entries = lanes.penalty_rows.Entries;

pub const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// entry i takes sub[i] from row row[i]'s logit of tok[i] in fp32, rounded once (CUDA's tf_penalize_rows)
    \\kernel void fz_penalize_rows(device bfloat* logits [[buffer(0)]], device const uint* row [[buffer(1)]],
    \\    device const uint* tok [[buffer(2)]], device const float* sub [[buffer(3)]],
    \\    constant uint2& c [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    \\  if (i >= c.y) return;
    \\  device bfloat* x = logits + ulong(row[i]) * c.x + tok[i];
    \\  *x = bfloat(float(*x) - sub[i]);
    \\}
;

/// The round's entries and the shared buffer the kernel reads them from (rows, then tokens, then amounts).
pub const Rows = struct {
    pipe: mtl.Pipeline,
    dev: mtl.Device,
    buf: ?mtl.Buffer = null,
    cap: usize = 0, // entries `buf` holds
    entries: Entries = .{},

    pub fn init(dev: mtl.Device) !Rows {
        const lib = try mtl.Library.fromSource(dev, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return .{ .pipe = try mtl.Pipeline.init(dev, lib, "fz_penalize_rows", false), .dev = dev };
    }

    pub fn deinit(p: *Rows, gpa: std.mem.Allocator) void {
        p.entries.deinit(gpa);
        if (p.buf) |b| b.deinit();
        p.pipe.deinit();
    }

    /// Each penalized window's rows from its start; false when none has penalties (no pass runs: main's tokens).
    pub fn fill(p: *Rows, gpa: std.mem.Allocator, windows: []const be.Window, drafts: []const []const u32, starts: []const usize) !bool {
        p.entries.clear();
        for (windows, drafts, starts) |w, d, row| try p.entries.stageWindow(gpa, w, d, row);
        const n = p.entries.len();
        if (n == 0) return false;
        if (n > p.cap) { // the last round's command buffer has finished: its buffer may go
            const cap = @max(n, 2 * p.cap, 4096);
            const b = try p.dev.buffer(cap * 12, fz.opts);
            if (p.buf) |old| old.deinit();
            p.buf = b;
            p.cap = cap;
        }
        const words = p.buf.?.slice(u32, 3 * p.cap);
        @memcpy(words[0..n], p.entries.rows.items);
        @memcpy(words[p.cap..][0..n], p.entries.toks.items);
        @memcpy(words[2 * p.cap ..][0..n], @as([*]const u32, @ptrCast(p.entries.subs.items.ptr))[0..n]); // fp32 bits
        return true;
    }

    /// fz_penalize_rows over the filled entries of `logits` (rows of VOCAB bf16), before the argmax.
    pub fn encode(p: *const Rows, enc: mtl.ComputeEncoder, logits: fz.Buf) void {
        const n = p.entries.len();
        const c = [2]u32{ fz.VOCAB, @intCast(n) };
        enc.setPipeline(p.pipe);
        enc.setBuffer(logits.b, logits.off, 0);
        enc.setBuffer(p.buf.?, 0, 1);
        enc.setBuffer(p.buf.?, p.cap * 4, 2);
        enc.setBuffer(p.buf.?, p.cap * 8, 3);
        enc.setBytes(std.mem.asBytes(&c), 4);
        enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

/// A verify's penalty pass for `batch_forward.encode` (`b`: the backend, its sessions' rows filled); null: none.
pub fn round(b: anytype, windows: []const be.Window, starts: []const usize) !?*const Rows {
    var any = false;
    for (windows) |w| any = any or w.stream.penalties.active();
    if (!any) return null; // a round without penalties encodes what main's does
    var drafts: [fz.MAXR][]const u32 = undefined;
    for (windows, drafts[0..windows.len]) |w, *d| d.* = (try b.session(w.stream)).tokens[1..w.rows()];
    if (b.e.penalties == null) b.e.penalties = try Rows.init(b.e.r.device);
    const p = &b.e.penalties.?;
    return if (try p.fill(b.gpa, windows, drafts[0..windows.len], starts)) p else null;
}

test "Flash Next penalties: a greedy row's pick and a sampled row's draw are the host draws over the lowered row" {
    const gpa = std.testing.allocator;
    const host = @import("host_sample.zig");
    var e: Entries = .{};
    defer e.deinit(gpa);
    var s: lanes.Stream = undefined;
    const ctx = [_]u32{ 7, 7, 3, 3, 9 }; // prompt 7 7, reply 3 3, pending 9 at cache_len 4
    s.context = .fromOwnedSlice(@constCast(&ctx));
    s.prompt_len = 2;
    s.penalties = .{ .presence = 1.5, .frequency = 0.5 };
    var logits: [64]u16 = @splat(0x3f80); // 1.0
    logits[3] = 0x4040; // 3.0, the argmax before penalties
    logits[9] = 0x4000; // 2.0
    logits[5] = 0x3fc0; // 1.5
    try e.stageOne(gpa, &s, 4, 9, 0);
    var row = logits;
    e.applyRow(0, &row);
    // 3 held twice: 3 - 2.5 = 0.5; 9 once: 2 - 2 = 0; 5 untouched leads
    try std.testing.expectEqual(@as(u16, 0x3f00), row[3]);
    try std.testing.expectEqual(@as(u16, 0), row[9]);
    try std.testing.expectEqual(@as(u32, 5), try host.reference(gpa, &row, .{ .seed = 0, .temperature = 0 }, 5));
    // the sampled draw over the lowered row equals the reference's, at every seed and top-k
    for (0..64) |seed| for ([_]u32{ 0, 1, 4, 64 }) |k| {
        const p: lanes.Sampling = .{ .seed = seed, .temperature = 0.7, .top_k = k, .top_p = 0.8 };
        try std.testing.expectEqual(try host.reference(gpa, &row, p, 5), try host.draw(gpa, &row, p, 5));
    };
    // zero penalties leave the row as main's
    s.penalties = .{};
    e.clear();
    try e.stageOne(gpa, &s, 4, 9, 0);
    row = logits;
    e.applyRow(0, &row);
    try std.testing.expectEqualSlices(u16, &logits, &row);
}

test "Flash Next penalties: a round lowers each penalized window's cells at its plan rows, with its own drafts" {
    if (comptime @hasDecl(mtl, "flashnext_recording")) try recordedRound() else return error.SkipZigTest;
}

/// The real round, fill and encode on the recording device; the dispatch it records applied as the kernel does.
fn recordedRound() !void {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    mtl.allocator = arena.allocator();
    const Session = struct { tokens: [4]u32 };
    const Fake = struct {
        gpa: std.mem.Allocator,
        e: struct { penalties: ?Rows = null, r: struct { device: mtl.Device = .{} } = .{} } = .{},
        streams: [2]*const lanes.Stream,
        sessions: [2]Session,
        pub fn session(f: *@This(), s: *const lanes.Stream) !*Session {
            return &f.sessions[std.mem.indexOfScalar(*const lanes.Stream, &f.streams, s) orelse return error.UnknownStream];
        }
    };
    var plain: lanes.Stream = undefined; // no penalties: its rows stay the forward's
    plain.context = .fromOwnedSlice(@constCast(&[_]u32{ 2, 3 }));
    plain.prompt_len = 1;
    plain.penalties = .{};
    var held: lanes.Stream = undefined; // prompt 1, reply 5 5 6, pending 9; two held drafts 5 7 from its session
    held.context = .fromOwnedSlice(@constCast(&[_]u32{ 1, 5, 5, 6 }));
    held.prompt_len = 1;
    held.penalties = .{ .presence = 1.5, .frequency = 0.5 };
    var f: Fake = .{ .gpa = gpa, .streams = .{ &plain, &held }, .sessions = .{ .{ .tokens = .{ 4, 3, 0, 0 } }, .{ .tokens = .{ 9, 5, 7, 0 } } } };
    defer if (f.e.penalties) |*p| p.deinit(gpa);
    const windows = [_]be.Window{
        .{ .stream = &plain, .pending = 4, .held = 0, .tokens = &.{3}, .parents = null, .positions = &.{ 2, 3 } },
        .{ .stream = &held, .pending = 9, .held = 2, .tokens = &.{}, .parents = null, .positions = &.{ 5, 6, 7 } },
    };
    const starts = [_]usize{ 0, 2 };
    try std.testing.expect(try round(&f, windows[0..1], starts[0..1]) == null); // no penalties: no pass, no pipeline
    try std.testing.expect(f.e.penalties == null);
    const p = (try round(&f, &windows, &starts)).?;
    const logits: fz.Buf = .{ .b = try f.e.r.device.buffer(5 * fz.VOCAB * 2, fz.opts) };
    const words = logits.b.slice(u16, 5 * fz.VOCAB);
    @memset(words, 0x4080); // 4.0
    mtl.penalized = null;
    p.encode(.{}, logits);
    const d = mtl.penalized.?; // the kernel's arithmetic over what was bound: entry i lowers row[i] * c.x + tok[i]
    try std.testing.expectEqual([2]u32{ fz.VOCAB, @intCast(d.n) }, d.c);
    try std.testing.expectEqual(logits.b.id, d.bound[0].b.id);
    const rows: [*]const u32 = @ptrCast(@alignCast(d.bound[1].b.contents() + d.bound[1].off));
    const toks: [*]const u32 = @ptrCast(@alignCast(d.bound[2].b.contents() + d.bound[2].off));
    const subs: [*]const f32 = @ptrCast(@alignCast(d.bound[3].b.contents() + d.bound[3].off));
    for (rows[0..d.n], toks[0..d.n], subs[0..d.n]) |row, tok, sub| {
        const cell = @as(usize, row) * fz.VOCAB + tok;
        words[cell] = lanes.penalty_rows.lower(words[cell], sub);
    }
    // plan rows 2..4: 5 (twice in the reply) loses 2.5, then 3 with rows 3-4's draft 5; 6, 9 and row 4's 7 lose 2
    const want = try gpa.alloc(u16, 5 * fz.VOCAB);
    defer gpa.free(want);
    @memset(want, 0x4080);
    for ([_][3]u32{ .{ 2, 5, 0x3fc0 }, .{ 3, 5, 0x3f80 }, .{ 4, 5, 0x3f80 }, .{ 4, 7, 0x4000 } }) |x| want[x[0] * fz.VOCAB + x[1]] = @intCast(x[2]);
    for (2..5) |row| for ([_]usize{ 6, 9 }) |tok| {
        want[row * fz.VOCAB + tok] = 0x4000;
    };
    try std.testing.expect(std.mem.eql(u16, want, words));
}
