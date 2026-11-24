//! tf-cuda-test qwen-prefill-cap MODEL IDS_FILE: a 12,000-row Qwen prompt capped at 512 rows, against its uncut pass.
const std = @import("std");
const lanes = @import("lanes");
const q = @import("qwen35");
const check = @import("check.zig");
const request_starts = @import("request_start_checks.zig");

const prompt_rows = 12000;
const cap: u32 = 512;
const draws = 64; // greedy tokens compared after the prompt: the first from its last row, then 63 decode rows
const key_chunk = 256; // window_attention.chunk_keys: full attention reads its keys in chunks of this many rows
const peer_from = 5000; // the live stream's prompt: 300 rows of the file, sharing no prefix with the long prompt
const peer_rows = 300;
const peer_draws = 192; // above any case's steps plus `draws`: the peer decodes after each step and target token
const context = prompt_rows + peer_draws + 96;

/// A cut layout: the request's chunk starts, the last capped piece's rows, and whether cuts fall inside key chunks.
const Case = struct { name: []const u8, starts: []const u32, tail: u32, inside_keys: bool };

const cases = [_]Case{
    .{ .name = "grid chunks", .starts = &.{}, .tail = 224, .inside_keys = false },
    .{ .name = "600-row chunk (512 + 88)", .starts = &.{ 4096, 4696 }, .tail = 136, .inside_keys = true },
    .{ .name = "off-grid start", .starts = &.{4133}, .tail = 187, .inside_keys = true },
    // cold chunks of 512 plus 1, 63, 64, 65, 127, 128, 129, 255, 256 and 257 rows: second pieces cross each tile
    .{ .name = "tile edges", .starts = &.{ 513, 1088, 1664, 2241, 2880, 3520, 4161, 4928, 5696, 6465 }, .tail = 415, .inside_keys = true },
};

const Ctx = struct { gpu: check.Gpu, l: *q.native.Lanes, b: lanes.backend.Backend, vocab: usize };

/// What a pass leaves: its cut ends, the prompt's last logits and each decode row's, and the greedy tokens.
const Pass = struct {
    cuts: std.ArrayList(u32) = .empty,
    logits: []u16,
    tokens: [draws]u32 = undefined,

    fn deinit(p: *Pass, gpa: std.mem.Allocator) void {
        p.cuts.deinit(gpa);
        gpa.free(p.logits);
    }
};

/// The newest forward's one kept row, read from the device.
fn deviceRow(c: *const Ctx, out: []u16) !void {
    const e = c.l.e;
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(out.ptr, e.logits, out.len * 2), "cuMemcpyDtoH");
}

/// The second stream: decoding one greedy token a round on the same engine, each row's logits hashed.
const Peer = struct {
    c: *const Ctx,
    s: lanes.Stream,
    tokens: [peer_draws]u32 = undefined,
    rows: [peer_draws][32]u8 = undefined,
    n: usize = 0,

    fn init(c: *const Ctx, ids: []const u32) !Peer {
        return .{ .c = c, .s = try lanes.Stream.init(c.gpu.gpa, .{ .id = "peer", .prompt = ids, .max_new = peer_draws, .drafts = false }) };
    }

    fn deinit(p: *Peer) void {
        p.c.b.vtable.release(p.c.b.ptr, &p.s);
        p.s.deinit(p.c.gpu.gpa);
    }

    fn start(p: *Peer) !void {
        const b = p.c.b;
        while (!try b.vtable.prefill_step.?(b.ptr, &p.s, 0)) {}
        const cache = try p.c.l.cacheOf(&p.s);
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(cache.logits), &p.rows[0], .{});
        p.tokens[0] = try b.vtable.read(b.ptr, try b.vtable.first(b.ptr, &p.s, peer_rows));
        p.n = 1;
    }

    fn round(p: *Peer) !void {
        try check.expect(p.n < peer_draws, "qwen-prefill-cap: the live stream ran out of its {d} tokens", .{peer_draws});
        const b = p.c.b;
        const handle = try b.vtable.queue(b.ptr, &p.s, .{ .value = p.tokens[p.n - 1] }, peer_rows + p.n);
        const row = try p.c.gpu.gpa.alloc(u16, p.c.vocab);
        defer p.c.gpu.gpa.free(row);
        try deviceRow(p.c, row);
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(row), &p.rows[p.n], .{});
        p.tokens[p.n] = try b.vtable.read(b.ptr, handle);
        p.n += 1;
    }

    /// Every token and row so far equals the same stream's run alone.
    fn same(p: *const Peer, alone: *const Peer, what: []const u8) !void {
        try check.sameBytes(what, std.mem.sliceAsBytes(p.tokens[0..p.n]), std.mem.sliceAsBytes(alone.tokens[0..p.n]));
        for (p.rows[0..p.n], alone.rows[0..p.n], 0..) |got, want, i|
            try check.expect(std.mem.eql(u8, &got, &want), "{s}: logits of its token {d} differ", .{ what, i });
    }
};

/// One prompt pass under `max_rows` (0: cold cuts), then its greedy tokens; `peer` decodes after each step and token.
fn pass(c: *const Ctx, ids: []const u32, starts: []const u32, max_rows: u32, peer: ?*Peer) !Pass {
    const gpa = c.gpu.gpa;
    const b = c.b;
    var out: Pass = .{ .logits = try gpa.alloc(u16, draws * c.vocab) };
    errdefer out.deinit(gpa);
    var s = try lanes.Stream.init(gpa, .{ .id = "long", .prompt = ids, .max_new = draws, .drafts = false, .chunks = starts });
    defer s.deinit(gpa);
    defer b.vtable.release(b.ptr, &s);
    while (true) {
        const done = try b.vtable.prefill_step.?(b.ptr, &s, max_rows);
        try out.cuts.append(gpa, s.fill_at.load(.monotonic));
        if (done) break;
        if (peer) |p| try p.round();
    }
    const cache = try c.l.cacheOf(&s);
    @memcpy(out.logits[0..c.vocab], cache.logits);
    out.tokens[0] = try b.vtable.read(b.ptr, try b.vtable.first(b.ptr, &s, ids.len));
    for (1..draws) |i| {
        if (peer) |p| try p.round();
        const handle = try b.vtable.queue(b.ptr, &s, .{ .value = out.tokens[i - 1] }, ids.len + i);
        try deviceRow(c, out.logits[i * c.vocab ..][0..c.vocab]);
        out.tokens[i] = try b.vtable.read(b.ptr, handle);
    }
    return out;
}

/// The capped cuts: at most `cap` rows each, every cold cut kept, the expected last piece, cuts inside key chunks.
fn cutsHold(case: Case, cold: []const u32, got: []const u32) !void {
    try check.expect(got.len > cold.len, "{s}: {d} capped steps for {d} cold ones: the cap did not reach the pass", .{ case.name, got.len, cold.len });
    try check.expect(got[got.len - 1] == prompt_rows, "{s}: the pass ended at row {d}", .{ case.name, got[got.len - 1] });
    var from: u32 = 0;
    var inside = false;
    for (got) |end| {
        try check.expect(end > from and end - from <= cap, "{s}: a step of {d} rows ending at {d}", .{ case.name, end - from, end });
        inside = inside or (end != prompt_rows and end % key_chunk != 0); // the prompt's end is not a cut
        from = end;
    }
    for (cold) |end| try check.expect(std.mem.indexOfScalar(u32, got, end) != null, "{s}: cold cut {d} did not land", .{ case.name, end });
    const last = got[got.len - 1] - got[got.len - 2];
    try check.expect(last == case.tail, "{s}: the last piece has {d} rows, expected {d}", .{ case.name, last, case.tail });
    try check.expect(inside == case.inside_keys, "{s}: cuts inside a {d}-row key chunk: {any}, expected {any}", .{ case.name, key_chunk, inside, case.inside_keys });
}

fn equal(what: []const u8, got: *const Pass, want: *const Pass) !void {
    try check.sameBytes(what, std.mem.sliceAsBytes(got.logits), std.mem.sliceAsBytes(want.logits));
    try check.sameBytes(what, std.mem.sliceAsBytes(&got.tokens), std.mem.sliceAsBytes(&want.tokens));
}

/// One layout: uncut, capped alone (a difference here is the cut's), then capped beside the live stream.
fn runCase(c: *const Ctx, ids: []const u32, peer_ids: []const u32, alone: *const Peer, case: Case) !void {
    const gpa = c.gpu.gpa;
    var whole = try pass(c, ids, case.starts, 0, null);
    defer whole.deinit(gpa);
    var solo = try pass(c, ids, case.starts, cap, null);
    defer solo.deinit(gpa);
    try cutsHold(case, whole.cuts.items, solo.cuts.items);
    var buf: [120]u8 = undefined;
    try equal(try std.fmt.bufPrint(&buf, "{s}: capped alone", .{case.name}), &solo, &whole);
    var peer = try Peer.init(c, peer_ids);
    defer peer.deinit();
    try peer.start();
    var live = try pass(c, ids, case.starts, cap, &peer);
    defer live.deinit(gpa);
    try check.expect(std.mem.eql(u32, live.cuts.items, solo.cuts.items), "{s}: the cuts moved beside a live stream", .{case.name});
    try equal(try std.fmt.bufPrint(&buf, "{s}: capped beside a live stream", .{case.name}), &live, &whole);
    try peer.same(alone, try std.fmt.bufPrint(&buf, "{s}: the live stream", .{case.name}));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(whole.logits), &digest, .{});
    check.pass("qwen-prefill-cap {s}: {d} cold steps, {d} capped, last piece {d} rows, {d} logit rows and {d} tokens equal, live stream's {d} tokens equal (logits sha256 {x})", .{ case.name, whole.cuts.items.len, live.cuts.items.len, case.tail, draws, draws, peer.n, digest });
}

/// MODEL's text path over the first 12,000 ids of IDS_FILE, every layout, and a control that a changed prompt shows.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const ids = try request_starts.readIds(gpu, ids_path, std.fmt.comptimePrint("{d}", .{prompt_rows}));
    defer gpa.free(ids);
    const loaded = try q.native.openLanes(gpa, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = false });
    defer loaded.deinit(loaded.ctx);
    const l: *q.native.Lanes = @ptrCast(@alignCast(loaded.ctx));
    try check.expect(l.e.scratch_rows == q.engine.grid_rows, "qwen-prefill-cap: unset TF_QWEN35_CHUNK_SOLO: the uncut pass must take {d}-row steps, not {d}", .{ q.engine.grid_rows, l.e.scratch_rows });
    try request_starts.validateIds(ids, l.e.g.vocab);
    const c: Ctx = .{ .gpu = gpu, .l = l, .b = loaded.backend, .vocab = l.e.g.vocab };
    const peer_ids = ids[peer_from..][0..peer_rows];
    var alone = try Peer.init(&c, peer_ids);
    defer alone.deinit();
    try alone.start();
    while (alone.n < peer_draws) try alone.round();
    for (cases) |case| try runCase(&c, ids, peer_ids, &alone, case);
    // the control: one changed prompt token must change the bytes this cell compares
    const changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[prompt_rows - 2] = if (changed[prompt_rows - 2] + 1 == c.vocab) 0 else changed[prompt_rows - 2] + 1;
    var base = try pass(&c, ids, &.{}, 0, null);
    defer base.deinit(gpa);
    var other = try pass(&c, changed, &.{}, 0, null);
    defer other.deinit(gpa);
    try check.expect(!std.mem.eql(u16, base.logits, other.logits), "qwen-prefill-cap: a changed prompt token left the compared logits equal", .{});
    check.pass("qwen-prefill-cap: {d} layouts equal their uncut pass; a changed prompt token changes the compared logits", .{cases.len});
}
