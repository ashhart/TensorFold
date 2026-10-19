//! cuda/tree_round.zig against core/tree_round.zig: picks (rowArgmax), batched trees (oracle, decode), the commits.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Gpu = check.Gpu;

const tr = cuda.tree_round;
const ref = core.tree_round;

comptime {
    // a CUDA round takes up to tr.max_rows rows, the core's helpers more (the 27B's Metal rounds), in the same fields
    std.debug.assert(@sizeOf(tr.Pick) == @sizeOf(ref.Pick) and tr.max_rows <= ref.max_rows);
}

/// The core's result in CUDA's layout: its first tr.max_rows path and token entries, the rest of a CUDA round's.
fn narrow(w: ref.Result) tr.Result {
    var r: tr.Result = undefined;
    const info = @typeInfo(tr.Result).@"struct";
    inline for (info.field_names, info.field_types) |name, T| @field(r, name) = switch (@typeInfo(T)) {
        .array => @field(w, name)[0..tr.max_rows].*,
        else => @field(w, name),
    };
    return r;
}

/// A CUDA result as the core's decode reads it, its arrays' entries past tr.max_rows zero.
fn wide(g: tr.Result) ref.Result {
    var r: ref.Result = .{};
    const info = @typeInfo(tr.Result).@"struct";
    inline for (info.field_names, info.field_types) |name, T| switch (@typeInfo(T)) {
        .array => @memcpy(@field(r, name)[0..tr.max_rows], &@field(g, name)),
        else => @field(r, name) = @field(g, name),
    };
    return r;
}

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7FFF +% ((b >> 16) & 1)) >> 16);
}

const nan: u16 = 0x7FC0;
const pos_inf: u16 = 0x7F80;
const neg_inf: u16 = 0xFF80;
/// A large finite value written past each row's vocab: a kernel reading the padding picks it.
const decoy: u16 = 0x7F00;

/// Row r by r's kind: spread, coarse ties, a NaN, +inf, -inf, all -inf, the max last, ties at both ends.
pub fn fillRow(rng: std.Random, r: usize, row: []u16) void {
    const v = row.len;
    for (row) |*h| h.* = switch (r % 8) {
        1 => bf16(-@floor(rng.float(f32) * 4.0)),
        5 => neg_inf,
        else => bf16(rng.floatNorm(f32) * 3.0),
    };
    switch (r % 8) {
        2 => row[rng.uintLessThan(usize, v)] = nan,
        3 => row[rng.uintLessThan(usize, v)] = pos_inf,
        4 => row[rng.uintLessThan(usize, v)] = neg_inf,
        6 => row[v - 1] = bf16(100.0),
        7 => {
            row[0] = bf16(100.0);
            row[v - 1] = bf16(100.0);
        },
        else => {},
    }
}

fn picks(gpu: Gpu, k: *const tr.Kernels, s: cuda.Stream) !void {
    const gpa = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x7AEE_0001);
    const rng = prng.random();
    const rows = 24; // more than one tree's 16: a round's argmax covers every tree
    var tested: usize = 0;
    for ([_]usize{ 1, 7, 255, 256, 257, 50_000, 151_936 }) |vocab| {
        const stride = vocab + 64;
        const host = try gpa.alloc(u16, rows * stride);
        defer gpa.free(host);
        @memset(host, decoy);
        for (0..rows) |r| fillRow(rng, r, host[r * stride ..][0..vocab]);
        var logits = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        defer logits.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * @sizeOf(tr.Pick));
        defer out.free();
        try out.fill8(0xA5, s.handle);
        try k.argmax(s, logits.ptr, out.ptr, rows, vocab, stride);
        try s.synchronize();
        var got: [rows]tr.Pick = undefined;
        try out.download(0, std.mem.sliceAsBytes(&got));
        var want: [rows]ref.Pick = undefined;
        for (&want, 0..) |*w, r| w.* = ref.rowArgmax(host[r * stride ..][0..vocab]);
        try check.sameBytes("argmax picks", std.mem.sliceAsBytes(&got), std.mem.sliceAsBytes(&want));
        tested += rows;
    }
    check.pass("tree-round argmax: {d} rows over 7 vocabularies equal rowArgmax (ties, NaN, +-inf, padding)", .{tested});
}

/// A round of `count` random trees in the kernels' layout, with the host reference's per-tree slices.
pub const Round = struct {
    tokens: std.ArrayList(u32) = .empty,
    parents: std.ArrayList(i32) = .empty,
    picks: std.ArrayList(tr.Pick) = .empty,
    eos: std.ArrayList(u32) = .empty,
    trees: std.ArrayList(tr.Tree) = .empty,

    pub fn deinit(self: *Round, gpa: std.mem.Allocator) void {
        self.tokens.deinit(gpa);
        self.parents.deinit(gpa);
        self.picks.deinit(gpa);
        self.eos.deinit(gpa);
        self.trees.deinit(gpa);
    }

    /// Tree t's expected result: the oracle's, or a bad tree where the oracle or a CUDA round's rows refuse the shape.
    pub fn want(self: *const Round, t: usize, vocab: u32) !tr.Result {
        const x = self.trees.items[t];
        const bad = narrow(.{ .status = @backingInt(ref.Status.bad_tree), .stop = @backingInt(ref.Stop.invalid) });
        if (x.rows > tr.max_rows) return bad;
        const p = ref.Match{ .rows = x.rows, .vocab = vocab, .budget = x.budget, .eos_count = x.eos_count };
        const rows = @min(x.rows, self.tokens.items.len - x.first);
        return narrow(ref.oracle(p, self.tokens.items[x.first..][0..rows], self.parents.items[x.first..][0..rows], @ptrCast(self.picks.items[x.first..][0..rows]), self.eos.items[x.eos_first..][0..x.eos_count]) catch |e| switch (e) {
            error.BadRoundShape => return bad,
        });
    }
};

pub const vocab_small: u32 = 6; // small enough that drafts match the picks often

pub fn buildRound(gpa: std.mem.Allocator, rng: std.Random, count: usize) !Round {
    var round: Round = .{};
    errdefer round.deinit(gpa);
    for (0..count) |t| {
        const chain = t % 97 == 50; // a full-depth path: every draft matches, 16 rows kept
        const rows: u32 = if (t % 97 == 13) 0 else if (t % 97 == 14) 17 else if (chain) 16 else 1 + rng.uintLessThan(u32, 16);
        const first: u32 = @intCast(round.tokens.items.len);
        for (0..rows) |r| {
            try round.parents.append(gpa, if (r == 0) -1 else if (chain) @intCast(r - 1) else @intCast(rng.uintLessThan(usize, r)));
            try round.tokens.append(gpa, rng.uintLessThan(u32, vocab_small));
        }
        // picks follow a child most of the time, so paths run deep
        for (0..rows) |r| {
            var token = rng.uintLessThan(u32, vocab_small);
            if (chain or rng.uintLessThan(u32, 10) < 7) {
                for (r + 1..rows) |c| if (round.parents.items[first + c] == @as(i32, @intCast(r))) {
                    token = round.tokens.items[first + c];
                    if (rng.boolean()) break;
                };
            }
            try round.picks.append(gpa, .{ .token = token, .nonfinite = 0 });
        }
        const eos_first: u32 = @intCast(round.eos.items.len);
        const eos_count: u32 = if (chain or rng.boolean()) 0 else 1 + rng.uintLessThan(u32, 2);
        for (0..eos_count) |_| try round.eos.append(gpa, rng.uintLessThan(u32, vocab_small));
        if (rows > 0 and !chain) switch (t % 41) {
            3 => round.picks.items[first + rng.uintLessThan(u32, rows)] = .{ .token = ref.invalid_token, .nonfinite = @as(u32, 1) << @intCast(rng.uintLessThan(u32, 3)) },
            5 => round.picks.items[first + rng.uintLessThan(u32, rows)].nonfinite = 8,
            7 => round.picks.items[first + rng.uintLessThan(u32, rows)].token = vocab_small,
            11 => if (rows > 1) {
                round.parents.items[first + rows - 1] = @intCast(rows - 1);
            },
            17 => round.tokens.items[first + rng.uintLessThan(u32, rows)] = vocab_small + 3,
            19 => if (eos_count > 0) {
                round.eos.items[eos_first] = vocab_small;
            },
            else => {},
        };
        try round.trees.append(gpa, .{ .first = first, .rows = rows, .budget = if (chain) 16 + rng.uintLessThan(u32, 2) else rng.uintLessThan(u32, 18), .eos_first = eos_first, .eos_count = eos_count });
    }
    return round;
}

/// The round on the GPU: its inputs, and results / keep / counts / gathered rows / heads for every tree.
const Device = struct {
    tokens: cuda.DeviceBuffer,
    parents: cuda.DeviceBuffer,
    picks: cuda.DeviceBuffer,
    eos: cuda.DeviceBuffer,
    trees: cuda.DeviceBuffer,
    results: cuda.DeviceBuffer,

    fn init(gpu: Gpu, round: *const Round) !Device {
        // a zero-length round still needs addresses: one spare element each
        const eos_bytes = std.mem.sliceAsBytes(round.eos.items);
        return .{
            .tokens = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(round.tokens.items)),
            .parents = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(round.parents.items)),
            .picks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(round.picks.items)),
            .eos = if (eos_bytes.len > 0) try cuda.DeviceBuffer.fromHost(gpu.d, eos_bytes) else try cuda.DeviceBuffer.alloc(gpu.d, 4),
            .trees = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(round.trees.items)),
            .results = try cuda.DeviceBuffer.alloc(gpu.d, round.trees.items.len * @sizeOf(tr.Result)),
        };
    }

    fn deinit(self: *Device) void {
        inline for (.{ "tokens", "parents", "picks", "eos", "trees", "results" }) |f| @field(self, f).free();
    }

    fn match(self: *const Device, k: *const tr.Kernels, s: cuda.Stream, count: usize) !void {
        try self.results.fill8(0xA5, s.handle); // every field is written: a missed one shows (on s: a legacy-stream fill races it)
        try k.match(s, self.tokens.ptr, self.parents.ptr, self.picks.ptr, self.eos.ptr, self.trees.ptr, self.results.ptr, vocab_small, count);
        try s.synchronize();
    }
};

fn results(gpu: Gpu, dev: *const Device, count: usize) ![]tr.Result {
    const out = try gpu.gpa.alloc(tr.Result, count);
    try dev.results.download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// Tree t's kept round rows: its verified path, for an ok result that consumed any.
fn kept(r: *const tr.Result) usize {
    return if (r.status == 0 and r.consumed_count > 0) r.consumed_count else 0;
}

fn matches(gpu: Gpu, k: *const tr.Kernels, s: cuda.Stream) !void {
    const gpa = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x7AEE_0002);
    const count = 4096;
    var round = try buildRound(gpa, prng.random(), count);
    defer round.deinit(gpa);
    var dev = try Device.init(gpu, &round);
    defer dev.deinit();
    try dev.match(k, s, count);
    const got = try results(gpu, &dev, count);
    defer gpa.free(got);
    var by_stop: [4]usize = @splat(0);
    var by_status: [4]usize = @splat(0);
    var deepest: u32 = 0;
    for (got, 0..) |*g, t| {
        const w = try round.want(t, vocab_small);
        try check.sameBytes("tree result", std.mem.asBytes(g), std.mem.asBytes(&w));
        const x = round.trees.items[t];
        if (x.rows >= 1 and x.rows <= tr.max_rows) {
            const d = ref.decode(&wide(g.*), .{ .rows = x.rows, .vocab = vocab_small, .budget = x.budget, .eos_count = x.eos_count }) catch |e| {
                std.debug.print("FAIL tree {d}: decode {s}\n", .{ t, @errorName(e) });
                return error.TestFailed;
            };
            by_stop[@backingInt(d.stop)] += 1;
            by_status[@backingInt(d.status)] += 1;
            deepest = @max(deepest, g.matched_count);
        }
    }
    for (by_stop, 0..) |n, i| try check.expect(n > 0, "the round never stopped by {s}", .{@tagName(std.enums.fromInt(ref.Stop, @as(u32, @intCast(i))).?)});
    for (by_status, 0..) |n, i| try check.expect(n > 0, "the round never had status {s}", .{@tagName(std.enums.fromInt(ref.Status, @as(u32, @intCast(i))).?)});
    try check.expect(deepest == tr.max_rows - 1, "no path matched more than {d} drafts", .{deepest});
    check.pass("tree-round match: {d} batched trees equal the oracle and decode (stops bonus/length/eos/invalid {any}, deepest {d})", .{ count, by_stop, deepest });

    // a tree alone gives its batched result: the batch shares nothing between trees
    for (0..64) |t| {
        const one = round.trees.items[t];
        var single = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&one));
        defer single.free();
        try dev.results.fill8(0x5A, s.handle);
        try k.match(s, dev.tokens.ptr, dev.parents.ptr, dev.picks.ptr, dev.eos.ptr, single.ptr, dev.results.ptr, vocab_small, 1);
        try s.synchronize();
        var alone: tr.Result = undefined;
        try dev.results.download(0, std.mem.asBytes(&alone));
        try check.sameBytes("tree alone", std.mem.asBytes(&alone), std.mem.asBytes(&got[t]));
    }
    check.pass("tree-round match: 64 trees alone equal their batched results", .{});

    // mutation: one followed pick changed must change that tree's result, and the comparison must see it
    const victim = for (got, 0..) |*g, t| {
        if (g.status == 0 and g.matched_count >= 2) break t;
    } else unreachable;
    const at = round.trees.items[victim].first + got[victim].path[1];
    const saved = round.picks.items[at];
    var bent = saved;
    bent.token = (saved.token + 1) % vocab_small;
    try dev.picks.upload(at * @sizeOf(tr.Pick), std.mem.asBytes(&bent));
    try dev.match(k, s, count);
    const mutated = try results(gpu, &dev, count);
    defer gpa.free(mutated);
    try check.expect(!std.mem.eql(u8, std.mem.asBytes(&mutated[victim]), std.mem.asBytes(&got[victim])), "a changed pick on tree {d}'s path left its result alone", .{victim});
    try dev.picks.upload(at * @sizeOf(tr.Pick), std.mem.asBytes(&saved));
    check.pass("tree-round match: a changed pick on a verified path is caught", .{});

    try commits(gpu, k, s, &round, &dev, got);
}

fn word(row: usize, c: usize) u16 {
    return @truncate((row *% 0x9E37 +% c *% 0x85EB +% 1) | 1);
}

fn commits(gpu: Gpu, k: *const tr.Kernels, s: cuda.Stream, round: *const Round, dev: *const Device, got: []const tr.Result) !void {
    const gpa = gpu.gpa;
    const count = got.len;
    try dev.match(k, s, count);
    const total = round.tokens.items.len;

    var keep = try cuda.DeviceBuffer.alloc(gpu.d, count * tr.max_rows * 4);
    defer keep.free();
    var counts = try cuda.DeviceBuffer.alloc(gpu.d, count * 4);
    defer counts.free();
    try keep.fill8(0xA5, s.handle);
    try counts.fill8(0xA5, s.handle);
    try k.keepRows(s, dev.results.ptr, dev.trees.ptr, keep.ptr, counts.ptr, count);
    try s.synchronize();
    const keep_got = try gpa.alloc(i32, count * tr.max_rows);
    defer gpa.free(keep_got);
    const counts_got = try gpa.alloc(u32, count);
    defer gpa.free(counts_got);
    try keep.download(0, std.mem.sliceAsBytes(keep_got));
    try counts.download(0, std.mem.sliceAsBytes(counts_got));
    const keep_want = try gpa.alloc(i32, count * tr.max_rows);
    defer gpa.free(keep_want);
    const counts_want = try gpa.alloc(u32, count);
    defer gpa.free(counts_want);
    for (got, 0..) |*r, t| {
        const n = kept(r);
        counts_want[t] = @intCast(n);
        for (0..tr.max_rows) |i| keep_want[t * tr.max_rows + i] = if (i < n) @intCast(round.trees.items[t].first + r.path[i]) else -1;
    }
    try check.sameBytes("keep rows", std.mem.sliceAsBytes(keep_got), std.mem.sliceAsBytes(keep_want));
    try check.sameBytes("keep counts", std.mem.sliceAsBytes(counts_got), std.mem.sliceAsBytes(counts_want));

    const width = 300; // not a multiple of the 256-column blocks
    const stride = width + 8;
    const rows_host = try gpa.alloc(u16, total * stride);
    defer gpa.free(rows_host);
    for (0..total) |row| for (0..stride) |c| {
        rows_host[row * stride + c] = word(row, c);
    };
    var src = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(rows_host));
    defer src.free();
    const sentinel: u16 = 0xA5A5;

    // gather reads packed rows (width apart): the first total * width words of a packed copy
    const packed_host = try gpa.alloc(u16, total * width);
    defer gpa.free(packed_host);
    for (0..total) |row| @memcpy(packed_host[row * width ..][0..width], rows_host[row * stride ..][0..width]);
    var packed_rows = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(packed_host));
    defer packed_rows.free();
    var gathered = try cuda.DeviceBuffer.alloc(gpu.d, count * tr.max_rows * width * 2);
    defer gathered.free();
    try gathered.fill8(0xA5, s.handle);
    try k.gatherRows(s, dev.results.ptr, dev.trees.ptr, packed_rows.ptr, gathered.ptr, width, count);
    var head = try cuda.DeviceBuffer.alloc(gpu.d, count * width * 2);
    defer head.free();
    try head.fill8(0xA5, s.handle);
    try k.keepHead(s, dev.results.ptr, dev.trees.ptr, src.ptr, head.ptr, width, stride, count);
    try s.synchronize();

    const gather_got = try gpa.alloc(u16, count * tr.max_rows * width);
    defer gpa.free(gather_got);
    try gathered.download(0, std.mem.sliceAsBytes(gather_got));
    const head_got = try gpa.alloc(u16, count * width);
    defer gpa.free(head_got);
    try head.download(0, std.mem.sliceAsBytes(head_got));
    const gather_want = try gpa.alloc(u16, count * tr.max_rows * width);
    defer gpa.free(gather_want);
    const head_want = try gpa.alloc(u16, count * width);
    defer gpa.free(head_want);
    @memset(gather_want, sentinel);
    @memset(head_want, sentinel);
    var moved: usize = 0;
    for (got, 0..) |*r, t| {
        const n = kept(r);
        const first = round.trees.items[t].first;
        for (0..n) |i| @memcpy(gather_want[(t * tr.max_rows + i) * width ..][0..width], packed_host[(first + r.path[i]) * width ..][0..width]);
        if (n > 0) @memcpy(head_want[t * width ..][0..width], rows_host[(first + r.path[n - 1]) * stride ..][0..width]);
        moved += n;
    }
    try check.sameBytes("gathered rows", std.mem.sliceAsBytes(gather_got), std.mem.sliceAsBytes(gather_want));
    try check.sameBytes("kept heads", std.mem.sliceAsBytes(head_got), std.mem.sliceAsBytes(head_want));
    check.pass("tree-round commit: keep rows, gather ({d} rows x {d}) and heads equal the host; nothing else written", .{ moved, width });
}

pub fn run(gpu: Gpu) !void {
    var m = try cuda.Module.load(gpu.d, cuda.kernels.tree_round);
    defer m.unload();
    const k = try tr.Kernels.resolve(m);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    try picks(gpu, &k, s);
    try matches(gpu, &k, s);
}
