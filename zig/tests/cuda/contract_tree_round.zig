//! tree_round.cu (#734) under the contract on CUDA: picks, verified trees and commits, byte for byte against the host.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const rt = @import("tree_round_tests.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const tr = cuda.tree_round;
const ref = core.tree_round;

pub const Which = enum { argmax, match, keep, gather, head };

fn noSums(_: c.Shape) usize {
    return 0;
}

const pick_rule = "a row holding NaN or an infinity picks no token and flags it (1 NaN, 2 +inf, 4 -inf); ties go to the lower token";
const match_rule = "a flagged pick or a malformed tree gives core/tree_round.oracle's invalid result, byte for byte";
const move_rule = "a tree whose result is not ok moves nothing; kept rows move bit for bit, NaN and infinities included";

/// argmax: rows x vocab (k); match: trees (rows) of at most 16 rows (k); commits: trees (rows) x row width (n).
pub fn spec(which: Which) c.Spec {
    return switch (which) {
        // a 27B tree (16 rows), eight streams' trees, Nemotron's 131,072 tokens
        .argmax => .{ .name = "cuda tree_round tf_round_bf16_argmax", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pick_rule }, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 16, .n = 1, .k = 248320 }, .{ .rows = 128, .n = 1, .k = 248320 }, .{ .rows = 48, .n = 1, .k = 131072 } } },
        .match => .{ .name = "cuda tree_round tf_tree_round_match", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = match_rule }, .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{ .{ .rows = 8, .n = 1, .k = 16 }, .{ .rows = 200, .n = 1, .k = 16 } } },
        .keep => .{ .name = "cuda tree_round tf_round_keep_rows", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = move_rule }, .served = &.{ .{ .rows = 8, .n = 16, .k = 16 }, .{ .rows = 200, .n = 16, .k = 16 } } },
        // hidden rows of the 27B (5,120) and Nemotron (2,688)
        .gather => .{ .name = "cuda tree_round tf_round_gather_rows", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = move_rule }, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 8, .n = 5120, .k = 16 }, .{ .rows = 64, .n = 2688, .k = 16 } } },
        // the next round's first logits row
        .head => .{ .name = "cuda tree_round tf_round_keep_head", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = move_rule }, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 8, .n = 248320, .k = 16 }, .{ .rows = 48, .n = 131072, .k = 16 } } },
    };
}

/// A large finite value past each row's vocab: a kernel reading the padding picks it.
const decoy: u16 = 0x7F00;
const pad = 8;

pub const Round = struct {
    rig: rig_.Rig,
    which: Which,
    k: *const tr.Kernels,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    round: ?rt.Round = null,
    results: []tr.Result = &.{},
    words: []u16 = &.{},
    in: [6]u64 = @splat(0),
    out: [2]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Round) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *Round) void {
        const a = self.rig.gpu.gpa;
        if (self.round) |*r| r.deinit(a);
        self.round = null;
        a.free(self.results);
        self.results = &.{};
        a.free(self.words);
        self.words = &.{};
    }

    pub fn kernel(self: *Round) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Round);

    fn stride(self: *const Round) usize {
        return self.shape.k + pad;
    }

    pub fn prepare(self: *Round, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.drop();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.which == .argmax) {
            self.words = try a.alloc(u16, shape.rows * self.stride());
            @memset(self.words, decoy);
            // gate 1: rows 0-3 hold NaN, +inf, -inf, and every -inf; the other gates also see them every eighth row
            for (0..shape.rows) |r| rt.fillRow(rng, if (fill == .nonfinite) r + 2 else r, self.words[r * self.stride() ..][0..shape.k]);
            self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.words));
            self.out[0] = try self.rig.output(shape.rows * @sizeOf(tr.Pick));
            return true;
        }
        self.round = try rt.buildRound(a, rng, shape.rows);
        const rd = &self.round.?;
        if (fill == .nonfinite) for (rd.trees.items[0..@min(4, shape.rows)], 0..) |t, i| if (t.rows > 0 and t.rows <= tr.max_rows) {
            rd.picks.items[t.first + t.rows - 1] = .{ .token = ref.invalid_token, .nonfinite = ([_]u32{ 1, 2, 4, 6 })[i] };
        };
        if (self.which == .match) {
            const lists = [_][]const u8{ std.mem.sliceAsBytes(rd.tokens.items), std.mem.sliceAsBytes(rd.parents.items), std.mem.sliceAsBytes(rd.picks.items), std.mem.sliceAsBytes(rd.eos.items), std.mem.sliceAsBytes(rd.trees.items) };
            for (lists, 0..) |l, i| self.in[i] = try self.rig.input(l);
            self.out[0] = try self.rig.output(shape.rows * @sizeOf(tr.Result));
            return true;
        }
        self.results = try a.alloc(tr.Result, shape.rows);
        for (self.results, 0..) |*r, t| r.* = try rd.want(t, rt.vocab_small);
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.results));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(rd.trees.items));
        const total = rd.tokens.items.len;
        switch (self.which) {
            .keep => {
                self.out[0] = try self.rig.output(shape.rows * tr.max_rows * 4);
                self.out[1] = try self.rig.output(shape.rows * 4);
            },
            .gather, .head => {
                const width = if (self.which == .head) shape.n + pad else shape.n;
                self.words = try a.alloc(u16, @max(total, 1) * width);
                for (self.words) |*w| w.* = rng.int(u16);
                if (fill == .nonfinite) for (self.words, 0..) |*w, i| switch (i % 7) {
                    1 => w.* = 0x7FC0,
                    2 => w.* = 0x7F80,
                    3 => w.* = 0xFF80,
                    else => {},
                };
                self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.words));
                const rows: usize = if (self.which == .head) 1 else tr.max_rows;
                self.out[0] = try self.rig.output(shape.rows * rows * shape.n * 2);
            },
            else => unreachable,
        }
        return true;
    }

    pub fn run(self: *Round) !void {
        const s = self.rig.s;
        const n = self.shape.rows;
        switch (self.which) {
            .argmax => try self.k.argmax(s, self.in[0], self.out[0], n, self.shape.k, self.stride()),
            .match => try self.k.match(s, self.in[0], self.in[1], self.in[2], self.in[3], self.in[4], self.out[0], rt.vocab_small, n),
            .keep => try self.k.keepRows(s, self.in[0], self.in[1], self.out[0], self.out[1], n),
            .gather => try self.k.gatherRows(s, self.in[0], self.in[1], self.in[2], self.out[0], self.shape.n, n),
            .head => try self.k.keepHead(s, self.in[0], self.in[1], self.in[2], self.out[0], self.shape.n, self.shape.n + pad, n),
        }
        try s.synchronize();
    }

    pub fn count(self: *Round) usize {
        const n = self.shape.rows;
        return switch (self.which) {
            .argmax => n * @sizeOf(tr.Pick),
            .match => n * @sizeOf(tr.Result),
            .keep => n * (tr.max_rows + 1) * 4,
            .gather => n * tr.max_rows * self.shape.n * 2,
            .head => n * self.shape.n * 2,
        };
    }

    pub fn reference(_: *Round, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The host's bytes (core/tree_round's rowArgmax and oracle, the kept rows); 0xff where nothing writes.
    pub fn check(self: *Round, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try self.rig.outputBytes();
        const want = try a.alloc(u8, got.len);
        defer a.free(want);
        @memset(want, 0xff);
        const n = self.shape.rows;
        switch (self.which) {
            .argmax => for (0..n) |r| {
                const p = ref.rowArgmax(self.words[r * self.stride() ..][0..self.shape.k]);
                @memcpy(want[r * 8 ..][0..8], std.mem.asBytes(&p));
            },
            .match => for (0..n) |t| {
                const w = try self.round.?.want(t, rt.vocab_small);
                @memcpy(want[t * @sizeOf(tr.Result) ..][0..@sizeOf(tr.Result)], std.mem.asBytes(&w));
            },
            .keep => {
                const keep = std.mem.bytesAsSlice(i32, want[0 .. n * tr.max_rows * 4]);
                const counts = std.mem.bytesAsSlice(u32, want[n * tr.max_rows * 4 ..]);
                for (self.results, 0..) |*r, t| {
                    const kept = keptRows(r);
                    counts[t] = @intCast(kept);
                    for (0..tr.max_rows) |i| keep[t * tr.max_rows + i] = if (i < kept) @intCast(self.round.?.trees.items[t].first + r.path[i]) else -1;
                }
            },
            .gather, .head => {
                const w = self.shape.n;
                const words = std.mem.bytesAsSlice(u16, want);
                for (self.results, 0..) |*r, t| {
                    const kept = keptRows(r);
                    const first = self.round.?.trees.items[t].first;
                    if (self.which == .gather) {
                        for (0..kept) |i| @memcpy(words[(t * tr.max_rows + i) * w ..][0..w], self.words[(first + r.path[i]) * w ..][0..w]);
                    } else if (kept > 0) @memcpy(words[t * w ..][0..w], self.words[(first + r.path[kept - 1]) * (w + pad) ..][0..w]);
                }
            },
        }
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} is 0x{x:0>2} where the host gives 0x{x:0>2}", .{ at, got.len, got[at], want[at] });
        return null;
    }
};

/// A tree's kept round rows: its verified path, for an ok result that consumed any.
fn keptRows(r: *const tr.Result) usize {
    return if (r.status == 0 and r.consumed_count > 0) r.consumed_count else 0;
}
