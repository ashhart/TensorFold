//! Tree gates without a trained drafter: `TreeOracle` drafts the serial decode with a sibling that loses or wins.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const ln = @import("lanes.zig");
const dspark = @import("dspark.zig");
const tree = @import("tree.zig");
const costs_mod = @import("costs.zig");

pub const TreeOracle = struct {
    serial: []const u32,
    prompt_len: u64,
    vocab: u32,
    /// zero Markov heads: a candidate's draft logit is its base logit
    w1: []u16,
    w2: []u16,
    round: u64 = 0,
    off_anchor: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, serial: []const u32, prompt_len: u64, vocab: u32) !TreeOracle {
        const w1 = try gpa.alloc(u16, vocab);
        errdefer gpa.free(w1);
        const w2 = try gpa.alloc(u16, vocab);
        @memset(w1, 0);
        @memset(w2, 0);
        return .{ .serial = serial, .prompt_len = prompt_len, .vocab = vocab, .w1 = w1, .w2 = w2 };
    }

    pub fn deinit(x: *TreeOracle, gpa: std.mem.Allocator) void {
        gpa.free(x.w1);
        gpa.free(x.w2);
    }

    pub fn pass(x: *TreeOracle) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }
    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov };
    fn ingest(_: *anyopaque, _: u32, _: u64, _: iface.Taps, _: []const u32) anyerror!void {}
    fn reset(_: *anyopaque, _: u32) void {}

    fn self(p: *anyopaque) *TreeOracle {
        return @ptrCast(@alignCast(p));
    }

    fn markov(p: *anyopaque) ?dspark.Markov {
        const x = self(p);
        return .{ .w1 = x.w1, .w2 = x.w2, .rank = 1 };
    }

    fn at(x: *const TreeOracle, pos: u64) u32 {
        if (pos < x.prompt_len) return x.vocab - 1;
        const j = pos - x.prompt_len;
        return if (j < x.serial.len) x.serial[@intCast(j)] else x.vocab - 1;
    }

    /// Round r: r % 4 == 3 corrupts draft 0 (sibling wins), r % 4 == 1 draft (r / 4) % block; candidates [draft, alt].
    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        for (asks, out) |a, o| {
            if (a.anchor != x.at(a.start)) x.off_anchor += 1;
            const b = o.drafts.len;
            for (o.drafts, o.conf, 0..) |*d, *c, i| {
                d.* = x.at(a.start + 1 + i);
                c.* = 0.9;
            }
            const wrong: ?usize = switch (x.round % 4) {
                1 => @intCast((x.round / 4) % b),
                3 => 0,
                else => null,
            };
            if (wrong) |i| o.drafts[i] = (o.drafts[i] + 1) % x.vocab;
            x.round += 1;
            const cand = o.cand orelse continue;
            const base = o.base.?;
            const c = cand.len / b;
            @memset(cand, -1);
            @memset(base, 0.0);
            for (0..b) |i| {
                const truth = x.at(a.start + 1 + i);
                cand[i * c] = @intCast(o.drafts[i]);
                base[i * c] = 2.0;
                if (c < 2) continue;
                cand[i * c + 1] = @intCast(if (o.drafts[i] != truth) truth else (truth + 1) % x.vocab);
                base[i * c + 1] = 1.9;
            }
        }
    }
};

pub const Result = struct { tokens: []u32, drafted: u64, accepted: u64, trees: u64, sib_wins: u64, rounds: u64 };

/// `max_new` tokens after `prompt` through lanes (one slot, greedy) with tree knobs `set`, default cost table.
pub fn generate(gpa: std.mem.Allocator, target: iface.Target, pass: iface.Pass, shape: dspark.Shape, prompt: []const u32, max_new: u32, set: tree.Settings) !Result {
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, target, pass, c, .{ .shape = shape, .slots = 1, .siblings = set.siblings, .sib_rows = set.sib_rows, .dup = set.dup, .spec = try @import("spec.zig").Settings.fromEnv() });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var s = try lanes.Stream.init(gpa, .{ .id = "tree-gate", .prompt = prompt, .max_new = max_new, .drafts = true });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    x.reportSpec();
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .drafted = e.drafted, .accepted = e.accepted, .trees = x.policy.trees, .sib_wins = x.policy.sib_wins, .rounds = s.rounds };
}
