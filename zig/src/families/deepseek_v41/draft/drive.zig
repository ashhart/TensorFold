//! One stream through lanes over any `Target` for gates; passes `NoDraft` (drafts off) and `Oracle` (corrupted serial).
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("iface.zig");
const ln = @import("lanes.zig");
const dspark = @import("dspark.zig");
const costs_mod = @import("costs.zig");

pub const Result = struct {
    tokens: []u32,
    drafted: u64,
    accepted: u64,
    /// Oracle only: asks whose anchor was not the serial decode's token there (the stream had already diverged)
    off_anchor: u64 = 0,
};

pub const NoDraft = struct {
    pub fn pass(x: *NoDraft) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }
    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov };
    fn ingest(_: *anyopaque, _: u32, _: u64, _: iface.Taps, _: []const u32) anyerror!void {}
    fn propose(_: *anyopaque, _: []const iface.Ask, _: []iface.Proposal) anyerror!void {
        return error.NoDrafter;
    }
    fn reset(_: *anyopaque, _: u32) void {}
    fn markov(_: *anyopaque) ?dspark.Markov {
        return null;
    }
};

/// Drafts `serial[j]` (position prompt_len + j); round r corrupts draft (r/4)%block if r%4==1, draft 0 if r%4==3.
pub const Oracle = struct {
    serial: []const u32,
    prompt_len: u64,
    vocab: u32,
    round: u64 = 0,
    off_anchor: u64 = 0,

    pub fn pass(x: *Oracle) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }
    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = NoDraft.reset, .markov = NoDraft.markov };
    fn ingest(_: *anyopaque, _: u32, _: u64, _: iface.Taps, _: []const u32) anyerror!void {}

    fn at(x: *const Oracle, pos: u64) u32 {
        if (pos < x.prompt_len) return x.vocab - 1;
        const j = pos - x.prompt_len;
        return if (j < x.serial.len) x.serial[@intCast(j)] else x.vocab - 1;
    }

    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x: *Oracle = @ptrCast(@alignCast(p));
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
        }
    }
};

/// `max_new` tokens after `prompt` through lanes (one slot, greedy), the target's first choice included.
pub fn generate(gpa: std.mem.Allocator, target: iface.Target, pass: iface.Pass, shape: dspark.Shape, prompt: []const u32, max_new: u32, drafts: bool) !Result {
    var c = try costs_mod.defaults(gpa, 64);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, target, pass, c, .{ .shape = shape, .slots = 1, .spec = try @import("spec.zig").Settings.fromEnv() });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    var s = try lanes.Stream.init(gpa, .{ .id = "gate", .prompt = prompt, .max_new = max_new, .drafts = drafts });
    defer s.deinit(gpa);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    x.reportSpec();
    return .{ .tokens = try gpa.dupe(u32, s.emitted()), .drafted = e.drafted, .accepted = e.accepted };
}
