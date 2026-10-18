//! The learner over a stand-in host and trainer on the CPU: a whole pocket lesson read, fitted, stepped, saved.
const std = @import("std");
const learner = @import("learner.zig");
const families = @import("families.zig");
const fam = @import("family.zig");
const train_mode = @import("train_mode.zig");
const lesson_route = @import("lesson_route.zig");
const clock = @import("pocket_clock.zig");
const neuron = @import("neuron.zig");

const Example = @import("lesson_examples.zig").Example;
const Seq = train_mode.Seq;
const Span = train_mode.Span;
const Result = train_mode.Result;
const d = 64;
const layers = 8;

/// A trainer with the pocket's hooks and nothing of Nemotron's routed blocks (no `sites`).
const Toy = struct {
    spans: [4]Span = undefined,
    pack: usize = 64,
    count: clock.Count = .{},
    prefixes: struct {
        pub const most = 4;
    } = .{},
    nudge: train_mode.Nudge = .{},
    from: [layers]u32 = @splat(std.math.maxInt(u32)),
    rows: std.ArrayList(u64) = .empty, // each pass row's hash of the tokens through it
    gpa: std.mem.Allocator,
    scratch: [d]u16 = undefined,
    push: [d]f32 = undefined,
    delta: [d]f32 = undefined,
    gates: usize = 0, // gate scales read

    pub fn packs(_: *const Toy) bool {
        return false;
    }

    pub fn prefixSlots(_: *Toy, _: usize) !void {}

    fn lay(t: *Toy, seqs: []const Seq) !void {
        t.rows.clearRetainingCapacity();
        for (seqs, 0..) |q, i| {
            t.spans[i] = .{ .at = t.rows.items.len, .rows = q.ids.len - 1 - q.skip, .skip = q.skip, .slot = q.slot };
            var h = std.hash.Wyhash.init(0);
            for (q.ids[0 .. q.ids.len - 1], 0..) |id, r| {
                h.update(std.mem.asBytes(&id));
                if (r >= q.skip) try t.rows.append(t.gpa, h.final());
            }
        }
        t.count.passes += 1;
        t.count.rows += t.rows.items.len;
    }

    pub fn pass(t: *Toy, seqs: []const Seq, mode: train_mode.Mode, _: lesson_route.Turn, out: []Result) !void {
        try t.lay(seqs);
        for (out[0..seqs.len]) |*r| r.* = .{ .loss = if (mode == .keys) 0 else 0.5, .recalled = true };
    }

    pub fn step(t: *Toy, ids: []const u32, start: usize, mode: train_mode.Mode, turn: lesson_route.Turn) !Result {
        var out: [1]Result = undefined;
        try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide }}, mode, turn, &out);
        return out[0];
    }

    /// Row `row`'s input at layer `l`: unit-sized noise from the row's hash, so different contexts read apart.
    pub fn normed(t: *Toy, l: usize, row: usize) []const u16 {
        var prng = std.Random.DefaultPrng.init(t.rows.items[row] ^ (@as(u64, l) *% 0x9e3779b97f4a7c15));
        for (&t.scratch) |*x| x.* = @truncate(@as(u32, @bitCast(1 + prng.random().floatNorm(f32))) >> 16);
        return &t.scratch;
    }

    pub fn residual(t: *Toy, l: usize, row: usize) []const u16 {
        return t.normed(l + 100, row);
    }

    pub fn pushAt(t: *Toy, _: usize, key: usize, l: usize) []const f32 {
        for (&t.push, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt((i + key + l) % 7)) - 0.03;
        return &t.push;
    }

    pub fn deltaAt(t: *Toy, _: usize, _: usize, _: usize) []f32 {
        return &t.delta;
    }

    pub fn gateAt(t: *Toy, _: usize, _: usize) f32 {
        t.gates += 1;
        return 0.25;
    }
};

/// A host over the toy trainer for family `F`, counting what the learner hands it.
fn Host(comptime F: fam.Family) type {
    return struct {
        pub const Trainer = Toy;
        pub const max_rows = 64;
        pub const family = F;
        pub const Ctx = *Seen;

        pub const Seen = struct { fail_save: enum { none, before, partial } = .none, partial: bool = false, dir: []const u8 = &.{}, disk: u64 = 0, uploads: usize = 0, saved: usize = 0, gate_ok: bool = true, up_ok: bool = true, written: []const usize = &.{} };

        pub fn layout(_: Ctx) fam.Layout {
            var every: fam.Layers = .empty;
            for (0..layers) |l| every.set(l);
            return .fromExperts(d, 128, layers, every);
        }

        pub fn trainer(gpa: std.mem.Allocator, _: std.Io, _: Ctx) !*Toy {
            const t = try gpa.create(Toy);
            t.* = .{ .gpa = gpa };
            return t;
        }

        pub fn pocketSaveBegin(s: Ctx, _: std.mem.Allocator, io: std.Io) !void {
            if (s.dir.len > 0) try @import("save_guard").begin(io, s.dir);
        }
        pub fn pocketSaveFinish(s: Ctx, _: std.mem.Allocator, io: std.Io) !void {
            if (s.dir.len > 0) try @import("save_guard").finish(io, s.dir);
        }

        pub fn free(t: *Toy, gpa: std.mem.Allocator) void {
            t.rows.deinit(gpa);
            gpa.destroy(t);
        }

        pub fn seconds(_: std.Io) f64 {
            return 0;
        }

        pub fn pocket(s: Ctx, _: usize, _: []const f32, _: []const f32, _: usize) !void {
            s.uploads += 1;
        }

        pub fn pocketRead(_: Ctx, _: usize, _: []f32, _: []f32) usize {
            return 0;
        }

        pub fn pocketSave(s: Ctx, _: std.mem.Allocator, _: std.Io, written: []const usize, _: usize, up: []f32, _: []f32) !usize {
            s.saved += 1;
            s.written = written;
            try writeImage(s, &.{}, up);
            return 1;
        }

        /// The forward's rows before a save: each gate row sharp^2 times its up row (g = sharp u, w = u / sharp).
        pub fn pocketSwiglu(s: Ctx, _: usize, gate: []const f32, up: []const f32, _: []const f32, _: usize) !void {
            s.uploads += 1;
            if (s.saved > 0) return; // after a save the rows are as written, each rounded its own way
            for (gate, up) |g, w| if (@abs(g - F.neuron.sharp * F.neuron.sharp * w) > 1e-5 * @abs(g) + 1e-30) {
                s.up_ok = false;
            };
        }

        pub fn pocketReadSwiglu(_: Ctx, _: usize, _: []f32, _: []f32, _: []f32) usize {
            return 0;
        }

        fn writeImage(s: Ctx, gate: []const f32, up: []const f32) !void {
            if (s.fail_save == .before) return error.InjectedSave;
            var h = std.hash.Wyhash.init(0);
            h.update(std.mem.sliceAsBytes(gate));
            h.update(std.mem.sliceAsBytes(up));
            const image = h.final();
            s.disk = if (s.fail_save == .partial) image ^ 1 else image;
            if (s.dir.len > 0) {
                var dir = try std.Io.Dir.cwd().openDir(std.testing.io, s.dir, .{});
                defer dir.close(std.testing.io);
                try dir.writeFile(std.testing.io, .{ .sub_path = "saved.bin", .data = std.mem.asBytes(&s.disk) });
            }
            if (s.fail_save == .partial) {
                s.partial = true;
                return error.InjectedSave;
            }
        }

        pub fn pocketSaveSwiglu(s: Ctx, _: std.mem.Allocator, _: std.Io, written: []const usize, _: usize, gate: []f32, up: []f32, _: []f32) !usize {
            s.saved += 1;
            s.written = written;
            try writeImage(s, gate, up);
            for (gate, up) |g, w| if (@abs(g - F.neuron.sharp * F.neuron.sharp * w) > 1e-5 * @abs(g) + 1e-30) {
                s.gate_ok = false;
            };
            // handed back as a writer would: rounded, so the learner takes u from the up rows as written
            return 1;
        }
    };
}

/// A fact's question and answer: token `who` picks whose fact it is, so its rows read apart from others'.
fn qa(gpa: std.mem.Allocator, who: u32, k: u32) !Example {
    const ids = try gpa.dupe(u32, &.{ 1, 2, who, 40 + k, 5, 90 + who, 91, 92 });
    return .{ .ids = ids, .start = 5, .decide = 4 };
}

fn run(comptime F: fam.Family) !void {
    const gpa = std.testing.allocator;
    const H = Host(F);
    var seen: H.Seen = .{};
    var l = learner.Learner(H).init(gpa, std.testing.io, &seen);
    defer l.deinit();
    try std.testing.expect(!l.readsConversations()); // its trainer has no taps
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var train: [6]Example = undefined;
    var held: [2]Example = undefined;
    var other: [8]Example = undefined;
    for (&train, 0..) |*x, k| x.* = try qa(a, 7, @intCast(k));
    for (&held, 0..) |*x, k| x.* = try qa(a, 7, @intCast(10 + k));
    for (&other, 0..) |*x, k| x.* = try qa(a, 20 + @as(u32, @intCast(k)), @intCast(k));
    try l.begin(.{ .train = &train, .held = &held, .keep = &other });
    var report: ?learner.Report = null;
    var steps: usize = 0;
    while (true) : (steps += 1) {
        const s = l.step();
        if (s.report) |r| report = r;
        if (s.done) break;
        try std.testing.expect(steps < 10_000);
    }
    try std.testing.expect(report != null and report.? == .learned);
    try std.testing.expect(report.?.learned.recalled);
    var placed: [64]learner.Learner(H).Placed = undefined;
    const got = l.placed(&placed);
    try std.testing.expect(got.len > 0);
    for (got) |x| try std.testing.expect(x.layer > 0 and x.layer < layers - 1);
    try std.testing.expect(seen.uploads > 0);
    // A reported verdict already used the exact saved form; rounding again changes no weight bit.
    const before = state(&l.pockets);
    try l.pockets.asWritten(l.trainer.?, @import("whole.zig").Hooks(H){ .b = &seen, .gpa = gpa, .io = std.testing.io });
    try std.testing.expectEqual(before, state(&l.pockets));
    // the save: through the family's own hook, every learning layer handed over
    try l.begin(.{ .save = true });
    while (!l.step().done) {}
    try std.testing.expectEqual(@as(usize, 1), seen.saved);
    try std.testing.expectEqual(@as(usize, layers - 1), seen.written.len);
    try std.testing.expect(seen.gate_ok and seen.up_ok);
    if (F.neuron.gated) try std.testing.expect(l.trainer.?.gates > 0);
}

test "a pocket lesson over a stand-in relu^2 host: read, fitted, stepped, judged, saved" {
    try run(families.nemotron_h);
}

test "a pocket lesson over a stand-in gated SwiGLU host: its rows handed over as gate sharp u and up u / sharp, the gate read" {
    try run(@import("families.zig").qwen4_exp);
}

fn state(p: anytype) u64 {
    var h = std.hash.Wyhash.init(0);
    inline for (.{ "gate", "up", "down", "used" }) |f| h.update(std.mem.sliceAsBytes(@field(p, f).items));
    return h.final();
}

fn saveFailure(comptime F: fam.Family, comptime allocate_fails: bool, partial: bool) !void {
    const Observe = struct {
        pub fn accepted(_: @This()) void {}
        pub fn failed(_: @This(), _: bool) !void {}
    };
    return observedSaveFailure(F, allocate_fails, partial, Observe{});
}
pub fn observedSaveFailure(comptime F: fam.Family, comptime allocate_fails: bool, partial: bool, observer: anytype) !void {
    const a = std.testing.allocator;
    const H = Host(F);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    var seen: H.Seen = .{ .dir = dir };
    var l = learner.Learner(H).init(a, std.testing.io, &seen);
    defer l.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const al = arena.allocator();
    var examples: [6]Example = undefined;
    for (&examples, 0..) |*ex, i| ex.* = try qa(al, 7, @intCast(i));
    const held = [_]Example{ try qa(al, 7, 10), try qa(al, 7, 11) };
    try l.begin(.{ .train = &examples, .held = &held });
    var graph_learned = false;
    for (0..10000) |_| {
        const step = l.step();
        if (step.report) |report| if (report == .learned) {
            graph_learned = report.learned.recalled;
            if (graph_learned) observer.accepted();
        };
        if (step.done) break;
    } else return error.NoVerdict;
    try std.testing.expect(graph_learned and l.kept_rounds > 0 and l.pocketed);
    const accepted = state(&l.pockets);
    seen.fail_save = if (partial) .partial else .before;
    if (allocate_fails) seen.fail_save = .none;
    try l.begin(.{ .save = true });
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    if (allocate_fails) l.pockets.gpa = failing.allocator();
    const failed = l.step();
    l.pockets.gpa = a;
    try observer.failed(failed.stale);
    try std.testing.expect(failed.done and failed.report != null and failed.report.? == .failed);
    try std.testing.expectEqual(accepted, state(&l.pockets));
    try std.testing.expect(graph_learned and failed.stale);
    try std.testing.expectEqual(!allocate_fails and partial, seen.partial);
    try std.testing.expectError(error.CheckpointSaveUncertain, @import("save_guard").check(std.testing.io, dir));
    try std.testing.expectError(error.PocketSaveUncertain, l.begin(.{ .train = &examples, .held = &held }));
    seen.fail_save = .none;
    try l.begin(.{ .save = true });
    const repaired = l.step();
    try std.testing.expect(repaired.done and repaired.report == null and !repaired.stale);
    try std.testing.expectEqual(accepted, state(&l.pockets));
    try @import("save_guard").check(std.testing.io, dir);
    const disk = try tmp.dir.readFileAlloc(std.testing.io, "saved.bin", a, .limited(32));
    defer a.free(disk);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&seen.disk), disk);
    try l.begin(.{ .train = &examples, .held = &held });
    l.abort();
}
test "Save recovery: allocation before save preserves an accepted lesson and blocks new work until retry" {
    inline for (.{ families.qwen4_exp, families.qwen3_5 }) |F| try saveFailure(F, true, false);
}
test "Save recovery: writer error before write preserves an accepted lesson and blocks new work until retry" {
    inline for (.{ families.qwen4_exp, families.qwen3_5 }) |F| try saveFailure(F, false, false);
}
test "Save recovery: partial throwing save preserves the accepted lesson and retry reconciles disk" {
    inline for (.{ families.qwen4_exp, families.qwen3_5 }) |F| try saveFailure(F, false, true);
}

test "Trainer admission: an ungated dense SwiGLU lesson uses the identical judged save gate" {
    try run(@import("families.zig").qwen3_5);
}
