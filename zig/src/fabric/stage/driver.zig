//! A stage's boot and rounds on the ring (RFC #2005 v1) for any engine: `Driver(B)` over an engine binding `B`.
const std = @import("std");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const payload = @import("payload.zig");
const Ring = @import("ring.zig").Ring;

pub const Error = error{ StreamOpen, NoFreeSlot, UnknownStream, BadAccepted, OtherPosition, ResumeNotInV1, CutsNotInV1, OtherEcho, PromptParts, TooManyStreams };

/// A stream's unsettled window: its first position and its rows.
pub const Window = struct { start: u32, rows: u32 };

/// The boundary's planes of `rows` rows in host memory, sent or filled in place by the ring.
pub const Hop = struct {
    ctx: *anyopaque,
    give: *const fn (ctx: *anyopaque, rows: usize, bytes: []const u8) anyerror!void,
    take: *const fn (ctx: *anyopaque, rows: usize, bytes: []u8) anyerror!void,
};

/// What rank 0's caller asks of a stream in one round.
pub const Entry = struct { id: u32, op: plan.Op, rows: u32 = 0, accepted: u32 = 0 };

/// The checkpoint's and layout's fingerprint: SHA-256 over config.json, the engine version and every rank's range.
pub fn fingerprint(version: []const u8, config: []const u8, ranges: []const boot.Range) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(config);
    h.update(version);
    for (ranges) |r| {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u32, b[0..4], r.first, .little);
        std.mem.writeInt(u32, b[4..8], r.end, .little);
        h.update(&b);
    }
    return h.finalResult();
}

/// The last stage's copy of rank 0's plan, which must be byte for byte the one sent.
fn echoed(ring: *Ring, round: u64, sent: []const u8, buf: []u8) !void {
    const h = try ring.recv(.plan, round, buf);
    if (!std.mem.eql(u8, buf[0..@intCast(h.bytes)], sent)) return error.OtherEcho;
}

/// The ring's hop for one round: rows to the next rank, rows from the one before (on rank 0, after the plan's echo).
const Link = struct {
    ring: *Ring,
    round: u64,
    bound: boot.Boundary,
    plan: plan.Plan,
    /// Rank 0: the plan it sent, which the last stage's echo must equal.
    sent: ?[]const u8 = null,
    echo: []u8 = &.{},

    fn hop(l: *Link) Hop {
        return .{ .ctx = l, .give = give, .take = take };
    }

    fn give(ctx: *anyopaque, rows: usize, bytes: []const u8) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(ctx));
        try l.ring.send(.rows, l.round, @intCast(rows), &.{bytes});
    }

    fn take(ctx: *anyopaque, rows: usize, bytes: []u8) anyerror!void {
        const l: *Link = @ptrCast(@alignCast(ctx));
        if (l.sent) |p| {
            try echoed(l.ring, l.round, p, l.echo);
            l.sent = null;
        }
        const h = try l.ring.recv(.rows, l.round, bytes);
        const r = try plan.checkRows(h, l.plan, l.bound);
        if (r.rows != rows) return error.OtherRowCount;
    }
};

/// Boot rounds 1 and 2: descriptions, then ready reports, each appended; rank 0 checks all; the next round returned.
pub fn bootRing(gpa: std.mem.Allocator, ring: *Ring, own: *const boot.Description, model: *const boot.Model, ranges: []const boot.Range, backend: payload.Backend) !u64 {
    try boot.checkLayout(model.layers, ranges);
    const world = ring.world;
    const most = world * (boot.describe_fixed + 4 * boot.most_taps + boot.most_version);
    const buf = try gpa.alloc(u8, most);
    defer gpa.free(buf);
    var mine: [boot.describe_fixed + 4 * boot.most_taps + boot.most_version]u8 = undefined;
    const d = try boot.encodeDescription(own, &mine);
    var ready: [boot.ready_size]u8 = undefined;
    const rd = try boot.encodeReady(.{ .rank = own.rank, .loaded = true, .warm = true }, &ready);
    for ([_][]const u8{ d, rd }, 1..) |msg, round| {
        if (ring.rank == 0) {
            try ring.send(.plan, round, 0, &.{msg});
            const h = try ring.recv(.plan, round, buf);
            try checkAll(buf[0..@intCast(h.bytes)], world, model, ranges, backend, round == 1);
        } else {
            const h = try ring.recv(.plan, round, buf);
            try ring.send(.plan, round, 0, &.{ buf[0..@intCast(h.bytes)], msg });
        }
    }
    return 3;
}

/// Every stage's description (or ready report) in rank order, each against rank 0's plan for it.
fn checkAll(all: []const u8, world: u32, model: *const boot.Model, ranges: []const boot.Range, backend: payload.Backend, describes: bool) !void {
    var at: usize = 0;
    for (0..world) |rank| {
        if (describes) {
            if (all.len < at + boot.describe_fixed) return error.BadLength;
            const n = boot.describe_fixed + 4 * @as(usize, all[at + 20]) + all[at + 21];
            if (all.len < at + n) return error.BadLength;
            const desc = try boot.decodeDescription(all[at..][0..n]);
            const want: boot.Expect = .{ .model = model, .rank = @intCast(rank), .world = world, .backend = backend, .range = ranges[rank] };
            try boot.check(want, &desc);
            at += n;
        } else {
            if (all.len < at + boot.ready_size) return error.BadLength;
            const r = try boot.decodeReady(all[at..][0..boot.ready_size]);
            if (r.rank != rank or !r.ready()) return error.NotReady;
            at += boot.ready_size;
        }
    }
    if (at != all.len) return error.BadLength;
}

/// The slots and rounds of a stage whose engine binding is `B`; `Fake` in driver_test.zig documents each decl.
pub fn Driver(comptime B: type) type {
    return struct {
        const Engine = B.Engine;
        const Cache = B.Cache;

        /// One stream's state on this stage: its cache (its own layers only) and its state generation.
        pub const Slot = struct { cache: Cache, id: u32 = 0, used: bool = false, generation: u32 = 0 };

        /// A stage's streams: the same code on rank 0 and on every follower, so each applies a plan's keeps alike.
        pub const Slots = struct {
            e: *Engine,
            all: []Slot,

            pub fn init(gpa: std.mem.Allocator, e: *Engine, n: usize, context: u32) !Slots {
                const all = try gpa.alloc(Slot, n);
                var made: usize = 0;
                errdefer {
                    for (all[0..made]) |*s| B.close(gpa, &s.cache);
                    gpa.free(all);
                }
                for (all) |*s| {
                    s.* = .{ .cache = try B.open(gpa, e, context) };
                    made += 1;
                }
                return .{ .e = e, .all = all };
            }

            pub fn deinit(s: *Slots, gpa: std.mem.Allocator) void {
                for (s.all) |*x| B.close(gpa, &x.cache);
                gpa.free(s.all);
            }

            pub fn find(s: *Slots, id: u32) ?*Slot {
                for (s.all) |*x| if (x.used and x.id == id) return x;
                return null;
            }

            /// The stream's previous window kept to its first `accepted` rows (a keep advances the generation).
            fn settle(s: *Slots, x: *Slot, accepted: u32) !void {
                const w = B.window(&x.cache) orelse return if (accepted != 0) error.BadAccepted;
                if (accepted > w.rows) return error.BadAccepted;
                try B.rollback(s.e, &x.cache, accepted);
                x.generation +%= 1;
            }

            /// One stream entry before the round's forward: opened, kept and checked, or closed; its slot with rows.
            pub fn apply(s: *Slots, st: plan.Stream) !?*Slot {
                switch (st.op) {
                    .open => {
                        if (s.find(st.id) != null) return error.StreamOpen;
                        const x = for (s.all) |*x| {
                            if (!x.used) break x;
                        } else return error.NoFreeSlot;
                        try B.reset(s.e, &x.cache);
                        x.* = .{ .cache = x.cache, .id = st.id, .used = true };
                        try plan.sameGeneration(x.generation, st);
                        return x;
                    },
                    .@"continue" => {
                        const x = s.find(st.id) orelse return error.UnknownStream;
                        try s.settle(x, st.accepted);
                        try plan.sameGeneration(x.generation, st);
                        if (B.position(&x.cache) != st.first) return error.OtherPosition;
                        return if (st.rows > 0) x else null;
                    },
                    .close, .cancel => {
                        const x = s.find(st.id) orelse return error.UnknownStream;
                        try plan.sameGeneration(x.generation, st);
                        x.used = false;
                        return null;
                    },
                    .@"resume" => return error.ResumeNotInV1, // a prompt cache across stages is step 7
                }
            }
        };

        /// A plan's forward parts, in stream order.
        const Round = struct {
            parts: [B.most_streams]B.Part = undefined,
            n: usize = 0,

            fn add(r: *Round, s: *Slots, st: plan.Stream) !void {
                const x = try s.apply(st) orelse return;
                r.parts[r.n] = .{ .cache = &x.cache, .rows = st.rows };
                r.n += 1;
            }
        };

        /// This stage's boot description.
        pub fn describe(e: *const Engine, rank: u32, world: u32, range: boot.Range, parallel: u32, context: u32) !boot.Description {
            return .{
                .rank = rank,
                .world = world,
                .version = try boot.Text.init(B.version),
                .backend = B.backend,
                .kv_dtype = B.kvDtype(e),
                .range = range,
                .boundary = B.boundary(e),
                .shapes = B.shapes(parallel, context),
                .taps = .{},
                .kv_rows = context,
                .states = parallel,
            };
        }

        /// Rank 0: the plan of each round to the ring, its own streams applied alike, then the forward with the hop.
        pub const Lead = struct {
            e: *Engine,
            /// null: one engine, no ring (the whole-model reference runs the same rounds).
            ring: ?*Ring,
            slots: Slots,
            limits: plan.Limits,
            round: u64,
            mem: []u8,

            pub fn init(gpa: std.mem.Allocator, e: *Engine, ring: ?*Ring, parallel: u32, context: u32, first_round: u64) !Lead {
                if (parallel > B.most_streams) return error.TooManyStreams;
                var slots = try Slots.init(gpa, e, parallel, context);
                errdefer slots.deinit(gpa);
                const mem = try gpa.alloc(u8, 2 * plan.planSize(parallel, 0));
                const limits = plan.Limits.of(B.shapes(parallel, context));
                return .{ .e = e, .ring = ring, .slots = slots, .limits = limits, .round = first_round, .mem = mem };
            }

            pub fn deinit(l: *Lead, gpa: std.mem.Allocator) void {
                l.slots.deinit(gpa);
                gpa.free(l.mem);
            }

            pub fn cache(l: *Lead, id: u32) ?*Cache {
                return if (l.slots.find(id)) |x| &x.cache else null;
            }

            /// One round of `mode` over `entries` (ascending ids): every stage applies them, then one forward of `ids`.
            pub fn step(l: *Lead, entries: []const Entry, ids: []const u32, mode: plan.Mode) !void {
                var streams: [B.most_streams]plan.Stream = undefined;
                if (entries.len > streams.len) return error.TooManyStreams;
                for (entries, streams[0..entries.len]) |en, *st| st.* = l.expect(en);
                const half = l.mem.len / 2;
                const p = try plan.encodeMode(mode, streams[0..entries.len], &.{}, l.limits, l.mem[0..half]);
                defer l.round += 1;
                var r: Round = .{};
                for (streams[0..entries.len]) |st| try r.add(&l.slots, st); // rank 0 checks what it planned
                const ring = l.ring orelse {
                    if (r.n > 0) try B.forward(l.e, r.parts[0..r.n], ids, mode, null);
                    return;
                };
                try ring.send(.plan, l.round, 0, &.{p});
                if (r.n == 0) return echoed(ring, l.round, p, l.mem[half..]);
                const decoded = try plan.decode(p, l.limits);
                var link: Link = .{ .ring = ring, .round = l.round, .bound = B.boundary(l.e), .plan = decoded, .sent = p, .echo = l.mem[half..] };
                try B.forward(l.e, r.parts[0..r.n], ids, mode, link.hop());
            }

            /// The plan's entry for `en`: where the stream stands once its previous window keeps `accepted` rows.
            fn expect(l: *Lead, en: Entry) plan.Stream {
                var st: plan.Stream = .{ .id = en.id, .op = en.op, .first = 0, .rows = en.rows, .accepted = en.accepted };
                const x = l.slots.find(en.id) orelse return st;
                st.generation = x.generation;
                if (en.op != .@"continue") return st;
                st.first = B.position(&x.cache);
                if (B.window(&x.cache)) |w| {
                    st.first = w.start + en.accepted;
                    st.generation +%= 1;
                }
                return st;
            }

            /// A keep-alive: a plan with no streams, which every stage forwards and none acts on.
            pub fn beat(l: *Lead) !void {
                try l.step(&.{}, &.{}, .window);
            }
        };

        /// A follower's rounds until rank 0 stops the ring: each plan passed on, its streams applied, its rows run.
        pub fn serve(gpa: std.mem.Allocator, e: *Engine, ring: *Ring, parallel: u32, context: u32, first_round: u64) !void {
            var slots = try Slots.init(gpa, e, parallel, context);
            defer slots.deinit(gpa);
            const limits = plan.Limits.of(B.shapes(parallel, context));
            const buf = try gpa.alloc(u8, plan.planSize(parallel, 0));
            defer gpa.free(buf);
            const zeros = try gpa.alloc(u32, limits.rows); // later stages read no token ids
            defer gpa.free(zeros);
            @memset(zeros, 0);
            var round = first_round;
            while (true) : (round += 1) {
                const h = ring.recv(.plan, round, buf) catch |err| return if (err == error.Stopped) {} else err;
                const bytes = buf[0..@intCast(h.bytes)];
                const p = try plan.decode(bytes, limits);
                try ring.send(.plan, round, 0, &.{bytes}); // on round, so the last stage echoes it to rank 0
                if (p.cuts != 0) return error.CutsNotInV1;
                var r: Round = .{};
                for (0..p.streams) |i| try r.add(&slots, p.stream(i));
                if (r.n == 0) continue;
                if (p.mode == .prompt and r.n != 1) return error.PromptParts;
                var link: Link = .{ .ring = ring, .round = round, .bound = B.boundary(e), .plan = p };
                try B.forward(e, r.parts[0..r.n], zeros[0..p.rows], p.mode, link.hop());
            }
        }
    };
}

test {
    _ = @import("driver_test.zig");
}
