//! A stage's boot and rounds on the ring (RFC #2005 v1) for any engine: `Driver(B)` over an engine binding `B`.
const std = @import("std");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const payload = @import("payload.zig");
const Ring = @import("ring.zig").Ring;
const inbox = @import("inbox.zig");
const keepalive = @import("keepalive.zig");
const hops = @import("link.zig");
const Link = hops.Link;
const echoed = hops.echoed;

pub const Error = error{ StreamOpen, NoFreeSlot, UnknownStream, BadAccepted, OtherPosition, ResumeNotInV1, CutsNotInV1, OtherEcho, PromptParts, TooManyStreams, BadDepth, TooManyInFlight, NothingInFlight, StreamInFlight, RoundsInFlight, Unsettled, KeptTwice, StageLost };

/// A stream's unsettled window: its first position and its rows.
pub const Window = struct { start: u32, rows: u32 };

/// The boundary's planes of `rows` rows in host memory, sent or filled in place by the ring.
pub const Hop = hops.Hop;

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
        pub const Slot = struct {
            cache: Cache,
            id: u32 = 0,
            used: bool = false,
            generation: u32 = 0,
            /// Rank 0 only: an open, close or cancel applied here for the next plan.
            queued: ?plan.Op = null,
            /// Rank 0 only: the rows of the last window its caller already kept.
            kept: ?u32 = null,
        };

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
                if (x.kept) |k| { // rank 0 rolled back when its caller kept
                    if (accepted != k) return error.BadAccepted;
                    x.kept = null;
                    x.generation +%= 1;
                    return;
                }
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
                        return if (st.rows > 0) x else null; // an open with no rows runs no forward
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
            gpa: std.mem.Allocator,
            /// Rounds in flight after `pipe`, oldest at `first`; none without it, so `step` runs as before.
            flights: []Flight = &.{},
            first: usize = 0,
            count: usize = 0,
            ib: ?*inbox.Inbox = null,
            /// The keep-alive's clock, and the round it last saw (null: not polled yet).
            beats: keepalive.Beat = .init(0),
            beat_round: ?u64 = null,
            /// The ring error that ended a round after its plan left: every later call is refused `StageLost`.
            lost: ?anyerror = null,

            pub fn init(gpa: std.mem.Allocator, e: *Engine, ring: ?*Ring, parallel: u32, context: u32, first_round: u64) !Lead {
                if (parallel > B.most_streams) return error.TooManyStreams;
                var slots = try Slots.init(gpa, e, parallel, context);
                errdefer slots.deinit(gpa);
                const mem = try gpa.alloc(u8, 2 * plan.planSize(parallel, 0));
                const limits = plan.Limits.of(B.shapes(parallel, context));
                return .{ .e = e, .ring = ring, .slots = slots, .limits = limits, .round = first_round, .mem = mem, .gpa = gpa };
            }

            pub fn deinit(l: *Lead, gpa: std.mem.Allocator) void {
                if (l.ib) |ib| {
                    if (ib.pending() > 0) l.ring.?.stop(); // a failed round: the ring ends, and the thread's wait with it
                    ib.stop();
                    gpa.destroy(ib);
                }
                for (l.flights) |*f| f.free(gpa);
                gpa.free(l.flights);
                l.slots.deinit(gpa);
                gpa.free(l.mem);
            }

            pub fn cache(l: *Lead, id: u32) ?*Cache {
                return if (l.live(id)) |x| &x.cache else null;
            }

            /// The stream's slot unless a close or cancel of it is queued.
            fn live(l: *Lead, id: u32) ?*Slot {
                const x = l.slots.find(id) orelse return null;
                return if (x.queued == null or x.queued.? == .open) x else null;
            }

            /// A stream opened on rank 0 now, its cache returned; the followers open it with the next plan.
            pub fn open(l: *Lead, id: u32) !*Cache {
                try l.alive();
                if (l.live(id) != null) return error.StreamOpen;
                if (l.slots.find(id) != null or (l.free() == null and l.ending())) try l.beat(); // a queued close first
                const x = l.free() orelse return error.NoFreeSlot;
                try B.reset(l.e, &x.cache);
                x.* = .{ .cache = x.cache, .id = id, .used = true, .queued = .open };
                return &x.cache;
            }

            fn free(l: *Lead) ?*Slot {
                for (l.slots.all) |*x| if (!x.used) return x;
                return null;
            }

            fn ending(l: *Lead) bool {
                for (l.slots.all) |x| if (x.used and x.queued != null and x.queued.? != .open) return true;
                return false;
            }

            /// A stream ended on rank 0 now (`op` close or cancel); the followers end it with the next plan.
            pub fn end(l: *Lead, id: u32, op: plan.Op) !void {
                try l.alive();
                if (op != .close and op != .cancel) return error.BadOp;
                const x = l.live(id) orelse return error.UnknownStream;
                if (l.flying(id)) return error.StreamInFlight;
                if (x.queued != null) { // opened since the last plan: the followers never saw it
                    x.* = .{ .cache = x.cache };
                    return;
                }
                x.queued = op;
            }

            /// The stream's last window already kept to `accepted` rows on rank 0; the next plan carries the count.
            pub fn kept(l: *Lead, id: u32, accepted: u32) !void {
                try l.alive();
                const x = l.live(id) orelse return error.UnknownStream;
                if (l.flying(id)) return error.StreamInFlight;
                if (B.window(&x.cache) != null) return error.Unsettled;
                if (x.kept != null) return error.KeptTwice;
                x.kept = accepted;
            }

            /// Every round in flight completed, oldest first.
            pub fn drain(l: *Lead) !void {
                try l.alive();
                while (l.count > 0) try l.complete();
            }

            /// A keep-alive if none of the plans seen by these polls went out within the interval; true if sent.
            pub fn beatIfDue(l: *Lead, now_ns: u64) !bool {
                try l.alive();
                const quiet = l.beat_round == l.round and l.count == 0 and l.beats.due(now_ns);
                if (quiet) try l.beat();
                if (quiet or l.beat_round != l.round) l.beats.sent(now_ns); // a plan since the last poll
                l.beat_round = l.round;
                return quiet;
            }

            /// The plan's streams: `entries` (ascending ids) and the queued opens, closes and cancels, merged by id.
            fn planned(l: *Lead, entries: []const Entry, out: *[B.most_streams]plan.Stream) ![]plan.Stream {
                var q: [B.most_streams]plan.Stream = undefined;
                var nq: usize = 0;
                for (l.slots.all) |x| if (x.used and x.queued != null) {
                    var at = nq;
                    while (at > 0 and q[at - 1].id > x.id) : (at -= 1) q[at] = q[at - 1];
                    q[at] = .{ .id = x.id, .op = x.queued.?, .first = 0, .rows = 0, .generation = x.generation };
                    nq += 1;
                };
                var i: usize = 0;
                var j: usize = 0;
                var n: usize = 0;
                while (i < entries.len or j < nq) : (n += 1) {
                    if (n == out.len) return error.TooManyStreams;
                    if (j == nq or (i < entries.len and entries[i].id < q[j].id)) {
                        out[n] = try l.expect(entries[i]);
                        i += 1;
                    } else if (i == entries.len or q[j].id < entries[i].id) {
                        out[n] = q[j];
                        j += 1;
                    } else { // the queued open's first rows
                        if (q[j].op != .open) return error.UnknownStream;
                        out[n] = try l.expect(entries[i]);
                        out[n].op = .open;
                        i += 1;
                        j += 1;
                    }
                }
                return out[0..n];
            }

            /// Rank 0's own part of a planned stream: a queued op was applied when asked, the rest as every stage does.
            fn own(l: *Lead, r: *Round, st: plan.Stream) !void {
                const x = l.slots.find(st.id) orelse return r.add(&l.slots, st);
                const q = x.queued orelse return r.add(&l.slots, st);
                x.queued = null;
                const go: plan.Stream = .{ .id = st.id, .op = .@"continue", .first = 0, .rows = st.rows };
                if (q == .open) return r.add(&l.slots, go);
                x.* = .{ .cache = x.cache };
            }

            /// One round of `mode` over `entries` (ascending ids): every stage applies them, then one forward of `ids`.
            pub fn step(l: *Lead, entries: []const Entry, ids: []const u32, mode: plan.Mode) !void {
                try l.alive();
                if (l.flights.len > 0) {
                    if (l.count > 0) return error.RoundsInFlight;
                    try l.issue(entries, ids, mode);
                    return l.complete();
                }
                var buf: [B.most_streams]plan.Stream = undefined;
                if (entries.len > buf.len) return error.TooManyStreams;
                const streams = try l.planned(entries, &buf);
                const half = l.mem.len / 2;
                const p = try plan.encodeMode(mode, streams, &.{}, l.limits, l.mem[0..half]);
                defer l.round += 1;
                var r: Round = .{};
                for (streams) |st| try l.own(&r, st); // rank 0 checks what it planned
                const ring = l.ring orelse {
                    if (r.n > 0) try B.forward(l.e, r.parts[0..r.n], ids, mode, null);
                    return;
                };
                const decoded = try plan.decode(p, l.limits);
                ring.send(.plan, l.round, 0, &.{p}) catch |err| return l.fail(err);
                if (r.n == 0) return echoed(ring, l.round, p, l.mem[half..]) catch |err| return l.fail(err);
                var link: Link = .{ .ring = ring, .round = l.round, .bound = B.boundary(l.e), .plan = decoded, .sent = p, .echo = l.mem[half..] };
                B.forward(l.e, r.parts[0..r.n], ids, mode, link.hop()) catch |err| return l.fail(err);
            }

            fn alive(l: *const Lead) !void {
                if (l.lost != null) return error.StageLost;
            }

            /// A round that failed after its plan left: the ring is out of step, so it is ended and the error kept.
            fn fail(l: *Lead, err: anyerror) anyerror {
                if (l.lost == null) l.ring.?.stop(); // the followers see the end at once, not at their timeout
                l.lost = err;
                return err;
            }

            /// The plan's entry for `en`: where the stream stands once its previous window keeps `accepted` rows.
            fn expect(l: *Lead, en: Entry) !plan.Stream {
                var st: plan.Stream = .{ .id = en.id, .op = en.op, .first = 0, .rows = en.rows, .accepted = en.accepted };
                const x = l.slots.find(en.id) orelse return st;
                st.generation = x.generation;
                if (en.op != .@"continue") return st;
                st.first = B.position(&x.cache);
                if (x.kept) |k| { // already kept on rank 0: the count is the caller's keep
                    if (en.accepted != 0 and en.accepted != k) return error.BadAccepted;
                    st.accepted = k;
                    st.generation +%= 1;
                    return st;
                }
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

            /// Up to `depth` rounds in flight: rank 0 sends round r + 1 while the later stages run round r.
            pub fn pipe(l: *Lead, io: std.Io, depth: usize) !void {
                if (depth < 1 or depth > inbox.Inbox.most or l.flights.len > 0) return error.BadDepth;
                const flights = try l.gpa.alloc(Flight, depth);
                var made: usize = 0;
                errdefer {
                    for (flights[0..made]) |*f| f.free(l.gpa);
                    l.gpa.free(flights);
                }
                const size = plan.planSize(l.limits.streams, 0);
                for (flights) |*f| {
                    f.* = .{ .held = try B.hold(l.gpa, l.e) };
                    made += 1;
                    f.ids = try l.gpa.alloc(u32, l.limits.rows);
                    f.sent = try l.gpa.alloc(u8, size);
                    f.echo = try l.gpa.alloc(u8, size);
                }
                if (l.ring) |ring| {
                    const ib = try l.gpa.create(inbox.Inbox);
                    errdefer l.gpa.destroy(ib);
                    ib.* = .{ .io = io, .ring = ring };
                    try ib.start();
                    l.ib = ib;
                }
                l.flights = flights;
            }

            /// One round sent and its streams applied, rank 0's layers run; `complete` takes its rows back in order.
            pub fn issue(l: *Lead, entries: []const Entry, ids: []const u32, mode: plan.Mode) !void {
                try l.alive();
                if (l.count == l.flights.len) return error.TooManyInFlight;
                if (entries.len > B.most_streams) return error.TooManyStreams;
                for (entries) |en| if (l.flying(en.id)) return error.StreamInFlight; // its next plan needs this round's keep
                const f = &l.flights[(l.first + l.count) % l.flights.len];
                var buf: [B.most_streams]plan.Stream = undefined;
                const streams = try l.planned(entries, &buf);
                const p = try plan.encodeMode(mode, streams, &.{}, l.limits, f.sent);
                f.round = l.round;
                f.sent_len = p.len;
                f.mode = mode;
                f.r = .{};
                f.n_ids = ids.len;
                @memcpy(f.ids[0..ids.len], ids);
                for (entries, f.streams[0..entries.len]) |en, *id| id.* = en.id;
                f.n_streams = entries.len;
                defer l.round += 1;
                for (streams) |st| try l.own(&f.r, st);
                const ring = l.ring orelse {
                    l.count += 1; // one engine: the forward runs at `complete`, in order
                    return;
                };
                const decoded = try plan.decode(p, l.limits);
                const bound = B.boundary(l.e);
                const bytes: u64 = if (f.r.n == 0) 0 else (try plan.Rows.of(bound, decoded.rows)).total();
                f.rows = try l.gpa.alloc(u8, @intCast(bytes));
                f.job = .{ .round = l.round, .echo = f.echo, .rows = f.rows };
                // posted before the plan leaves: the last stage may answer at once
                l.ib.?.post(&f.job) catch |err| return l.fail(err);
                l.count += 1;
                ring.send(.plan, l.round, 0, &.{p}) catch |err| return l.fail(err);
                if (f.r.n == 0) return;
                f.link = .{ .ring = ring, .round = l.round, .bound = bound, .plan = decoded, .sent = p, .job = &f.job, .ib = l.ib };
                B.issue(l.e, f.r.parts[0..f.r.n], f.ids[0..ids.len], mode, f.link.hop(), &f.held) catch |err| return l.fail(err);
            }

            /// The oldest round in flight: its rows back from the last stage, then rank 0's head over them.
            pub fn complete(l: *Lead) !void {
                try l.alive();
                if (l.count == 0) return error.NothingInFlight;
                const f = &l.flights[l.first];
                defer {
                    l.gpa.free(f.rows);
                    f.rows = &.{};
                    f.n_streams = 0;
                    l.first = (l.first + 1) % l.flights.len;
                    l.count -= 1;
                }
                const parts = f.r.parts[0..f.r.n];
                if (l.ring == null) return if (f.r.n > 0) B.forward(l.e, parts, f.ids[0..f.n_ids], f.mode, null);
                if (f.r.n > 0) return B.complete(l.e, parts, f.mode, f.link.hop(), &f.held) catch |err| return l.fail(err);
                l.ib.?.wait(&f.job) catch |err| return l.fail(err);
                if (!std.mem.eql(u8, f.echo[0..f.job.echo_len], f.sent[0..f.sent_len])) return l.fail(error.OtherEcho);
            }

            pub fn inFlight(l: *const Lead) usize {
                return l.count;
            }

            fn flying(l: *const Lead, id: u32) bool {
                for (0..l.count) |i| {
                    const f = &l.flights[(l.first + i) % l.flights.len];
                    if (std.mem.indexOfScalar(u32, f.streams[0..f.n_streams], id) != null) return true;
                }
                return false;
            }
        };

        /// One round in flight on rank 0: its plan, parts and ids, the inbox's job and the engine's held state.
        const Flight = struct {
            round: u64 = 0,
            mode: plan.Mode = .window,
            r: Round = .{},
            ids: []u32 = &.{},
            n_ids: usize = 0,
            streams: [B.most_streams]u32 = undefined,
            n_streams: usize = 0,
            sent: []u8 = &.{},
            /// The plan's length in `sent`: the echo must be the whole plan, as `echoed` checks.
            sent_len: usize = 0,
            echo: []u8 = &.{},
            rows: []u8 = &.{},
            link: Link = undefined,
            job: inbox.Job = undefined,
            held: B.Held,

            fn free(f: *Flight, gpa: std.mem.Allocator) void {
                B.unhold(gpa, &f.held);
                for ([_][]u8{ f.sent, f.echo, f.rows }) |b| gpa.free(b);
                gpa.free(f.ids);
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
