//! The MoE route plan under the contract on HIP: tf_moe_route's members and items, exactly, in each mode it serves.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const moe = hip.moe;

/// Who asks for the plan: a prompt pass (GEMM tile from 33 or 64 rows, or no stacked shared expert), decode, a lane.
pub const Mode = enum { prompt_33, prompt_64, apart, decode, lane };

/// The real launch, or a control's: the shared expert cut at the routed tile, or a start of 33 planned from 64.
pub const Fault = enum { none, shared_cut, start_64 };

pub const Case = struct { mode: Mode, fault: Fault = .none };

/// Each mode, then a control on each rule: a prompt's stacked shared expert whole, and gfx1151's start at 33.
pub const cases = [_]Case{
    .{ .mode = .prompt_33 },
    .{ .mode = .prompt_64 },
    .{ .mode = .apart },
    .{ .mode = .decode },
    .{ .mode = .lane },
    .{ .mode = .prompt_64, .fault = .shared_cut },
    .{ .mode = .prompt_33, .fault = .start_64 },
};

fn none(_: c.Shape) usize {
    return 0;
}

const rule = "int32 ids in and out, nothing to be NaN, infinite or summed: each expert's pairs in pick order, " ++
    "cut into items of its tile in expert order, then empty items of the last expert";

fn name(case: Case) []const u8 {
    return switch (case.fault) {
        .shared_cut => "control: tf_moe_route cuts a prompt pass's stacked shared expert at the routed 8-row tile",
        .start_64 => "control: tf_moe_route plans a prompt pass from 64 rows where the GEMM tile starts at 33",
        .none => switch (case.mode) {
            .prompt_33 => "hip tf_moe_route (a prompt pass, the GEMM tile from 33 rows)",
            .prompt_64 => "hip tf_moe_route (a prompt pass, the GEMM tile from 64 rows)",
            .apart => "hip tf_moe_route (a prompt pass, no shared expert stacked)",
            .decode => "hip tf_moe_route (decode and the MTP draft head)",
            .lane => "hip tf_moe_route (a lane round)",
        },
    };
}

/// Rows around the GEMM tile's starts and its 128 rows: the 35B-A3B's 256 experts and the shared one, top 8.
const prompt = [_]c.Shape{
    .{ .rows = 40, .n = 257, .k = 9 },
    .{ .rows = 9, .n = 257, .k = 9 },
    .{ .rows = 32, .n = 257, .k = 9 },
    .{ .rows = 33, .n = 257, .k = 9 },
    .{ .rows = 56, .n = 257, .k = 9 },
    .{ .rows = 63, .n = 257, .k = 9 },
    .{ .rows = 64, .n = 257, .k = 9 },
    .{ .rows = 127, .n = 257, .k = 9 },
    .{ .rows = 128, .n = 257, .k = 9 },
    .{ .rows = 130, .n = 257, .k = 9 },
    // Flash Next: 512 experts and the shared one, top 10
    .{ .rows = 33, .n = 513, .k = 11 },
    .{ .rows = 41, .n = 513, .k = 11 },
    .{ .rows = 63, .n = 513, .k = 11 },
    .{ .rows = 64, .n = 513, .k = 11 },
};

/// No shared expert stacked: Nemotron's 128 experts, top 6, and the 35B-A3B's 257 ids as a tp rank plans them.
const apart = [_]c.Shape{
    .{ .rows = 40, .n = 128, .k = 6 },
    .{ .rows = 9, .n = 128, .k = 6 },
    .{ .rows = 32, .n = 128, .k = 6 },
    .{ .rows = 33, .n = 128, .k = 6 },
    .{ .rows = 63, .n = 128, .k = 6 },
    .{ .rows = 64, .n = 128, .k = 6 },
    .{ .rows = 130, .n = 128, .k = 6 },
    .{ .rows = 40, .n = 257, .k = 9 },
};

/// Decode rounds and the MTP draft head, a few streams' rows to wide rounds: 8-row items below 64, shared too.
const decode = [_]c.Shape{
    .{ .rows = 8, .n = 257, .k = 9 },
    .{ .rows = 2, .n = 257, .k = 9 },
    .{ .rows = 4, .n = 257, .k = 9 },
    .{ .rows = 16, .n = 257, .k = 9 },
    .{ .rows = 40, .n = 257, .k = 9 },
    .{ .rows = 63, .n = 257, .k = 9 },
    .{ .rows = 64, .n = 257, .k = 9 },
    .{ .rows = 8, .n = 513, .k = 11 },
    .{ .rows = 41, .n = 513, .k = 11 },
};

/// A lane round's rows: 8-row items at every count, the shared expert's too.
const lane = [_]c.Shape{
    .{ .rows = 40, .n = 257, .k = 9 },
    .{ .rows = 2, .n = 257, .k = 9 },
    .{ .rows = 9, .n = 257, .k = 9 },
    .{ .rows = 63, .n = 257, .k = 9 },
    .{ .rows = 64, .n = 257, .k = 9 },
    .{ .rows = 130, .n = 257, .k = 9 },
    .{ .rows = 41, .n = 513, .k = 11 },
};

/// rows: the prompt's or round's rows; n: the experts the plan cuts, a stacked shared one last; k: pairs a row.
pub fn spec(case: Case) c.Spec {
    return .{
        .name = name(case),
        .accum = .none,
        .depth = none,
        .out = .raw,
        .nan = .{ .stated = rule },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "integer ids and counts: there are no sums to round" },
        // rows around the GEMM tile's default start, and experts where the kernel's 64 count segments become 16
        .tiles = .{ .{ .size = 64, .min = 2 }, .{ .size = 256, .min = 16 }, .{} },
        .served = switch (case.mode) {
            .prompt_33, .prompt_64 => &prompt,
            .apart => &apart,
            .decode => &decode,
            .lane => &lane,
        },
    };
}

/// The cut the engine asks of the route in the mode, by moe.zig's rules: the routed experts' tile, the shared one's.
fn asked(mode: Mode, rows: usize) [2]usize {
    const start = moe.card_prompt_from;
    return switch (mode) {
        .prompt_33 => .{ moe.tileOf(rows, false, true, 33), moe.sharedTileOf(rows, false, true) },
        .prompt_64 => .{ moe.tileOf(rows, false, true, 64), moe.sharedTileOf(rows, false, true) },
        .apart => .{ moe.tileOf(rows, false, true, start), moe.tileOf(rows, false, true, start) },
        .decode => .{ moe.tileOf(rows, false, false, start), moe.sharedTileOf(rows, false, false) },
        .lane => .{ moe.tileOf(rows, true, false, start), moe.sharedTileOf(rows, true, false) },
    };
}

/// The cut this contract states: 8-row items below the GEMM tile's start, 128 from it; a prompt's shared expert at 128.
fn wanted(mode: Mode, rows: usize) [2]usize {
    const from: usize = switch (mode) {
        .prompt_33 => 33,
        .prompt_64, .decode => 64,
        .apart => moe.card_prompt_from,
        .lane => std.math.maxInt(usize),
    };
    const tile: usize = if (rows < from) 8 else 128;
    return .{ tile, if (mode == .prompt_33 or mode == .prompt_64) 128 else tile };
}

fn word(bytes: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, bytes[4 * i ..][0..4], .little);
}

/// shape.rows rows of shape.k picks over shape.n experts: the members [pairs] and items [capacity][3] they plan.
pub const Route = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    case: Case,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    picks: []i32 = &.{},
    capacity: usize = 0,
    in: u64 = 0,
    members: u64 = 0,
    items: u64 = 0,
    msg: [240]u8 = undefined,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, case: Case) Route {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .case = case, .arena = .init(a) };
    }

    pub fn deinit(self: *Route) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    /// A control keeps its mode's spec under its own name and must go red at gate 3's exact compare.
    pub fn kernel(self: *Route) g.Kernel {
        const control: ?g.Gate = if (self.case.fault != .none) .shapes else null;
        return .{ .spec = spec(self.case), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Route);

    /// The shared expert stacks as the last id, and every row's last pick.
    fn stacked(self: *const Route) bool {
        return self.case.mode != .apart;
    }

    /// Distinct picks a row, half from six hot experts so some fill several 8-row items; gate 1: the id range's edges.
    pub fn prepare(self: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const hot = 6;
        const routed = if (self.stacked()) shape.n -| 1 else shape.n;
        const top = if (self.stacked()) shape.k -| 1 else shape.k;
        // the launcher plans up to 1,024 experts; the plan's route starts at 2 rows (one row takes the select's)
        if (shape.rows < 2 or shape.n > 1024 or top < 2 or top / 2 > hot or routed < top + hot) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.picks = try a.alloc(i32, shape.rows * shape.k);
        for (0..shape.rows) |r| {
            const row = self.picks[r * shape.k ..][0..shape.k];
            for (row[0..top], 0..) |*e, j| {
                if (fill == .nonfinite) {
                    e.* = @intCast(if (j < top / 2) j else routed - top + j);
                    continue;
                }
                const pool: usize = if (j < top / 2) hot else routed;
                while (true) {
                    e.* = @intCast(rng.uintLessThan(usize, pool));
                    if (std.mem.indexOfScalar(i32, row[0..j], e.*) == null) break;
                }
            }
            if (self.stacked()) row[top] = @intCast(routed);
        }
        self.in = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        // as serving sizes it: the bucket's pairs at the 8-row cut, an upper bound on any plan's items
        self.capacity = moe.capacity(moe.bucket(shape.rows) * shape.k, shape.n);
        self.members = try self.rig.output(self.picks.len * 4);
        self.items = try self.rig.output(self.capacity * 12);
        return true;
    }

    pub fn run(self: *Route) !void {
        const s = self.shape;
        var tile, var shared_tile = asked(self.case.mode, s.rows);
        switch (self.case.fault) {
            .none => {},
            .shared_cut => shared_tile = tile,
            .start_64 => tile = moe.tileOf(s.rows, false, true, 64),
        }
        try self.l.tf_moe_route(
            @ptrFromInt(self.in),
            @intCast(self.picks.len),
            @intCast(s.n),
            @intCast(tile),
            if (self.stacked()) @intCast(s.n - 1) else -1,
            @intCast(shared_tile),
            @ptrFromInt(self.members),
            @ptrFromInt(self.items),
            @intCast(self.capacity),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Route) usize {
        return self.picks.len + 3 * self.capacity;
    }

    pub fn reference(_: *Route, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn say(self: *Route, comptime fmt: []const u8, args: anytype) !?[]const u8 {
        return try std.fmt.bufPrint(&self.msg, fmt, args);
    }

    /// Every member and every item's expert, first and count against the plan this contract states, then the rest.
    pub fn check(self: *Route, _: g.Gate) !?[]const u8 {
        const s = self.shape;
        const members = try self.arena.allocator().dupe(u8, try self.rig.bytes(self.members, self.picks.len * 4));
        const items = try self.rig.bytes(self.items, self.capacity * 12);
        const tile, const shared_tile = wanted(self.case.mode, s.rows);
        var place: usize = 0;
        var slot: usize = 0;
        for (0..s.n) |e| {
            const first = place;
            for (self.picks, 0..) |p, pair| if (p == e) {
                const got = word(members, place);
                if (got != pair) return self.say("member {d} is pair {d}; expert {d}'s in pick order is {d}", .{ place, got, e, pair });
                place += 1;
            };
            const cut = if (self.stacked() and e == s.n - 1) shared_tile else tile;
            var at = first;
            while (at < place) : (slot += 1) {
                if (slot == self.capacity) return self.say("expert {d}'s items run past {d} slots", .{ e, self.capacity });
                const want = [3]usize{ e, at, @min(place - at, cut) };
                for (want, 0..) |v, f| {
                    const got = word(items, 3 * slot + f);
                    if (got != v) return self.say("item {d} field {d} is {d}, the plan's {d} (expert {d}, {d} pairs an item)", .{ slot, f, got, v, e, cut });
                }
                at += want[2];
            }
        }
        for (slot..self.capacity) |x| {
            const e = word(items, 3 * x);
            const pairs = word(items, 3 * x + 2);
            if (e != s.n - 1 or pairs != 0) return self.say("empty item {d} is expert {d} with {d} pairs, not the last with none", .{ x, e, pairs });
        }
        return null;
    }
};
