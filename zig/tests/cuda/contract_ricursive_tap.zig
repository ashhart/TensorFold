//! Ripple's CUDA tap in the contract: tf_rc_tap's chunks and rounds into the slots' rings, bit for bit, and controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rc = @import("ricursive_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig"); // the family-neutral Gated wrapper and fills

const c = core.contract.spec;
const g = core.contract.gates;
const kern = rc.kernels;
const Kernels = kern.Kernels;
const Function = cuda.Function;

const hidden = @import("contract_ricursive_glue.zig").hidden;

const tap_ring = rc.model.tap_rows; // 2,064 rows a slot
const tap_cols = 5; // the drafter's tapped layers (1, 20, 39, 58, 77)
const tap_col = 2; // the column the registrations write
const tap_streams = 4; // a round's streams: row r stream r % 4's, each in one of 5 shuffled slots (the fifth never named)
const tap_apart = tap_ring + 1; // the streams' last positions apart: a ring row further round each
const tap_nan = "a bf16 copy: NaN patterns and both infinities land in the ring as stored";
const tap_tile: c.Tile = .{ .size = 2048, .step = 8, .min = 8 }; // a block's 256 chunks of 8 values

/// tf_rc_tap's calls: a prompt chunk (one slot, positions from p0) or a round (a position and a slot a row).
pub const TapForm = enum { chunk, round };
/// A control's planted fault (ricursive.cu's tap bits).
pub const TapFault = enum { column, ring, swap };

/// Chunk: rows x width x the first position, in slot 1 of 2 (slot 0 never named).
const tap_chunk_served = [_]c.Shape{
    .{ .rows = 16, .n = hidden, .k = tap_ring - 8 }, // across the wrap: ring rows 2,056 to 2,063, then 0 to 7
    .{ .rows = 1, .n = hidden, .k = tap_ring - 1 },
    .{ .rows = 333, .n = hidden, .k = 4000 },
    .{ .rows = 2048, .n = hidden, .k = 1000 },
    .{ .rows = 2048, .n = hidden, .k = 262_144 - 2048 }, // the longest context's last chunk
};

/// Round: rows x width x the first stream's last position + 1 (the others end 2,065, 4,130 and 6,195 below it).
const tap_round_served = [_]c.Shape{
    .{ .rows = 64, .n = hidden, .k = 4 * tap_ring + 2 }, // verify windows of 16 ending at ring rows 1, 0, 2,063, 2,062
    .{ .rows = 4, .n = hidden, .k = 4 * tap_ring + 2 }, // a decode round: a row a stream
    .{ .rows = 7, .n = hidden, .k = 3 * tap_apart + 3 }, // windows of 2, 2, 2 and 1, the last stream's row at position 2
    .{ .rows = 128, .n = hidden, .k = 100_007 }, // the backend's widest round
    .{ .rows = 16, .n = hidden, .k = 262_144 }, // the longest context checked
};

pub fn tapSpec(form: TapForm, fault: ?TapFault) c.Spec {
    const name = if (fault) |f| switch (f) {
        .column => "control: cuda ricursive tf_rc_tap into the next tap column",
        .ring => "control: cuda ricursive tf_rc_tap one ring row past pos % 2,064",
        .swap => "control: cuda ricursive tf_rc_tap with rows 0 and 1 swapped",
    } else switch (form) {
        .chunk => "cuda ricursive tf_rc_tap, a prompt chunk (one slot from p0) into 2,064-row rings",
        .round => "cuda ricursive tf_rc_tap, a round (4 streams' rows, a position and a slot each) into 2,064-row rings",
    };
    const served: []const c.Shape = switch (form) {
        .chunk => &tap_chunk_served,
        .round => &tap_round_served,
    };
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .nan = .{ .stated = tap_nan }, .tiles = .{ .{}, tap_tile, .{} }, .served = served };
}

/// Rows stream i holds in a round of `rows`: rows i, i + 4, ...
fn tapStreamRows(rows: usize, i: usize) usize {
    return (rows + tap_streams - 1 - i) / tap_streams;
}

/// Rows into tap column 2 of 5 at ring row pos % 2,064 of each row's slot, exactly; every other ring row keeps poison.
pub const Tap = struct {
    kern: *const Kernels,
    f: Function,
    form: TapForm,
    fault: ?TapFault = null,
    shape: c.Shape = undefined,
    resid: []u16 = &.{},
    at: []i32 = &.{}, // row r's position
    of: []i32 = &.{}, // row r's slot
    slots: usize = 0,
    dev: [3]u64 = @splat(0), // R, then a round's positions and slots
    out: u64 = 0,

    pub fn spec(self: *const Tap) c.Spec {
        return tapSpec(self.form, self.fault);
    }

    pub fn control(self: *const Tap) ?g.Gate {
        return if (self.fault != null) .shapes else null;
    }

    pub fn free(self: *Tap, a: std.mem.Allocator) void {
        a.free(self.resid);
        a.free(self.at);
        a.free(self.of);
        self.resid = &.{};
        self.at = &.{};
        self.of = &.{};
    }

    /// R in [-4, 4); non-finite: NaN, +Inf and -Inf in its first rows.
    pub fn prepare(self: *Tap, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const rows = shape.rows;
        const n = shape.n;
        const round = self.form == .round;
        if (rows == 0 or n == 0 or n % 8 != 0) return false;
        if (round) {
            const most = tapStreamRows(rows, 0);
            if (most > tap_ring or shape.k < (tap_streams - 1) * tap_apart + most) return false;
        } else if (rows > tap_ring) return false;
        self.free(a);
        self.shape = shape;
        self.resid = try ref.bfs(a, rng, rows * n, -4, 4);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |x, r| if (r < rows) {
            self.resid[r * n + (7 * r + 1) % n] = c.toBf16(x);
        };
        self.at = try a.alloc(i32, rows);
        self.of = try a.alloc(i32, rows);
        if (round) {
            self.slots = tap_streams + 1;
            var ids: [tap_streams + 1]i32 = undefined;
            for (&ids, 0..) |*x, i| x.* = @intCast(i);
            rng.shuffle(i32, &ids); // stream i in slot ids[i]
            for (self.at, self.of, 0..) |*x, *s, r| {
                const i = r % tap_streams;
                x.* = @intCast(shape.k - i * tap_apart - tapStreamRows(rows, i) + r / tap_streams);
                s.* = ids[i];
            }
        } else {
            self.slots = 2;
            for (self.at, self.of, 0..) |*x, *s, r| {
                x.* = @intCast(shape.k + r);
                s.* = 1;
            }
        }
        self.dev = .{ try ref.input(rig, self.resid), 0, 0 };
        if (round) self.dev[1..].* = .{ try ref.input(rig, self.at), try ref.input(rig, self.of) };
        self.out = try rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *Tap, rig: *rig_.Rig) !void {
        const chunk = self.form == .chunk;
        try kern.tapBy(self.f, rig.s, .{ .resid = self.dev[0], .ring = self.out, .pos = self.dev[1], .slot = self.dev[2], .p0 = if (chunk) self.shape.k else 0, .s0 = if (chunk) 1 else 0, .rows = self.shape.rows, .width = self.shape.n, .taps = tap_cols, .col = tap_col, .ring_rows = tap_ring });
    }

    pub fn count(self: *const Tap) usize {
        return self.slots * tap_ring * tap_cols * self.shape.n;
    }

    pub fn check(self: *const Tap, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const want = try a.alloc(u16, self.count());
        defer a.free(want);
        @memset(want, 0xffff); // the output's poison where no row goes
        for (self.at, self.of, 0..) |p, s, r| {
            const row = (@as(usize, @intCast(s)) * tap_ring + @as(usize, @intCast(p)) % tap_ring) * tap_cols + tap_col;
            @memcpy(want[row * n ..][0..n], self.resid[r * n ..][0..n]);
        }
        const bytes = std.mem.sliceAsBytes(want);
        if (got.len != bytes.len) return try std.fmt.bufPrint(msg, "{d} output bytes where {d} are due", .{ got.len, bytes.len });
        const at = std.mem.indexOfDiff(u8, got, bytes) orelse return null;
        const e = at / 2;
        const row = e / n;
        return try std.fmt.bufPrint(msg, "slot {d} ring row {d} column {d} value {d}: {x:0>4} where {x:0>4} is due", .{ row / tap_cols / tap_ring, row / tap_cols % tap_ring, row % tap_cols, e % n, std.mem.readInt(u16, got[e * 2 ..][0..2], .little), want[e] });
    }
};
