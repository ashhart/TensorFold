//! tree_round.cu's tf_round_pick_ids under the contract (a draft chain's next ids), and a control without the fallback.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const vocab = 262144;
const fallback = 0;
const invalid: u32 = 0xffffffff;

/// A registration: the served kernel (keep null), or the control dropping the fallback.
pub const Entry = struct { name: []const u8, keep: ?i32 = null };

pub const entries = [_]Entry{
    .{ .name = "cuda tree_round tf_round_pick_ids (a draft chain's next ids)" },
    .{ .name = "control: picks to ids without the fallback", .keep = 0 },
};

fn depth(_: c.Shape) usize {
    return 0;
}

const rule = "a pick with no token (NaN or an infinity in its row) or one past the vocabulary gives the fallback id";

pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .none,
        .depth = depth,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 256 }, .{}, .{} },
        // the assistant's hold (up to 32 streams), one row, the DFlash hold's rows
        .served = &.{ .{ .rows = 32, .n = 1, .k = vocab }, .{ .rows = 1, .n = 1, .k = vocab }, .{ .rows = 128, .n = 1, .k = vocab } },
    };
}

pub const PickIds = struct {
    rig: rig_.Rig,
    entry: Entry,
    round: *const cuda.tree_round.Kernels,
    ctl: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    rows: usize = 0,
    picks: []const u32 = &.{},
    in: u64 = 0,
    out: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *PickIds) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(PickIds);

    pub fn kernel(self: *PickIds) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.keep != null) .shapes else null };
    }

    /// Tokens across the vocabulary; a row in five (three: nonfinite fill) has none, a row in eleven is past it.
    pub fn prepare(self: *PickIds, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k != vocab) return false;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const picks = try a.alloc(u32, 2 * shape.rows);
        for (0..shape.rows) |r| {
            const none = r % (if (fill == .nonfinite) @as(usize, 3) else 5) == 1;
            picks[2 * r] = if (none) invalid else if (r % 11 == 4) vocab + rng.uintLessThan(u32, 1000) else rng.uintLessThan(u32, vocab);
            picks[2 * r + 1] = if (none) 1 + rng.uintLessThan(u32, 7) else 0;
        }
        self.rows = shape.rows;
        self.picks = picks;
        self.in = try self.rig.input(std.mem.sliceAsBytes(picks));
        self.out = try self.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(self: *PickIds) !void {
        const s = self.rig.s;
        if (self.ctl) |f| {
            var args: cuda.Args = .{};
            args.add(self.in);
            args.add(self.out);
            args.add(@as(u32, @intCast(self.rows)));
            args.add(@as(u32, vocab));
            args.add(@as(c_int, fallback));
            args.add(@as(c_int, self.entry.keep.?));
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((self.rows + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &args);
        } else try self.round.pickIds(s, self.in, self.out, self.rows, vocab, fallback);
        try s.synchronize();
    }

    pub fn count(self: *PickIds) usize {
        return self.rows * 4;
    }

    pub fn reference(_: *PickIds, out: []f64, mag: []f64) !void {
        @memset(out, 0);
        @memset(mag, 0);
    }

    /// Every id the pick's token, or the fallback where the pick has none or is past the vocabulary, byte for byte.
    pub fn check(self: *PickIds, _: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(i32, try self.rig.bytes(self.out, self.rows * 4));
        for (0..self.rows) |r| {
            const t = self.picks[2 * r];
            const want: i32 = if (t >= vocab) fallback else @intCast(t);
            if (got[r] != want) return try std.fmt.bufPrint(&self.msg, "ids[{d}] = {d} for pick {d}, wants {d}", .{ r, got[r], t, want });
        }
        return null;
    }
};
