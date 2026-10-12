//! DSpark's statics layout, ring rows and blocks, and the overlapped ingest's renamed scratch roles.

const std = @import("std");
const calls = @import("block_calls.zig");
const config = @import("config.zig");
const Config = config.Config;
const Arg = calls.Arg;
const buffers = @import("buffers.zig");

/// The rows of one ring (a power of two holding the window and a block).
pub fn ringRows(cfg: *const Config) i64 {
    return @intCast(std.math.ceilPowerOfTwo(u32, cfg.window + cfg.dspark_block) catch unreachable);
}

/// The DSpark blocks' layer indices (after the backbone).
pub fn blocks(cfg: *const Config, out: []u32) []u32 {
    const k = @min(out.len, cfg.mtp_layers);
    for (out[0..k], 0..) |*b, i| b.* = cfg.layers + @as(u32, @intCast(i));
    return out[0..k];
}

/// "w.ds.i64" / "w.ds.i32" element offsets for one slot of `n` rows; `lay`: a several-slot pass (`Layout.of`).
pub const Statics = struct {
    n: i64,
    window: i64,
    lay: ?Layout = null,
    pub fn ids(s: Statics) i64 {
        return if (s.lay) |l| l.ids else 0;
    }
    pub fn positions(s: Statics) i64 {
        return if (s.lay) |l| l.positions else s.n;
    }
    pub fn ipar(s: Statics) i64 {
        return 2 * s.n;
    }
    pub fn readRows(s: Statics) i64 {
        return if (s.lay) |l| l.read else 2 * s.n + 4;
    }
    pub fn writeRows(s: Statics) i64 {
        return if (s.lay) |l| l.write else 3 * s.n + 4;
    }
    pub fn len64(s: Statics) i64 {
        return if (s.lay) |l| l.len64 else 4 * s.n + 4;
    }
    pub fn start(_: Statics) i64 {
        return 0;
    }
    pub fn tokens(s: Statics) i64 {
        return if (s.lay) |l| l.tokens else 1;
    }
    pub fn counts(s: Statics) i64 {
        return if (s.lay) |l| l.counts else 1 + s.n * s.window;
    }
    pub fn lo(s: Statics) i64 {
        return if (s.lay) |l| l.lo else 1 + s.n * s.window + s.n;
    }
    pub fn hi(s: Statics) i64 {
        return if (s.lay) |l| l.hi else 1 + s.n * s.window + 2 * s.n;
    }
    pub fn anchor(s: Statics) i64 {
        return 1 + s.n * s.window + 3 * s.n;
    }
    pub fn fpar(s: Statics) i64 {
        return 2 + s.n * s.window + 3 * s.n;
    }
    pub fn len32(s: Statics) i64 {
        return if (s.lay) |l| l.len32 else 5 + s.n * s.window + 3 * s.n;
    }
};

/// A pass over `slots` slots of `n` rows: arrays back to back, each keeping the one-slot pass's 16-byte residue.
pub const Layout = struct {
    ids: i64,
    positions: i64,
    read: i64,
    write: i64,
    len64: i64,
    tokens: i64,
    counts: i64,
    lo: i64,
    hi: i64,
    len32: i64,
    /// the i32 elements the launches read (staged up to here: the role ends here)
    end32: i64,

    /// The first offset >= `at` with the one-slot offset's residue mod `m` elements.
    fn place(at: i64, one: i64, m: i64) i64 {
        return at + @mod(one - at, m);
    }

    pub fn of(slots: i64, n: i64, window: i64) Layout {
        const one: Statics = .{ .n = n, .window = window };
        const R = slots * n;
        var l: Layout = undefined;
        // DraftInputs' arrays in its order and sizes (the chain's parameters, starts and anchors kept as gaps)
        l.ids = place(0, one.ids(), 2);
        l.positions = place(l.ids + R, one.positions(), 2);
        l.read = place(l.positions + R + 4 * slots, one.readRows(), 2);
        l.write = place(l.read + R, one.writeRows(), 2);
        l.len64 = l.write + R;
        l.tokens = place(slots, one.tokens(), 4);
        l.counts = place(l.tokens + R * window, one.counts(), 4);
        l.lo = place(l.counts + R, one.lo(), 4);
        l.hi = place(l.lo + R, one.hi(), 4);
        l.end32 = l.hi + R;
        l.len32 = l.end32 + 4 * slots;
        return l;
    }
};

/// Several slots' DSpark state: each block's ring holds `ring_slots` stacked rings; ingest from `taps_role`.
pub const Slots = struct { ring_slots: i64 = 1, slot: i64 = 0, taps_role: []const u8 = "w.taps" };

/// TF_DSV41_DRAFT_OVERLAP: `cs` (an ingest) with scratch roles renamed "w.ovl.<role>" so it can run on its own stream.
pub fn sideRoles(a: std.mem.Allocator, cs: []const calls.Call) ![]calls.Call {
    return renameRoles(a, cs, "w.ovl.");
}

/// `cs` with every role sideRole names renamed `prefix` ++ role.
pub fn renameRoles(a: std.mem.Allocator, cs: []const calls.Call, prefix: []const u8) ![]calls.Call {
    const out = try a.dupe(calls.Call, cs);
    for (out) |*c| {
        const args = try a.dupe(calls.Named, c.args);
        for (args) |*x| x.arg = try sideArg(a, x.arg, prefix);
        c.args = args;
    }
    return out;
}

/// A role sideRoles renames.
pub fn sideRole(role: []const u8) bool {
    return buffers.scopeOf(role) != .persistent and !std.mem.eql(u8, role, "w.taps");
}

fn sideArg(a: std.mem.Allocator, x: Arg, prefix: []const u8) !Arg {
    return switch (x) {
        .t => |t| .{ .t = try sideTensor(a, t, prefix) },
        .opaque_table => |t| .{ .opaque_table = try sideTensor(a, t, prefix) },
        .list => |items| blk: {
            const l = try a.alloc(Arg, items.len);
            for (items, l) |y, *z| z.* = try sideArg(a, y, prefix);
            break :blk .{ .list = l };
        },
        else => x,
    };
}

fn sideTensor(a: std.mem.Allocator, t: calls.Tensor, prefix: []const u8) !calls.Tensor {
    var u = t;
    switch (t.role) {
        .buf => |r| if (sideRole(r)) {
            u.role = .{ .buf = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, r }) };
        },
        else => {},
    }
    return u;
}
