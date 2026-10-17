//! Option types the block and forward options carry: the fused dense prefill GEMM's choices, the x3gm v2 mode.
const std = @import("std");
const dk = @import("dsv41_kernels");

const pfd = dk.ops.pfd;

/// TF_DSV41_GM_V2 (gm2pf.zig): prefill's routed experts on x3gm v2; off: v1. The same bits either way.
pub const GmMode = enum { off, one, split };

/// The SMs the dense prefill heuristic prices waves on, on every GPU (the pick never changes a bit).
pub const pfd_sms: usize = 48;

/// The fused dense prefill GEMM's choices (pfdense.pick): an override, the JSON table, else the heuristic.
pub const PfDense = struct {
    entries: []const Entry = &.{},
    /// TF_DSV41_PF_DENSE_CFG: (id, group) for every shape
    override: ?[2]u32 = null,

    pub const Entry = struct { k: u32, n: u32, cfg: u32, group: u32 = 8, small: ?u32 = null, small_rows: u32 = 0 };

    /// pfdense.why_not's shape and width rules (the codebook is mul1 for every dense group of our packs).
    pub fn eligible(k: i64, n: i64, k2: u32, lanes: bool) bool {
        const widths: []const u32 = if (lanes) &.{ 8, 10, 12 } else &.{ 8, 10, 12, 16 };
        if (std.mem.indexOfScalar(u32, widths, k2) == null) return false;
        return k > 0 and n > 0 and @rem(k, 128) == 0 and @rem(n, 128) == 0;
    }

    /// (cfg, group) for an [m, k] @ [k, n] call: override, table, else heuristic (also when a stage does not divide K).
    pub fn pick(p: *const PfDense, k: usize, n: usize, k2: u32, m: usize) [2]u32 {
        var got: ?[2]u32 = p.override;
        if (got == null) for (p.entries) |e| {
            if (e.k != k or e.n != n) continue;
            const c = if (e.small != null and m <= e.small_rows) e.small.? else e.cfg;
            got = .{ c, e.group };
            break;
        };
        if (got) |g| if (g[0] < pfd.cfgs.len and k % (16 * pfd.cfgs[g[0]][3]) == 0) return g;
        const h = pfd.heuristic(k, n, k2, m, pfd_sms);
        return .{ @intCast(h[0]), @intCast(h[1]) };
    }

    /// pfdense._table_file: `{"K,N": {"cfg": id, "group": g, "small": id, "small_rows": r}}`.
    pub fn parseTable(a: std.mem.Allocator, json: []const u8) ![]const Entry {
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadTable;
        if (parsed != .object) return error.BadTable;
        const out = try a.alloc(Entry, parsed.object.count());
        var it = parsed.object.iterator();
        var i: usize = 0;
        while (it.next()) |kv| : (i += 1) {
            var ks = std.mem.tokenizeScalar(u8, kv.key_ptr.*, ',');
            const k = std.fmt.parseInt(u32, std.mem.trim(u8, ks.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            const n = std.fmt.parseInt(u32, std.mem.trim(u8, ks.next() orelse return error.BadTable, " "), 10) catch return error.BadTable;
            if (kv.value_ptr.* != .object) return error.BadTable;
            const o = kv.value_ptr.object;
            out[i] = .{ .k = k, .n = n, .cfg = try field(o, "cfg") orelse return error.BadTable };
            if (try field(o, "group")) |g| out[i].group = g;
            out[i].small = try field(o, "small");
            out[i].small_rows = (try field(o, "small_rows")) orelse 0;
        }
        return out;
    }

    fn field(o: std.json.ObjectMap, name: []const u8) !?u32 {
        const v = o.get(name) orelse return null;
        return switch (v) {
            .integer => |x| std.math.cast(u32, x) orelse error.BadTable,
            else => error.BadTable,
        };
    }

    pub fn parseOverride(raw: []const u8) ![2]u32 {
        var it = std.mem.tokenizeScalar(u8, raw, ',');
        const id = std.fmt.parseInt(u32, std.mem.trim(u8, it.next() orelse return error.BadKnob, " "), 10) catch return error.BadKnob;
        const g: u32 = if (it.next()) |x| std.fmt.parseInt(u32, std.mem.trim(u8, x, " "), 10) catch return error.BadKnob else 8;
        if (it.next() != null or id >= pfd.cfgs.len or g < 1 or g > 1024) return error.BadKnob;
        return .{ id, g };
    }
};
