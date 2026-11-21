//! GLM's decode select (glm_select.metal's mark and merge) against glm_index_select's key lists, as forward.zig picks.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

const top = 512; // index_topk 2048 / kpool 4
const width = 2048 + 4 - 1; // the key list a row: 2,048 tokens and the causal tail
const cap = 3072; // glm_select.metal's CAP and forward.zig's select_cap
const marks = 8192; // forward.zig's select_marks: between the two, glm_index_select alone
const span = 4096;
const SelectArgs = extern struct { p0: u32, top: u32, width: u32, s_stride: u32, i_stride: u32 };

pub const Path = enum { direct, marked };

/// Rows: the window's rows; n: the blocks kept; k: the window's first position.
fn served(path: Path) []const c.Shape {
    return switch (path) {
        .direct => &.{ .{ .rows = 16, .n = top, .k = 8180 }, .{ .rows = 1, .n = top, .k = 0 }, .{ .rows = 1, .n = top, .k = 2044 }, .{ .rows = 3, .n = top, .k = 2046 }, .{ .rows = 16, .n = top, .k = 2040 }, .{ .rows = 5, .n = top, .k = 4097 }, .{ .rows = 2, .n = top, .k = 12286 }, .{ .rows = 16, .n = top, .k = 12275 } },
        .marked => &.{ .{ .rows = 16, .n = top, .k = 65530 }, .{ .rows = 1, .n = top, .k = 32772 }, .{ .rows = 16, .n = top, .k = 32761 }, .{ .rows = 4, .n = top, .k = 40000 }, .{ .rows = 3, .n = top, .k = 120000 }, .{ .rows = 16, .n = top, .k = 131000 }, .{ .rows = 2, .n = top, .k = 262140 }, .{ .rows = 16, .n = top, .k = 262137 }, .{ .rows = 4, .n = top, .k = 300000 } },
    };
}

fn most(s: c.Shape) usize {
    return (s.k + s.rows) / 4;
}

fn stride(s: c.Shape) usize {
    return (s.k + s.rows + 1) / 4 + 3;
}

fn words(s: c.Shape) u32 {
    return tf.glm.state.selectWords(@intCast(s.k + s.rows + 64), 4);
}

pub const Case = struct {
    a: std.mem.Allocator,
    path: Path,
    control: bool,
    mark: mtl.Pipeline,
    merge: mtl.Pipeline,
    old: mtl.Pipeline,

    /// A control's fault: direct rows read each block's score one block late; marked blocks listed one block late.
    const faults = [2][2][]const u8{ .{ "s[t + m * T2]", "s[t + m * T2 + 1]" }, .{ "cblock[at++] = ushort(32 * w + ctz(m))", "cblock[at++] = ushort(32 * w + ctz(m) + 1)" } };

    pub fn init(s: *bits.Suite, path: Path, control: bool) !Case {
        const loader = try s.production();
        const text = try loader.require(sources.glm_select);
        const f = faults[@backingInt(path)];
        const own = if (control) try bits.edited(s.a, text, "void glm_select_merge", f[0], f[1]) else try s.a.dupe(u8, text);
        defer s.a.free(own);
        const mark = try bits.pipeline(s.device, own, "glm_select_mark");
        errdefer mark.deinit();
        const merge = try bits.pipeline(s.device, own, "glm_select_merge");
        errdefer merge.deinit();
        return .{ .a = s.a, .path = path, .control = control, .mark = mark, .merge = merge, .old = try bits.pipeline(s.device, try loader.require(sources.glm_glue), "glm_index_select") };
    }
    pub fn deinit(x: *Case) void {
        x.mark.deinit();
        x.merge.deinit();
        x.old.deinit();
    }
    pub fn name(path: Path, control: bool) []const u8 {
        return switch (path) {
            .direct => if (control) "control: metal glm launch select merge within 3,072 blocks, each score read a block late" else "metal glm launch select merge within 3,072 blocks",
            .marked => if (control) "control: metal glm launch select mark and merge past 8,192 blocks, marks listed a block late" else "metal glm launch select mark and merge past 8,192 blocks",
        };
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.path, x.control), "glm_index_select", "glm_index_select's lists: its key order (-0 below +0, NaNs by their bits), ties to the earliest blocks, the causal tail and -1 padding", served(x.path), "a selection compares keys and sums none");
    }
    pub fn layout(x: *const Case, s: c.Shape) ?bits.Layout {
        const m = most(s);
        const ok = switch (x.path) {
            .direct => m <= cap,
            .marked => m > marks,
        };
        if (!ok or s.rows == 0 or s.rows > 16 or s.n != top) return null;
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * stride(s) * 4, .int);
        _ = l.add(.output, s.rows * width * 4, .int);
        _ = l.add(.scratch, s.rows * @as(usize, words(s)) * 4, .int);
        return l;
    }
    /// Whether glm_select.metal's floor samples block i of n (256 runs of 8 at even spacing).
    fn sampled(i: usize, n: usize) bool {
        var q: usize = 0;
        while (q < 2048) : (q += 8) {
            const at = q / 8 * n / 256 / 8 * 8;
            if (i >= at and i < at + 8) return true;
        }
        return false;
    }
    /// Each row a pattern of its own (ties, specials, ramps, a hot span, a floor that overshoots); gate 1 specials.
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const scores = std.mem.bytesAsSlice(f32, data[0]);
        for (scores) |*v| v.* = rnd.float(f32) * 100 - 50;
        for (0..s.rows) |row| {
            const n = (s.k + row + 1) / 4;
            const pattern = if (f == .nonfinite) 2 else (seed + row) % 9;
            for (scores[row * stride(s) ..][0..n], 0..) |*v, i| v.* = switch (pattern) {
                0 => rnd.float(f32) * 8 - 2,
                1 => @floatFromInt(rnd.uintLessThan(u32, 6)),
                2 => switch (rnd.uintLessThan(u32, 16)) {
                    0 => std.math.nan(f32),
                    1 => -std.math.nan(f32),
                    2 => 0.0,
                    3 => -0.0,
                    4 => std.math.inf(f32),
                    5 => -std.math.inf(f32),
                    else => rnd.float(f32) - 0.5,
                },
                3 => @floatFromInt(i / 3),
                4 => @floatFromInt(n - i / 2),
                5 => if (i >= n / 3 and i < n / 3 + 700) 5.0 + rnd.float(f32) else rnd.float(f32),
                6 => 1.5,
                7 => if (sampled(i, n)) 3.0 + rnd.float(f32) else rnd.float(f32),
                else => @as(f32, @floatFromInt(rnd.intRangeAtMost(i32, -1000, 1000))) / 64,
            };
        }
    }
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const sel = SelectArgs{ .p0 = @intCast(s.k), .top = top, .width = width, .s_stride = @intCast(stride(s)), .i_stride = width };
        const n: usize = s.rows;
        if (!new) {
            e.setPipeline(x.old);
            e.setBuffer(r[0].buf, r[0].off, 0);
            e.setBuffer(r[1].buf, r[1].off, 1);
            e.setValue(sel, 2);
            return e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(1024, 1, 1));
        }
        if (most(s) > cap) { // forward.zig's selectKeys past select_marks: each span's blocks at or above the floor
            e.setPipeline(x.mark);
            e.setBuffer(r[0].buf, r[0].off, 0);
            e.setBuffer(r[2].buf, r[2].off, 1);
            e.setValue(sel, 2);
            e.setValue(words(s), 3);
            e.dispatchGroups(mtl.Size.of((most(s) + span - 1) / span, n, 1), mtl.Size.of(256, 1, 1));
        }
        e.setPipeline(x.merge);
        e.setBuffer(r[0].buf, r[0].off, 0);
        e.setBuffer(r[1].buf, r[1].off, 1);
        e.setValue(sel, 2);
        e.setValue(words(s), 3);
        e.setBuffer(r[2].buf, r[2].off, 4);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(512, 1, 1));
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]Path{ .direct, .marked }) |path| for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(path, control))) try s.add(Case, try Case.init(s, path, control));
    };
}

/// Gate 5's registrations `glm launch select <function>`: glm_select.metal as the loader compiles it.
pub const prefix = "glm launch select ";
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !bits.Texts {
    return .{ .here = sources.glm_select, .pre_m5 = sources.glm_select };
}
