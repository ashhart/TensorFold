//! Lane packs: words tiled, scale pairs group-major, and one run per bit width when a fused stack mixes widths.
const std = @import("std");
const st = @import("../../core/safetensors.zig");
const affine = @import("affine.zig");
const pio = @import("pack_io.zig");
const widen = @import("pack_widen.zig");

/// Which of a quantized linear's three tensors.
pub const Part = enum { weight, scales, biases };

/// A checkpoint quantized linear's three tensors with its verified affine spec.
pub const Q = struct {
    w: st.Tensor,
    s: st.Tensor,
    b: st.Tensor,
    rows: usize,
    k: usize,
    kw: usize,
    spec: affine.Spec,

    pub fn part(r: Q, comptime which: Part) []const u8 {
        return switch (which) {
            .weight => r.w.bytes,
            .scales => r.s.bytes,
            .biases => r.b.bytes,
        };
    }
};

/// The lane pipeline's word permutation: rows tiled nt at a time, a group's words innermost (lane_qmm.tile_weight).
pub fn tileWeight(gpa: std.mem.Allocator, flat: []const u8, rows: usize, kw: usize, nt: usize, w: usize) ![]u8 {
    const words = std.mem.bytesAsSlice(u32, flat);
    const out = try gpa.alloc(u8, flat.len);
    errdefer gpa.free(out);
    const out_words = std.mem.bytesAsSlice(u32, out);
    const groups = kw / w;
    for (0..rows / nt) |t| {
        for (0..groups) |g| {
            for (0..nt) |j| {
                for (0..w) |c| {
                    out_words[((t * groups + g) * nt + j) * w + c] = words[(t * nt + j) * kw + g * w + c];
                }
            }
        }
    }
    return out;
}

/// (K/GS, N, 2) bf16 (scale, bias) pairs, group-major, over a row-concatenated stack (lane_qmm.pack_scales).
pub fn packScales(gpa: std.mem.Allocator, members: []const Q) ![]u8 {
    var rows: usize = 0;
    for (members) |m| rows += m.rows;
    const kg = members[0].s.dim(1);
    const out = try gpa.alloc(u8, kg * rows * 4);
    errdefer gpa.free(out);
    const out16 = std.mem.bytesAsSlice(u16, out);
    var at: usize = 0;
    for (members) |m| {
        const s16 = std.mem.bytesAsSlice(u16, m.part(.scales));
        const b16 = std.mem.bytesAsSlice(u16, m.part(.biases));
        for (0..kg) |g| {
            for (0..m.rows) |j| {
                out16[(g * rows + at + j) * 2] = s16[j * kg + g];
                out16[(g * rows + at + j) * 2 + 1] = b16[j * kg + g];
            }
        }
        at += m.rows;
    }
    return out;
}

/// Row-concatenated members in the checkpoint's own MLX layout: a pure byte move per member.
pub fn concatPart(gpa: std.mem.Allocator, members: []const Q, comptime part: Part) ![]u8 {
    var n: usize = 0;
    for (members) |m| n += m.part(part).len;
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    var at: usize = 0;
    for (members) |m| {
        @memcpy(out[at..][0..m.part(part).len], m.part(part));
        at += m.part(part).len;
    }
    return out;
}

/// One bit width's members, in member order, at their smallest group: one product, as decode.py's `_stacked`.
pub const Run = struct { bits: u8, group: usize, members: u16, rows: usize };

/// Runs by bit width in first-seen order, adjacent or not, so split_k(N, K) sees 0.6.6's N; group 128 reads as 64.
pub fn runsOf(specs: []const affine.Spec, rows: []const usize, out: []Run) ![]Run {
    if (specs.len == 0 or specs.len != rows.len) return error.EmptyLaneStack;
    if (specs.len > 16) return error.TooManyLaneMembers;
    var n: usize = 0;
    for (specs, rows, 0..) |s, r, i| {
        const bit = @as(u16, 1) << @intCast(i);
        for (out[0..n]) |*run| {
            if (run.bits != s.bits) continue;
            run.members |= bit;
            run.rows += r;
            run.group = @min(run.group, s.group);
            break;
        } else {
            if (n == out.len) return error.TooManyLaneRuns;
            out[n] = .{ .bits = s.bits, .group = s.group, .members = bit, .rows = r };
            n += 1;
        }
    }
    for (out[0..n]) |*run| run.group = @min(run.group, 64);
    return out[0..n];
}

/// Whether a run of `rows` columns is tiled: 32 at a time when they divide, else MLX's layout (decode.py's `nt`).
pub fn tiled(rows: usize) bool {
    return rows % 32 == 0;
}

/// One width's `.wq` (tiled, or MLX's layout when ragged) and `.sbt` group-major pairs (decode.py's lane pipeline).
fn putRun(out: *pio.Out, gpa: std.mem.Allocator, name: []const u8, members: []const Q) !void {
    var rows: usize = 0;
    for (members) |m| rows += m.rows;
    const spec = members[0].spec;
    if (rows % 4 != 0) return error.LaneRowsNotFours; // lane_matmul's column pairs and scale pairs
    const w = spec.group * spec.bits / 32;
    const a = out.arena.allocator();
    const wq_name = try std.fmt.allocPrint(a, "{s}.wq", .{name});
    const sbt_name = try std.fmt.allocPrint(a, "{s}.sbt", .{name});
    const flat = try concatPart(gpa, members, .weight);
    defer gpa.free(flat);
    // put owns each blob from the call on, failing or not
    try out.put(wq_name, "U32", &.{ rows, members[0].kw }, if (tiled(rows)) try tileWeight(gpa, flat, rows, members[0].kw, 32, w) else try gpa.dupe(u8, flat));
    try out.put(sbt_name, "BF16", &.{ members[0].s.dim(1), rows, 2 }, try packScales(gpa, members));
}

/// A lane projection; any width but 6/32 adds a `.runs` table, and several widths `.rN` per run and `.members`.
pub fn putLane(out: *pio.Out, gpa: std.mem.Allocator, name: []const u8, given: []const Q) !void {
    putStack(out, gpa, name, given) catch |e| {
        std.log.warn("lane {s}: {s}", .{ name, @errorName(e) }); // the pack refusal names its lane
        return e;
    };
}

fn putStack(out: *pio.Out, gpa: std.mem.Allocator, name: []const u8, given: []const Q) !void {
    var specs: [16]affine.Spec = undefined;
    var rows: [16]usize = undefined;
    if (given.len > specs.len) return error.TooManyLaneRuns;
    for (given, 0..) |m, i| {
        specs[i] = m.spec;
        rows[i] = m.rows;
    }
    var buf: [8]Run = undefined;
    const runs = try runsOf(specs[0..given.len], rows[0..given.len], &buf);
    const a = out.arena.allocator();
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const work = scratch.allocator();
    var place: [16][3]u32 = undefined; // run, row within the run, row within the fused output
    var fused: u32 = 0;
    for (runs, 0..) |r, ri| {
        var storage: [16]Q = undefined;
        var count: usize = 0;
        var at: u32 = 0;
        for (given, 0..) |m, i| {
            if ((r.members >> @intCast(i)) & 1 == 0) continue;
            storage[count] = try widen.regroup(work, m, r.group);
            count += 1;
            place[i] = .{ @intCast(ri), at, 0 };
            at += @intCast(m.rows);
        }
        const run_name = if (runs.len == 1) name else try std.fmt.allocPrint(a, "{s}.r{d}", .{ name, ri });
        try putRun(out, gpa, run_name, storage[0..count]);
    }
    // the uniform packs keep their bytes
    if (runs.len == 1 and runs[0].bits == 6 and runs[0].group == 32 and tiled(runs[0].rows)) return;
    const runs_name = try std.fmt.allocPrint(a, "{s}.runs", .{name});
    const table = try gpa.alloc(u32, runs.len * 3);
    for (runs, 0..) |r, i| table[i * 3 ..][0..3].* = .{ r.bits, @intCast(r.group), @intCast(r.rows) };
    try out.put(runs_name, "U32", &.{ runs.len, 3 }, table);
    if (runs.len == 1) return;
    for (given, 0..) |m, i| {
        place[i][2] = fused;
        fused += @intCast(m.rows);
    }
    const members_name = try std.fmt.allocPrint(a, "{s}.members", .{name});
    const members = try gpa.alloc(u32, given.len * 3);
    for (place[0..given.len], 0..) |p, i| members[i * 3 ..][0..3].* = p;
    try out.put(members_name, "U32", &.{ given.len, 3 }, members);
}

test {
    _ = @import("pack_lane_test.zig");
}
