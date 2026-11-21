//! GLM's prompt index scores (glm_index_nax.metal, 16 rows x 64 blocks a simdgroup) against main's 32 x 32.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

/// Main's glm_index_nax.metal: 64-row tiles of 4 simdgroups, 32 rows x 32 blocks each.
pub const main_index = @embedFile("baselines/glm_index_nax_main.metal.txt");

const HI = 32; // indexer heads
const DI = 128; // a head's dims, a pooled key's
const QR = 20480; // qr_proj's row: 64 heads' q_nope, then the indexer's queries
const QOFF = 64 * 256;
const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32, rows: u32 };

/// Rows: a pass's rows (select_rows at most); k: its first position; n: a key's dims. Both tile heights are served.
const served = [_]c.Shape{ .{ .rows = 256, .n = DI, .k = 16384 }, .{ .rows = 1, .n = DI, .k = 2047 }, .{ .rows = 37, .n = DI, .k = 2048 }, .{ .rows = 128, .n = DI, .k = 4100 }, .{ .rows = 129, .n = DI, .k = 6141 }, .{ .rows = 255, .n = DI, .k = 33001 }, .{ .rows = 256, .n = DI, .k = 90000 }, .{ .rows = 200, .n = DI, .k = 130872 } };

fn stride(s: c.Shape) usize {
    return (s.k + s.rows + 1) / 4 + 3;
}

pub const Case = struct {
    control: bool,
    new: mtl.Pipeline,
    old: mtl.Pipeline,

    /// The loader's library as it compiles it (nax.h inlined for this GPU), and main's the same way.
    pub fn text(a: std.mem.Allocator, device: mtl.Device) ![]u8 {
        return tf.frags.source(device, a, sources.glm_index_nax);
    }

    pub fn init(s: *bits.Suite, control: bool) !Case {
        const loader = try s.production();
        const own = try text(s.a, s.device);
        defer s.a.free(own);
        _ = try loader.require(own);
        const body = if (control) try bits.edited(s.a, own, "void glm_index_scores_nax(", "const int tm = 16 * int(sg);", "const int tm = 16 * int(sg / 2);") else try s.a.dupe(u8, own);
        defer s.a.free(body);
        const main_text = try tf.frags.source(s.device, s.a, main_index);
        defer s.a.free(main_text);
        const new = try bits.pipeline(s.device, body, "glm_index_scores_nax");
        errdefer new.deinit();
        return .{ .control = control, .new = new, .old = try bits.pipeline(s.device, main_text, "glm_index_scores_nax") };
    }
    pub fn deinit(x: *Case) void {
        x.new.deinit();
        x.old.deinit();
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "control: metal glm launch index scores, two simdgroups on each 16 rows" else "metal glm launch index scores glm_index_scores_nax, 16 rows x 64 blocks a simdgroup";
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.control), "main's glm_index_scores_nax (32 rows x 32 blocks a simdgroup)", "main's bytes in every score written, NaN payloads and infinities from the queries and weights included", &served, "the same tensor-unit products a head, the same head order and fused sums; only the tiles' rows change");
    }
    pub fn layout(_: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.rows > 256 or s.n != DI or s.k + s.rows + 1 < 4) return null;
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * QR * 2, .bf16);
        _ = l.add(.input, s.rows * HI * 2, .bf16);
        _ = l.add(.input, stride(s) * DI * 2, .bf16);
        _ = l.add(.output, s.rows * stride(s) * 4, .f32);
        return l;
    }
    /// Queries, weights, keys; gate 1 and every third shape plant specials in queries and weights (NaN keys relu to 0).
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        for (std.mem.bytesAsSlice(u16, data[0])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        for (std.mem.bytesAsSlice(u16, data[1])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.4);
        const pool = std.mem.bytesAsSlice(u16, data[2]);
        for (pool) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        const q = std.mem.bytesAsSlice(u16, data[0]);
        const w = std.mem.bytesAsSlice(u16, data[1]);
        for (bits.special_bf16, 0..) |v, i| {
            const r = (i * 37) % s.rows;
            q[r * QR + QOFF + (i * 131) % (HI * DI)] = v;
            w[((i * 53 + 1) % s.rows) * HI + (i * 7) % HI] = v;
        }
        pool[(seed % stride(s)) * DI + 5] = 0x7fc0;
    }
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const most = (s.k + s.rows) / 4;
        e.setPipeline(if (new) x.new else x.old);
        e.setBuffer(r[0].buf, r[0].off + QOFF * 2, 0);
        e.setBuffer(r[1].buf, r[1].off, 1);
        e.setBuffer(r[2].buf, r[2].off, 2);
        e.setBuffer(r[3].buf, r[3].off, 3);
        e.setValue(ScoreArgs{ .p0 = @intCast(s.k), .q_stride = QR, .w_stride = HI, .s_stride = @intCast(stride(s)), .rows = @intCast(s.rows) }, 4);
        const tile: usize = if (new) tf.glm.prompt.scoreRows(@intCast(most), @intCast(s.rows)) else 64; // selectKeys', main's
        e.dispatchGroups(mtl.Size.of((most + 63) / 64, (s.rows + tile - 1) / tile, 1), mtl.Size.of(2 * tile, 1, 1));
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(control))) try s.add(Case, try Case.init(s, control));
    }
}

/// Gate 5's registration `glm launch index <function>`: the loader's index library here, and as a pre-M5 GPU builds it.
pub const prefix = "glm launch index ";
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, _: []const u8) !bits.Texts {
    return .{ .here = try Case.text(a, device), .pre_m5 = try tf.frags.precision.source(a, false, sources.glm_index_nax, sources.nax, false) };
}
