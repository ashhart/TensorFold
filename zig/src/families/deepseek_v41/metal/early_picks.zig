//! Behind-the-verify picks on GPU: greedy, raced keyed, and filtered (top_k/top_p/min_p) per-row picks.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const sources = @import("kernel_sources");

const lanes = host.lanes;
const threads = 1024;
pub const max_topk = 256;

/// A place in a buffer.
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

/// Per-row sampling parameters for `tf_dsv41_early_filtered` (matches early_filtered.metal TfEarlyRow).
pub const Row = extern struct {
    base: u64 = 0,
    t: f32 = 1,
    sampled: u32 = 0,
    top_k: u32 = 0,
    top_p: f32 = 1,
    min_p: f32 = 0,
    pad: u32 = 0,

    pub fn of(smp: ?lanes.Sampling, draw: u64) Row {
        const s = smp orelse return .{};
        if (s.temperature <= 0) return .{};
        return .{
            .base = lanes.sampling.keyBase(s.seed, draw),
            .t = @floatCast(@max(s.temperature, 1e-6)),
            .sampled = 1,
            .top_k = s.top_k,
            .top_p = @floatCast(s.top_p),
            .min_p = @floatCast(s.min_p),
        };
    }
};

/// Per-slot accept descriptor for `tf_dsv41_early_accept_multi` (matches early_filtered.metal TfEarlySlot).
pub const Slot = extern struct {
    r0: u32 = 0,
    n: u32 = 0,
    block: u32 = 0,
    skew: u32 = 0,
    pos_lo: u32 = 0,
    pos_hi: u32 = 0,
};

comptime {
    if (@sizeOf(Row) != 32 or @sizeOf(Slot) != 24) @compileError("TfEarlyRow / TfEarlySlot layout mismatch");
}

pub const Picks = struct {
    pick: mtl.Pipeline,
    keyed: mtl.Pipeline,
    accept: mtl.Pipeline,
    filtered: mtl.Pipeline,
    accept_multi: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Picks {
        const lib = try mtl.Library.fromSource(device, sources.dsv41.keyed ++ sources.dsv41.early, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pick = try mtl.Pipeline.init(device, lib, "tf_dsv41_early_pick", false);
        errdefer pick.deinit();
        const keyed = try mtl.Pipeline.init(device, lib, "tf_dsv41_early_keyed", false);
        errdefer keyed.deinit();
        const accept = try mtl.Pipeline.init(device, lib, "tf_dsv41_early_accept", false);
        errdefer accept.deinit();
        const flib = try mtl.Library.fromSource(device, sources.dsv41.keyed ++ sources.dsv41.early_filtered, mtl.CompileOptions.mlx());
        defer flib.deinit();
        const filtered = try mtl.Pipeline.init(device, flib, "tf_dsv41_early_filtered", false);
        errdefer filtered.deinit();
        const accept_multi = try mtl.Pipeline.init(device, flib, "tf_dsv41_early_accept_multi", false);
        return .{ .pick = pick, .keyed = keyed, .accept = accept, .filtered = filtered, .accept_multi = accept_multi };
    }

    pub fn deinit(p: *Picks) void {
        p.pick.deinit();
        p.keyed.deinit();
        p.accept.deinit();
        p.filtered.deinit();
        p.accept_multi.deinit();
    }

    /// Each of `n` rows' pick into `out`: greedy, or keyed by `smp` with a key base a row in `keys` (`bases`).
    pub fn encode(p: *const Picks, e: mtl.ComputeEncoder, logits: Ref, out: Ref, keys: Ref, n: u32, vocab: u32, smp: ?lanes.Sampling) void {
        if (smp) |s| {
            e.setPipeline(p.keyed);
            e.setValue(extern struct { vocab: u32, t: f32 }{ .vocab = vocab, .t = @floatCast(@max(s.temperature, 1e-6)) }, 0);
            e.setBuffer(keys.buf, keys.off, 3);
        } else {
            e.setPipeline(p.pick);
            e.setValue(vocab, 0);
        }
        e.setBuffer(logits.buf, logits.off, 1);
        e.setBuffer(out.buf, out.off, 2);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(threads, 1, 1));
    }

    /// Unified per-row picks for `n` rows (greedy, raced, min_p, top_k <= 256, and pure nucleus) in one dispatch.
    pub fn encodeRows(p: *const Picks, e: mtl.ComputeEncoder, logits: Ref, out: Ref, rows_ref: Ref, n: u32, vocab: u32) void {
        e.setPipeline(p.filtered);
        e.setValue(vocab, 0);
        e.setBuffer(logits.buf, logits.off, 1);
        e.setBuffer(out.buf, out.off, 2);
        e.setBuffer(rows_ref.buf, rows_ref.off, 3);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(threads, 1, 1));
    }
};

/// The key bases of a keyed window's rows, one a draw position (`keyBase(seed, draw)`).
pub fn bases(s: lanes.Sampling, draws: []const u64, out: []u64) void {
    for (out, draws) |*b, d| b.* = lanes.sampling.keyBase(s.seed, d);
}

/// Whether the host races a sampled row (sampling.zig's `race`): no top-k under the vocabulary, nucleus or min_p.
pub fn raced(s: lanes.Sampling, vocab: u32) bool {
    return s.temperature > 0 and s.min_p <= 0 and (s.top_k == 0 or s.top_k >= vocab) and !(s.top_p > 0 and s.top_p < 1);
}

/// Whether `tf_dsv41_early_filtered` handles this sampling config on GPU (greedy, raced, min_p, top_k <= 256, nucleus).
pub fn supported(smp: ?lanes.Sampling, vocab: u32) bool {
    const s = smp orelse return true;
    if (s.temperature <= 0) return true;
    return s.top_k == 0 or s.top_k <= max_topk or s.top_k >= vocab;
}

test "the keyed pick races only rows the host races" {
    try std.testing.expect(raced(.{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 1000));
    try std.testing.expect(raced(.{ .seed = 1, .temperature = 0.7, .top_k = 1000, .top_p = 0.0 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 1.0, .top_k = 20, .top_p = 1.0 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 0.95 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.05 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1.0 }, 1000));
    try std.testing.expect(supported(.{ .seed = 1, .temperature = 0.8, .top_k = 20, .top_p = 0.95, .min_p = 0.05 }, 129280));
    try std.testing.expect(supported(.{ .seed = 1, .temperature = 0.8, .top_k = 0, .top_p = 0.95 }, 129280));
    try std.testing.expect(!supported(.{ .seed = 1, .temperature = 0.8, .top_k = 500, .top_p = 0.95 }, 129280));
}
