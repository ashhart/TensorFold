//! A window's picks on the GPU behind its verify: the host's greedy rule, or its keyed race over the whole row.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const sources = @import("kernel_sources");

const lanes = host.lanes;
const threads = 1024;

/// A place in a buffer.
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Picks = struct {
    pick: mtl.Pipeline,
    keyed: mtl.Pipeline,
    accept: mtl.Pipeline,

    pub fn init(device: mtl.Device) !Picks {
        const lib = try mtl.Library.fromSource(device, sources.dsv41.keyed ++ sources.dsv41.early, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pick = try mtl.Pipeline.init(device, lib, "tf_dsv41_early_pick", false);
        errdefer pick.deinit();
        const keyed = try mtl.Pipeline.init(device, lib, "tf_dsv41_early_keyed", false);
        errdefer keyed.deinit();
        return .{ .pick = pick, .keyed = keyed, .accept = try mtl.Pipeline.init(device, lib, "tf_dsv41_early_accept", false) };
    }

    pub fn deinit(p: *Picks) void {
        p.pick.deinit();
        p.keyed.deinit();
        p.accept.deinit();
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
};

/// The key bases of a keyed window's rows, one a draw position (`keyBase(seed, draw)`).
pub fn bases(s: lanes.Sampling, draws: []const u64, out: []u64) void {
    for (out, draws) |*b, d| b.* = lanes.sampling.keyBase(s.seed, d);
}

/// Whether the host races a sampled row (sampling.zig's `race`): no top-k under the vocabulary, nucleus or min_p.
pub fn raced(s: lanes.Sampling, vocab: u32) bool {
    return s.temperature > 0 and s.min_p <= 0 and (s.top_k == 0 or s.top_k >= vocab) and !(s.top_p > 0 and s.top_p < 1);
}

test "the keyed pick races only rows the host races" {
    try std.testing.expect(raced(.{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, 1000));
    try std.testing.expect(raced(.{ .seed = 1, .temperature = 0.7, .top_k = 1000, .top_p = 0.0 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 1.0, .top_k = 20, .top_p = 1.0 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 0.95 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.05 }, 1000));
    try std.testing.expect(!raced(.{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1.0 }, 1000));
}
