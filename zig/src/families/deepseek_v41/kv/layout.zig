//! DeepSeek-V4.1's paged cache families: comp.L (compressed KV) and index_k.L (index keys) per KV source layer.

const std = @import("std");
const sessions = @import("sessions");
const Family = sessions.Family;

pub const KvFormat = enum { fp8, fp4 };
pub const IndexFormat = enum { fp8, bf16 };

pub fn compRowBytes(f: KvFormat) u32 {
    return switch (f) {
        .fp8 => 584,
        .fp4 => 384,
    };
}

pub fn indexRowBytes(f: IndexFormat) u32 {
    return switch (f) {
        .fp8 => 132,
        .bf16 => 256,
    };
}

pub const Options = struct {
    kv: KvFormat = .fp8,
    index: IndexFormat = .fp8,
    /// TF_DSV41_KV_SPLIT: the comp families sharded by logical page over the TP ranks
    split: bool = false,
    page: u32 = 256,
};

pub const max_sources = 8;

pub const Layout = struct {
    opts: Options = .{},
    n: u32 = 0,
    fams: [2 * max_sources]Family = undefined,
    names: [2 * max_sources][16]u8 = undefined,
    /// the KV source layers, in order
    sources: [max_sources]u32 = undefined,
    nsources: u32 = 0,

    /// The families of every KV source layer of a config (duck-typed: `layers`, `isKvSource`, `compressRatio`).
    pub fn fromConfig(cfg: anytype, opts: Options) !Layout {
        var l: Layout = .{ .opts = opts };
        var layer: u32 = 0;
        while (layer < cfg.layers) : (layer += 1) {
            if (!cfg.isKvSource(layer)) continue;
            try l.add(layer, cfg.compressRatio(layer));
        }
        if (l.nsources == 0) return error.NoKvSources;
        return l;
    }

    /// One KV source layer at a compression ratio (1 or 2).
    pub fn add(l: *Layout, layer: u32, ratio: u32) !void {
        if (l.nsources == max_sources) return error.TooManySources;
        if (ratio == 0 or l.opts.page % ratio != 0) return error.BadRatio;
        l.sources[l.nsources] = layer;
        l.nsources += 1;
        l.push("comp", layer, .{ .name = "", .ratio = ratio, .row_bytes = compRowBytes(l.opts.kv), .split = l.opts.split });
        l.push("index_k", layer, .{ .name = "", .ratio = ratio, .row_bytes = indexRowBytes(l.opts.index) });
    }

    fn push(l: *Layout, kind: []const u8, layer: u32, f: Family) void {
        const name = std.fmt.bufPrint(&l.names[l.n], "{s}.{d}", .{ kind, layer }) catch unreachable;
        l.fams[l.n] = f;
        l.fams[l.n].name = name;
        l.n += 1;
    }

    /// Rebinds the names after the value moved (the families name into this value's own storage).
    pub fn families(l: *Layout) []const Family {
        for (l.fams[0..l.n], 0..) |*f, i| f.name = l.names[i][0..f.name.len];
        return l.fams[0..l.n];
    }

    /// The pool's layout (sessions.Layout) over these families.
    pub fn pool(l: *Layout) sessions.Layout {
        return .{ .families = l.families(), .page = l.opts.page };
    }

    /// The family index of comp.L / index_k.L.
    pub fn compOf(l: *const Layout, layer: u32) ?u32 {
        for (l.sources[0..l.nsources], 0..) |s, i| if (s == layer) return @intCast(2 * i);
        return null;
    }

    pub fn indexOf(l: *const Layout, layer: u32) ?u32 {
        return if (l.compOf(layer)) |c| c + 1 else null;
    }

    /// A rank's pool bytes a token (world: the TP ranks the split families shard over).
    pub fn bytesPerToken(l: *Layout, world: u32) f64 {
        return l.pool().bytesPerToken(if (l.opts.split) world else 1);
    }
};

/// The release topology (config.json's kv_source_layer_ids and compress_ratios), for tests and offline pricing.
pub const Release = struct {
    layers: u32 = 43,
    pub fn isKvSource(_: Release, layer: u32) bool {
        return layer == 2 or layer == 8 or layer == 14 or layer == 20;
    }
    pub fn compressRatio(_: Release, layer: u32) u32 {
        return if (layer < 2 or layer >= 40) 0 else if (layer < 20) 2 else 1;
    }
};

test "the release's families: 1,790 B a token replicated, 1,060 split, the FP4 and bf16 variants" {
    var l = try Layout.fromConfig(Release{}, .{});
    try std.testing.expectEqual(@as(u32, 8), l.n);
    try std.testing.expectEqualStrings("comp.2", l.families()[0].name);
    try std.testing.expectEqualStrings("index_k.20", l.families()[7].name);
    try std.testing.expectEqual(@as(u32, 1), l.families()[6].ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 1790), l.bytesPerToken(2), 1e-9);
    var s = try Layout.fromConfig(Release{}, .{ .split = true });
    try std.testing.expectApproxEqAbs(@as(f64, 1060), s.bytesPerToken(2), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1790), s.bytesPerToken(1), 1e-9);
    var f4 = try Layout.fromConfig(Release{}, .{ .kv = .fp4, .split = true });
    try std.testing.expectApproxEqAbs(@as(f64, 810), f4.bytesPerToken(2), 1e-9); // A + FP4 comp
    var bf = try Layout.fromConfig(Release{}, .{ .index = .bf16 });
    try std.testing.expectApproxEqAbs(@as(f64, 2100), bf.bytesPerToken(1), 1e-9);
    try std.testing.expectEqual(@as(?u32, 6), l.compOf(20));
    try std.testing.expectEqual(@as(?u32, 3), l.indexOf(8));
    const moved = l;
    var m = moved;
    try std.testing.expectEqualStrings("comp.14", m.families()[4].name);
}
