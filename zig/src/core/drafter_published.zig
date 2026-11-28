//! A drafter release published before format v1 (Kolibri-1's two on the Hub) as a v1 Config, under the same checks.

const std = @import("std");
const Io = std.Io;
const release = @import("drafter_release.zig");
const layout = @import("drafter_layout.zig");
const Why = release.Why;

/// Context rows a published drafter attends to: the window both were trained with (not in their config.json).
pub const window: u32 = 2048;
/// The weights' file name in a published release (not in their config.json).
pub const weights = "model.safetensors";

/// A published config.json's architecture and what it is in the format.
const Arch = struct { name: []const u8, as: release.Architecture, names: layout.Names };
const archs = [_]Arch{
    .{ .name = "kolibri1-eagle3-drafter", .as = .eagle3, .names = .{ .layers = "blocks", .own_target = true } },
    .{ .name = "kolibri1-block-drafter", .as = .@"block-drafter", .names = .{ .own_target = true, .unused = &.{"inp.weight"} } },
};

/// The published config.json, field for field (unknown fields refused).
const Published = struct {
    architecture: []const u8,
    target: []const u8,
    drafter: struct {
        hidden: u32,
        heads: u32,
        head_dim: u32,
        ffn: u32,
        layers: u32,
        taps: u32,
        eps: f64,
        theta: f64,
        block: u32 = 1,
        chain_ctx: bool = false,
        row0_chain: bool = false,
        pred_rank: u32 = 0,
        spine_rank: u32 = 0,
    },
    taps: []const u32, // 0-based layers whose normed outputs the drafter reads
    draft_vocab: []const u8,
    draft_vocab_size: u32,
    serving: ?struct { depth: u32, quant: release.Weight, head: release.Weight } = null,
    training_tokens: ?u64 = null,
    training_steps: ?u64 = null,
};

/// The target a published release names by repo, with the adapter of the target the server loaded.
pub const Target = struct { repo: []const u8, adapter: release.Adapter };

/// A published release read whole: its Config in the format's terms, its draft vocabulary, how the file names tensors.
pub const Release = struct {
    manifest: release.Manifest,
    vocab: []u32,
    names: layout.Names,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, target: Target, why: *Why) !Release {
        const text = try dir.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
        defer gpa.free(text);
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const got = try parse(arena.allocator(), text, target, why);
        var m: release.Manifest = .{ .arena = arena, .config = got.config };
        try release.validate(&m.config, why);
        try release.check(&m.config, target.adapter, why);
        const vt = try dir.readFileAlloc(io, m.config.draft_vocab.file, gpa, .limited(64 << 20));
        defer gpa.free(vt);
        const vocab = try release.parseVocab(&m.config, gpa, vt, why);
        errdefer gpa.free(vocab);
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const header = try layout.readHeader(scratch.allocator(), io, dir, weights, why);
        try layout.checkWeightsNamed(scratch.allocator(), &m.config, header, got.names, why);
        return .{ .manifest = m, .vocab = vocab, .names = got.names };
    }

    pub fn deinit(r: *Release, gpa: std.mem.Allocator) void {
        gpa.free(r.vocab);
        r.manifest.deinit();
        r.* = undefined;
    }
};

/// Whether `text` is a published config.json (its architecture one of `archs`); a v1 release says "format".
pub fn declared(text: []const u8) bool {
    const Probe = struct { architecture: []const u8 = "" };
    var buf: [4096]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&buf);
    const p = std.json.parseFromSliceLeaky(Probe, fba.allocator(), text, .{ .ignore_unknown_fields = true }) catch return false;
    for (archs) |a| if (std.mem.eql(u8, a.name, p.architecture)) return true;
    return false;
}

/// The published `text` as the format's Config over `target` (names in `a`); validate and check still run after.
pub fn parse(a: std.mem.Allocator, text: []const u8, target: Target, why: *Why) !struct { config: release.Config, names: layout.Names } {
    const p = std.json.parseFromSliceLeaky(Published, a, text, .{ .allocate = .alloc_always }) catch |e| {
        why.set("config.json does not parse as a published drafter release ({s})", .{@errorName(e)});
        return error.BadDrafterRelease;
    };
    const arch = for (archs) |x| {
        if (std.mem.eql(u8, x.name, p.architecture)) break x;
    } else return refuse(why, "architecture \"{s}\" is not a published drafter release ({s} or {s})", .{ p.architecture, archs[0].name, archs[1].name });
    if (!std.mem.eql(u8, p.target, target.repo)) return refuse(why, "the release drafts for {s}; the loaded target is {s}", .{ p.target, target.repo });
    const s = p.drafter;
    if (s.taps != p.taps.len) return refuse(why, "drafter taps {d}, but {d} layers listed", .{ s.taps, p.taps.len });
    if (arch.as == .@"block-drafter" and !(s.chain_ctx and s.row0_chain)) return refuse(why, "a block drafter serves with chain_ctx and row0_chain only", .{});
    const ids = try a.alloc(u32, p.taps.len);
    for (ids, p.taps) |*id, layer| id.* = layer + 1; // a layer's output is the stream after layer + 1 layers
    const t = target.adapter;
    const serving: release.Serving = if (p.serving) |v| .{ .depth = v.depth, .body = v.quant, .head = v.head } else .{ .depth = 2, .body = .bf16, .head = .bf16 };
    return .{ .config = .{
        .format = release.format,
        .version = release.version,
        .architecture = arch.as,
        .target = .{ .repo = p.target, .revision = "", .layers = t.layers, .hidden = t.hidden, .vocab = t.vocab, .embed_scale = t.embed_scale, .tied = t.tied, .softcap = t.softcap, .head_bias = t.head_bias },
        .taps = .{ .ids = ids, .stream = .residual, .normed = true, .dtype = .bf16, .row_bytes = @intCast(ids.len * @as(usize, t.hidden) * 2) },
        .drafter = .{ .hidden = s.hidden, .heads = s.heads, .head_dim = s.head_dim, .ffn = s.ffn, .layers = s.layers, .block = s.block, .window = window, .eps = s.eps, .theta = s.theta, .pred_rank = s.pred_rank, .spine_rank = s.spine_rank },
        .draft_vocab = .{ .file = p.draft_vocab, .size = p.draft_vocab_size },
        .weights = weights,
        .serving = serving,
    }, .names = arch.names };
}

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) error{BadDrafterRelease} {
    why.set(fmt, args);
    return error.BadDrafterRelease;
}

test {
    _ = @import("drafter_published_test.zig");
}
