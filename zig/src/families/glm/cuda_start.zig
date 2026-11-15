//! Connect rank liveness before resolving page settings or constructing the engine.
const std = @import("std");

/// Every rank resolves its page mode within the same watched failure boundary as engine construction.
pub fn open(io: std.Io, p: anytype, text: ?[]const u8, initial: anytype, watch: anytype, args: anytype, comptime after: anytype, comptime report: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    if (p.world > 1) try watch.start(io, p.rank, p.world);
    return resolved(p, text, initial, args, after, report) catch |err| {
        watch.fail(@errorName(err));
        return err;
    };
}

fn resolved(p: anytype, text: ?[]const u8, initial: anytype, args: anytype, comptime after: anytype, comptime report: anytype) @typeInfo(@TypeOf(after)).@"fn".return_type.? {
    var settings = initial;
    settings.pages = try paging(text, p.world, settings.kv8);
    settings.kv_split = try splitting(settings.kv_split, p.world, settings.pages, settings.kv_host != 0);
    report(args, Report{ .text = text, .rank = p.rank, .world = p.world, .kv8 = settings.kv8, .pages = settings.pages });
    return after(args, settings);
}

pub fn log(_: anytype, selection: Report) void {
    std.log.info("glm kv selection: {f}", .{selection});
}

/// A configuration selection, before construction; the resolved boolean is the one passed to the engine.
pub const Report = struct {
    text: ?[]const u8,
    rank: usize,
    world: usize,
    kv8: bool,
    pages: bool,

    pub fn format(r: Report, writer: *std.Io.Writer) !void {
        const v = r.text orelse "auto";
        const mode = if (std.mem.eql(u8, v, "auto")) "auto" else if (std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "0")) "off" else "on";
        try writer.print("{s}; TF_KV_PAGES={s} ({s}); rank {d} of {d}; cache {s}", .{
            if (r.pages) "pool pages" else "window per stream",
            mode,
            if (r.text == null) "default" else "env",
            r.rank,
            r.world,
            if (r.kv8) "fp8" else "bf16",
        });
    }
};

/// TF_KV_PAGES as `open` takes it: empty or whitespace-only text is unset, so `auto`.
pub fn pagesText() ?[:0]const u8 {
    return @import("core").profile.given(.TF_KV_PAGES);
}

/// TF_GLM_KV_SPLIT at `world` ranks: refused at one rank (first: auto pages are off there), off pages, host pages.
pub fn splitting(on: bool, world: usize, pages: bool, host: bool) !bool {
    if (!on) return false;
    if (world < 2) return error.KvSplitNeedsRanks;
    if (!pages) return error.KvSplitNeedsPages;
    if (host) return error.KvSplitOnHost;
    return true;
}

/// TF_GLM_KV_SPLIT with TF_KV_SHARE is refused: the shared store's pages would be of any owner's class.
pub fn sharing(split: bool, share: usize) !void {
    if (split and share > 0) return error.KvSplitShare;
}

/// TF_GLM_KV_SPLIT with kept prompt states is refused: a state would copy latent rows only their owner holds.
pub fn keeping(split: bool, keeps: bool) !void {
    if (split and keeps) return error.KvSplitKeeps;
}

/// TF_KV_PAGES keeps FP8 on whole windows; explicit pages with FP8 are refused on every rank.
pub fn paging(text: ?[]const u8, world: usize, kv8: bool) !bool {
    const v = text orelse "auto";
    if (std.mem.eql(u8, v, "auto")) return world >= 2 and !kv8;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return false;
    if (kv8) return error.Fp8OnPages;
    return true;
}

/// TF_GLM_KV_SPLIT's refusals in the server's words (`cuda_native.explain` asks here for the errors it does not name).
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.KvSplitNeedsRanks => "TF_GLM_KV_SPLIT splits rows over ranks: it needs TF_GLM_WORLD 2 or more",
        error.KvSplitNeedsPages => "TF_GLM_KV_SPLIT needs KV pages: unset TF_KV_PAGES or set it to auto or on (fp8 KV keeps windows)",
        error.KvSplitOnHost => "TF_GLM_KV_SPLIT keeps the KV pages on the device: unset TF_GLM_KV_HOST",
        error.KvSplitShare => "TF_GLM_KV_SPLIT books each stream's pages from its owner's class, and TF_KV_SHARE's pages come from any: unset one",
        error.KvSplitKeeps => "TF_GLM_KV_SPLIT keeps each stream's latent rows on one rank, and kept prompt states do not carry them yet: serve with --prompt-cache-gib 0 and without --learn-conversations, or without TF_GLM_KV_SPLIT",
        error.KvSplitRoom, error.KvSplitNoRoom, error.KvSplitLocal => "TF_GLM_KV_SPLIT: a forward over owner-only rows had no room to exchange them (a bug: the log names the step)",
        error.ClassTooSmall => "TF_GLM_KV_SPLIT: one owner class of the KV pages holds less than one stream's window (the log gives the numbers): lower --context, raise --parallel, or serve without TF_GLM_KV_SPLIT",
        error.ResidueMismatch => "a rank was sent a KV page of another owner's class (a bug: the ranks' books disagree)",
        error.BadClasses => "the KV pages do not split into the owner classes (a multiple of the ranks, at most 16)",
        else => null,
    };
}
