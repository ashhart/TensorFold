//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.

const std = @import("std");
const cuda = @import("cuda");
const heat = @import("heat");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    kv8: bool = false, // --kv-cache fp8, supplied by the native options rather than an environment override
    pages: bool = false, // TF_KV_PAGES as `open` resolves it (with the world and --kv-cache): every sequence on pages
    prompt_fixed_k: bool = false, // TF_GLM_PROMPT_K: one (default) or shape (experimental fixed K slices)
    tiled: bool = true, // TF_GLM_MM: 1 (the bf16 linears on glm_mm, tiled at load) or 0
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows), ring or gather
    ring: bool = false, // TF_GLM_PREFILL_REDUCE=ring: split, its exchanges passed along the ring (cuda_relay)
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole).
    index_split: u32 = 0, // TF_GLM_INDEX_SPLIT: from this position on, rank 0 splits a prompt chunk's indexer rows (0: never)
    heat: ?heat.Bands = null, // TF_HEAT_HIGH and TF_HEAT_LOW (upstream's names): a prompt chunk waits on every rank's hottest zone
    layers: u32 = 0, // TF_GLM_LAYERS: the first this many layers only, a thin model for one-GPU checks (0: all)
    dense_bits: u8 = 16, // TF_GLM_DENSE_BITS: the dense linears packed at load to 4, 6 or 8 bits (16: bf16)
    head_bits: u8 = 16, // TF_GLM_HEAD_BITS: the head, likewise

    /// The settings `get` (an environment lookup) gives; `problem` names a value that is not one of the choices.
    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        var s: Settings = .{};
        if (get("TF_GLM_PROMPT_K")) |v| {
            if (std.mem.eql(u8, v, "shape")) s.prompt_fixed_k = true else if (!std.mem.eql(u8, v, "one")) {
                problem.* = "TF_GLM_PROMPT_K: one or shape";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_MM")) |v| {
            if (std.mem.eql(u8, v, "0")) s.tiled = false else if (!std.mem.eql(u8, v, "1")) {
                problem.* = "TF_GLM_MM: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PREFILL_REDUCE")) |v| {
            if (std.mem.eql(u8, v, "split")) s.split = true else if (std.mem.eql(u8, v, "ring")) {
                s.split = true;
                s.ring = true;
            } else if (std.mem.eql(u8, v, "gather")) s.split = false else {
                problem.* = "TF_GLM_PREFILL_REDUCE: split, ring or gather";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KDA_DECODE_WIDE")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kda_wide = true else if (std.mem.eql(u8, v, "0")) s.kda_wide = false else {
                problem.* = "TF_GLM_KDA_DECODE_WIDE: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_OVERLAP_PIECES")) |v| {
            s.pieces = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.pieces < 1 or s.pieces > 8) {
                problem.* = "TF_GLM_OVERLAP_PIECES: 1 to 8";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_INDEX_SPLIT")) |v| {
            s.index_split = std.fmt.parseInt(u32, v, 10) catch {
                problem.* = "TF_GLM_INDEX_SPLIT: the position a prompt chunk's indexer rows split from, or 0 for never";
                return error.BadSetting;
            };
        }
        if (get("TF_GLM_LAYERS")) |v| {
            s.layers = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.layers == 0) {
                problem.* = "TF_GLM_LAYERS: the first how many layers, 1 or more";
                return error.BadSetting;
            }
        }
        inline for (.{ .{ "TF_GLM_DENSE_BITS", "dense_bits" }, .{ "TF_GLM_HEAD_BITS", "head_bits" } }) |e| if (get(e[0])) |v| {
            const b = std.fmt.parseInt(u8, v, 10) catch {
                problem.* = e[0] ++ ": 4, 6, 8 or 16";
                return error.BadSetting;
            };
            @field(s, e[1]) = switch (b) {
                4, 6, 8, 16 => b,
                0 => 16, // the old family's spelling of bf16
                else => {
                    problem.* = e[0] ++ ": 4, 6, 8 or 16";
                    return error.BadSetting;
                },
            };
        };
        s.heat = heat.bandsFrom(get("TF_HEAT_HIGH"), get("TF_HEAT_LOW")) catch {
            problem.* = "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high";
            return error.BadSetting;
        };
        if (s.prompt_fixed_k and !s.tiled) {
            problem.* = "TF_GLM_PROMPT_K=shape requires TF_GLM_MM=1";
            return error.BadSetting;
        }
        return s;
    }
};

/// The most ranks a world holds: rank 0's greedy pick gathers every rank's top-1 into one host array of this size.
pub const max_world = 16;

/// The process environment's value of `name` (null when unset).
pub fn environment(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// What every rank must agree on, as bytes the communicator carries.
pub const Fingerprint = extern struct {
    layout: u32 = 12, // this struct's version
    world: u32,
    rank: u32,
    kv8: u32 = 0,
    pages: u32 = 0, // TF_KV_PAGES: the engine's own sequence is made on pages at init, so every rank pages alike
    prompt_fixed_k: u32,
    tiled: u32,
    split: u32, // TF_GLM_PREFILL_REDUCE: 0 gather, 1 split, 2 ring
    kda_wide: u32,
    pieces: u32,
    layers: u32, // TF_GLM_LAYERS (0: all)
    mtp: u32, // 1: the MTP head loads and runs (drafts on); 2: and DFlash2 (--drafter), whose rounds gather
    vision: u32 = 0, // --vision's image tokens (0: off): rank 0 sends image rows every rank must hold
    video_tokens: u32 = 0,
    max_images: u32 = 0,
    max_videos: u32 = 0,
    max_frames: u32 = 0,
    bits: u32 = 16 << 8 | 16, // TF_GLM_DENSE_BITS << 8 | TF_GLM_HEAD_BITS: every rank's matmuls read the same codes
    capacity: u64, // tokens a sequence holds
    heat_high: f64, // the heat bands in degrees (0 and 0: off): every rank must wait alike
    heat_low: f64,
    config: [32]u8, // sha256 of config.json
    program: [32]u8, // sha256 of this executable
    checkpoint: [64]u8, // hex digest of the checkpoint's metadata and every shard byte, through the digest cache

    pub const Field = enum {
        layout,
        rank,
        world,
        kv8,
        pages,
        prompt_fixed_k,
        tiled,
        split,
        kda_wide,
        pieces,
        layers,
        mtp,
        vision,
        video_tokens,
        max_images,
        max_videos,
        max_frames,
        bits,
        capacity,
        heat_high,
        heat_low,
        config,
        program,
        checkpoint,
    };

    pub fn of(
        world: usize,
        rank: usize,
        s: Settings,
        capacity: usize,
        mtp: bool,
        config: [32]u8,
        program: [32]u8,
        checkpoint_digest: [32]u8,
    ) Fingerprint {
        const f: Fingerprint = .{
            .kv8 = @intFromBool(s.kv8),
            .pages = @intFromBool(s.pages),
            .world = @intCast(world),
            .rank = @intCast(rank),
            .prompt_fixed_k = @intFromBool(s.prompt_fixed_k),
            .tiled = @intFromBool(s.tiled),
            .split = if (!s.split) 0 else if (s.ring) 2 else 1,
            .kda_wide = @intFromBool(s.kda_wide),
            .pieces = s.pieces,
            .layers = s.layers,
            .mtp = @intFromBool(mtp),
            .bits = @as(u32, s.dense_bits) << 8 | s.head_bits,
            .capacity = capacity,
            .heat_high = if (s.heat) |b| b.high_c else 0,
            .heat_low = if (s.heat) |b| b.low_c else 0,
            .config = config,
            .program = program,
            .checkpoint = std.fmt.bytesToHex(checkpoint_digest, .lower),
        };
        return f;
    }
};

pub const Mismatch = struct { field: Fingerprint.Field, rank: usize };

/// The first field where rank `r`'s fingerprint differs from rank 0's (or holds the wrong rank), null when all agree.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    for (all, 0..) |f, r| {
        if (f.layout != all[0].layout) return .{ .field = .layout, .rank = r };
        if (f.rank != r) return .{ .field = .rank, .rank = r };
        const fields = .{
            "kv8",
            "pages",
            "prompt_fixed_k",
            "world",
            "tiled",
            "split",
            "kda_wide",
            "pieces",
            "layers",
            "mtp",
            "vision",
            "video_tokens",
            "max_images",
            "max_videos",
            "max_frames",
            "bits",
            "capacity",
            "heat_high",
            "heat_low",
            "config",
            "program",
            "checkpoint",
        };
        inline for (fields) |name| {
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(f, name)), std.mem.asBytes(&@field(all[0], name))))
                return .{ .field = @field(Fingerprint.Field, name), .rank = r };
        }
    }
    return null;
}

/// sha256 of a file's bytes (read whole: config.json, the executable).
pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return hashBytes(bytes);
}

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return out;
}

/// Every rank's `mine` in rank order, over one ncclAllGather.
pub fn gather(
    comptime T: type,
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    rank: usize,
    mine: T,
    out: []T,
) !void {
    const size = @sizeOf(T);
    var dev = try cuda.DeviceBuffer.alloc(d, size * out.len);
    defer dev.free();
    try dev.uploadAsync(rank * size, std.mem.asBytes(&mine), stream.handle);
    const gathered = lib.api.ncclAllGather(dev.ptr + rank * size, dev.ptr, size, .u8, comm, stream.handle);
    try lib.check(gathered, "ncclAllGather");
    try stream.synchronize();
    try dev.download(0, std.mem.sliceAsBytes(out));
}

/// All-gathers `mine` over `comm`; any disagreement is error.RankMismatch on every rank, printing what differs.
pub fn agree(
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    mine: Fingerprint,
    out: []Fingerprint,
) !void {
    try gather(Fingerprint, d, lib, comm, stream, mine.rank, mine, out);
    if (mismatch(out)) |m| {
        std.debug.print("REFUSE rank settings: rank {d}'s {s} differs from rank 0's ({s})\n", .{ m.rank, @tagName(m.field), explain(m.field) });
        return error.RankMismatch;
    }
}

/// Before loading: direct split reaches every rank, so each pair is probed; the ring only passes rows to neighbours.
pub fn probeStart(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, s: Settings, rank: usize, world: usize) !void {
    if (s.split and !s.ring) try probeSplit(d, lib, comm, stream, rank, world);
}

/// Before loading, split's peers: one value each way with each, in rank order (no pair waits on a later one).
fn probeSplit(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, rank: usize, world: usize) !void {
    var dev = try cuda.DeviceBuffer.alloc(d, 8);
    defer dev.free();
    for (0..world) |peer| if (peer != rank) {
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        try lib.check(lib.api.ncclSend(dev.ptr, 1, .f32, @intCast(peer), comm, stream.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(dev.ptr + 4, 1, .f32, @intCast(peer), comm, stream.handle), "ncclRecv");
        lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd") catch return unreachablePeer(rank, peer);
        stream.synchronize() catch return unreachablePeer(rank, peer);
    };
}

fn unreachablePeer(rank: usize, peer: usize) error{SplitUnreachable} {
    std.debug.print("REFUSE TF_GLM_PREFILL_REDUCE=split: rank {d} cannot exchange with rank {d}; split sends every rank's rows to every other, and a fabric where some ranks reach only their neighbours (a ring of four without a switch) needs TF_GLM_PREFILL_REDUCE=ring or gather\n", .{ rank, peer });
    return error.SplitUnreachable;
}

fn explain(f: Fingerprint.Field) []const u8 {
    return switch (f) {
        .layout => "a different build of this check",
        .rank => "a rank number given twice or out of order",
        .world => "the ranks were started with different world sizes",
        .kv8 => "--kv-cache fp8",
        .pages => "TF_KV_PAGES (unset or auto pages from two ranks)",
        .prompt_fixed_k => "TF_GLM_PROMPT_K",
        .tiled => "TF_GLM_MM",
        .split => "TF_GLM_PREFILL_REDUCE",
        .kda_wide => "TF_GLM_KDA_DECODE_WIDE",
        .pieces => "TF_GLM_OVERLAP_PIECES",
        .layers => "TF_GLM_LAYERS",
        .mtp => "--no-drafts given to some ranks and not others",
        .vision => "--vision (or its image tokens) given to some ranks and not others",
        .video_tokens => "--vision-video-tokens",
        .max_images => "the image count limit",
        .max_videos => "the video count limit",
        .max_frames => "the frame limit",
        .bits => "TF_GLM_DENSE_BITS or TF_GLM_HEAD_BITS",
        .capacity => "the sequence capacity (the context)",
        .heat_high => "TF_HEAT_HIGH",
        .heat_low => "TF_HEAT_LOW",
        .config => "config.json",
        .program => "the executable",
        .checkpoint => "checkpoint metadata or shard bytes",
    };
}

pub fn mediaFingerprint(f: *Fingerprint, limits: @import("cuda_images.zig").Limits) void {
    f.vision = limits.image_tokens;
    f.video_tokens = limits.video_tokens;
    f.max_images = limits.max_images;
    f.max_videos = limits.max_videos;
    f.max_frames = limits.max_frames;
}

/// The checkpoint digest a rank sends: its metadata and every shard byte, read again only for a file that changed.
pub fn artifact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    return @import("core").checkpoint_identity.fullIdentity(gpa, io, dir);
}
