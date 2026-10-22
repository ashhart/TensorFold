//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.

const std = @import("std");
const cuda = @import("cuda");
const heat = @import("heat");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    kv8: bool = false, // --kv-cache fp8, supplied by the native options rather than an environment override
    prompt_fixed_k: bool = false, // TF_GLM_PROMPT_K: one (default) or shape (experimental fixed K slices)
    tiled: bool = true, // TF_GLM_MM: 1 (the bf16 linears on glm_mm, tiled at load) or 0
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows), ring or gather
    ring: bool = false, // TF_GLM_PREFILL_REDUCE=ring: split, its exchanges passed along the ring (cuda_relay)
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole).
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
    layout: u32 = 11, // this struct's version
    world: u32,
    rank: u32,
    kv8: u32 = 0,
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

fn fake(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "gather";
    if (std.mem.eql(u8, name, "TF_GLM_OVERLAP_PIECES")) return "2";
    if (std.mem.eql(u8, name, "TF_HEAT_HIGH")) return "92";
    if (std.mem.eql(u8, name, "TF_HEAT_LOW")) return "88";
    if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) return "4";
    if (std.mem.eql(u8, name, "TF_GLM_HEAD_BITS")) return "8";
    return null;
}

fn ringed(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "ring";
    return null;
}

fn bad(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "sum";
    return null;
}

fn typo(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) return "abc";
    return null;
}

test "settings from the environment: the defaults, the choices, and a refusal naming a bad value" {
    var problem: []const u8 = "";
    const s = try Settings.read(&fake, &problem);
    try std.testing.expect(!s.split and s.kda_wide and s.pieces == 2 and s.heat.?.high_c == 92 and s.heat.?.low_c == 88);
    try std.testing.expect(s.dense_bits == 4 and s.head_bits == 8);
    try std.testing.expectError(error.BadSetting, Settings.read(&bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PREFILL_REDUCE: split, ring or gather", problem);
    const r = try Settings.read(&ringed, &problem);
    try std.testing.expect(r.split and r.ring);
    try std.testing.expectEqual(@as(u32, 2), Fingerprint.of(2, 0, r, 8192, true, @splat(0), @splat(0), hashBytes("/c")).split);

    try std.testing.expectError(error.BadSetting, Settings.read(&typo, &problem)); // a typo is refused, not read as bf16
    try std.testing.expectEqualStrings("TF_GLM_DENSE_BITS: 4, 6, 8 or 16", problem);
}

test "fingerprints: agreeing ranks pass; the first differing field and rank are named" {
    const h0: [32]u8 = @splat(1);
    const h1: [32]u8 = @splat(2);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37")),
        Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37")),
    };
    try std.testing.expect(mismatch(&all) == null);
    all[1].program = h1;
    try std.testing.expectEqual(Fingerprint.Field.program, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .split = false }, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Mismatch{ .field = .split, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Fingerprint.Field.rank, mismatch(&all).?.field);
    const heat_on: Settings = .{ .heat = .{ .high_c = 92, .low_c = 88 } };
    all[1] = Fingerprint.of(2, 1, heat_on, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Fingerprint.Field.heat_high, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/other"));
    try std.testing.expectEqual(Fingerprint.Field.checkpoint, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, false, h0, h0, hashBytes("/hub/snapshots/423acf37")); // --no-drafts on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .mtp, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    all[1].vision = 4096; // --vision on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .vision, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{ .dense_bits = 4 }, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37")); // packed on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .bits, .rank = 1 }, mismatch(&all).?);
}

/// A world seen from `rank`: a group that sends to or receives from a rank it cannot reach fails at its end.
const Fabric = struct {
    var world: usize = 4;
    var rank: usize = 0;
    var switched = false; // every pair reaches the other; else a ring without a switch, each rank its two neighbours
    var failing = false;
    var posted: usize = 0;
    var allocated: usize = 0;

    fn reaches(peer: c_int) bool {
        const p: usize = @intCast(peer);
        return switched or (p + 1) % world == rank or (rank + 1) % world == p;
    }
    fn post(peer: c_int) cuda.nccl.Result {
        posted += 1;
        if (!reaches(peer)) failing = true;
        return 0;
    }
    fn start() callconv(.c) cuda.nccl.Result {
        failing = false;
        return 0;
    }
    fn end() callconv(.c) cuda.nccl.Result {
        return @intFromBool(failing);
    }
    fn send(_: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return post(peer);
    }
    fn recv(_: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        return post(peer);
    }
    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        allocated += 1;
        p.* = 0x10000;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        allocated -= 1;
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    /// One rank's start probes on this fabric, with `s`; the sends and receives it posted.
    fn probe(s: Settings, r: usize, w: usize, all: bool) !usize {
        var api: cuda.abi.Api = undefined;
        api.cuMemAlloc_v2 = alloc;
        api.cuMemFree_v2 = free;
        api.cuStreamSynchronize = sync;
        const d: cuda.Driver = .{ .lib = undefined, .api = api };
        var nccl_api: cuda.nccl.Api = undefined;
        nccl_api.ncclGroupStart = &start;
        nccl_api.ncclGroupEnd = &end;
        nccl_api.ncclSend = &send;
        nccl_api.ncclRecv = &recv;
        const lib: cuda.nccl.Library = .{ .lib = undefined, .api = nccl_api };
        world = w;
        rank = r;
        switched = all;
        posted = 0;
        const got = probeStart(&d, &lib, null, .{ .d = &d, .handle = null }, s, r, w);
        try std.testing.expectEqual(@as(usize, 0), allocated); // the probe's buffer is freed, refused or not
        try got;
        return posted;
    }
};

test "a direct split start on a ring of four without a switch is refused before loading, at every rank" {
    for (0..4) |r| try std.testing.expectError(error.SplitUnreachable, Fabric.probe(.{}, r, 4, false));
    // the same start on four ranks behind a switch probes each pair once each way
    for (0..4) |r| try std.testing.expectEqual(@as(usize, 6), try Fabric.probe(.{}, r, 4, true));
    // two ranks are each other's neighbours: a direct split starts on a ring of two
    for (0..2) |r| try std.testing.expectEqual(@as(usize, 2), try Fabric.probe(.{}, r, 2, false));
}

test "a ring start on four ranks that reach only their neighbours loads without the all-peer probe" {
    var problem: []const u8 = "";
    const ring = try Settings.read(&ringed, &problem);
    for (0..4) |r| try std.testing.expectEqual(@as(usize, 0), try Fabric.probe(ring, r, 4, false));
    // gather needs no probe either
    for (0..4) |r| try std.testing.expectEqual(@as(usize, 0), try Fabric.probe(.{ .split = false }, r, 4, false));
}

pub fn mediaFingerprint(f: *Fingerprint, limits: @import("cuda_images.zig").Limits) void {
    f.vision = limits.image_tokens;
    f.video_tokens = limits.video_tokens;
    f.max_images = limits.max_images;
    f.max_videos = limits.max_videos;
    f.max_frames = limits.max_frames;
}
test "every media limit is part of the ranks' agreement" {
    const L = @import("cuda_images.zig").Limits;
    const base: L = .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 };
    inline for (.{ "video_tokens", "max_images", "max_videos", "max_frames" }) |field| {
        var all = [_]Fingerprint{
            Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
            Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same")),
        };
        mediaFingerprint(&all[0], base);
        mediaFingerprint(&all[1], base);
        try std.testing.expect(mismatch(&all) == null);
        var changed = base;
        @field(changed, field) += 1;
        mediaFingerprint(&all[1], changed);
        try std.testing.expect(mismatch(&all) != null);
        try std.testing.expectEqual(@as(usize, 1), mismatch(&all).?.rank);
    }
}

test "every scheduling and identity field that differs stops the ranks" {
    const zero = Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same"));
    const one = Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), hashBytes("/same"));
    inline for (.{ "layout", "world", "rank", "split", "kda_wide", "pieces", "layers", "mtp", "vision", "video_tokens", "max_images", "max_videos", "max_frames", "capacity", "heat_high", "heat_low" }) |name| {
        var all = [_]Fingerprint{ zero, one };
        @field(all[1], name) += 1;
        try std.testing.expectEqual(@field(Fingerprint.Field, name), mismatch(&all).?.field);
        std.mem.swap(Fingerprint, &all[0], &all[1]);
        all[0].rank = 0;
        all[1].rank = if (std.mem.eql(u8, name, "rank")) 0 else 1;
        try std.testing.expect(mismatch(&all) != null);
    }
    inline for (.{ "config", "program", "checkpoint" }) |name| {
        var all = [_]Fingerprint{ zero, one };
        @field(all[1], name)[0] = 1;
        try std.testing.expectEqual(@field(Fingerprint.Field, name), mismatch(&all).?.field);
    }
}

/// The checkpoint digest a rank sends: its metadata and every shard byte, read again only for a file that changed.
pub fn artifact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    return @import("core").checkpoint_identity.fullIdentity(gpa, io, dir);
}

test "ranks with different TF_GLM_MM choices disagree" {
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, hashBytes("/model")), Fingerprint.of(2, 1, .{ .tiled = false }, 8192, true, h, h, hashBytes("/model")) };
    try std.testing.expect(mismatch(&all) != null);
}
fn digestFixture(io: std.Io, dir: std.Io.Dir, sub: []const u8, byte: u8) !void {
    try dir.createDirPath(io, sub);
    var d = try dir.openDir(io, sub, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const header = "{\"dense\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}";
    var bytes: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], byte);
    try d.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
}

test "ranks disagree on tensor bytes under identical headers, and agree on a copy in a folder of the same name" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try digestFixture(io, tmp.dir, "a/model", 1);
    try digestFixture(io, tmp.dir, "b/model", 2);
    try digestFixture(io, tmp.dir, "c/model", 1);
    var paths: [3][:0]u8 = undefined;
    for ([_][]const u8{ "a/model", "b/model", "c/model" }, &paths) |sub, *p| p.* = try tmp.dir.realPathFileAlloc(io, sub, a);
    defer for (paths) |p| a.free(p);
    const h: [32]u8 = @splat(7);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, h, h, try artifact(a, io, paths[0])), Fingerprint.of(2, 1, .{}, 8192, true, h, h, try artifact(a, io, paths[1])) };
    try std.testing.expectEqual(Mismatch{ .field = .checkpoint, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h, h, try artifact(a, io, paths[2]));
    try std.testing.expect(mismatch(&all) == null);
}

test "prompt K experiment is explicit and all ranks must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn one(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "one" else null;
        }
        fn untiled(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "shape" else if (std.mem.eql(u8, key, "TF_GLM_MM")) "0" else null;
        }

        fn shape(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "shape" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "auto" else null;
        }
    };
    var problem: []const u8 = "";
    const selected = try Settings.read(Reader.shape, &problem);
    try std.testing.expect(selected.prompt_fixed_k);
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).prompt_fixed_k);
    try std.testing.expect(!(try Settings.read(Reader.one, &problem)).prompt_fixed_k);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.untiled, &problem));
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0)), Fingerprint.of(2, 1, selected, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(Fingerprint.Field.prompt_fixed_k, mismatch(&all).?.field);
    all[0].prompt_fixed_k = 1;
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
}
