//! cuda_ranks' checks: settings read from the environment, fingerprints agreeing or naming a field, the start probes.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;
const hashBytes = ranks.hashBytes;
const probeStart = ranks.probeStart;
const mediaFingerprint = ranks.mediaFingerprint;
const artifact = ranks.artifact;

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
    all[1] = Fingerprint.of(2, 1, .{ .pages = true }, 8192, true, h0, h0, hashBytes("/hub/snapshots/423acf37"));
    try std.testing.expectEqual(Mismatch{ .field = .pages, .rank = 1 }, mismatch(&all).?); // TF_KV_PAGES on one rank
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
    inline for (.{ "layout", "world", "rank", "pages", "split", "kda_wide", "pieces", "layers", "mtp", "vision", "video_tokens", "max_images", "max_videos", "max_frames", "capacity", "heat_high", "heat_low" }) |name| {
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
