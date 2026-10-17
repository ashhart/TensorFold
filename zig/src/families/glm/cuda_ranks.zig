//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.

const std = @import("std");
const cuda = @import("cuda");
const heat = @import("heat");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The forward's knobs a rank reads from its environment (the Python family's names and defaults).
pub const Settings = struct {
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows) or gather
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole).
    heat: ?heat.Bands = null, // TF_HEAT_HIGH and TF_HEAT_LOW (upstream's names): a prompt chunk waits on every rank's hottest zone
    layers: u32 = 0, // TF_GLM_LAYERS: the first this many layers only, a thin model for one-GPU checks (0: all)

    /// The settings `get` (an environment lookup) gives; `problem` names a value that is not one of the choices.
    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        var s: Settings = .{};
        if (get("TF_GLM_PREFILL_REDUCE")) |v| {
            if (std.mem.eql(u8, v, "split")) s.split = true else if (std.mem.eql(u8, v, "gather")) s.split = false else {
                problem.* = "TF_GLM_PREFILL_REDUCE: split or gather";
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
        s.heat = heat.bandsFrom(get("TF_HEAT_HIGH"), get("TF_HEAT_LOW")) catch {
            problem.* = "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high";
            return error.BadSetting;
        };
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
    layout: u32 = 6, // this struct's version
    world: u32,
    rank: u32,
    split: u32,
    kda_wide: u32,
    pieces: u32,
    layers: u32, // TF_GLM_LAYERS (0: all)
    mtp: u32, // 1: the MTP head loads and runs (drafts on); 2: and DFlash2 (--drafter), whose rounds gather
    vision: u32 = 0, // --vision's image tokens (0: off): rank 0 sends image rows every rank must hold
    capacity: u64, // tokens a sequence holds
    heat_high: f64, // the heat bands in degrees (0 and 0: off): every rank must wait alike
    heat_low: f64,
    config: [32]u8, // sha256 of config.json
    program: [32]u8, // sha256 of this executable
    checkpoint: [64]u8, // the checkpoint folder's last name (the snapshot's revision), zero padded

    pub const Field = enum {
        layout,
        rank,
        world,
        split,
        kda_wide,
        pieces,
        layers,
        mtp,
        vision,
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
        checkpoint_dir: []const u8,
    ) Fingerprint {
        var f: Fingerprint = .{
            .world = @intCast(world),
            .rank = @intCast(rank),
            .split = @intFromBool(s.split),
            .kda_wide = @intFromBool(s.kda_wide),
            .pieces = s.pieces,
            .layers = s.layers,
            .mtp = @intFromBool(mtp),
            .capacity = capacity,
            .heat_high = if (s.heat) |b| b.high_c else 0,
            .heat_low = if (s.heat) |b| b.low_c else 0,
            .config = config,
            .program = program,
            .checkpoint = @splat(0),
        };
        const name = std.fs.path.basename(std.mem.trimEnd(u8, checkpoint_dir, "/"));
        const n = @min(name.len, f.checkpoint.len);
        @memcpy(f.checkpoint[0..n], name[0..n]);
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
            "world",
            "split",
            "kda_wide",
            "pieces",
            "layers",
            "mtp",
            "vision",
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

fn explain(f: Fingerprint.Field) []const u8 {
    return switch (f) {
        .layout => "a different build of this check",
        .rank => "a rank number given twice or out of order",
        .world => "the ranks were started with different world sizes",
        .split => "TF_GLM_PREFILL_REDUCE",
        .kda_wide => "TF_GLM_KDA_DECODE_WIDE",
        .pieces => "TF_GLM_OVERLAP_PIECES",
        .layers => "TF_GLM_LAYERS",
        .mtp => "--no-drafts given to some ranks and not others",
        .vision => "--vision (or its image tokens) given to some ranks and not others",
        .capacity => "the sequence capacity (the context)",
        .heat_high => "TF_HEAT_HIGH",
        .heat_low => "TF_HEAT_LOW",
        .config => "config.json",
        .program => "the executable",
        .checkpoint => "the checkpoint's snapshot folder",
    };
}

fn fake(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "gather";
    if (std.mem.eql(u8, name, "TF_GLM_OVERLAP_PIECES")) return "2";
    if (std.mem.eql(u8, name, "TF_HEAT_HIGH")) return "92";
    if (std.mem.eql(u8, name, "TF_HEAT_LOW")) return "88";
    return null;
}

fn bad(name: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, name, "TF_GLM_PREFILL_REDUCE")) return "sum";
    return null;
}

test "settings from the environment: the defaults, the choices, and a refusal naming a bad value" {
    var problem: []const u8 = "";
    const s = try Settings.read(&fake, &problem);
    try std.testing.expect(!s.split and s.kda_wide and s.pieces == 2 and s.heat.?.high_c == 92 and s.heat.?.low_c == 88);
    try std.testing.expectError(error.BadSetting, Settings.read(&bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PREFILL_REDUCE: split or gather", problem);
}

test "fingerprints: agreeing ranks pass; the first differing field and rank are named" {
    const h0: [32]u8 = @splat(1);
    const h1: [32]u8 = @splat(2);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, "/hub/snapshots/423acf37/"),
        Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, "/hub/snapshots/423acf37"),
    };
    try std.testing.expect(mismatch(&all) == null);
    all[1].program = h1;
    try std.testing.expectEqual(Fingerprint.Field.program, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{ .split = false }, 8192, true, h0, h0, "/hub/snapshots/423acf37");
    try std.testing.expectEqual(Mismatch{ .field = .split, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 0, .{}, 8192, true, h0, h0, "/hub/snapshots/423acf37");
    try std.testing.expectEqual(Fingerprint.Field.rank, mismatch(&all).?.field);
    const heat_on: Settings = .{ .heat = .{ .high_c = 92, .low_c = 88 } };
    all[1] = Fingerprint.of(2, 1, heat_on, 8192, true, h0, h0, "/hub/snapshots/423acf37");
    try std.testing.expectEqual(Fingerprint.Field.heat_high, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, "/hub/snapshots/other");
    try std.testing.expectEqual(Fingerprint.Field.checkpoint, mismatch(&all).?.field);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, false, h0, h0, "/hub/snapshots/423acf37"); // --no-drafts on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .mtp, .rank = 1 }, mismatch(&all).?);
    all[1] = Fingerprint.of(2, 1, .{}, 8192, true, h0, h0, "/hub/snapshots/423acf37");
    all[1].vision = 4096; // --vision on rank 1 only
    try std.testing.expectEqual(Mismatch{ .field = .vision, .rank = 1 }, mismatch(&all).?);
}
