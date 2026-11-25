//! The execution witness's first CUDA path: Qwen3.5-2B, one rank, MLX affine 4-bit, text only, every weight byte.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const cuda_disk = @import("cuda_disk.zig");
const disk_identity = @import("disk_identity.zig");
const weights_witness = @import("weights_witness.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;

/// Whether `o` may take the first path before the weights load: the flag, dense Qwen, one rank, text, nothing learned.
pub fn wanted(comptime F: type, o: api.Open) bool {
    if (comptime F != @import("qwen35").native) return false;
    return weights_witness.wanted(o);
}

/// What the CUDA path binds beside the weights: the kernels it loads, the GPU's facts and the family's.
pub const Cuda = struct {
    pub const domain = "tensorfold cuda engine witness v1";
    kernels: ?[]const u8 = null,
    device: cuda_disk.Device = undefined,
    facts: []const u64 = &.{},
    profile: disk_identity.Profile = .{ .family = "qwen3_5", .backend = "cuda" },
    /// The SM count the kernels' tiles are weighed against, the window, the streams and the staggered segments.
    words: [4]u64 = @splat(0),

    /// A captured set's aot.json and every cubin it names, stamped beside the checkpoint's files.
    pub fn stamp(b: *Cuda, a: Allocator, io: std.Io, paths: *std.ArrayList([]const u8)) !void {
        const k = try a.dupe(u8, b.kernels orelse return);
        b.kernels = k;
        try capturedPaths(a, io, k, paths);
    }

    /// The kept identity's components: the embedded kernels, a captured set, the GPU, the family and the profile.
    pub fn keep(b: *const Cuda, h: *Sha256, a: Allocator, io: std.Io) !void {
        try cuda_disk.components(h, a, io, .{ .binary = null, .checkpoint = null, .kernels = b.kernels, .device = b.device, .facts = b.facts, .probed = null, .profile = b.profile });
    }

    pub fn tail(b: *const Cuda) []const u8 {
        return std.mem.sliceAsBytes(&b.words);
    }
};

/// One engine's attestation: stamps from before its load, facts from after it, the hash once the server is ready.
pub const Witness = weights_witness.Witness(Cuda);

/// Stamps config.json, the index, every shard it names and a captured set's aot.json and cubins, before the load.
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, kernels: ?[]const u8) !*Witness {
    return Witness.open(gpa, io, dir, .{ .kernels = kernels });
}

/// `open`, or when a file cannot be stamped a witness that reads hash_failed; only OutOfMemory fails the load.
pub fn openOrFailed(gpa: Allocator, io: std.Io, dir: []const u8, kernels: ?[]const u8) error{OutOfMemory}!*Witness {
    return Witness.openOrFailed(gpa, io, dir, .{ .kernels = kernels });
}

/// After the load: the path the loaded geometry takes, this GPU's facts and the family's; false: no path.
pub fn bind(w: *Witness, comptime K: type, target: anytype, ctx: *const cuda.Context, window: u64, streams: u64, segments: u64) bool {
    const path = K.witnessPath(target) orelse return false;
    const a = w.arena.allocator();
    const name_buf = a.alloc(u8, 256) catch return false;
    const device = cuda_disk.deviceOf(ctx, name_buf) catch return false;
    const sms = ctx.attribute(.multiprocessor_count) catch return false;
    const facts = K.facts(target);
    set(w, path, device, a.dupe(u64, &facts) catch return false, .{ @intCast(sms), window, streams, segments });
    return true;
}

/// The facts `bind` reads from the device and family, given directly.
pub fn set(w: *Witness, path: []const u8, device: cuda_disk.Device, facts: []const u64, words: [4]u64) void {
    w.path = path;
    w.backend.device = device;
    w.backend.facts = facts;
    w.backend.words = words;
}

/// The witness `F`'s load took onto its path, or null (any other family, geometry or format: unsupported_path).
pub fn bound(comptime F: type, w: ?*Witness, loaded: anytype, ctx: *const cuda.Context, words: [3]u64) ?*Witness {
    if (comptime F != @import("qwen35").native) return null;
    const x = w orelse return null;
    return if (bind(x, F.kept_disk, loaded.target, ctx, words[0], words[1], words[2])) x else null;
}

/// A witness the host did not adopt (a refusal or failure after `open`), freed.
pub const release = Witness.release;

/// `Info.witness` for an adopted witness on its path; null otherwise (unsupported_path).
pub const source = Witness.sourceOf;

/// A captured set's manifest and each cubin it names, as `aot_identity.write` reads them.
fn capturedPaths(a: Allocator, io: std.Io, dir: []const u8, out: *std.ArrayList([]const u8)) !void {
    const manifest = try std.fs.path.join(a, &.{ dir, "aot.json" });
    try out.append(a, manifest);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, manifest, a, .limited(1 << 24));
    const specs = try cuda.aot.parseSpecs(a, text);
    for (specs.value.kernels) |k| try out.append(a, try std.fmt.allocPrint(a, "{s}/cubins/{s}.cubin", .{ dir, k.hash }));
}

test {
    _ = @import("cuda_witness_test.zig");
}
