//! The execution witness's Metal path: Qwen3.5-2B, one Mac, MLX affine 4-bit, text only, every weight byte.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen35;
const disk_identity = @import("disk_identity.zig");
const weights_witness = @import("weights_witness.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const field = weights_witness.field;

/// The one geometry `q.config` admits: the 2B, every module affine 4-bit in groups of 64, text only.
pub const path = "qwen3_5-2b/metal/tp1/mlx-q4g64/text";

/// Whether `o` may take the path before the weights load: the flag, one Mac, text, nothing learned or drafted.
pub const wanted = weights_witness.wanted;

/// What the Metal path binds beside the weights: the kernels, how this OS compiles them, the GPU, the profile.
pub const Metal = struct {
    pub const domain = "tensorfold metal engine witness v1";
    /// Every library the load compiles, as `kernels` hashes them.
    kernels: [32]u8 = @splat(0),
    /// The chip, hashed with the OS build as Metal's kept states key them: the OS's compiler builds the GPU code.
    device: []const u8 = "",
    tensor_units: bool = false,
    /// The compile options every library takes on this OS: math mode, math functions and language version.
    compile: [3]u64 = @splat(0),
    profile: disk_identity.Profile = .{ .family = "qwen3_5", .backend = "metal" },
    /// The window, the streams, the prompt step, the lane rows the exactness check left and KV pages (1) or not (0).
    words: [5]u64 = @splat(0),

    /// Nothing beyond the checkpoint: the kernels are source text in the binary, compiled at load.
    pub fn stamp(_: *Metal, _: Allocator, _: std.Io, _: *std.ArrayList([]const u8)) !void {}

    pub fn keep(b: *const Metal, h: *Sha256, _: Allocator, _: std.Io) !void {
        field(h, &b.kernels);
        field(h, std.mem.sliceAsBytes(&b.compile));
        disk_identity.machine(h, b.device);
        field(h, &.{@intFromBool(b.tensor_units)});
        disk_identity.profile(h, b.profile);
    }

    pub fn tail(b: *const Metal) []const u8 {
        return std.mem.sliceAsBytes(&b.words);
    }
};

/// One engine's attestation: stamps from before its load, facts from after it, the hash once the server is ready.
pub const Witness = weights_witness.Witness(Metal);

/// Stamps config.json, the index and every shard it names, before the load.
pub fn open(gpa: Allocator, io: std.Io, dir: []const u8) !*Witness {
    return Witness.open(gpa, io, dir, .{});
}

/// `open`, or when a file cannot be stamped a witness that reads hash_failed; only OutOfMemory fails the load.
pub fn openOrFailed(gpa: Allocator, io: std.Io, dir: []const u8) error{OutOfMemory}!*Witness {
    return Witness.openOrFailed(gpa, io, dir, .{});
}

/// Every library `q.kernels.load` compiles: its source, its functions in pipeline order and a projection's tile.
pub fn kernels() [32]u8 {
    var h = Sha256.init(.{});
    for (q.kernels.libraries) |l| {
        field(&h, l.source);
        for (l.functions) |f| field(&h, f);
        const tile = [2]u64{ l.columns, l.threads };
        field(&h, std.mem.sliceAsBytes(&tile));
    }
    return h.finalResult();
}

/// After the load: the kernels, this Mac's GPU and compile options, and the served shape; false: no memory for them.
pub fn bind(w: *Witness, device: mtl.Device, words: [5]u64) bool {
    const o = q.kernels.options();
    const compile = [3]u64{ @intCast(@backingInt(o.math_mode)), @intCast(@backingInt(o.math_functions)), o.language };
    const name = w.arena.allocator().dupe(u8, std.mem.span(device.name())) catch return false;
    set(w, kernels(), name, device.tensorUnits(), compile, words);
    return true;
}

/// The facts `bind` reads from the binary and the Mac, given directly.
pub fn set(w: *Witness, kernel_digest: [32]u8, device: []const u8, tensor_units: bool, compile: [3]u64, words: [5]u64) void {
    w.path = path;
    w.backend.kernels = kernel_digest;
    w.backend.device = device;
    w.backend.tensor_units = tensor_units;
    w.backend.compile = compile;
    w.backend.words = words;
}

/// The witness the load took onto the path, or null (no memory for the facts: unsupported_path).
pub fn bound(w: ?*Witness, device: mtl.Device, words: [5]u64) ?*Witness {
    const x = w orelse return null;
    return if (bind(x, device, words)) x else null;
}

/// A witness the host did not adopt (a refusal or failure after `open`), freed.
pub const release = Witness.release;

/// `Info.witness` for an adopted witness on its path; null otherwise (unsupported_path).
pub const source = Witness.sourceOf;

test {
    _ = @import("metal_witness_test.zig");
}
