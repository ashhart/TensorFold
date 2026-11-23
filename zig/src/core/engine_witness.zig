//! What an engine attests for the execution witness: its supported path and a digest of its facts, or why it cannot.

/// Why a witness is unknown; the read endpoint names it.
pub const Reason = enum { pending, unsupported_path, feature_excluded, parked, reloading, files_changed, hash_failed };

/// A supported path's label (at most 64 bytes, static) and the digest of everything the engine binds for it.
pub const Known = struct { path: []const u8, digest: [32]u8 };

/// The engine's attestation now: still hashing, known, or unknown with its reason.
pub const Facts = union(enum) { pending, unknown: Reason, known: Known };

/// `Info.witness`: read on any thread while this engine is resident; the server never reads a parked engine's.
pub const Source = struct {
    ctx: *anyopaque,
    read: *const fn (ctx: *anyopaque) Facts,
    /// Starts the engine's hashing: the server calls it once it is ready, and again for each engine a reload publishes.
    begin: ?*const fn (ctx: *anyopaque) void = null,

    pub fn facts(s: Source) Facts {
        return s.read(s.ctx);
    }

    pub fn start(s: Source) void {
        if (s.begin) |f| f(s.ctx);
    }
};
