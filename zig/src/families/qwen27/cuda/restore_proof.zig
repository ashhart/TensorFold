//! Bounded provenance binds a successful full-N restore to one stream and its exclusively owned Seq.
const std = @import("std");
pub const Proof = struct { stream: *const anyopaque, seq: *const anyopaque, tokens: usize };
pub const Proofs = struct {
    items: [16]?Proof = @splat(null),

    pub fn mark(p: *Proofs, stream: *const anyopaque, seq: *const anyopaque, tokens: usize) !void {
        if (tokens == 0 or tokens > 512) return error.InvalidSeqRestoreProof;
        for (&p.items) |*slot| if (slot.*) |existing| {
            if (existing.stream == stream) return error.DuplicateSeqRestoreProof;
        };
        for (&p.items) |*slot| if (slot.* == null) {
            slot.* = .{ .stream = stream, .seq = seq, .tokens = tokens };
            return;
        };
        return error.SeqRestoreProofCapacity;
    }

    /// A read-only binding check leaves the release-time pool proof intact.
    pub fn matches(p: *const Proofs, stream: *const anyopaque, seq: *const anyopaque, tokens: usize) bool {
        for (p.items) |slot| if (slot) |proof| {
            if (proof.stream == stream) return proof.seq == seq and proof.tokens == tokens;
        };
        return false;
    }

    pub fn remove(p: *Proofs, stream: *const anyopaque) ?Proof {
        for (&p.items) |*slot| if (slot.*) |existing| {
            if (existing.stream != stream) continue;
            slot.* = null;
            return existing;
        };
        return null;
    }
};

test "provenance binds stream and Seq once and is cleared before pointer reuse" {
    var p: Proofs = .{};
    var streams: [2]u8 = @splat(0);
    var seqs: [2]u8 = @splat(0);
    try p.mark(&streams[0], &seqs[0], 14);
    try std.testing.expect(p.remove(&streams[1]) == null);
    try std.testing.expectError(error.DuplicateSeqRestoreProof, p.mark(&streams[0], &seqs[1], 14));
    const proof = p.remove(&streams[0]).?;
    try std.testing.expect(proof.seq == @as(*const anyopaque, &seqs[0]) and proof.seq != @as(*const anyopaque, &seqs[1]) and proof.tokens == 14);
    try std.testing.expect(p.remove(&streams[0]) == null);
    try p.mark(&streams[0], &seqs[1], 31);
    const next = p.remove(&streams[0]).?;
    try std.testing.expect(next.seq == @as(*const anyopaque, &seqs[1]) and next.tokens == 31);
}

test "fixed proof ownership refuses overflow and unsupported restoration lengths" {
    var p: Proofs = .{};
    var streams: [17]u8 = @splat(0);
    var seqs: [17]u8 = @splat(0);
    try std.testing.expectError(error.InvalidSeqRestoreProof, p.mark(&streams[0], &seqs[0], 0));
    try std.testing.expectError(error.InvalidSeqRestoreProof, p.mark(&streams[0], &seqs[0], 513));
    for (0..16) |i| try p.mark(&streams[i], &seqs[i], 14);
    try std.testing.expectError(error.SeqRestoreProofCapacity, p.mark(&streams[16], &seqs[16], 14));
    _ = p.remove(&streams[0]);
    try p.mark(&streams[16], &seqs[16], 31);
}

test "read-only first-logit proof rejects another owner, Seq and position without consuming release authority" {
    var proofs: Proofs = .{};
    var streams: [2]u8 = @splat(0);
    var seqs: [2]u8 = @splat(0);
    try proofs.mark(&streams[0], &seqs[0], 14);
    try std.testing.expect(proofs.matches(&streams[0], &seqs[0], 14));
    try std.testing.expect(!proofs.matches(&streams[1], &seqs[0], 14));
    try std.testing.expect(!proofs.matches(&streams[0], &seqs[1], 14));
    try std.testing.expect(!proofs.matches(&streams[0], &seqs[0], 31));
    try std.testing.expect(proofs.matches(&streams[0], &seqs[0], 14));
    _ = proofs.remove(&streams[0]);
    try std.testing.expect(!proofs.matches(&streams[0], &seqs[0], 14));
}
