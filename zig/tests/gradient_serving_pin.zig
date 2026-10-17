//! Byte identities of the maintained #952 serving kernels and launch wrappers, with only additive exports removed.
const std = @import("std");
const manifest = @embedFile("manifest.tsv");
const Pin = struct { path: []const u8, hash: []const u8, mode: []const u8 };
const count = blk: {
    @setEvalBranchQuota(2000000);
    break :blk std.mem.count(u8, manifest, "\n");
};
fn pins() [count]Pin {
    @setEvalBranchQuota(2000000);
    var out: [count]Pin = undefined;
    var lines = std.mem.tokenizeScalar(u8, manifest, '\n');
    var i: usize = 0;
    while (lines.next()) |line| : (i += 1) {
        var cols = std.mem.splitScalar(u8, line, '\t');
        out[i] = .{ .path = cols.next().?, .hash = cols.next().?, .mode = cols.next().? };
    }
    return out;
}
const all = blk: {
    @setEvalBranchQuota(2000000);
    break :blk pins();
};
test "serving identities or the explicit learning-extension identities" {
    @setEvalBranchQuota(2000000);
    inline for (all) |pin| {
        const bytes = @embedFile(pin.path);
        var data: []const u8 = bytes;
        if (!std.mem.eql(u8, pin.mode, "raw")) {
            const marker = if (std.mem.eql(u8, pin.mode, "family")) "pub const fn_back" else "pub const gdn_train";
            data = std.mem.trim(u8, bytes[0..std.mem.indexOf(u8, bytes, marker).?], " \r\n\t");
        }
        var sha = std.crypto.hash.sha2.Sha256.init(.{});
        if (std.mem.eql(u8, pin.mode, "root")) {
            var lines = std.mem.tokenizeScalar(u8, data, '\n');
            while (lines.next()) |line| {
                const s = std.mem.trim(u8, line, " \r\t");
                if (s.len == 0 or std.mem.eql(u8, s, "_ = gdn_train;") or std.mem.eql(u8, s, "_ = train_q4;") or std.mem.eql(u8, s, "_ = train_attn;")) continue;
                sha.update(s);
                sha.update("\n");
            }
        } else sha.update(data);
        var digest: [32]u8 = undefined;
        sha.final(&digest);
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (!std.mem.eql(u8, &hex, pin.hash) and !approved(pin, bytes)) {
            std.debug.print("serving source changed: {s}\n", .{pin.path});
            return error.TestUnexpectedResult;
        }
    }
}

fn approved(pin: Pin, bytes: []const u8) bool {
    var lines = std.mem.tokenizeScalar(u8, @embedFile("approved.tsv"), '\n');
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        if (!std.mem.eql(u8, cols.next().?, pin.path)) continue;
        const wanted = cols.next().?;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), wanted)) return false;
        const prefix = std.fmt.parseInt(usize, cols.next().?, 10) catch return false;
        if (prefix > 0) {
            if (prefix > bytes.len) return false;
            std.crypto.hash.sha2.Sha256.hash(bytes[0..prefix], &digest, .{});
            if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), pin.hash)) return false;
        }
        return true;
    }
    return false;
}
