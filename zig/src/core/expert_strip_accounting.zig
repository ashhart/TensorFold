//! Exact host-read and publication counters with checksummed portable snapshots.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Totals = struct {
    read_calls: u64 = 0,
    read_bytes: u64 = 0,
    verified_stripes: u64 = 0,
    verified_bytes: u64 = 0,
    committed_tensors: u64 = 0,
    committed_bytes: u64 = 0,
    fallback_selections: u64 = 0,
    cancelled_tensors: u64 = 0,
    failed_tensors: u64 = 0,
};

const fields = 9;
const prefix = 9;
pub const encoded_size = prefix + fields * 8 + 32;
const magic = "TFSTRIPE";

fn put(totals: *const Totals, out: []u8) void {
    const values = [_]u64{
        totals.read_calls,
        totals.read_bytes,
        totals.verified_stripes,
        totals.verified_bytes,
        totals.committed_tensors,
        totals.committed_bytes,
        totals.fallback_selections,
        totals.cancelled_tensors,
        totals.failed_tensors,
    };
    for (values, 0..) |value, i| std.mem.writeInt(u64, out[prefix + i * 8 ..][0..8], value, .little);
}

pub fn encode(totals: Totals) [encoded_size]u8 {
    var out: [encoded_size]u8 = undefined;
    @memcpy(out[0..8], magic);
    out[8] = 1;
    put(&totals, out[0 .. encoded_size - 32]);
    var digest: [32]u8 = undefined;
    Sha256.hash(out[0 .. encoded_size - 32], &digest, .{});
    @memcpy(out[encoded_size - 32 ..], &digest);
    return out;
}

pub fn decode(bytes: []const u8) !Totals {
    if (bytes.len != encoded_size or !std.mem.eql(u8, bytes[0..8], magic) or bytes[8] != 1)
        return error.InvalidAccountingSnapshot;
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes[0 .. encoded_size - 32], &digest, .{});
    if (!std.mem.eql(u8, &digest, bytes[encoded_size - 32 ..])) return error.CorruptAccountingSnapshot;
    var values: [fields]u64 = undefined;
    for (&values, 0..) |*value, i| value.* = std.mem.readInt(u64, bytes[prefix + i * 8 ..][0..8], .little);
    return .{
        .read_calls = values[0],
        .read_bytes = values[1],
        .verified_stripes = values[2],
        .verified_bytes = values[3],
        .committed_tensors = values[4],
        .committed_bytes = values[5],
        .fallback_selections = values[6],
        .cancelled_tensors = values[7],
        .failed_tensors = values[8],
    };
}

fn add(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.AccountingOverflow;
}

fn addCommitted(totals: *Totals, bytes: u64) !void {
    const tensors = try add(totals.committed_tensors, 1);
    const all_bytes = try add(totals.committed_bytes, bytes);
    totals.committed_tensors = tensors;
    totals.committed_bytes = all_bytes;
}

pub const Accounting = struct {
    totals: Totals = .{},

    pub fn snapshot(a: *const Accounting) Totals {
        return a.totals;
    }

    pub fn restore(a: *Accounting, totals: Totals) void {
        a.totals = totals;
    }

    pub fn canCommit(a: *const Accounting, bytes: u64) !void {
        var copy = a.totals;
        try addCommitted(&copy, bytes);
    }

    pub fn addRead(a: *Accounting, bytes: usize) !void {
        const calls = try add(a.totals.read_calls, 1);
        const read_bytes = try add(a.totals.read_bytes, bytes);
        a.totals.read_calls = calls;
        a.totals.read_bytes = read_bytes;
    }

    pub fn addVerified(a: *Accounting, bytes: usize) !void {
        const stripes = try add(a.totals.verified_stripes, 1);
        const verified_bytes = try add(a.totals.verified_bytes, bytes);
        a.totals.verified_stripes = stripes;
        a.totals.verified_bytes = verified_bytes;
    }

    pub fn addFallback(a: *Accounting) !void {
        a.totals.fallback_selections = try add(a.totals.fallback_selections, 1);
    }

    pub fn commit(a: *Accounting, bytes: u64) void {
        addCommitted(&a.totals, bytes) catch unreachable;
    }

    pub fn recordFailure(a: *Accounting, cancelled: bool) !void {
        if (cancelled) {
            a.totals.cancelled_tensors = try add(a.totals.cancelled_tensors, 1);
        } else {
            a.totals.failed_tensors = try add(a.totals.failed_tensors, 1);
        }
    }
};
