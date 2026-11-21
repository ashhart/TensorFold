//! Host-only exact reads, checksums and atomic replica selection; all device DMA stays behind Sink.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const types = @import("expert_strip_types.zig");
const planning = @import("expert_strip_plan.zig");
fn cyclic(index: usize, shift: usize, modulus: usize) usize {
    const until_wrap = modulus - index;
    return if (shift >= until_wrap) shift - until_wrap else index + shift;
}

const accounting = @import("expert_strip_accounting.zig");

pub const Reader = struct {
    plan: *const planning.Plan,
    selected: []std.atomic.Value(usize),

    pub fn init(a: std.mem.Allocator, plan: *const planning.Plan) !Reader {
        const selected = try a.alloc(std.atomic.Value(usize), plan.stripes.len);
        for (selected) |*choice| choice.* = .init(0);
        return .{ .plan = plan, .selected = selected };
    }

    pub fn deinit(r: *Reader, a: std.mem.Allocator) void {
        a.free(r.selected);
        r.* = undefined;
    }

    pub fn selectedCandidate(r: *const Reader, stripe: usize) !usize {
        if (stripe >= r.selected.len) return error.InvalidStripe;
        return r.selected[stripe].load(.acquire);
    }

    fn cancelled(cancel: ?*const types.Cancellation) bool {
        return if (cancel) |token| token.isCancelled() else false;
    }

    fn checkCancel(cancel: ?*const types.Cancellation) !void {
        if (cancelled(cancel)) return error.Cancelled;
    }

    fn sameArtifact(host: types.Host, candidate: types.Candidate, expected: types.Identity) !void {
        const actual = try host.artifact(host.ctx, candidate);
        if (!actual.eql(expected)) return error.WrongArtifact;
    }

    fn attempt(
        r: *Reader,
        host: types.Host,
        candidate: types.Candidate,
        destination: []u8,
        expected: types.Digest,
        cancel: ?*const types.Cancellation,
        stats: *accounting.Accounting,
    ) !void {
        try checkCancel(cancel);
        try sameArtifact(host, candidate, r.plan.identity);
        try checkCancel(cancel);
        var done: usize = 0;
        while (done < destination.len) {
            try checkCancel(cancel);
            const offset = std.math.add(u64, candidate.offset, done) catch return error.InvalidTensorSpan;
            const n = try host.read(host.ctx, candidate, offset, destination[done..], cancel);
            if (n == 0) return error.IncompleteRead;
            if (n > destination.len - done) return error.InvalidReadCount;
            try stats.addRead(n);
            done += n;
        }
        try checkCancel(cancel);
        var digest: types.Digest = undefined;
        Sha256.hash(destination, &digest, .{});
        if (!std.mem.eql(u8, &digest, &expected)) return error.ChecksumMismatch;
        try sameArtifact(host, candidate, r.plan.identity);
        try checkCancel(cancel);
    }

    fn readStripe(
        r: *Reader,
        host: types.Host,
        stripe_index: usize,
        destination: []u8,
        checksum: types.Digest,
        cancel: ?*const types.Cancellation,
        stats: *accounting.Accounting,
    ) !void {
        const stripe = r.plan.stripes[stripe_index];
        const current = r.selected[stripe_index].load(.acquire);
        if (current >= stripe.candidates.len) return error.InvalidFallbackSelection;
        var failure: ?anyerror = null;
        for (0..stripe.candidates.len) |attempt_index| {
            const candidate_index = cyclic(current, attempt_index, stripe.candidates.len);
            const candidate = stripe.candidates[candidate_index];
            r.attempt(host, candidate, destination, checksum, cancel, stats) catch |err| {
                if (err == error.Cancelled or err == error.AccountingOverflow) return err;
                failure = failure orelse err;
                continue;
            };
            if (candidate_index != current) {
                if (r.selected[stripe_index].cmpxchgStrong(current, candidate_index, .acq_rel, .acquire) == null)
                    try stats.addFallback();
            }
            try stats.addVerified(destination.len);
            return;
        }
        return failure orelse error.NoReplica;
    }

    pub fn read(
        r: *Reader,
        host: types.Host,
        output: []u8,
        sink: ?types.Sink,
        cancel: ?*const types.Cancellation,
        stats: *accounting.Accounting,
    ) !void {
        r.run(host, output, sink, cancel, stats) catch |err| {
            stats.recordFailure(err == error.Cancelled) catch return error.AccountingOverflow;
            return err;
        };
    }

    fn run(
        r: *Reader,
        host: types.Host,
        output: []u8,
        sink: ?types.Sink,
        cancel: ?*const types.Cancellation,
        stats: *accounting.Accounting,
    ) !void {
        if (output.len != r.plan.tensor_bytes) return error.InvalidOutputSize;
        if (sink) |s| if (s.stage == null or s.commit == null) return error.InvalidDeviceSink;
        try checkCancel(cancel);
        try stats.canCommit(r.plan.tensor_bytes);
        var staged = false;
        errdefer if (staged) if (sink) |s| if (s.abort) |abort| abort(s.ctx);
        for (r.plan.stripes, 0..) |stripe, i| {
            const at: usize = @intCast(stripe.tensor_offset);
            const destination = output[at..][0..stripe.length];
            try r.readStripe(host, i, destination, stripe.checksum, cancel, stats);
            try checkCancel(cancel);
            if (sink) |s| {
                const device = try r.plan.deviceForStripe(i);
                staged = true;
                try s.stage.?(s.ctx, device, i, destination);
            }
        }
        try checkCancel(cancel);
        if (sink) |s| {
            try s.commit.?(s.ctx);
            staged = false;
        }
        stats.commit(r.plan.tensor_bytes);
    }
};
