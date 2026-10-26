//! Bounded expert slabs. The owner drains GPU consumers before demand and joins reads before release.
const std = @import("std");
const cache = @import("cache.zig");
const reader = @import("reader.zig");
pub const components = 9;
pub const worker_stack_bytes = 256 * 1024;
pub const Expert = [components]reader.Span;

pub fn Arena(comptime Reader: type) type {
    return struct {
        const Self = @This();
        pub const Job = struct { ticket: cache.Ticket, failure: ?anyerror = null };
        a: std.mem.Allocator,
        source: Reader,
        experts: []const Expert,
        strides: [components]usize,
        slabs: [components][]u8,
        clock: cache.Cache,
        jobs: []Job,
        job_count: usize = 0,
        worker: ?std.Thread = null,

        /// Slabs belong to the GPU owner. No separate selected-expert weight table or staging copy is made.
        pub fn init(a: std.mem.Allocator, source: Reader, experts: []const Expert, strides: [components]usize, slabs: [components][]u8) !Self {
            return initMapped(a, source, experts, strides, slabs, null);
        }

        pub fn initMapped(a: std.mem.Allocator, source: Reader, experts: []const Expert, strides: [components]usize, slabs: [components][]u8, map: ?[]i32) !Self {
            if (experts.len == 0) return error.NoExperts;
            if (strides[0] == 0 or slabs[0].len % strides[0] != 0) return error.InvalidSlab;
            const capacity = slabs[0].len / strides[0];
            for (strides, slabs) |stride, slab| {
                if (stride == 0 or slab.len != try std.math.mul(usize, capacity, stride)) return error.InvalidSlab;
            }
            for (experts) |expert| for (expert, strides) |span, stride| {
                if (span.length > stride) return error.InvalidSlab;
                try reader.validate(span, try source.fileSize(span.file), span.length);
            };
            if (map) |m| if (m.len != experts.len) return error.InvalidPoolShape;
            var clock = if (map) |m| try cache.Cache.initMap(a, m, capacity) else try cache.Cache.init(a, experts.len, capacity);
            errdefer clock.deinit();
            const jobs = try a.alloc(Job, capacity);
            return .{ .a = a, .source = source, .experts = experts, .strides = strides, .slabs = slabs, .clock = clock, .jobs = jobs };
        }

        pub fn deinit(s: *Self) void {
            s.join() catch {};
            s.clock.release();
            s.clock.deinit();
            s.a.free(s.jobs);
        }

        fn destinations(s: *Self, t: cache.Ticket) [components][]u8 {
            var out: [components][]u8 = undefined;
            for (&out, s.strides, s.slabs, s.experts[t.key]) |*dst, stride, slab, span|
                dst.* = slab[t.slot * stride ..][0..span.length];
            return out;
        }

        fn execute(s: *Self) void {
            for (s.jobs[0..s.job_count]) |*job| {
                const dst = s.destinations(job.ticket);
                reader.expert(s.source, &s.experts[job.ticket.key], &dst) catch |err| {
                    job.failure = err;
                };
            }
        }

        /// Worker writes only reserved slab ranges. Publication and cache policy stay on the owner thread.
        pub fn join(s: *Self) !void {
            if (s.worker) |worker| {
                worker.join();
                s.worker = null;
            }
            var failure: ?anyerror = null;
            for (s.jobs[0..s.job_count]) |job| {
                if (job.failure) |err| {
                    try s.clock.abort(job.ticket);
                    failure = failure orelse err;
                } else try s.clock.publish(job.ticket);
            }
            s.job_count = 0;
            if (failure) |err| return err;
        }

        fn reserve(s: *Self, key: usize, predicted: bool) !void {
            if (try s.clock.reserve(key, predicted)) |ticket| {
                s.jobs[s.job_count] = .{ .ticket = ticket };
                s.job_count += 1;
            }
        }

        /// Caller first drains all commands using the preceding leases; successful return protects this union.
        pub fn demand(s: *Self, keys: []const usize) !void {
            try s.join();
            s.clock.release();
            if (keys.len == 0) return error.NoExperts;
            try s.clock.protect(keys);
            errdefer s.clock.release();
            for (keys) |key| s.reserve(key, false) catch |err| {
                for (s.jobs[0..s.job_count]) |job| try s.clock.abort(job.ticket);
                s.job_count = 0;
                return err;
            };
            s.execute();
            try s.join();
            for (keys) |key| if (try s.clock.resident(key) == null) return error.ExpertNotReady;
        }

        /// Capacity pressure drops predictions, never demand. Input IDs are copied into owned read tickets.
        pub fn forecast(s: *Self, keys: []const usize) !void {
            for (keys) |key| if (key >= s.experts.len) return error.InvalidExpert;
            try s.join();
            for (keys) |key| s.reserve(key, true) catch |err| switch (err) {
                error.NoEvictableSlot => break,
                else => return err,
            };
            if (s.job_count == 0) return;
            s.worker = std.Thread.spawn(.{ .stack_size = worker_stack_bytes }, execute, .{s}) catch |err| {
                for (s.jobs[0..s.job_count]) |job| try s.clock.abort(job.ticket);
                s.job_count = 0;
                return err;
            };
        }
    };
}
