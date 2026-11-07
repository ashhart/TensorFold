//! Bounded expert slabs. The owner drains GPU consumers before demand and joins reads before release.
const std = @import("std");
const cache = @import("cache.zig");
const reader = @import("reader.zig");
pub const components = 9;
/// The engine threads' stack: glibc refuses a smaller one when the binary's static TLS needs more (EINVAL).
pub const worker_stack_bytes = 16 << 20;
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
        ahead: []Job, // a forecast's reads, which its worker takes in turn
        ahead_count: usize = 0,
        ahead_taken: usize = 0, // of them, what the worker had when demand stopped it
        ahead_settled: usize = 0, // of them, what the owner has published or given back
        stopped: bool = true,
        next: std.atomic.Value(usize) = .init(0), // the worker's next read
        finished: std.atomic.Value(usize) = .init(0), // the reads it has finished
        worker: ?std.Thread = null,
        file_sizes: []u64,
        ahead_sizes: []u64, // the worker's own, as it can read beside demand
        io: reader.Stats = .{},
        ahead_io: reader.Stats = .{},
        demand_misses: u64 = 0,
        forecast_misses: u64 = 0,
        demand_keys: u64 = 0,
        unique_demand_keys: u64 = 0,
        worker_joins: u64 = 0,
        cancelled_forecasts: u64 = 0,

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
            const bound = try std.math.mul(usize, experts.len, components);
            var files: usize = 0;
            for (experts) |expert| for (expert, strides) |span, stride| {
                if (span.length > stride) return error.InvalidSlab;
                if (span.file >= bound) return error.NoFile;
                files = @max(files, span.file + 1);
            };
            const sizes = try a.alloc(u64, files);
            errdefer a.free(sizes);
            const ahead_sizes = try a.alloc(u64, files);
            errdefer a.free(ahead_sizes);
            @memset(sizes, reader.unchecked);
            for (experts) |expert| for (expert) |span| {
                if (sizes[span.file] == reader.unchecked) sizes[span.file] = try source.fileSize(span.file);
                try reader.validate(span, sizes[span.file], span.length);
            };
            if (map) |m| if (m.len != experts.len) return error.InvalidPoolShape;
            var clock = if (map) |m| try cache.Cache.initMap(a, m, capacity) else try cache.Cache.init(a, experts.len, capacity);
            errdefer clock.deinit();
            const jobs = try a.alloc(Job, capacity);
            errdefer a.free(jobs);
            const ahead = try a.alloc(Job, capacity);
            return .{ .a = a, .source = source, .experts = experts, .strides = strides, .slabs = slabs, .clock = clock, .jobs = jobs, .ahead = ahead, .file_sizes = sizes, .ahead_sizes = ahead_sizes };
        }

        pub fn deinit(s: *Self) void {
            s.join() catch {};
            s.clock.release();
            s.clock.deinit();
            s.a.free(s.jobs);
            s.a.free(s.ahead);
            s.a.free(s.file_sizes);
            s.a.free(s.ahead_sizes);
        }

        fn readAhead(s: *Self) void {
            while (true) {
                const i = s.next.fetchAdd(1, .acq_rel);
                if (i >= s.ahead_count) return;
                const job = &s.ahead[i];
                reader.batch(s.source, @as(*[1]Job, @ptrCast(job)), s.experts, s.strides, s.slabs, s.ahead_sizes, &s.ahead_io) catch |err| {
                    job.failure = err;
                };
                s.finished.store(i + 1, .release);
            }
        }

        /// The worker takes no more reads: those it never started give their slots back.
        fn stopAhead(s: *Self) !void {
            if (s.stopped) return;
            s.stopped = true;
            s.ahead_taken = @min(s.next.swap(s.ahead_count, .acq_rel), s.ahead_count);
            for (s.ahead[s.ahead_taken..s.ahead_count]) |job| {
                try s.clock.giveBack(job.ticket);
                s.cancelled_forecasts += 1;
            }
        }

        /// Publication stays on the owner's thread: the reads the worker has finished; returns their first failure.
        fn settleAhead(s: *Self) !void {
            var failure: ?anyerror = null;
            const finished = s.finished.load(.acquire);
            for (s.ahead[s.ahead_settled..finished]) |job| {
                if (job.failure) |err| {
                    try s.clock.abort(job.ticket);
                    failure = failure orelse err;
                } else try s.clock.publish(job.ticket);
            }
            s.ahead_settled = finished;
            if (failure) |err| return err;
        }

        /// The key of the read a stopped worker is still inside.
        fn aheadKey(s: *const Self) ?usize {
            if (s.worker == null or !s.stopped) return null;
            const at = s.finished.load(.acquire);
            return if (at < s.ahead_taken) s.ahead[at].ticket.key else null;
        }

        pub fn readingAhead(s: *const Self) bool {
            return s.aheadKey() != null;
        }

        /// Stops any forecast and waits for the read it has under way; every reservation is settled on return.
        pub fn join(s: *Self) !void {
            try s.stopAhead();
            if (s.worker) |worker| {
                worker.join();
                s.worker = null;
                s.worker_joins += 1;
                s.io.calls += s.ahead_io.calls;
                s.io.bytes += s.ahead_io.bytes;
                s.io.logical_bytes += s.ahead_io.logical_bytes;
                s.io.size_checks += s.ahead_io.size_checks;
                s.ahead_io = .{};
            }
            return s.settleAhead();
        }

        fn execute(s: *Self) void {
            reader.batch(s.source, s.jobs[0..s.job_count], s.experts, s.strides, s.slabs, s.file_sizes, &s.io) catch |err| {
                for (s.jobs[0..s.job_count]) |*job| job.failure = err;
            };
        }

        fn finishDemand(s: *Self) !void {
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

        fn reserve(s: *Self, key: usize) !void {
            if (try s.clock.reserve(key, false)) |ticket| {
                s.jobs[s.job_count] = .{ .ticket = ticket };
                s.job_count += 1;
            }
        }

        /// Caller first drains all commands using the preceding leases; successful return protects this union.
        pub fn demand(s: *Self, keys: []const usize) !void {
            for (keys) |key| if (key >= s.experts.len) {
                try s.join();
                return error.InvalidExpert;
            };
            // a read this union needs is waited for; any other runs on beside demand's own
            try s.stopAhead();
            try s.settleAhead();
            const running = s.aheadKey();
            if (running == null or std.mem.indexOfScalar(usize, keys, running.?) != null) try s.join();
            s.clock.release();
            if (keys.len == 0) return error.NoExperts;
            try s.clock.protect(keys);
            errdefer s.clock.release();
            s.demand_keys += keys.len;
            s.unique_demand_keys += s.clock.current_count;
            std.mem.sort(usize, s.clock.current[0..s.clock.current_count], {}, std.sort.asc(usize));
            for (s.clock.current[0..s.clock.current_count]) |key| s.reserve(key) catch |err| {
                for (s.jobs[0..s.job_count]) |job| try s.clock.abort(job.ticket);
                s.job_count = 0;
                return err;
            };
            s.demand_misses += s.job_count;
            s.execute();
            try s.finishDemand();
            for (s.clock.current[0..s.clock.current_count]) |key| if (try s.clock.resident(key) == null) return error.ExpertNotReady;
        }

        /// Capacity pressure drops predictions, never demand. Input IDs are copied into owned read tickets.
        pub fn forecast(s: *Self, keys: []const usize) !void {
            for (keys) |key| if (key >= s.experts.len) return error.InvalidExpert;
            try s.stopAhead();
            if (s.readingAhead()) return; // the last forecast's read is still under way: the disk has enough to do
            try s.join();
            var count: usize = 0;
            for (keys) |key| {
                const ticket = s.clock.reserve(key, true) catch |err| switch (err) {
                    error.NoEvictableSlot => break,
                    else => return err,
                } orelse continue;
                s.ahead[count] = .{ .ticket = ticket };
                count += 1;
            }
            if (count == 0) return;
            s.forecast_misses += count;
            s.ahead_count = count;
            s.ahead_taken = 0;
            s.ahead_settled = 0;
            s.next.store(0, .release);
            s.finished.store(0, .release);
            s.stopped = false;
            s.worker = std.Thread.spawn(.{ .stack_size = worker_stack_bytes }, readAhead, .{s}) catch |err| {
                for (s.ahead[0..count]) |job| try s.clock.giveBack(job.ticket);
                s.stopped = true;
                s.ahead_count = 0;
                return err;
            };
        }
    };
}
