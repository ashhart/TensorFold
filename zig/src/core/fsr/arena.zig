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
        pub const Job = struct { ticket: cache.Ticket, failure: ?anyerror = null, cancelled: bool = false };
        a: std.mem.Allocator,
        source: Reader,
        experts: []const Expert,
        strides: [components]usize,
        slabs: [components][]u8,
        clock: cache.Cache,
        jobs: []Job,
        job_count: usize = 0,
        worker: ?std.Thread = null,
        file_sizes: []u64,
        io: reader.Stats = .{},
        demand_misses: u64 = 0,
        forecast_misses: u64 = 0,
        demand_keys: u64 = 0,
        unique_demand_keys: u64 = 0,
        worker_joins: u64 = 0,
        wanted: []u64,
        epoch: u64 = 0,
        urgent: std.atomic.Value(bool) = .init(false),
        forecasting: bool = false,
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
            const wanted = try a.alloc(u64, experts.len);
            @memset(wanted, 0);
            return .{ .a = a, .source = source, .experts = experts, .strides = strides, .slabs = slabs, .clock = clock, .jobs = jobs, .file_sizes = sizes, .wanted = wanted };
        }

        pub fn deinit(s: *Self) void {
            s.join() catch {};
            s.clock.release();
            s.clock.deinit();
            s.a.free(s.jobs);
            s.a.free(s.file_sizes);
            s.a.free(s.wanted);
        }

        fn execute(s: *Self) void {
            if (s.forecasting) {
                for (s.jobs[0..s.job_count]) |*job| {
                    if (s.urgent.load(.acquire) and s.wanted[job.ticket.key] != s.epoch) {
                        job.cancelled = true;
                        continue;
                    }
                    reader.batch(s.source, @as(*[1]Job, @ptrCast(job)), s.experts, s.strides, s.slabs, s.file_sizes, &s.io) catch |err| {
                        job.failure = err;
                    };
                }
                return;
            }
            reader.batch(s.source, s.jobs[0..s.job_count], s.experts, s.strides, s.slabs, s.file_sizes, &s.io) catch |err| {
                for (s.jobs[0..s.job_count]) |*job| job.failure = err;
            };
        }

        /// Worker writes only reserved slab ranges. Publication and cache policy stay on the owner thread.
        pub fn join(s: *Self) !void {
            if (s.worker) |worker| {
                worker.join();
                s.worker = null;
                s.worker_joins += 1;
            }
            var failure: ?anyerror = null;
            for (s.jobs[0..s.job_count]) |job| {
                if (job.cancelled) {
                    try s.clock.abort(job.ticket);
                    s.cancelled_forecasts += 1;
                } else if (job.failure) |err| {
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
            for (keys) |key| if (key >= s.experts.len) {
                try s.join();
                return error.InvalidExpert;
            };
            if (s.worker != null) {
                if (s.epoch == std.math.maxInt(u64)) @memset(s.wanted, std.math.maxInt(u64));
                s.epoch +%= 1;
                for (keys) |key| s.wanted[key] = s.epoch;
                s.urgent.store(true, .release);
            }
            try s.join();
            s.forecasting = false;
            s.clock.release();
            if (keys.len == 0) return error.NoExperts;
            try s.clock.protect(keys);
            errdefer s.clock.release();
            s.demand_keys += keys.len;
            s.unique_demand_keys += s.clock.current_count;
            std.mem.sort(usize, s.clock.current[0..s.clock.current_count], {}, std.sort.asc(usize));
            for (s.clock.current[0..s.clock.current_count]) |key| s.reserve(key, false) catch |err| {
                for (s.jobs[0..s.job_count]) |job| try s.clock.abort(job.ticket);
                s.job_count = 0;
                return err;
            };
            s.demand_misses += s.job_count;
            s.execute();
            try s.join();
            for (s.clock.current[0..s.clock.current_count]) |key| if (try s.clock.resident(key) == null) return error.ExpertNotReady;
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
            s.forecast_misses += s.job_count;
            s.urgent.store(false, .release);
            s.forecasting = true;
            s.worker = std.Thread.spawn(.{ .stack_size = worker_stack_bytes }, execute, .{s}) catch |err| {
                for (s.jobs[0..s.job_count]) |job| try s.clock.abort(job.ticket);
                s.job_count = 0;
                return err;
            };
        }
    };
}
