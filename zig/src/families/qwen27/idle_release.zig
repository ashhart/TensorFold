//! Host-backed snapshots for temporarily releasing whole prepared-weight buffers.
const std = @import("std");

pub fn createModel(comptime Model: type, a: std.mem.Allocator) !*Model {
    const model = try a.create(Model);
    model.idle_release = null;
    return model;
}

pub fn Owner(comptime Metal: type) type {
    return struct {
        const Self = @This();
        const scratch_bytes = 8 << 20;
        const max_buffer_bytes = 4 << 30;

        pub const Buffer = Metal.Buffer;
        pub const Events = struct {
            ctx: ?*anyopaque = null,
            beforeRelease: ?*const fn (?*anyopaque, []const Buffer) anyerror!void = null,
            afterRestore: ?*const fn (?*anyopaque, []const Buffer) anyerror!void = null,
        };

        const Extent = struct {
            offset: u64 = 0,
            digest: [32]u8 = @splat(0),
        };

        allocator: std.mem.Allocator,
        io: std.Io,
        quiet_ns: i64,
        last_work_ns: i64 = 0,
        idle_started: bool = false,
        buffers: []Buffer,
        extents: []Extent,
        scratch: []u8,
        run_path: []u8,
        weights_path: []u8,
        snapshot_ready: bool = false,
        released: bool = false,
        events: Events = .{},
        mutex: std.Io.Mutex = .init,

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            root: []const u8,
            buffers: []const Metal.Buffer,
            quiet_ns: i64,
        ) !?*Self {
            if (quiet_ns <= 0) return null;
            var count: usize = 0;
            for (buffers) |buffer| {
                if (buffer.length() != 0 and buffer.length() <= max_buffer_bytes) count += 1;
            }

            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            const owned_buffers = try allocator.alloc(Buffer, count);
            errdefer allocator.free(owned_buffers);
            const extents = try allocator.alloc(Extent, count);
            errdefer allocator.free(extents);
            const scratch = try allocator.alloc(u8, scratch_bytes);
            errdefer allocator.free(scratch);

            var token: u64 = undefined;
            io.random(std.mem.asBytes(&token));
            var name_buf: [24]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "run-{x:0>16}", .{token});
            const run_path = try std.fs.path.join(allocator, &.{ root, name });
            errdefer allocator.free(run_path);
            const weights_path = try std.fs.path.join(allocator, &.{ run_path, "weights.bin" });
            errdefer allocator.free(weights_path);

            var at: usize = 0;
            for (buffers) |buffer| {
                if (buffer.length() == 0 or buffer.length() > max_buffer_bytes) continue;
                owned_buffers[at] = buffer;
                extents[at] = .{};
                at += 1;
            }
            self.* = .{
                .allocator = allocator,
                .io = io,
                .quiet_ns = quiet_ns,
                .buffers = owned_buffers,
                .extents = extents,
                .scratch = scratch,
                .run_path = run_path,
                .weights_path = weights_path,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.mutex.lockUncancelable(self.io);
            std.Io.Dir.cwd().deleteTree(self.io, self.run_path) catch {};
            self.mutex.unlock(self.io);
            self.allocator.free(self.weights_path);
            self.allocator.free(self.run_path);
            self.allocator.free(self.scratch);
            self.allocator.free(self.extents);
            self.allocator.free(self.buffers);
            self.allocator.destroy(self);
        }

        pub fn setEvents(self: *Self, events: Events) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            self.events = events;
        }

        pub fn idle(self: *Self, now_ns: i64) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.released or now_ns < self.last_work_ns) return;
            if (!self.idle_started) {
                self.last_work_ns = now_ns;
                self.idle_started = true;
                return;
            }
            if (now_ns - self.last_work_ns < self.quiet_ns) return;
            self.last_work_ns = now_ns;
            if (!self.snapshot_ready) self.serialize() catch return;
            if (self.events.beforeRelease) |before| {
                before(self.events.ctx, self.buffers) catch return;
            }
            for (self.buffers) |buffer| buffer.makeVolatile();
            self.released = true;
        }

        pub fn restore(self: *Self, now_ns: i64) !void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            defer {
                self.last_work_ns = now_ns;
                self.idle_started = false;
            }
            if (!self.released) return;
            const file = try std.Io.Dir.cwd().openFile(self.io, self.weights_path, .{});
            defer file.close(self.io);
            for (self.buffers) |buffer| buffer.makeNonVolatile();

            for (self.buffers, self.extents) |buffer, extent| {
                var hash = std.crypto.hash.sha2.Sha256.init(.{});
                const len = buffer.length();
                var at: usize = 0;
                while (at < len) {
                    const want = @min(self.scratch.len, len - at);
                    const got = try file.readPositionalAll(self.io, self.scratch[0..want], extent.offset + at);
                    if (got != want) return error.ShortSnapshot;
                    hash.update(self.scratch[0..got]);
                    @memcpy(buffer.contents()[at..][0..got], self.scratch[0..got]);
                    at += got;
                }
                var digest: [32]u8 = undefined;
                hash.final(&digest);
                if (!std.mem.eql(u8, &digest, &extent.digest)) return error.SnapshotDigestMismatch;
            }
            if (self.events.afterRestore) |after| try after(self.events.ctx, self.buffers);
            self.released = false;
        }

        pub fn releasedBytes(self: *Self) usize {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (!self.released) return 0;
            var total: usize = 0;
            for (self.buffers) |buffer| total += buffer.length();
            return total;
        }

        pub fn isReleased(self: *Self) bool {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            return self.released;
        }

        fn serialize(self: *Self) !void {
            try std.Io.Dir.cwd().createDirPath(self.io, self.run_path);
            const file = try std.Io.Dir.cwd().createFile(self.io, self.weights_path, .{ .exclusive = true });
            var complete = false;
            defer {
                file.close(self.io);
                if (!complete) std.Io.Dir.cwd().deleteFile(self.io, self.weights_path) catch {};
            }
            var offset: u64 = 0;
            for (self.buffers, self.extents) |buffer, *extent| {
                var hash = std.crypto.hash.sha2.Sha256.init(.{});
                const len = buffer.length();
                var at: usize = 0;
                while (at < len) {
                    const n = @min(self.scratch.len, len - at);
                    const bytes = buffer.contents()[at..][0..n];
                    hash.update(bytes);
                    try file.writePositionalAll(self.io, bytes, offset + at);
                    at += n;
                }
                hash.final(&extent.digest);
                extent.offset = offset;
                offset += len;
            }
            try file.sync(self.io);
            complete = true;
            self.snapshot_ready = true;
        }
    };
}
