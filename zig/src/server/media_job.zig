//! Owner-bound durable media jobs; an external executor starts and cancels work by serial.
const std = @import("std");
const disk = @import("media_job_store.zig");
const Allocator = std.mem.Allocator;
pub const Kind = disk.Kind;
pub const State = disk.State;
pub const Owner = disk.Owner;
const Record = disk.Record;

pub const Limits = struct {
    jobs: usize = 64,
    running: usize = 4,
    per_owner: usize = 16,
    max_request_bytes: usize = 16 << 20,
    request_bytes: usize = 256 << 20,
    retained_results: usize = 128,
    result_bytes: usize = 1024 << 20,
};

pub const Output = struct { media_type: []const u8, bytes: []const u8 };
pub const Outcome = union(enum) { succeeded: []const Output, failed: []const u8, cancelled };
pub const StartError = error{ Busy, Closed };

/// Request and id remain registry-owned until finish returns and must not be retained afterward.
pub const Invocation = struct {
    registry: *Registry,
    serial: u64,
    id: []const u8,
    kind: Kind,
    request: []const u8,

    pub fn finish(inv: Invocation, outcome: Outcome) !void {
        return inv.registry.finish(inv.serial, outcome);
    }
};

pub const Runner = struct {
    ctx: *anyopaque,
    start: *const fn (*anyopaque, Invocation) StartError!void,
    cancel: ?*const fn (*anyopaque, u64) void = null,
};

pub const Submission = struct { id: []u8, serial: u64 };
pub const CancelStatus = enum { cancelled, requested, already_requested, already_cancelled, finished };
pub const ArtifactInfo = struct { index: usize, media_type: []const u8, bytes: u64, digest: [32]u8 };

pub const Snapshot = struct {
    id: []const u8,
    serial: u64,
    kind: Kind,
    created: i64,
    state: State,
    cancel_requested: bool,
    message: ?[]const u8,
    artifacts: []ArtifactInfo,

    pub fn deinit(s: *Snapshot, a: Allocator) void {
        if (s.id.len > 0) a.free(s.id);
        if (s.message) |message| a.free(message);
        for (s.artifacts) |artifact| if (artifact.media_type.len > 0) a.free(artifact.media_type);
        if (s.artifacts.len > 0) a.free(s.artifacts);
        s.* = undefined;
    }
};

pub const Result = struct {
    media_type: []const u8,
    bytes: []u8,
    digest: [32]u8,

    pub fn deinit(result: *Result, a: Allocator) void {
        a.free(result.media_type);
        a.free(result.bytes);
        result.* = undefined;
    }
};

pub const Stats = struct { active: usize, queued: usize, running: usize, request_bytes: usize, retained: usize, result_bytes: usize };

pub const Registry = struct {
    gpa: Allocator,
    io: std.Io,
    store: disk.Store,
    limits: Limits,
    mutex: std.Io.Mutex = .init,
    ended: std.Io.Condition = .init,
    jobs: std.ArrayList(*Record) = .empty,
    active: usize = 0,
    running: usize = 0,
    request_bytes: usize = 0,
    retained: usize = 0,
    result_bytes: usize = 0,
    next_serial: u64 = 1,
    runner: ?Runner = null,
    dispatching: bool = false,
    closed: bool = false,
    last_error: ?anyerror = null,

    pub fn init(r: *Registry, gpa: Allocator, io: std.Io, directory: []const u8, limits: Limits) !void {
        if (limits.jobs == 0 or limits.running == 0 or limits.per_owner == 0 or limits.retained_results == 0 or limits.result_bytes == 0) return error.InvalidLimits;
        r.* = .{ .gpa = gpa, .io = io, .store = undefined, .limits = limits };
        errdefer r.jobs.deinit(gpa);
        const record_cap = std.math.add(usize, limits.max_request_bytes, 128 * 1024) catch return error.InvalidLimits;
        try r.store.init(gpa, io, directory, record_cap);
        errdefer r.store.deinit();
        const record_total = std.math.add(usize, limits.jobs, limits.retained_results) catch return error.InvalidLimits;
        var loaded = try r.store.load(record_total);
        defer loaded.deinit(gpa);
        var moved: usize = 0;
        errdefer {
            r.freeJobs();
            for (loaded.items[moved..]) |*record| record.deinit(gpa);
        }
        std.mem.sort(Record, loaded.items, {}, struct {
            fn less(_: void, left: Record, right: Record) bool {
                return left.serial < right.serial;
            }
        }.less);
        for (loaded.items) |*loaded_record| {
            const record = try gpa.create(Record);
            record.* = loaded_record.*;
            r.jobs.append(gpa, record) catch |e| {
                gpa.destroy(record);
                return e;
            };
            moved += 1;
            errdefer {
                _ = r.jobs.pop();
                record.deinit(gpa);
                gpa.destroy(record);
            }
            try r.recover(record);
            try r.accountLoaded(record);
            if (r.active > r.limits.jobs) return error.JobLimit;
            if (r.request_bytes > r.limits.request_bytes) return error.RequestByteLimit;
            if (r.ownerActive(record.owner) > r.limits.per_owner) return error.OwnerLimit;
            r.next_serial = std.math.add(u64, record.serial, 1) catch return error.IdExhausted;
        }
        try r.enforceRetention();
    }

    /// Call shutdown and drain external executors first; deinit alone models a stopped process whose callbacks are gone.
    pub fn deinit(r: *Registry) void {
        r.freeJobs();
        r.jobs.deinit(r.gpa);
        r.store.deinit();
        r.* = undefined;
    }

    fn freeJobs(r: *Registry) void {
        for (r.jobs.items) |record| {
            record.deinit(r.gpa);
            r.gpa.destroy(record);
        }
        r.jobs.clearRetainingCapacity();
    }

    fn recover(r: *Registry, record: *Record) !void {
        if (record.request.len > r.limits.max_request_bytes) return error.RequestTooLarge;
        if ((!record.state.terminal() and record.request.len == 0) or (record.state.terminal() and record.request.len != 0)) return error.CorruptRecord;
        if (record.state == .queued and !record.cancel_requested) return;
        if (record.state == .queued or record.state == .running) {
            var next = record.*;
            next.state = if (record.cancel_requested) .cancelled else .failed;
            next.message = if (record.cancel_requested) null else try r.gpa.dupe(u8, "media generation was interrupted by server restart");
            next.request = &.{};
            next.cancel_requested = false;
            errdefer if (next.message) |message| r.gpa.free(message);
            try r.store.save(&next);
            if (record.request.len > 0) r.gpa.free(record.request);
            record.* = next;
        }
        if (record.state.terminal() and record.state != .failed and record.message != null) return error.CorruptRecord;
    }

    fn accountLoaded(r: *Registry, record: *Record) !void {
        if (record.state == .queued) {
            r.active += 1;
            r.request_bytes = try std.math.add(usize, r.request_bytes, record.request.len);
        } else if (record.state.terminal()) {
            r.retained += 1;
            r.result_bytes = try std.math.add(usize, r.result_bytes, record.resultBytes());
        }
    }

    fn find(r: *Registry, id: []const u8, owner: Owner) ?*Record {
        for (r.jobs.items) |record| if (std.mem.eql(u8, record.id, id) and std.meta.eql(record.owner, owner)) return record;
        return null;
    }

    fn serialRecord(r: *Registry, serial: u64) ?*Record {
        for (r.jobs.items) |record| if (record.serial == serial) return record;
        return null;
    }

    fn ownerActive(r: *Registry, owner: Owner) usize {
        var count: usize = 0;
        for (r.jobs.items) |record| count += @intFromBool(!record.state.terminal() and std.meta.eql(record.owner, owner));
        return count;
    }

    fn newId(r: *Registry, a: Allocator) ![]u8 {
        for (0..32) |_| {
            const id = try a.alloc(u8, disk.id_len);
            errdefer a.free(id);
            @memcpy(id[0..6], "media-");
            var random: [16]u8 = undefined;
            r.io.random(&random);
            const digits = "0123456789abcdef";
            for (random, 0..) |byte, index| {
                id[6 + index * 2] = digits[byte >> 4];
                id[7 + index * 2] = digits[byte & 15];
            }
            var unique = true;
            for (r.jobs.items) |record| if (std.mem.eql(u8, record.id, id)) {
                unique = false;
                break;
            };
            if (unique) return id;
            a.free(id);
        }
        return error.IdExhausted;
    }

    pub fn submit(r: *Registry, a: Allocator, owner: Owner, kind: Kind, request: []const u8) !Submission {
        if (request.len == 0) return error.InvalidRequest;
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        if (r.closed) return error.Closed;
        if (r.active >= r.limits.jobs) return error.JobLimit;
        if (r.ownerActive(owner) >= r.limits.per_owner) return error.OwnerLimit;
        if (request.len > r.limits.max_request_bytes) return error.RequestTooLarge;
        if (request.len > r.limits.request_bytes -| r.request_bytes) return error.RequestByteLimit;
        if (r.next_serial == std.math.maxInt(u64)) return error.IdExhausted;
        const record = try r.gpa.create(Record);
        errdefer r.gpa.destroy(record);
        record.* = .{ .serial = r.next_serial, .id = &.{}, .kind = kind, .owner = owner, .created = std.Io.Clock.real.now(r.io).toSeconds(), .state = .queued };
        errdefer record.deinit(r.gpa);
        record.id = try r.newId(r.gpa);
        record.request = try r.gpa.dupe(u8, request);
        const answer: Submission = .{ .id = try a.dupe(u8, record.id), .serial = record.serial };
        errdefer a.free(answer.id);
        try r.jobs.ensureUnusedCapacity(r.gpa, 1);
        try r.store.save(record);
        r.jobs.appendAssumeCapacity(record);
        r.next_serial += 1;
        r.active += 1;
        r.request_bytes += request.len;
        held = false;
        r.mutex.unlock(r.io);
        r.kick();
        return answer;
    }

    fn note(r: *Registry, e: anyerror) void {
        r.mutex.lockUncancelable(r.io);
        if (r.last_error == null) r.last_error = e;
        r.mutex.unlock(r.io);
    }

    fn kick(r: *Registry) void {
        r.dispatch() catch |e| r.note(e);
    }

    pub fn attach(r: *Registry, runner: Runner) !void {
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        if (r.running != 0) return error.ExecutorActive;
        r.runner = runner;
        held = false;
        r.mutex.unlock(r.io);
        return r.dispatch();
    }

    pub fn dispatch(r: *Registry) !void {
        r.mutex.lockUncancelable(r.io);
        if (r.dispatching or r.closed or r.runner == null) {
            r.mutex.unlock(r.io);
            return;
        }
        r.dispatching = true;
        while (!r.closed and r.running < r.limits.running) {
            var selected: ?*Record = null;
            for (r.jobs.items) |record| if (record.state == .queued) {
                selected = record;
                break;
            };
            const record = selected orelse break;
            const runner = r.runner.?;
            var next = record.*;
            next.state = .running;
            r.store.save(&next) catch |e| {
                r.dispatching = false;
                r.mutex.unlock(r.io);
                return e;
            };
            record.state = .running;
            r.running += 1;
            const invocation: Invocation = .{ .registry = r, .serial = record.serial, .id = record.id, .kind = record.kind, .request = record.request };
            r.mutex.unlock(r.io);
            runner.start(runner.ctx, invocation) catch |e| {
                r.mutex.lockUncancelable(r.io);
                r.dispatching = false;
                const current = r.serialRecord(invocation.serial);
                if (current == null or current.?.state != .running) {
                    r.mutex.unlock(r.io);
                    return;
                }
                switch (e) {
                    error.Busy => {
                        if (current.?.cancel_requested) {
                            r.mutex.unlock(r.io);
                            return r.finish(invocation.serial, .cancelled);
                        }
                        var queued = current.?.*;
                        queued.state = .queued;
                        r.store.save(&queued) catch |save_error| {
                            r.mutex.unlock(r.io);
                            return save_error;
                        };
                        current.?.state = .queued;
                        r.running -= 1;
                        r.mutex.unlock(r.io);
                        return;
                    },
                    error.Closed => {
                        r.mutex.unlock(r.io);
                        return r.finish(invocation.serial, .{ .failed = "the media executor is closed" });
                    },
                }
            };
            r.mutex.lockUncancelable(r.io);
        }
        r.dispatching = false;
        r.mutex.unlock(r.io);
    }

    fn normalizedFailure(message: []const u8) []const u8 {
        if (message.len == 0) return "media generation failed";
        if (message.len > disk.max_message) return "media generation failed with an oversized error detail";
        return message;
    }

    fn freeArtifacts(r: *Registry, artifacts: []disk.Artifact) void {
        for (artifacts) |artifact| if (artifact.media_type.len > 0) r.gpa.free(artifact.media_type);
        if (artifacts.len > 0) r.gpa.free(artifacts);
    }

    fn makeArtifacts(r: *Registry, record: *Record, outputs: []const Output) ![]disk.Artifact {
        const artifacts = try r.gpa.alloc(disk.Artifact, outputs.len);
        for (artifacts) |*artifact| artifact.* = .{ .media_type = &.{}, .bytes = 0, .digest = @splat(0) };
        errdefer freeArtifacts(r, artifacts);
        var published: usize = 0;
        errdefer for (0..published) |index| r.store.deleteArtifact(record.id, index) catch {};
        for (outputs, artifacts) |output, *artifact| {
            artifact.media_type = try r.gpa.dupe(u8, output.media_type);
            artifact.bytes = @intCast(output.bytes.len);
            artifact.digest = try r.store.saveArtifact(record.id, published, output.bytes);
            published += 1;
        }
        return artifacts;
    }

    pub fn finish(r: *Registry, serial: u64, outcome: Outcome) !void {
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        const record = r.serialRecord(serial) orelse return error.StaleCompletion;
        if (record.state != .running) return error.StaleCompletion;
        const request_bytes = record.request.len;
        var next = record.*;
        {
            errdefer if (next.message) |message| r.gpa.free(message);
            errdefer if (next.artifacts.ptr != record.artifacts.ptr) {
                for (next.artifacts, 0..) |_, index| r.store.deleteArtifact(next.id, index) catch {};
                freeArtifacts(r, next.artifacts);
            };
            next.request = &.{};
            next.cancel_requested = false;
            next.message = null;
            next.artifacts = &.{};
            if (record.cancel_requested) {
                next.state = .cancelled;
            } else switch (outcome) {
                .cancelled => next.state = .cancelled,
                .failed => |message| {
                    next.state = .failed;
                    next.message = try r.gpa.dupe(u8, normalizedFailure(message));
                },
                .succeeded => |outputs| {
                    var valid = outputs.len > 0 and outputs.len <= disk.max_artifacts;
                    var bytes: usize = 0;
                    for (outputs) |output| {
                        valid = valid and output.bytes.len > 0 and disk.validMediaType(output.media_type);
                        bytes = std.math.add(usize, bytes, output.bytes.len) catch {
                            valid = false;
                            break;
                        };
                    }
                    if (!valid or bytes > r.limits.result_bytes) {
                        next.state = .failed;
                        next.message = try r.gpa.dupe(u8, "the media result is empty, malformed, or exceeds the retained result byte quota");
                    } else {
                        next.state = .succeeded;
                        next.artifacts = try r.makeArtifacts(record, outputs);
                    }
                },
            }
            try r.store.save(&next);
        }
        if (record.request.len > 0) r.gpa.free(record.request);
        record.* = next;
        r.running -= 1;
        r.active -= 1;
        r.request_bytes -= request_bytes;
        r.retained += 1;
        r.result_bytes += record.resultBytes();
        r.ended.broadcast(r.io);
        const retention = r.enforceRetention();
        held = false;
        r.mutex.unlock(r.io);
        r.kick();
        return retention;
    }

    pub fn cancel(r: *Registry, id: []const u8, owner: Owner) !?CancelStatus {
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        const record = r.find(id, owner) orelse return null;
        if (record.state.terminal()) return if (record.state == .cancelled) .already_cancelled else .finished;
        if (record.state == .running) {
            if (record.cancel_requested) return .already_requested;
            var next = record.*;
            next.cancel_requested = true;
            try r.store.save(&next);
            record.cancel_requested = true;
            const runner = r.runner;
            const serial = record.serial;
            held = false;
            r.mutex.unlock(r.io);
            if (runner) |active| if (active.cancel) |call| call(active.ctx, serial);
            return .requested;
        }
        var next = record.*;
        next.state = .cancelled;
        next.request = &.{};
        try r.store.save(&next);
        const old_request = record.request;
        record.* = next;
        r.active -= 1;
        r.request_bytes -= old_request.len;
        r.retained += 1;
        if (old_request.len > 0) r.gpa.free(old_request);
        try r.enforceRetention();
        held = false;
        r.mutex.unlock(r.io);
        return .cancelled;
    }

    fn removeAt(r: *Registry, index: usize) void {
        const record = r.jobs.orderedRemove(index);
        r.retained -= 1;
        r.result_bytes -= record.resultBytes();
        record.deinit(r.gpa);
        r.gpa.destroy(record);
    }

    fn enforceRetention(r: *Registry) !void {
        while (r.retained > r.limits.retained_results or r.result_bytes > r.limits.result_bytes) {
            var oldest: ?usize = null;
            for (r.jobs.items, 0..) |record, index| if (record.state.terminal()) {
                if (oldest == null or record.serial < r.jobs.items[oldest.?].serial) oldest = index;
            };
            const index = oldest orelse return error.RetentionOverflow;
            try r.store.remove(r.jobs.items[index]);
            removeAt(r, index);
        }
    }

    pub fn remove(r: *Registry, id: []const u8, owner: Owner) !bool {
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        const record = r.find(id, owner) orelse return false;
        if (!record.state.terminal()) return error.JobActive;
        var index: usize = 0;
        while (r.jobs.items[index] != record) : (index += 1) {}
        try r.store.remove(record);
        removeAt(r, index);
        held = false;
        r.mutex.unlock(r.io);
        return true;
    }

    pub fn inspect(r: *Registry, a: Allocator, id: []const u8, owner: Owner) !?Snapshot {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        const record = r.find(id, owner) orelse return null;
        var snapshot: Snapshot = .{ .id = &.{}, .serial = record.serial, .kind = record.kind, .created = record.created, .state = record.state, .cancel_requested = record.cancel_requested, .message = null, .artifacts = &.{} };
        errdefer snapshot.deinit(a);
        snapshot.id = try a.dupe(u8, record.id);
        if (record.message) |message| snapshot.message = try a.dupe(u8, message);
        if (record.artifacts.len > 0) snapshot.artifacts = try a.alloc(ArtifactInfo, record.artifacts.len);
        for (snapshot.artifacts) |*artifact| artifact.* = .{ .index = 0, .media_type = &.{}, .bytes = 0, .digest = @splat(0) };
        for (record.artifacts, snapshot.artifacts, 0..) |artifact, *copy, index| {
            copy.index = index;
            copy.bytes = artifact.bytes;
            copy.digest = artifact.digest;
            copy.media_type = try a.dupe(u8, artifact.media_type);
        }
        return snapshot;
    }

    pub fn result(r: *Registry, a: Allocator, id: []const u8, owner: Owner, index: usize) !?Result {
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        const record = r.find(id, owner) orelse return null;
        if (record.state != .succeeded or index >= record.artifacts.len) return null;
        const artifact = record.artifacts[index];
        const media_type = try a.dupe(u8, artifact.media_type);
        held = false;
        r.mutex.unlock(r.io);
        errdefer a.free(media_type);
        const bytes = try r.store.readArtifact(a, id, index, artifact);
        return .{ .media_type = media_type, .bytes = bytes, .digest = artifact.digest };
    }

    pub fn stats(r: *Registry) Stats {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        var snapshot: Stats = .{ .active = r.active, .queued = 0, .running = r.running, .request_bytes = r.request_bytes, .retained = r.retained, .result_bytes = r.result_bytes };
        for (r.jobs.items) |record| snapshot.queued += @intFromBool(record.state == .queued);
        return snapshot;
    }

    pub fn takeError(r: *Registry) ?anyerror {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        const failure = r.last_error;
        r.last_error = null;
        return failure;
    }

    pub fn shutdown(r: *Registry) !void {
        r.mutex.lockUncancelable(r.io);
        var held = true;
        defer if (held) r.mutex.unlock(r.io);
        r.closed = true;
        var failure: ?anyerror = null;
        while (true) {
            var selected: ?*Record = null;
            for (r.jobs.items) |record| if (record.state == .running and !record.cancel_requested) {
                selected = record;
                break;
            };
            const record = selected orelse break;
            var next = record.*;
            next.cancel_requested = true;
            r.store.save(&next) catch |e| {
                failure = e;
            };
            record.cancel_requested = true;
            const runner = r.runner;
            const serial = record.serial;
            r.mutex.unlock(r.io);
            if (runner) |active| if (active.cancel) |call| call(active.ctx, serial);
            r.mutex.lockUncancelable(r.io);
        }
        while (r.running > 0) r.ended.waitUncancelable(r.io, &r.mutex);
        r.runner = null;
        r.dispatching = false;
        held = false;
        r.mutex.unlock(r.io);
        if (failure) |e| return e;
    }
};

test {
    _ = @import("media_job_test.zig");
}
