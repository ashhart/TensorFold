//! Fixed-capacity FIFO admission for one owner thread; cancellation tokens remain safe across threads.
const std = @import("std");
const types = @import("expert_strip_types.zig");
const planning = @import("expert_strip_plan.zig");
const reading = @import("expert_strip_reader.zig");
const accounting = @import("expert_strip_accounting.zig");

pub const Request = struct {
    user: u64,
    host: types.Host,
    output: []u8,
    sink: ?types.Sink = null,
    cancel: ?*const types.Cancellation = null,
};

pub const Ticket = struct {
    slot: usize,
    generation: u64,
};

pub const Completion = struct {
    ticket: Ticket,
    user: u64,
    result: ?anyerror,
};

pub const CancelResult = enum { queued, running, not_found };
const State = enum { free, queued, running };

const Job = struct {
    state: State = .free,
    generation: u64 = 0,
    sequence: u64 = 0,
    request: Request = undefined,
    cancel: types.Cancellation = .{},
};

pub const Queue = struct {
    a: std.mem.Allocator,
    reader: reading.Reader,
    jobs: []Job,
    count: usize = 0,
    running: ?usize = null,
    sequence: u64 = 0,
    accounting: accounting.Accounting = .{},

    pub fn init(a: std.mem.Allocator, plan: *const planning.Plan, capacity: usize) !Queue {
        if (capacity == 0) return error.InvalidQueueCapacity;
        var reader = try reading.Reader.init(a, plan);
        errdefer reader.deinit(a);
        const jobs = try a.alloc(Job, capacity);
        @memset(jobs, .{});
        return .{ .a = a, .reader = reader, .jobs = jobs };
    }

    pub fn deinit(q: *Queue) void {
        std.debug.assert(q.running == null);
        q.reader.deinit(q.a);
        q.a.free(q.jobs);
        q.* = undefined;
    }

    pub fn submit(q: *Queue, request: Request) !Ticket {
        if (q.running != null) return error.QueueBusy;
        if (q.count == q.jobs.len) return error.QueueFull;
        var slot: ?usize = null;
        for (q.jobs, 0..) |job, i| if (job.state == .free) {
            slot = i;
            break;
        };
        const i = slot orelse return error.QueueFull;
        const job = &q.jobs[i];
        job.generation +%= 1;
        if (job.generation == 0) job.generation = 1;
        q.sequence +%= 1;
        if (q.sequence == 0) return error.QueueSequenceExhausted;
        job.sequence = q.sequence;
        job.request = request;
        job.cancel.reset(request.cancel);
        job.state = .queued;
        q.count += 1;
        return .{ .slot = i, .generation = job.generation };
    }

    fn matches(job: *const Job, ticket: Ticket) bool {
        return job.state != .free and job.generation == ticket.generation;
    }

    pub fn cancel(q: *Queue, ticket: Ticket) !CancelResult {
        if (ticket.slot >= q.jobs.len) return .not_found;
        const job = &q.jobs[ticket.slot];
        if (!matches(job, ticket)) return .not_found;
        job.cancel.cancel();
        if (job.state == .running) return .running;
        job.state = .free;
        q.count -= 1;
        try q.accounting.recordFailure(true);
        return .queued;
    }

    pub fn pending(q: *const Queue) usize {
        return q.count - @intFromBool(q.running != null);
    }

    pub fn runNext(q: *Queue) ?Completion {
        if (q.running != null) return null;
        var next: ?usize = null;
        for (q.jobs, 0..) |job, i| if (job.state == .queued) {
            if (next == null or job.sequence < q.jobs[next.?].sequence) next = i;
        };
        const slot = next orelse return null;
        const job = &q.jobs[slot];
        job.state = .running;
        q.running = slot;
        const ticket: Ticket = .{ .slot = slot, .generation = job.generation };
        var result: ?anyerror = null;
        q.reader.read(job.request.host, job.request.output, job.request.sink, &job.cancel, &q.accounting) catch |err| {
            result = err;
        };
        const completion: Completion = .{ .ticket = ticket, .user = job.request.user, .result = result };
        job.state = .free;
        q.running = null;
        q.count -= 1;
        return completion;
    }

    pub fn snapshot(q: *const Queue) accounting.Totals {
        return q.accounting.snapshot();
    }

    pub fn restore(q: *Queue, totals: accounting.Totals) !void {
        if (q.count != 0) return error.QueueNotIdle;
        q.accounting.restore(totals);
    }
};
