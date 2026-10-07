//! The parity tests' engine: a request replays the script its prompt names, as script_scheduler.py does.
const std = @import("std");
const api = @import("engine_api");
const model_text = @import("server").model_text;
const Allocator = std.mem.Allocator;

pub const ScriptEngine = struct {
    gpa: Allocator,
    io: std.Io,
    text: model_text.Text,
    scripts: std.json.Value,
    info_: api.Info,
    mutex: std.Io.Mutex = .init,
    cancelled: std.AutoHashMapUnmanaged(api.Id, void) = .empty,
    active: u32 = 0,

    pub fn engine(e: *ScriptEngine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score } };
    }

    fn self(ctx: *anyopaque) *ScriptEngine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return .{};
    }

    /// The parity engine has no weights, so every label logit is zero.
    fn score(_: *anyopaque, _: []const u32, _: []const u32, logits: []f64) error{Failed}!f64 {
        for (logits) |*logit| logit.* = 0;
        return 0;
    }

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const e = self(ctx);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        out.* = .{ .running = e.active, .preemptions = 0 };
    }

    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const e = self(ctx);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.cancelled.put(e.gpa, id, {}) catch {};
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = self(ctx);
        e.mutex.lockUncancelable(e.io);
        e.active += 1;
        e.mutex.unlock(e.io);
        const t = std.Thread.spawn(.{}, run, .{ e, id, request, sink }) catch return error.Busy;
        t.detach();
    }

    fn isCancelled(e: *ScriptEngine, id: api.Id) bool {
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        return e.cancelled.contains(id);
    }

    /// The script a prompt names: ``@script=NAME`` in its text, else "default".
    fn scriptFor(e: *ScriptEngine, a: Allocator, prompt: []const u32) std.json.Value {
        const text = e.text.decode(a, prompt) catch "";
        var name: []const u8 = "default";
        if (std.mem.lastIndexOf(u8, text, "@script=")) |at| {
            var end = at + 8;
            while (end < text.len and (std.ascii.isAlphanumeric(text[end]) or text[end] == '_' or text[end] == '-')) end += 1;
            name = text[at + 8 .. end];
        }
        return e.scripts.object.get(name) orelse e.scripts.object.get("default").?;
    }

    fn run(e: *ScriptEngine, id: api.Id, r: *const api.Request, sink: api.Sink) void {
        var arena: std.heap.ArenaAllocator = .init(e.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const script = e.scriptFor(a, r.prompt);
        const cached: u32 = if (script.object.get("cached")) |c| @intCast(c.integer) else 0;
        const delay: i64 = if (script.object.get("delay_ms")) |d| d.integer else 0;
        const prefill_start = std.Io.Clock.awake.now(e.io).toNanoseconds();
        sink.event(sink.ctx, id, &.{ .prefilled = cached });
        var emitted: std.ArrayList(u32) = .empty;
        const prefill_done = std.Io.Clock.awake.now(e.io).toNanoseconds();
        var stats: api.Stats = .{ .prefill_seconds = @as(f64, @floatFromInt(@max(0, prefill_done - prefill_start))) / 1e9 };
        var reason: api.Reason = .stop;
        var done = false;
        for (script.object.get("chunks").?.array.items) |chunk| {
            if (delay > 0) std.Io.sleep(e.io, .fromMilliseconds(delay), .awake) catch {};
            if (e.isCancelled(id)) {
                reason = .cancelled;
                done = true;
                break;
            }
            var tokens: std.ArrayList(u32) = .empty;
            for (chunk.array.items) |entry| switch (entry) {
                .integer => |i| tokens.append(a, @intCast(i)) catch {},
                .string => |s| if (std.mem.eql(u8, s, "<EOS>")) {
                    if (r.eos.len > 0) tokens.append(a, std.mem.min(u32, r.eos)) catch {};
                } else tokens.appendSlice(a, e.text.encode(a, s, false) catch &.{}) catch {},
                else => {},
            };
            const start = emitted.items.len;
            for (tokens.items) |t| {
                emitted.append(a, t) catch {};
                if (std.mem.indexOfScalar(u32, r.eos, t) != null or (r.stop != null and r.stop.?.check(r.stop.?.ctx, emitted.items))) {
                    reason = .stop;
                    done = true;
                } else if (emitted.items.len >= r.max_tokens) {
                    reason = .length;
                    done = true;
                }
                if (done) break;
            }
            const landed: u32 = @intCast(emitted.items.len - start);
            if (landed > 0) {
                sink.event(sink.ctx, id, &.{ .tokens = emitted.items[start..] });
                stats.rounds += 1;
                stats.drafted += landed - 1;
                stats.accepted += landed - 1;
                stats.min_rows = if (stats.min_rows == 0) landed else @min(stats.min_rows, landed);
            }
            if (done) break;
        }
        e.mutex.lockUncancelable(e.io);
        e.active -= 1;
        _ = e.cancelled.remove(id);
        e.mutex.unlock(e.io);
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = reason, .stats = stats } });
    }
};
