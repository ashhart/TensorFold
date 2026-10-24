//! Optional token trace around the shared engine, with forget requests forwarded to the owner.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// The engine as served, with each finished reply's prompt and reply ids appended to TF_DSV41_TRACE_TOKENS (when set).
pub const Recorder = struct {
    gpa: Allocator,
    io: std.Io,
    inner: api.Engine,
    path: ?[]const u8,
    mutex: std.Io.Mutex = .init,
    live: std.AutoHashMapUnmanaged(api.Id, *Rec) = .empty,
    /// every recorded line so far (the file is rewritten whole: one reply at a time)
    log: std.ArrayList(u8) = .empty,

    /// `sampling`: the request's own (seed included), so `tf-dsv41 generate` replays a sampled reply as it was asked
    const Rec = struct { r: *Recorder, sink: api.Sink, prompt: []u32, sampling: ?api.Sampling = null, out: std.ArrayList(u32) = .empty, drafted: u64 = 0, structure: ?Structure = null };
    /// a structured request's grammar (TF_DSV41_GRAMMAR), so `generate` replays it under the same grammar
    const Structure = struct { kind: []const u8, text: []const u8 };

    pub fn init(gpa: Allocator, io: std.Io, inner: api.Engine) Recorder {
        return .{ .gpa = gpa, .io = io, .inner = inner, .path = if (std.c.getenv("TF_DSV41_TRACE_TOKENS")) |p| std.mem.span(p) else null };
    }

    pub fn deinit(r: *Recorder) void {
        var it = r.live.valueIterator();
        while (it.next()) |x| r.drop(x.*);
        r.live.deinit(r.gpa);
        r.log.deinit(r.gpa);
    }

    fn drop(r: *Recorder, x: *Rec) void {
        r.gpa.free(x.prompt);
        if (x.structure) |st| r.gpa.free(st.text);
        x.out.deinit(r.gpa);
        r.gpa.destroy(x);
    }

    pub fn engine(r: *Recorder) api.Engine {
        if (r.path == null) return r.inner;
        return .{ .ctx = r, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .forget_prompts = forget } };
    }

    fn forget(ctx: *anyopaque, span: []const u32) api.ForgetError!api.prompt_cache.Forgotten {
        const r = self(ctx);
        var out = try r.inner.forgetPrompts(span);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        // The operator's trace remains on disk and can contain prompts, reply tokens or structured output.
        if (r.path != null) out.failed += 1;
        return out;
    }

    fn self(ctx: *anyopaque) *Recorder {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return self(ctx).inner.info();
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        self(ctx).inner.cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        self(ctx).inner.status(out, stream_tokens);
    }
    fn memory(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        return self(ctx).inner.memory(reset_peak);
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r = self(ctx);
        const x = r.gpa.create(Rec) catch return error.Busy;
        x.* = .{ .r = r, .sink = sink, .sampling = request.sampling, .prompt = r.gpa.dupe(u32, request.prompt) catch {
            r.gpa.destroy(x);
            return error.Busy;
        } };
        if (request.structure) |st| x.structure = .{ .kind = @tagName(st.kind), .text = r.gpa.dupe(u8, st.text) catch {
            r.drop(x);
            return error.Busy;
        } };
        {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            r.live.put(r.gpa, id, x) catch {
                r.drop(x);
                return error.Busy;
            };
        }
        return r.inner.submit(id, request, .{ .ctx = x, .event = event });
    }

    fn event(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const x: *Rec = @ptrCast(@alignCast(ctx));
        const r = x.r;
        switch (ev.*) {
            .tokens => |t| x.out.appendSlice(r.gpa, t) catch {},
            .finished => |fin| {
                x.drafted = fin.stats.drafted;
                r.write(x);
            },
            else => {},
        }
        x.sink.event(x.sink.ctx, id, ev);
        if (ev.* == .finished) {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            _ = r.live.remove(id);
            r.drop(x);
        }
    }

    fn write(r: *Recorder, x: *const Rec) void {
        var line: std.Io.Writer.Allocating = .init(r.gpa);
        defer line.deinit();
        std.json.Stringify.value(.{ .prompt = x.prompt, .tokens = x.out.items, .drafted = x.drafted, .sampling = x.sampling, .structure = x.structure }, .{}, &line.writer) catch return;
        line.writer.writeByte('\n') catch return;
        r.log.appendSlice(r.gpa, line.written()) catch return;
        std.Io.Dir.cwd().writeFile(r.io, .{ .sub_path = r.path.?, .data = r.log.items }) catch {};
    }
};
