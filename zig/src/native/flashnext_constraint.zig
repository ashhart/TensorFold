//! Constrained replies borrow one Flash Next session; ordinary replies keep the engine's GPU-side driver.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const mtl = @import("metal");
const Host = @import("flashnext_host.zig").Host;
const run = @import("flashnext_constraint_run.zig");
const images = @import("flashnext_images.zig");
const VisPrompt = tf.flashnext_replay.vis_prompt.VisPrompt;

pub fn serve(h: *Host, job: anytype) void {
    const Job = @typeInfo(@TypeOf(job)).pointer.child;
    const Context = struct {
        h: *Host,
        job: *Job,
        fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            switch (e.*) {
                .prefilled => |cached| {
                    c.job.cached = cached;
                    const done = c.h.now();
                    c.h.lock();
                    if (done > c.job.began) c.h.prefill_rate = @as(f64, @floatFromInt(c.job.request.prompt.len - cached)) / (@as(f64, @floatFromInt(done - c.job.began)) / 1e9);
                    c.h.prefill_at = done;
                    c.h.prefilling = false;
                    c.job.prefilled = done;
                    c.h.unlock();
                    c.job.prefill_sent = true;
                    Host.emit(c.job, e.*);
                },
                .tokens => |tokens| {
                    Host.emit(c.job, e.*);
                    c.h.lock();
                    c.h.live_generated += tokens.len;
                    c.h.unlock();
                    c.h.noteDecoded(tokens.len);
                },
                else => Host.emit(c.job, e.*),
            }
        }
        fn cancelled(ptr: *anyopaque) bool {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            var context: Host.Ctx = .{ .h = c.h, .job = c.job };
            return Host.Ctx.cancelled(&context);
        }
    };
    var context: Context = .{ .h = h, .job = job };
    job.began = h.now();
    const finished = reply(h, job.id, job.request, .{ .sink = .{ .ctx = &context, .event = Context.event }, .cancelled = Context.cancelled }) catch |err| {
        if (!job.prefill_sent) Host.emit(job, .{ .prefilled = 0 });
        h.finish(job, .failed, .{}, @errorName(err));
        return;
    };
    if (!job.prefill_sent) Host.emit(job, .{ .prefilled = 0 });
    h.finish(job, finished.reason, finished.stats, finished.text());
}

fn reply(h: *Host, id: api.Id, request: *const api.Request, hooks: run.Hooks) !run.Finished {
    if (h.eng.r.tp != null) return error.BatchSpeedUpUnsupported;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const scope = @import("flashnext_constraint_state.zig").Scope(tf.flashnext_engine.Engine, mtl.Buffer).capture(h.eng);
    defer scope.restore(h.eng);
    h.eng.hostMode();
    h.eng.r.gdn_pipe = h.shared_gdn_pipe;
    var back = try tf.flashnext_backend.Backend.init(h.gpa, h.eng, &.{h.eng.r.window.capacity});
    defer back.deinit();
    if (h.vis != null) back.images = .{ .ctx = h, .build = buildImages, .release = releaseImages };
    h.prompt = request.prompt;
    var info = h.info_;
    info.lanes = 1;
    info.prefill_step = @intCast(h.eng.pr.step);
    return run.run(h.gpa, h.io, back.backend(), back.facts(), info, run.isolatedCache(h), id, request, hooks);
}

fn buildImages(ptr: *anyopaque, a: std.mem.Allocator, imgs: []const api.Image, prompt_len: usize, from: usize) anyerror!*VisPrompt {
    const h: *Host = @ptrCast(@alignCast(ptr));
    return images.build(h.vis.?, h.eng, a, imgs, prompt_len, from, true);
}

fn releaseImages(_: *anyopaque, p: *VisPrompt) void {
    p.rows.deinit();
}
