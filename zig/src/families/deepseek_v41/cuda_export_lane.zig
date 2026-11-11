//! Hand-off `Exporter` of the DeepSeek-V4.1 CUDA engine: `--mcdma-produce LINK` and TF_DSV41_HANDOFF_DUMP.
const std = @import("std");
const api = @import("engine_api");
const model = @import("cuda_model.zig");
const serve = @import("cuda_export_serve.zig");
const gpu = @import("cuda_export_gpu.zig");
const cx = @import("cuda_export.zig");
const split = @import("cuda_export_split.zig");
const ff = @import("forward_follow.zig");
const pc = api.prompt_cache;
const Handoff = api.LaneHost.Export.Handoff;
const log = std.log.scoped(.dsv41);

/// Where the exports are written when set: a directory (TF_DSV41_HANDOFF_DUMP).
pub const dump_env: [:0]const u8 = "TF_DSV41_HANDOFF_DUMP";

/// The dump directory, or null when unset or blank; a malformed value sets `problem` and fails.
pub fn dumpDir(problem: *[]const u8) !?[]const u8 {
    const env = @import("sys").environment;
    return env.path(dump_env, (env.Process{}).get(dump_env)) catch {
        problem.* = "TF_DSV41_HANDOFF_DUMP is malformed: it must be a directory path without control characters";
        return error.InvalidEnvironment;
    };
}

pub const Export = struct {
    gpa: std.mem.Allocator,
    source: gpu.Source,
    produce: *serve.Produce,
    f: *@import("forward.zig").Forward,
    /// a KV_SPLIT pool: this export is shard 0 and the followers export theirs (cuda_export_split.zig)
    split: bool,

    /// The exporter for `m`, or null with `problem` set; `link` is a `--mcdma-produce` name or `http`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model.Model, link: ?[]const u8, checkpoint: []const u8, problem: *[]const u8) !?*Export {
        const dump: []const u8 = (try dumpDir(problem)) orelse "";
        if (link == null and dump.len == 0) return null;
        if (gpu.refusal(m)) |why| {
            problem.* = why;
            return null;
        }
        const cap = @import("sys").environment.read(u64, "TF_PRODUCE_HELD_GIB", 32) catch {
            problem.* = "--mcdma-produce: TF_PRODUCE_HELD_GIB is malformed; the line above names its forms";
            return error.InvalidEnvironment;
        };
        const http = if (link) |l| std.mem.eql(u8, l, api.LaneHost.Remote.http_link) else false;
        const x = try gpa.create(Export);
        errdefer gpa.destroy(x);
        const shards: u32 = if (split.splitOf(m)) m.sessions.?.kx.world else 1;
        if (shards > 1 and http) {
            problem.* = "--mcdma-produce http exports a replicated KV pool: unset TF_DSV41_KV_SPLIT, or export on mcdma-rpcd links (one per rank)";
            return null;
        }
        x.* = .{ .gpa = gpa, .source = .{ .ss = m.sessions.? }, .produce = undefined, .f = &m.f, .split = shards > 1 };
        x.produce = try serve.Produce.open(gpa, io, x.source.source(), .{ .link = if (http) null else link, .http = http, .dump = if (dump.len > 0) dump else null, .model = checkpoint, .held_bytes = cap << 30, .shard = 0, .shards = shards }, problem);
        log.info("dsv41 handoff export: {s}{s}{s}", .{ if (link) |l| l else "no link", if (dump.len > 0) ", files under " else "", dump });
        return x;
    }

    /// Stops serving and frees what is held (after the lane host stopped).
    pub fn stop(x: *Export) void {
        x.produce.stop();
        x.gpa.destroy(x);
    }

    pub fn exporter(x: *Export) api.LaneHost.Export.Exporter {
        const over_http = x.produce.opts.http;
        return .{ .ptr = x, .snapshots = .{ .ptr = x, .vtable = &snapshots }, .exported = exported, .failed = failed, .serve = if (over_http) .{ .ptr = x, .open = readOpen, .release = readRelease } else null };
    }

    /// `GET /v1/handoff/<id>`: the held export's replies, a message at a time.
    fn readOpen(ptr: *anyopaque, id: [16]u8, why: *[]const u8) api.LaneHost.Export.Serve.OpenError!api.LaneHost.Export.Stream {
        const r = try self(ptr).produce.read(id, why);
        return .{ .ptr = r, .size = r.size(), .next = readNext, .close = readClose };
    }

    fn readNext(ptr: *anyopaque) anyerror!?[]const u8 {
        const r: *serve.Produce.Reading = @ptrCast(@alignCast(ptr));
        return r.next();
    }

    fn readClose(ptr: *anyopaque) void {
        const r: *serve.Produce.Reading = @ptrCast(@alignCast(ptr));
        r.close();
    }

    /// `DELETE /v1/handoff/<id>`.
    fn readRelease(ptr: *anyopaque, id: [16]u8) bool {
        return self(ptr).produce.discard(id);
    }

    const snapshots: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop };

    fn self(ptr: *anyopaque) *Export {
        return @ptrCast(@alignCast(ptr));
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return cx.estimate(at);
    }

    /// The active slot read back; the stream is armed on the engine thread at its cut, so only one slot stands there.
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return @ptrCast(try self(ptr).produce.save(at));
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.ExportsAreNotRestored;
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        self(ptr).produce.drop(@ptrCast(@alignCast(saved)));
    }

    fn exported(ptr: *anyopaque, h: Handoff, tokens: []const u32, saved: pc.Saved) void {
        const x = self(ptr);
        // the followers read their pages now, at the cut, while this rank converts its own
        if (x.split) ff.sendExport(x.f, h.id, tokens.len) catch |err| log.warn("handoff {x}: the followers were not asked for their shards ({t})", .{ &h.id, err });
        x.produce.exportedBg(h.id, tokens, @ptrCast(@alignCast(saved)));
    }

    fn failed(ptr: *anyopaque, h: Handoff, why: []const u8) void {
        const x = self(ptr);
        x.produce.failed(h.id, why);
        if (x.split) ff.sendExportFail(x.f, h.id) catch {};
    }
};
