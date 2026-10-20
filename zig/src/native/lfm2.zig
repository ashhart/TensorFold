//! d1 (LFM2-VL) on CUDA: an engine that scores decisions through the score hook and writes no text.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lfm2 = @import("lfm2");
const core = @import("core");
const Allocator = std.mem.Allocator;

pub const family: api.Family = .{ .model_type = "lfm2_vl", .formats = &.{"bf16"} };
const default_context = 32768;

const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    cfg: lfm2.config.Config,
    weights: lfm2.weights.Weights,
    model: lfm2.model.Model,
    vision: ?lfm2.vision.Vision = null,
    marks: ?lfm2.image.Marks = null, // the picture placeholders' ids; null: pictures refused
    window: u32,
    mutex: std.Io.Mutex = .init, // one prompt pass at a time: HTTP threads call score directly

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .name = "lfm2", .context_window = of(ctx).window, .thinking_budget = false, .decides = true, .pictures = of(ctx).vision != null };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed; // the server refuses text requests before they reach a decision engine
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
        const e = of(ctx);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.ctx.makeCurrent() catch return error.Failed;
        return e.model.score(prompt, labels, logits, mass) catch |err| {
            std.log.err("lfm2 decision pass failed: {s}", .{@errorName(err)});
            return error.Failed;
        };
    }

    /// Pictures decoded before the GPU is taken; branches packed into passes within the window and model limits.
    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        const e = of(ctx);
        const a = d.arena orelse return error.Failed;
        var trunk = d.trunk;
        const pictures = a.alloc(lfm2.image.Picture, d.pictures.len) catch return error.Failed;
        if (d.pictures.len > 0) {
            const marks = e.marks orelse return refuse(d, "this checkpoint has no vision tower");
            var ids: std.ArrayList(u32) = .empty;
            ids.appendSlice(a, d.trunk[0..d.at]) catch return error.Failed;
            for (d.pictures, pictures) |url, *pic| {
                var problem: []const u8 = "";
                const rgb = lfm2.image.load(e.io, a, url, &problem) catch return error.Failed;
                pic.* = lfm2.image.prepare(a, rgb orelse return refuse(d, problem)) catch return error.Failed;
                pic.appendIds(a, &ids, &marks) catch return error.Failed;
            }
            ids.appendSlice(a, d.trunk[d.at..]) catch return error.Failed;
            trunk = ids.items;
        }
        for (d.branches) |b| if (trunk.len + b.len > e.window)
            return refuse(d, std.fmt.allocPrint(a, "a decision sequence exceeds the {d}-token context", .{e.window}) catch return error.Failed);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.ctx.makeCurrent() catch return error.Failed;
        const placed = e.place(a, trunk, pictures) catch |err| {
            std.log.err("lfm2 picture features failed: {s}", .{@errorName(err)});
            return error.Failed;
        };
        d.read = trunk.len;
        var start: usize = 0;
        var at: usize = 0;
        while (start < d.branches.len) {
            var end = start + 1;
            var used = trunk.len + d.branches[start].len;
            var n = d.counts[start];
            while (end < d.branches.len and end - start < lfm2.model.branches_most and n + d.counts[end] <= lfm2.model.labels_most and used + d.branches[end].len <= e.window) : (end += 1) {
                used += d.branches[end].len;
                n += d.counts[end];
            }
            e.model.decide(trunk, d.branches[start..end], d.counts[start..end], d.labels[at..][0..n], d.logits[at..][0..n], placed) catch |err| {
                std.log.err("lfm2 decision pass failed: {s}", .{@errorName(err)});
                return error.Failed;
            };
            for (d.branches[start..end]) |b| d.read += b.len;
            start = end;
            at += n;
        }
    }

    /// The pictures' features in the model's picture rows and the trunk's placeholder runs they cover; null without.
    fn place(e: *Engine, a: Allocator, trunk: []const u32, pictures: []const lfm2.image.Picture) !?lfm2.model.Placed {
        if (pictures.len == 0) return null;
        const v = &e.vision.?;
        const out = e.model.pictureRows();
        var rows: usize = 0;
        for (pictures) |*p| rows += try v.features(e.model.stream, p, out + 2 * rows * e.cfg.hidden);
        var runs: std.ArrayList([2]usize) = .empty;
        var slots: usize = 0;
        for (trunk, 0..) |t, i| {
            if (t != e.marks.?.image) continue;
            slots += 1;
            if (runs.items.len > 0 and runs.items[runs.items.len - 1][0] + runs.items[runs.items.len - 1][1] == i) {
                runs.items[runs.items.len - 1][1] += 1;
            } else try runs.append(a, .{ i, 1 });
        }
        if (slots != rows) return error.PictureSlots;
        return .{ .features = out, .runs = runs.items };
    }

    fn refuse(d: *api.Decision, problem: []const u8) error{Refused} {
        d.problem = problem;
        return error.Refused;
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        e.ctx.makeCurrent() catch {};
        if (e.vision) |*v| v.deinit();
        e.model.deinit();
        e.weights.deinit(e.gpa);
        e.cfg.deinit(e.gpa);
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score, .decide = decide };
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const window: u32 = if (o.context) |c| blk: {
        if (c < 1 or c > default_context) {
            problem.* = try std.fmt.allocPrint(a, "--context {d}: d1 reads 1 to {d} tokens a decision", .{ c, default_context });
            return null;
        }
        break :blk @intCast(c);
    } else default_context;
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .window = window, .driver = undefined, .ctx = undefined, .cfg = undefined, .weights = undefined, .model = undefined };
    e.cfg = try lfm2.config.Config.load(gpa, io, o.dir, problem) orelse return null;
    errdefer e.cfg.deinit(gpa);
    e.driver = cuda.Driver.open() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.driver.close();
    e.ctx = try cuda.Context.init(&e.driver, 0);
    errdefer e.ctx.deinit();
    try e.ctx.makeCurrent();
    e.weights = try lfm2.weights.Weights.load(gpa, io, &e.driver, o.dir, &e.cfg);
    errdefer e.weights.deinit(gpa);
    e.model = try lfm2.model.Model.init(gpa, &e.driver, &e.cfg, &e.weights, window);
    errdefer e.model.deinit();
    if (e.weights.vision) |*vw| {
        const path = try std.fs.path.join(a, &.{ o.dir, "tokenizer.json" });
        var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
        defer tok.deinit();
        e.marks = lfm2.image.Marks.resolve(&tok) orelse {
            problem.* = "the LFM2 tokenizer lacks required picture marker tokens; image input cannot be served safely";
            return error.InvalidVisionMarkers;
        };
        if (e.marks) |*marks| if (marks.firstNonAdded(&tok)) |id| {
            problem.* = try std.fmt.allocPrint(a, "LFM2 picture marker token {d} is not an added token in this tokenizer; image input cannot be served safely", .{id});
            return error.InvalidVisionMarkers;
        };
        if (e.marks != null) e.vision = try lfm2.vision.Vision.init(gpa, &e.driver, &e.cfg, vw);
    }
    return .{ .engine = .{ .ctx = e, .vtable = &Engine.vtable }, .close = Engine.close, .ctx = e };
}
