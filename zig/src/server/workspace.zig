//! Source-bound workspace prompts on raw completions, using the ordinary lane prompt cache.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const cli = @import("cli.zig");
const model_text = @import("model_text.zig");
const wm = @import("core").workspace_manifest;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;

pub const Operation = enum { cold, warm, edit };

pub const Request = struct {
    op: Operation,
    sources: []const []const u8 = &.{},
    revision: ?wm.Revision = null,
};

pub const Service = struct {
    gpa: Allocator,
    store: wm.Store,
    identity: wm.Identity,
    model_hex: [64]u8,
    template_hex: [64]u8,
    cache_hex: [64]u8,
    mutex: std.Io.Mutex = .init,

    pub fn init(gpa: Allocator, io: std.Io, root: []const u8, state: []const u8, model: [32]u8, text: model_text.Text, info: api.Info, args: cli.Args) !*Service {
        try privateState(io, state);
        const service = try gpa.create(Service);
        errdefer gpa.destroy(service);
        const template = templateIdentity(text);
        const cache = cacheIdentity(info, args);
        service.* = .{
            .gpa = gpa,
            .store = try wm.Store.init(gpa, io, root, state),
            .identity = undefined,
            .model_hex = std.fmt.bytesToHex(model, .lower),
            .template_hex = std.fmt.bytesToHex(template, .lower),
            .cache_hex = std.fmt.bytesToHex(cache, .lower),
        };
        service.identity = .{ .model = &service.model_hex, .template = &service.template_hex, .cache = &service.cache_hex };
        return service;
    }

    pub fn deinit(service: *Service) void {
        const gpa = service.gpa;
        service.store.deinit();
        gpa.destroy(service);
    }
};

fn privateState(io: std.Io, path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const permissions: std.Io.File.Permissions = .fromMode(0o700);
    if (std.fs.path.isAbsolute(path)) std.Io.Dir.createDirAbsolute(io, path, permissions) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    } else cwd.createDir(io, path, permissions) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    const dir = try cwd.openDir(io, path, .{ .follow_symlinks = false });
    defer dir.close(io);
    const stat = try dir.stat(io);
    if (stat.kind != .directory or stat.permissions.toMode() & 0o077 != 0) return error.InsecureWorkspaceState;
}

fn templateIdentity(text: model_text.Text) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("TensorFold raw workspace template v1");
    hash.update(text.templateSource());
    return hash.finalResult();
}

pub fn modelIdentity(a: Allocator, io: std.Io, dir: []const u8, checkpoint: [32]u8, tokenizer: [32]u8) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("TensorFold workspace model v2");
    hash.update(&checkpoint);
    hash.update(&tokenizer);
    const path = try std.fs.path.join(a, &.{ dir, "tokenizer.json" });
    defer a.free(path);
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            hash.update("semantic assets only");
            return hash.finalResult();
        },
        else => return err,
    };
    defer file.close(io);
    hash.update("tokenizer.json bytes");
    var buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const count = file.readStreaming(io, &.{&buffer}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (count == 0) break;
        hash.update(buffer[0..count]);
    }
    return hash.finalResult();
}

fn cacheIdentity(info: api.Info, args: cli.Args) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("TensorFold raw workspace cache v1");
    var buffer: [128]u8 = undefined;
    for ([_][]const u8{ info.name, args.backend, args.kv_cache, args.prompt_precision, args.policy }) |part| {
        const length = std.fmt.bufPrint(&buffer, "{d}:", .{part.len}) catch unreachable;
        hash.update(length);
        hash.update(part);
    }
    for ([_]u64{ info.context_window, info.prefill_step, info.prefill_min_chunk, @intFromBool(info.prompt_cache) }) |number| {
        const text = std.fmt.bufPrint(&buffer, "{d};", .{number}) catch unreachable;
        hash.update(text);
    }
    return hash.finalResult();
}

pub fn parse(cx: *Cx, raw: json.Value, is_chat: bool, enabled: bool) errors.Refused!?Request {
    const value = raw.get("workspace") orelse return null;
    if (is_chat) return cx.refuse("workspace is supported only on raw completions");
    if (!enabled) return cx.refuse("workspace completions require --workspace and --workspace-state");
    if (value != .object) return cx.refuse("workspace must be an object with op and sources");
    if (raw.get("messages")) |messages| if (messages == .array and messages.array.len > 0) return cx.refuse("workspace cannot be combined with messages");
    const named = value.get("op") orelse return cx.refuse("workspace.op must be cold, edit or warm");
    if (named != .string) return cx.refuse("workspace.op must be cold, edit or warm");
    const op = std.meta.stringToEnum(Operation, named.string) orelse return cx.refuse("workspace.op must be cold, edit or warm");
    const sources_value = value.get("sources");
    if (op == .warm) {
        if (sources_value != null or raw.get("prompt") != null) return cx.refuse("a warm workspace takes neither prompt nor sources");
        return .{ .op = .warm };
    }
    if (raw.get("prompt") == null or raw.get("prompt").? != .string) return cx.refuse("a cold or edited workspace needs one string prompt");
    const array = if (sources_value) |sources| if (sources == .array) sources.array else return cx.refuse("workspace.sources must be an array of relative paths") else return cx.refuse("workspace.sources must be an array of relative paths");
    if (array.len == 0 or array.len > 4096) return cx.refuse("workspace.sources must name 1 through 4096 files");
    const sources = try cx.a.alloc([]const u8, array.len);
    for (array, sources, 0..) |item, *source, i| {
        if (item != .string) return cx.refuse("workspace.sources must contain only relative path strings");
        source.* = item.string;
        for (sources[0..i]) |prior| if (std.mem.eql(u8, prior, source.*)) return cx.refuse("workspace.sources must not repeat a path");
    }
    return .{ .op = op, .sources = sources };
}

fn stopped(gone: anytype) wm.Stop {
    const Adapter = struct {
        fn check(ctx: *anyopaque) bool {
            const live: *const @TypeOf(gone) = @ptrCast(@alignCast(ctx));
            return live.*.check();
        }
    };
    return .{ .ptr = @ptrCast(@constCast(&gone)), .check = Adapter.check };
}

fn refusal(cx: *Cx, err: anyerror) error{ Refused, Cancelled, OutOfMemory } {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Cancelled, error.Canceled => error.Cancelled,
        error.NamespaceMismatch => cx.refuse("the workspace manifest belongs to a different model, template, cache or workspace"),
        else => cx.refuse("the workspace source or manifest could not be used safely"),
    };
}

pub const Pending = struct {
    service: *Service,
    io: std.Io,
    call: Operation,
    revision: wm.Revision,

    pub fn deinit(p: *Pending) void {
        p.service.mutex.unlock(p.io);
        p.* = undefined;
    }
};

pub fn resolve(srv: anytype, cx: *Cx, call: Request, base_text: []const u8, gone: anytype) error{ Refused, Cancelled, OutOfMemory }!Pending {
    const service = srv.workspace.?;
    service.mutex.lockUncancelable(srv.io);
    errdefer service.mutex.unlock(srv.io);
    const stop = stopped(gone);
    var sources: []wm.Source = &.{};
    var base: []const u32 = &.{};
    if (call.op != .warm) {
        base = srv.text.encode(cx.a, base_text, true) catch |err| return switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.Template => cx.refuse("the workspace prompt could not be tokenized"),
        };
        sources = try cx.a.alloc(wm.Source, call.sources.len);
        for (call.sources, sources) |path, *source| {
            const text = service.store.readSource(cx.a, path, stop) catch |err| return refusal(cx, err);
            if (!std.unicode.utf8ValidateSlice(text.bytes)) return cx.refuse("workspace sources must contain valid UTF-8");
            const tokens = srv.text.encode(cx.a, text.bytes, false) catch |err| return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                error.Template => cx.refuse("a workspace source could not be tokenized"),
            };
            source.* = .{ .path = path, .tokens = tokens, .size = @intCast(text.bytes.len), .sha256 = text.sha256 };
        }
    }
    var store = service.store;
    store.gpa = cx.a;
    const revision = switch (call.op) {
        .cold => store.prepareCold(service.identity, base, sources, stop),
        .edit => store.prepareEdit(service.identity, base, sources, stop),
        .warm => store.warm(service.identity, stop),
    } catch |err| return refusal(cx, err);
    return .{ .service = service, .io = srv.io, .call = call.op, .revision = revision };
}

pub fn commit(cx: *Cx, p: *Pending, gone: anytype) error{ Refused, Cancelled, OutOfMemory }!void {
    if (p.call == .warm) return;
    var store = p.service.store;
    store.gpa = p.revision.manifest.gpa;
    store.commit(&p.revision.manifest, p.call == .edit, stopped(gone)) catch |err| return refusal(cx, err);
}

pub fn metadata(a: Allocator, reply: anytype, revision: wm.Revision) Allocator.Error!void {
    const detail = try json.newObject(a);
    const provenance = std.fmt.bytesToHex(revision.manifest.provenance, .lower);
    try detail.put(a, "provenance", .{ .string = try std.fmt.allocPrint(a, "{s}", .{&provenance}) });
    try detail.put(a, "manifest_reused_tokens", try json.intValue(a, revision.reused_tokens));
    try detail.put(a, "changed_files", try json.intValue(a, revision.changed_files));
    try reply.runtime.object.put(a, "workspace", .{ .object = detail });
}
