//! `tensorfold tfq3 MODEL`: pull or resolve any Safetensors checkpoint and quantize to TFQ3.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Environ = std.process.Environ.Map;
const hub = @import("hub.zig");
const pull = @import("pull.zig");
const output_guard = @import("tfq3_cli_output.zig");

pub const install_cmd = "see https://github.com/ashhart/TFQ#installation (install a pinned release and verify its published checksum)";

pub const usage =
    \\usage: tensorfold tfq3 MODEL [--out DIR] [--profile 4b|3.5b|3b] [--bits 3|4|5]
    \\                             [--target-gib GB] [--fit auto|GB] [--residual auto|bf16|q8]
    \\                             [--head-bits 0|4|5] [--ple-bits 3|4] [--trace FILE]
    \\                             [--nonmoe-model DIR] [--family auto|qwen27|flashnext|glm|generic]
    \\                             [--quantize-vision] [--quantize-drafter] [--quantize-embed]
    \\                             [--native] [--converter PATH] [--skip-existing] [--no-pull] [--dry-run]
    \\       tensorfold tfq3 inspect [-v] DIR
    \\       tensorfold tfq3 doctor [--converter PATH]
    \\
;

pub const Family = enum {
    qwen27,
    flashnext,
    glm,
    generic,

    pub fn name(self: Family) []const u8 {
        return switch (self) {
            .qwen27 => "qwen27",
            .flashnext => "flashnext",
            .glm => "glm",
            .generic => "generic",
        };
    }

    pub fn parse(text: []const u8) ?Family {
        if (std.mem.eql(u8, text, "qwen27")) return .qwen27;
        if (std.mem.eql(u8, text, "flashnext")) return .flashnext;
        if (std.mem.eql(u8, text, "glm")) return .glm;
        if (std.mem.eql(u8, text, "generic")) return .generic;
        return null;
    }

    pub fn fromModelType(model_type: []const u8) ?Family {
        if (std.mem.eql(u8, model_type, "qwen3_5") or std.mem.eql(u8, model_type, "qwen3_5_text")) return .qwen27;
        if (std.mem.eql(u8, model_type, "qwen4_exp")) return .flashnext;
        if (std.mem.eql(u8, model_type, "glm5_next")) return .glm;
        return null;
    }
};

pub const Residual = enum {
    auto,
    bf16,
    q8,

    pub fn name(self: Residual) []const u8 {
        return switch (self) {
            .auto => "auto",
            .bf16 => "bf16",
            .q8 => "q8",
        };
    }
};

pub const Mode = enum { help, convert, inspect, doctor };

pub const Options = struct {
    mode: Mode = .convert,
    model: []const u8 = "",
    out_dir: ?[]const u8 = null,
    family: ?Family = null,
    profile: []const u8 = "4b",
    bits: ?[]const u8 = null,
    target_gib: ?[]const u8 = null,
    fit: ?[]const u8 = null,
    residual: Residual = .auto,
    residual_set: bool = false,
    head_bits: []const u8 = "4",
    ple_bits: []const u8 = "3",
    ple_bits_set: bool = false,
    trace: ?[]const u8 = null,
    nonmoe_model: ?[]const u8 = null,
    converter: ?[]const u8 = null,
    quantize_vision: bool = false,
    quantize_drafter: bool = false,
    quantize_embed: bool = false,
    native: bool = false,
    skip_existing: bool = false,
    no_pull: bool = false,
    dry_run: bool = false,
    verbose: bool = false,

    pub fn wantsGeneric(self: Options) bool {
        return self.bits != null or self.target_gib != null or self.fit != null or
            self.quantize_vision or self.quantize_drafter or self.quantize_embed;
    }
};

pub const Hooks = struct {
    pull_fn: *const fn (*const Context, []const u8) anyerror!u8 = defaultPull,
    find_converter: *const fn (*const Context, ?[]const u8) anyerror!?[]const u8 = defaultFindConverter,
    spawn_fn: *const fn (*const Context, []const []const u8) anyerror!u8 = defaultSpawn,
};

pub const Context = struct {
    a: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    err: *std.Io.Writer,
    env: ?*const Environ = null,
    cache_override: ?[]const u8 = null,
    test_ctx: *anyopaque = undefined,
    hooks: Hooks = .{},
};

/// Parses argv after `tfq3`; returns null on usage errors.
pub fn parseArgs(args: []const []const u8) ?Options {
    if (args.len == 0) return null;
    if (std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h") or std.mem.eql(u8, args[0], "help")) {
        return .{ .mode = .help };
    }
    if (std.mem.eql(u8, args[0], "doctor")) {
        var o: Options = .{ .mode = .doctor };
        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--converter")) {
                if (i + 1 >= args.len) return null;
                i += 1;
                o.converter = args[i];
            } else return null;
        }
        return o;
    }
    if (std.mem.eql(u8, args[0], "inspect")) {
        var o: Options = .{ .mode = .inspect };
        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "-v") or std.mem.eql(u8, args[i], "--verbose")) {
                o.verbose = true;
            } else if (std.mem.eql(u8, args[i], "--converter")) {
                if (i + 1 >= args.len) return null;
                i += 1;
                o.converter = args[i];
            } else if (std.mem.startsWith(u8, args[i], "-") or o.model.len != 0) {
                return null;
            } else o.model = args[i];
        }
        return if (o.model.len > 0) o else null;
    }
    var o: Options = .{ .mode = .convert };
    var i: usize = if (std.mem.eql(u8, args[0], "convert")) 1 else 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return .{ .mode = .help };
        if (std.mem.eql(u8, arg, "--out") or std.mem.eql(u8, arg, "-o")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            o.out_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--family")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            o.family = if (std.mem.eql(u8, args[i], "auto")) null else Family.parse(args[i]) orelse return null;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (!validChoice(args[i], &.{ "4b", "3.5b", "3b" })) return null;
            o.profile = args[i];
        } else if (std.mem.eql(u8, arg, "--bits")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (!validChoice(args[i], &.{ "3", "4", "5" })) return null;
            o.bits = args[i];
        } else if (std.mem.eql(u8, arg, "--target-gib")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (parsePosF64(args[i]) == null) return null;
            o.target_gib = args[i];
        } else if (std.mem.eql(u8, arg, "--fit")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (!std.mem.eql(u8, args[i], "auto") and parsePosF64(args[i]) == null) return null;
            o.fit = args[i];
        } else if (std.mem.eql(u8, arg, "--residual")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (std.mem.eql(u8, args[i], "auto")) o.residual = .auto else if (std.mem.eql(u8, args[i], "bf16")) o.residual = .bf16 else if (std.mem.eql(u8, args[i], "q8")) o.residual = .q8 else return null;
            o.residual_set = true;
        } else if (std.mem.eql(u8, arg, "--head-bits")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (!validChoice(args[i], &.{ "0", "4", "5" })) return null;
            o.head_bits = args[i];
        } else if (std.mem.eql(u8, arg, "--ple-bits")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            if (!validChoice(args[i], &.{ "3", "4" })) return null;
            o.ple_bits = args[i];
            o.ple_bits_set = true;
        } else if (std.mem.eql(u8, arg, "--trace")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            o.trace = args[i];
        } else if (std.mem.eql(u8, arg, "--nonmoe-model")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            o.nonmoe_model = args[i];
        } else if (std.mem.eql(u8, arg, "--converter")) {
            if (i + 1 >= args.len) return null;
            i += 1;
            o.converter = args[i];
        } else if (std.mem.eql(u8, arg, "--quantize-vision")) {
            o.quantize_vision = true;
        } else if (std.mem.eql(u8, arg, "--quantize-drafter")) {
            o.quantize_drafter = true;
        } else if (std.mem.eql(u8, arg, "--quantize-embed")) {
            o.quantize_embed = true;
        } else if (std.mem.eql(u8, arg, "--native")) {
            o.native = true;
        } else if (std.mem.eql(u8, arg, "--skip-existing")) {
            o.skip_existing = true;
        } else if (std.mem.eql(u8, arg, "--no-pull") or std.mem.eql(u8, arg, "--no-download")) {
            o.no_pull = true;
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            o.dry_run = true;
        } else if (std.mem.startsWith(u8, arg, "-") or o.model.len != 0) {
            return null;
        } else o.model = arg;
    }
    return if (o.model.len > 0) o else null;
}

pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, env: ?*const Environ, args: []const []const u8) !u8 {
    const ctx: Context = .{ .a = a, .io = io, .out = out, .err = err, .env = env };
    return runWith(&ctx, args);
}

pub fn runWith(ctx: *const Context, args: []const []const u8) !u8 {
    const o = parseArgs(args) orelse {
        try ctx.err.writeAll(usage);
        return 2;
    };
    return switch (o.mode) {
        .help => blk: {
            try ctx.out.writeAll(usage);
            break :blk 0;
        },
        .doctor => runDoctor(ctx, o),
        .inspect => runInspect(ctx, o),
        .convert => runConvert(ctx, o),
    };
}

fn runDoctor(ctx: *const Context, o: Options) !u8 {
    const conv = try ctx.hooks.find_converter(ctx, o.converter);
    if (conv) |c| {
        try ctx.out.flush();
        return ctx.hooks.spawn_fn(ctx, &.{ c, "doctor" });
    }
    if (o.converter != null) {
        try missingConverter(ctx.err, o.converter, null);
        return 1;
    }
    try ctx.out.writeAll("native Zig TFQ3 encoder: unavailable (its dense layout has no production loader)\n");
    try ctx.out.print("GPU converter (tfq3):    not installed; for ~10x faster Metal/CUDA conversion:\n  {s}\n", .{install_cmd});
    return 0;
}

fn runInspect(ctx: *const Context, o: Options) !u8 {
    const dir = try expandHome(ctx.a, ctx.env, o.model);
    if (!hub.isDir(ctx.io, dir)) {
        try ctx.err.print("tensorfold tfq3: {s} is not a directory\n", .{dir});
        return 1;
    }
    if (try ctx.hooks.find_converter(ctx, o.converter)) |conv| {
        try ctx.out.flush();
        if (o.verbose) return ctx.hooks.spawn_fn(ctx, &.{ conv, "inspect", "-v", dir });
        return ctx.hooks.spawn_fn(ctx, &.{ conv, "inspect", dir });
    }
    if (o.converter) |explicit| {
        try missingConverter(ctx.err, explicit, null);
        return 1;
    }
    return inspectNative(ctx.a, ctx.io, ctx.out, ctx.err, dir);
}

fn runConvert(ctx: *const Context, o: Options) !u8 {
    const conv = if (o.native) null else try ctx.hooks.find_converter(ctx, o.converter);
    const use_native = o.native or (conv == null and o.converter == null);
    const expanded = try expandHome(ctx.a, ctx.env, o.model);
    var src_dir: []const u8 = "";
    var from_repo = false;
    var needs_pull = false;
    if (hub.isDir(ctx.io, expanded)) {
        src_dir = expanded;
    } else if (splitRepoSpec(o.model)) |spec| {
        from_repo = true;
        const root = try hub.cacheDir(ctx.a, ctx.env, ctx.cache_override);
        if (try hub.cachedSnapshotRev(ctx.a, ctx.io, root, spec.repo, spec.rev)) |snap| {
            src_dir = snap;
        } else if (o.no_pull) {
            try ctx.err.print("tensorfold tfq3: {s} is not in the Hugging Face cache; run: tensorfold pull {s}\n", .{ o.model, o.model });
            return 1;
        } else if (o.dry_run) {
            needs_pull = true;
            src_dir = try std.fmt.allocPrint(ctx.a, "<pull:{s}>", .{o.model});
        } else {
            if (conv == null and o.converter != null) {
                try missingConverter(ctx.err, o.converter, o.model);
                return 1;
            }
            if (use_native) return (try rejectNativeOptions(ctx.err, o, o.family)).?;
            try ctx.out.print("tensorfold tfq3: pulling {s} into Hugging Face cache...\n", .{o.model});
            try ctx.out.flush();
            const code = try ctx.hooks.pull_fn(ctx, o.model);
            if (code != 0) return code;
            src_dir = try hub.cachedSnapshotRev(ctx.a, ctx.io, root, spec.repo, spec.rev) orelse {
                try ctx.err.print("tensorfold tfq3: pulled {s} but no cached snapshot with config.json was found\n", .{o.model});
                return 1;
            };
        }
    } else {
        try ctx.err.print("tensorfold tfq3: {s} is neither a directory nor a Hugging Face repo id (owner/name)\n", .{o.model});
        return 2;
    }

    const out_dir = if (o.out_dir) |d| try expandHome(ctx.a, ctx.env, d) else try defaultOutDir(ctx.a, o.model, expanded, from_repo);
    var family = o.family;
    var residual = o.residual;
    var model_type: []const u8 = "unknown";
    if (!needs_pull) {
        model_type = detectModelType(ctx.a, ctx.io, src_dir);
        if (family == null) {
            family = if (o.wantsGeneric()) .generic else (Family.fromModelType(model_type) orelse .generic);
        }
        if (residual == .auto) {
            const q = hub.quantization(ctx.a, ctx.io, src_dir);
            residual = if (q != null and q.?.bits == 8) .q8 else .bf16;
        }
    }

    if (conv == null and o.converter != null and !o.dry_run) {
        try missingConverter(ctx.err, o.converter, o.model);
        return 1;
    }

    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(ctx.a, &.{ conv orelse "tfq3", "convert", src_dir, "--out", out_dir, "--no-pull" });
    if (family) |f| try argv.appendSlice(ctx.a, &.{ "--family", f.name() });
    if (family == null or family.? == .qwen27) {
        try argv.appendSlice(ctx.a, &.{ "--profile", o.profile, "--residual", residual.name(), "--head-bits", o.head_bits });
    } else if (family.? == .flashnext) {
        try argv.appendSlice(ctx.a, &.{ "--ple-bits", o.ple_bits });
    } else if (family.? == .generic) {
        try argv.appendSlice(ctx.a, &.{ "--profile", o.profile, "--head-bits", o.head_bits });
    }
    if (o.bits) |b| try argv.appendSlice(ctx.a, &.{ "--bits", b });
    if (o.target_gib) |g| try argv.appendSlice(ctx.a, &.{ "--target-gib", g });
    if (o.fit) |f| try argv.appendSlice(ctx.a, &.{ "--fit", f });
    if (o.quantize_vision) try argv.append(ctx.a, "--quantize-vision");
    if (o.quantize_drafter) try argv.append(ctx.a, "--quantize-drafter");
    if (o.quantize_embed) try argv.append(ctx.a, "--quantize-embed");
    if (o.trace) |t| try argv.appendSlice(ctx.a, &.{ "--trace", try expandHome(ctx.a, ctx.env, t) });
    if (o.nonmoe_model) |nm| try argv.appendSlice(ctx.a, &.{ "--nonmoe-model", try expandHome(ctx.a, ctx.env, nm) });
    if (o.skip_existing) try argv.append(ctx.a, "--skip-existing");

    if (o.dry_run) {
        if (needs_pull) try ctx.out.print("[dry-run] pull {s}\n", .{o.model});
        try ctx.out.print("source:    {s}\noutput:    {s}\nfamily:    {s} (model_type={s})\nconverter: {s}\n", .{
            src_dir,
            out_dir,
            if (family) |f| f.name() else "auto",
            model_type,
            if (use_native) "native (refused: no production loader)" else (conv orelse "not installed"),
        });
        try ctx.out.writeAll("[dry-run]");
        for (argv.items) |arg| try ctx.out.print(" {s}", .{arg});
        try ctx.out.writeByte('\n');
        if (conv == null) try ctx.out.print("install GPU converter: {s}\n", .{install_cmd});
        return 0;
    }
    if (use_native) return (try rejectNativeOptions(ctx.err, o, family)).?;
    if (try output_guard.check(ctx.a, ctx.io, ctx.err, src_dir, out_dir, o.skip_existing) != 0) return 1;
    try ctx.out.flush();
    return ctx.hooks.spawn_fn(ctx, argv.items);
}

fn rejectNativeOptions(err: *std.Io.Writer, o: Options, family: ?Family) !?u8 {
    if (std.mem.eql(u8, o.profile, "3.5b")) {
        try err.writeAll("tensorfold tfq3: native conversion does not support --profile 3.5b\n");
        return 1;
    }
    if (o.residual_set) {
        try err.writeAll("tensorfold tfq3: native conversion does not support explicit --residual semantics\n");
        return 1;
    }
    if (o.ple_bits_set) {
        try err.writeAll("tensorfold tfq3: native conversion does not support --ple-bits\n");
        return 1;
    }
    if (o.trace != null) {
        try err.writeAll("tensorfold tfq3: native conversion does not support --trace\n");
        return 1;
    }
    if (o.nonmoe_model != null) {
        try err.writeAll("tensorfold tfq3: native conversion does not support --nonmoe-model\n");
        return 1;
    }
    if (o.skip_existing) {
        try err.writeAll("tensorfold tfq3: native conversion does not support --skip-existing\n");
        return 1;
    }
    if (family) |f| if (f != .generic) {
        try err.print("tensorfold tfq3: native conversion does not support {s} family layout (--family {s})\n", .{ f.name(), f.name() });
        return 1;
    };
    try err.writeAll("tensorfold tfq3: native conversion is unavailable: its dense layout has no production loader; install the external converter\n");
    return 1;
}

/// Validates native output using the same table and shard loaders as inference.
pub fn inspectNative(a: Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, dir: []const u8) !u8 {
    return @import("tfq3_cli_inspect.zig").run(a, io, out, err, dir);
}

fn missingConverter(err: *std.Io.Writer, explicit: ?[]const u8, model: ?[]const u8) !void {
    if (explicit) |p| {
        try err.print("tensorfold tfq3: converter not found at {s}\n", .{p});
        return;
    }
    try err.print("tensorfold tfq3: converter 'tfq3' was not found.\nInstall it once (auto-detects Apple Silicon Metal or NVIDIA CUDA):\n  {s}\n", .{install_cmd});
    if (model) |m| try err.print("Then re-run:\n  tensorfold tfq3 {s}\n", .{m});
}

fn validChoice(val: []const u8, choices: []const []const u8) bool {
    for (choices) |c| if (std.mem.eql(u8, val, c)) return true;
    return false;
}

fn parsePosF64(s: []const u8) ?f64 {
    const v = std.fmt.parseFloat(f64, s) catch return null;
    const max_gib = @as(f64, @floatFromInt(std.math.maxInt(u64))) / @as(f64, @floatFromInt(hub.GiB));
    return if (std.math.isFinite(v) and v > 0.0 and v < max_gib) v else null;
}

fn splitRepoSpec(spec: []const u8) ?struct { repo: []const u8, rev: []const u8 } {
    const at = std.mem.indexOfScalar(u8, spec, '@');
    const repo = if (at) |idx| spec[0..idx] else spec;
    const rev = if (at) |idx| spec[idx + 1 ..] else "main";
    if (!hub.isRepoIdLike(repo) or !hub.revisionOk(rev)) return null;
    return .{ .repo = repo, .rev = rev };
}

fn expandHome(a: Allocator, env: ?*const Environ, path: []const u8) ![]const u8 {
    if (!std.mem.startsWith(u8, path, "~/")) return path;
    const home = hub.homeDir(env) orelse return path;
    return std.fs.path.join(a, &.{ home, path[2..] });
}

fn defaultOutDir(a: Allocator, raw: []const u8, expanded: []const u8, from_repo: bool) ![]const u8 {
    if (from_repo) {
        const spec = splitRepoSpec(raw).?;
        const slash = std.mem.indexOfScalar(u8, spec.repo, '/').?;
        return std.fmt.allocPrint(a, "./{s}-TFQ3", .{spec.repo[slash + 1 ..]});
    }
    return std.fmt.allocPrint(a, "{s}-TFQ3", .{std.mem.trimEnd(u8, expanded, "/\\")});
}

/// Reads root `model_type` or `text_config.model_type` from config.json.
pub fn detectModelType(a: Allocator, io: std.Io, dir: []const u8) []const u8 {
    const text = @import("checkpoint_config").optional(a, io, dir) orelse return "unknown";
    var parsed = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return "unknown";
    defer parsed.deinit();
    if (parsed.value != .object) return "unknown";
    if (parsed.value.object.get("model_type")) |mt| {
        if (mt == .string and mt.string.len > 0) return a.dupe(u8, mt.string) catch "unknown";
    }
    if (parsed.value.object.get("text_config")) |tc| {
        if (tc == .object) if (tc.object.get("model_type")) |mt| {
            if (mt == .string and mt.string.len > 0) return a.dupe(u8, mt.string) catch "unknown";
        };
    }
    return "unknown";
}

fn isFile(io: std.Io, path: []const u8) bool {
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return st.kind == .file or st.kind == .sym_link;
}

fn defaultPull(ctx: *const Context, spec: []const u8) anyerror!u8 {
    return pull.run(ctx.a, ctx.io, ctx.out, ctx.err, ctx.env, ctx.cache_override, spec);
}

fn defaultFindConverter(ctx: *const Context, explicit: ?[]const u8) anyerror!?[]const u8 {
    if (explicit) |p| {
        const full = try expandHome(ctx.a, ctx.env, p);
        if (std.fs.path.isAbsolute(full) or std.mem.indexOfAny(u8, full, "/\\") != null) {
            return if (isFile(ctx.io, full)) full else null;
        }
        return findOnPath(ctx, full);
    }
    const exe_name = if (comptime builtin.os.tag == .windows) "tfq3.exe" else "tfq3";
    if (std.process.executableDirPathAlloc(ctx.io, ctx.a)) |exe_dir| {
        const beside = try std.fs.path.join(ctx.a, &.{ exe_dir, exe_name });
        if (isFile(ctx.io, beside)) return beside;
    } else |_| {}
    if (hub.homeDir(ctx.env)) |home| {
        const local_bin = try std.fs.path.join(ctx.a, &.{ home, ".local", "bin", exe_name });
        if (isFile(ctx.io, local_bin)) return local_bin;
    }
    if (comptime builtin.os.tag != .windows) {
        for ([_][]const u8{ "/opt/homebrew/bin/tfq3", "/usr/local/bin/tfq3" }) |cand| {
            if (isFile(ctx.io, cand)) return cand;
        }
    }
    return findOnPath(ctx, exe_name);
}

fn findOnPath(ctx: *const Context, exe_name: []const u8) !?[]const u8 {
    if (hub.envValue(ctx.env, "PATH")) |path_env| {
        var it = std.mem.tokenizeScalar(u8, path_env, std.fs.path.delimiter);
        while (it.next()) |dir| {
            const cand = try std.fs.path.join(ctx.a, &.{ dir, exe_name });
            if (isFile(ctx.io, cand)) return cand;
        }
    }
    return null;
}

fn defaultSpawn(ctx: *const Context, argv: []const []const u8) anyerror!u8 {
    var child = try std.process.spawn(ctx.io, .{ .argv = argv, .environ_map = ctx.env });
    defer child.kill(ctx.io);
    const term = try child.wait(ctx.io);
    return switch (term) {
        .exited => |code| @intCast(@min(code, 255)),
        .signal => |sig| 128 + @as(u8, @intCast(@min(@backingInt(sig), 127))),
        else => 1,
    };
}
