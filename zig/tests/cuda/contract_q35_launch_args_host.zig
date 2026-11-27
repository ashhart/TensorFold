//! The q35 attention launches' argument lists against the .cu kernels' signatures, without a device.
const std = @import("std");
const qwen35 = @import("qwen35");
const cuda = @import("cuda");
const kernel = @import("pdl_kernel_scan");
const sig = qwen35.signatures;

const Tok = std.zig.Token;
const Span = struct { start: usize, end: usize };

/// One kernel's parameters as the .cu source declares them.
const Found = struct { symbol: []const u8, params: []const ?sig.Kind };

/// A launch site's handles, the signature it names and the argument tuple's elements.
const Site = struct {
    path: []const u8,
    handles: []const []const u8,
    named: ?[]const u8,
    dependent: bool,
    elements: []const []const u8,
};

fn identByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn word(text: []const u8, at: usize) []const u8 {
    return text[at..];
}

/// A parameter's kind: geometry struct, pointer, count, scale or Boolean mode.
fn classify(param: []const u8) ?sig.Kind {
    const text = std.mem.trim(u8, param, " \t\r\n");
    if (text.len == 0) return null;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    const first = it.next() orelse return null;
    if (std.mem.eql(u8, first, "Geo")) return .geo;
    if (std.mem.indexOfScalar(u8, text, '*') != null) return .ptr;
    if (std.mem.eql(u8, first, "int")) return .i32;
    if (std.mem.eql(u8, first, "float")) return .f32;
    if (std.mem.eql(u8, first, "bool")) return .boolean;
    return null;
}

/// The parameters of one kernel's list, split at the commas outside any brackets.
fn splitParams(a: std.mem.Allocator, text: []const u8, out: *std.ArrayList(?sig.Kind)) !void {
    var depth: usize = 0;
    var start: usize = 0;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (c == '(' or c == '[' or c == '{' or c == '<') depth += 1;
        if (c == ')' or c == ']' or c == '}' or c == '>') depth -= 1;
        if (c != ',' or depth != 0) continue;
        try out.append(a, classify(text[start..i]));
        start = i + 1;
    }
    if (start < text.len) try out.append(a, classify(text[start..]));
}

/// Every __global__ kernel in a CUDA source, with its parameter kinds.
fn kernelsIn(a: std.mem.Allocator, text: []const u8, out: *std.ArrayList(Found)) !void {
    const bounds = "__launch_bounds__";
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, "__global__")) |pos| {
        at = pos + "__global__".len;
        var open: ?usize = null;
        var i = at;
        while (i < text.len) : (i += 1) {
            if (text[i] != '(') continue;
            const before = if (i >= bounds.len) text[i - bounds.len .. i] else "";
            if (std.mem.eql(u8, before, bounds)) {
                var depth: usize = 0;
                while (i < text.len) : (i += 1) {
                    if (text[i] == '(') depth += 1;
                    if (text[i] == ')') {
                        depth -= 1;
                        if (depth == 0) break;
                    }
                }
                continue;
            }
            open = i;
            break;
        }
        const o = open orelse continue;
        var end = o;
        while (end > 0 and identByte(text[end - 1])) end -= 1;
        if (end == o) continue;
        var close = o;
        var depth: usize = 0;
        while (close < text.len) : (close += 1) {
            if (text[close] == '(') depth += 1;
            if (text[close] == ')') {
                depth -= 1;
                if (depth == 0) break;
            }
        }
        if (close >= text.len) continue;
        var params: std.ArrayList(?sig.Kind) = .empty;
        try splitParams(a, text[o + 1 .. close], &params);
        try out.append(a, .{ .symbol = text[end..o], .params = params.items });
    }
}

/// The source of a tuple literal's elements, split at its own level.
fn tupleElements(a: std.mem.Allocator, tokens: []const Tok, text: []const u8, open: usize) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var depth: usize = 1;
    var start = open + 1;
    var i = open + 1;
    while (i < tokens.len) : (i += 1) {
        const tag = tokens[i].tag;
        if (tag == .l_brace or tag == .l_paren or tag == .l_bracket) {
            depth += 1;
            continue;
        }
        if (tag == .r_brace or tag == .r_paren or tag == .r_bracket) {
            depth -= 1;
            if (depth == 0) {
                if (i > start) try out.append(a, std.mem.trim(u8, text[tokens[start].loc.start..tokens[i].loc.start], " \t\r\n"));
                break;
            }
            continue;
        }
        if (tag == .comma and depth == 1) {
            try out.append(a, std.mem.trim(u8, text[tokens[start].loc.start..tokens[i].loc.start], " \t\r\n"));
            start = i + 1;
        }
    }
    return out.items;
}

/// The kind a launch site's argument carries: geometry struct, pointer, count, scale or Boolean mode.
fn argKind(text: []const u8) sig.Kind {
    if (std.mem.startsWith(u8, text, "@as(f32")) return .f32;
    if (std.mem.startsWith(u8, text, "@as(c_int")) return .i32;
    if (std.mem.eql(u8, text, "geo")) return .geo;
    if (std.mem.eql(u8, text, "rows_only")) return .boolean;
    return .ptr;
}

/// The tuple literal of a local constant, if the file holds one by that name.
fn constTuple(tokens: []const Tok, text: []const u8, name: []const u8) ?usize {
    var i: usize = 0;
    while (i + 4 < tokens.len) : (i += 1) {
        if (tokens[i].tag != .keyword_const or tokens[i + 1].tag != .identifier) continue;
        if (!std.mem.eql(u8, text[tokens[i + 1].loc.start..tokens[i + 1].loc.end], name)) continue;
        if (tokens[i + 2].tag != .equal or tokens[i + 3].tag != .period or tokens[i + 4].tag != .l_brace) continue;
        return i + 4;
    }
    return null;
}

/// The nearest preceding constant expression, used for a locally selected kernel handle.
fn constValue(tokens: []const Tok, text: []const u8, name: []const u8, before: usize) ?Span {
    var result: ?Span = null;
    var i: usize = 0;
    while (i + 3 < before) : (i += 1) {
        if (tokens[i].tag != .keyword_const or tokens[i + 1].tag != .identifier or tokens[i + 2].tag != .equal) continue;
        if (!std.mem.eql(u8, text[tokens[i + 1].loc.start..tokens[i + 1].loc.end], name)) continue;
        var end = i + 3;
        while (end < before and tokens[end].tag != .semicolon) : (end += 1) {}
        if (end < before) result = .{ .start = i + 3, .end = end };
    }
    return result;
}

/// The handles and the signature name in one argument.
fn readArg(a: std.mem.Allocator, tokens: []const Tok, text: []const u8, span: Span, handles: *std.ArrayList([]const u8)) !?[]const u8 {
    var k = span.start;
    while (k + 4 < span.end) : (k += 1) {
        if (tokens[k].tag != .identifier or tokens[k + 1].tag != .period or tokens[k + 3].tag != .period) continue;
        if (tokens[k + 2].tag != .identifier or tokens[k + 4].tag != .identifier) continue;
        const middle = text[tokens[k + 2].loc.start..tokens[k + 2].loc.end];
        if (!std.mem.eql(u8, middle, "fns") and !std.mem.eql(u8, middle, "fp8") and !std.mem.eql(u8, middle, "mt2")) continue;
        try handles.append(a, text[tokens[k].loc.start..tokens[k + 4].loc.end]);
        k += 4;
    }
    k = span.start;
    while (k + 2 < span.end) : (k += 1) {
        if (tokens[k].tag != .identifier or tokens[k + 1].tag != .period or tokens[k + 2].tag != .identifier) continue;
        const handle = text[tokens[k].loc.start..tokens[k + 2].loc.end];
        if (sig.byHandle(handle) != null) try handles.append(a, handle);
    }
    if (span.end - span.start != 3) return null;
    if (tokens[span.start].tag != .identifier or tokens[span.start + 1].tag != .period or tokens[span.start + 2].tag != .identifier) return null;
    if (!std.mem.eql(u8, text[tokens[span.start].loc.start..tokens[span.start].loc.end], "sig")) return null;
    return text[tokens[span.start + 2].loc.start..tokens[span.start + 2].loc.end];
}

/// Every launch of a family kernel: its handles, the signature it names and the argument tuple's size.
fn sitesIn(a: std.mem.Allocator, path: []const u8, text: [:0]const u8, out: *std.ArrayList(Site)) !void {
    var lexer = std.zig.Tokenizer.init(text);
    var tokens: std.ArrayList(Tok) = .empty;
    while (true) {
        const token = lexer.next();
        if (token.tag == .eof) break;
        try tokens.append(a, token);
    }
    const list = tokens.items;
    var i: usize = 0;
    while (i < list.len) : (i += 1) {
        if (list[i].tag != .identifier) continue;
        const name = text[list[i].loc.start..list[i].loc.end];
        if (!std.mem.eql(u8, name, "run") and !std.mem.eql(u8, name, "runPdl") and !std.mem.eql(u8, name, "shared") and !std.mem.eql(u8, name, "launch") and !std.mem.eql(u8, name, "launchPdl")) continue;
        if (i + 1 >= list.len or list[i + 1].tag != .l_paren) continue;
        var spans: std.ArrayList(Span) = .empty;
        var depth: usize = 0;
        var start = i + 2;
        var j = i + 1;
        while (j < list.len) : (j += 1) {
            const tag = list[j].tag;
            if (tag == .l_paren or tag == .l_brace or tag == .l_bracket) {
                depth += 1;
                continue;
            }
            if (tag == .r_paren or tag == .r_brace or tag == .r_bracket) {
                depth -= 1;
                if (depth == 0) {
                    try spans.append(a, .{ .start = start, .end = j });
                    break;
                }
                continue;
            }
            if (tag == .comma and depth == 1) {
                try spans.append(a, .{ .start = start, .end = j });
                start = j + 1;
            }
        }
        if (spans.items.len == 0) continue;
        var handles: std.ArrayList([]const u8) = .empty;
        var named: ?[]const u8 = null;
        for (spans.items) |span| {
            const found = try readArg(a, list, text, span, &handles);
            if (found) |n| named = n;
            if (span.end - span.start == 1 and list[span.start].tag == .identifier) {
                const local = constValue(list, text, text[list[span.start].loc.start..list[span.start].loc.end], i);
                if (local) |value| _ = try readArg(a, list, text, value, &handles);
            }
        }
        if (handles.items.len == 0) continue;
        const last = spans.items[spans.items.len - 1];
        var elements: []const []const u8 = &.{};
        if (last.end - last.start >= 2 and list[last.start].tag == .period and list[last.start + 1].tag == .l_brace) {
            elements = try tupleElements(a, list, text, last.start + 1);
        } else if (last.end - last.start == 1 and list[last.start].tag == .identifier) {
            const open = constTuple(list, text, text[list[last.start].loc.start..list[last.start].loc.end]);
            if (open) |o| elements = try tupleElements(a, list, text, o);
        }
        try out.append(a, .{ .path = path, .handles = handles.items, .named = named, .elements = elements, .dependent = std.mem.eql(u8, name, "runPdl") or std.mem.eql(u8, name, "launchPdl") });
    }
}

/// Every source file under a directory, with its bytes.
fn sources(a: std.mem.Allocator, dir: std.Io.Dir, prefix: []const u8, want: []const u8, out: *std.ArrayList([:0]const u8)) !void {
    const io = std.testing.io;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ prefix, entry.name });
        if (entry.kind == .directory) {
            var child = try dir.openDir(io, entry.name, .{ .iterate = true });
            defer child.close(io);
            try sources(a, child, path, want, out);
        } else if (entry.kind == .file) {
            if (!std.mem.endsWith(u8, entry.name, want)) continue;
            const data = try dir.readFileAlloc(io, entry.name, a, .limited(16 * 1024 * 1024));
            try out.append(a, try a.dupeSentinel(u8, data, 0));
        }
    }
}

test "the q35 attention kernels' parameter lists match their .cu signatures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files: std.ArrayList([:0]const u8) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(std.testing.io, "zig/kernels/cuda", .{ .iterate = true });
    defer dir.close(std.testing.io);
    try sources(a, dir, "zig/kernels/cuda", ".cu", &files);
    var dir2 = try std.Io.Dir.cwd().openDir(std.testing.io, "zig/kernels/cuda", .{ .iterate = true });
    defer dir2.close(std.testing.io);
    try sources(a, dir2, "zig/kernels/cuda", ".cuh", &files);
    var found: std.ArrayList(Found) = .empty;
    for (files.items) |text| try kernelsIn(a, text, &found);
    var mismatches: usize = 0;
    var checked: usize = 0;
    for (sig.all) |s| {
        var seen: usize = 0;
        for (found.items) |f| {
            if (!std.mem.eql(u8, f.symbol, s.symbol)) continue;
            seen += 1;
            if (f.params.len != s.params.len) {
                std.debug.print("{s}: {s} takes {d} parameters, the table holds {d}\n", .{ s.symbol, s.symbol, f.params.len, s.params.len });
                mismatches += 1;
                continue;
            }
            for (f.params, 0..) |kind, p| {
                if (kind == null or kind.? != s.params[p]) {
                    std.debug.print("{s}: parameter {d} is {any}, the table holds {any}\n", .{ s.symbol, p, kind, s.params[p] });
                    mismatches += 1;
                }
            }
        }
        if (seen == 0) {
            std.debug.print("{s}: no .cu source declares it\n", .{s.symbol});
            mismatches += 1;
        }
        checked += 1;
    }
    try std.testing.expect(checked > 0 and found.items.len > 0);
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "every q35 attention launch passes its kernel's argument count" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var files: std.ArrayList([:0]const u8) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(std.testing.io, "zig/src/families/qwen3_5", .{ .iterate = true });
    defer dir.close(std.testing.io);
    try sources(a, dir, "zig/src/families/qwen3_5", ".zig", &files);
    var sites: std.ArrayList(Site) = .empty;
    for (files.items) |text| try sitesIn(a, "qwen3_5", text, &sites);
    var checked: usize = 0;
    var mismatches: usize = 0;
    for (sites.items) |site| {
        var known: usize = 0;
        for (site.handles) |handle| {
            if (sig.byHandle(handle) != null) known += 1;
        }
        if (known == 0) continue;
        const named = site.named orelse {
            std.debug.print("{s}: a launch names no signature\n", .{site.path});
            mismatches += 1;
            continue;
        };
        const list = sig.byName(named) orelse {
            std.debug.print("{s}: a launch names the unknown signature {s}\n", .{ site.path, named });
            mismatches += 1;
            continue;
        };
        for (site.handles) |handle| {
            const other = sig.byHandle(handle) orelse {
                std.debug.print("{s}: the handle {s} is not in the table\n", .{ site.path, handle });
                mismatches += 1;
                continue;
            };
            if (other.params.len != list.params.len) {
                std.debug.print("{s}: {s} takes {d} parameters, {s} takes {d}\n", .{ site.path, other.symbol, other.params.len, list.symbol, list.params.len });
                mismatches += 1;
            }
        }
        if (site.elements.len != list.params.len) {
            std.debug.print("{s}: the launch of {s} passes {d} arguments, the kernel takes {d}\n", .{ site.path, list.symbol, site.elements.len, list.params.len });
            mismatches += 1;
            continue;
        }
        for (site.elements, 0..) |element, p| {
            const kind = argKind(element);
            if (kind != list.params[p]) {
                std.debug.print("{s}: the launch of {s} passes {s} as argument {d}, the kernel takes {any}\n", .{ site.path, list.symbol, element, p, list.params[p] });
                mismatches += 1;
            }
        }
        checked += 1;
    }
    if (checked < 9) std.debug.print("checked {d} q35 attention launches of {d} sites\n", .{ checked, sites.items.len });
    try std.testing.expect(checked >= 9);
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "the launch scanner counts a tuple's elements and finds a local one" {
    const source =
        \\const args = .{ geo, e.qn, t_segs, t_params, ai, e.pm };
        \\const rows = .{ geo, e.qn, t_segs, t_params, ai, @as(c_int, @intCast(total)), e.pm, e.pl };
        \\try e.run(e.fns.attn_merge, .{ 1, 2, 1 }, .{ 32, 1, 1 }, args);
    ;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var sites: std.ArrayList(Site) = .empty;
    try sitesIn(a, "inline", try a.dupeSentinel(u8, source, 0), &sites);
    try std.testing.expectEqual(@as(usize, 1), sites.items.len);
    try std.testing.expectEqual(@as(usize, 6), sites.items[0].elements.len);
    try std.testing.expectEqualStrings("e.fns.attn_merge", sites.items[0].handles[0]);
}

test "the launch scanner follows local choices and direct engine handles" {
    const source =
        \\const selected = if (small) e.mt2.paged else e.fp8.attn_rows;
        \\const args = .{ geo, e.qn };
        \\try launch(e, selected, sig.attn_rows8, .{1,1,1}, .{128,1,1}, args);
    ;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var sites: std.ArrayList(Site) = .empty;
    try sitesIn(arena.allocator(), "inline", source, &sites);
    try std.testing.expectEqual(@as(usize, 1), sites.items.len);
    try std.testing.expectEqual(@as(usize, 2), sites.items[0].handles.len);
    try std.testing.expectEqualStrings("attn_rows8", sites.items[0].named.?);
}

test "Qwen3.5 prompt tiles predicate padded rows against the actual forward total" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const common = try std.Io.Dir.cwd().readFileAlloc(io, "zig/kernels/cuda/qwen35.cuh", a, .limited(1 << 20));
    defer a.free(common);
    const qwen = try std.Io.Dir.cwd().readFileAlloc(io, "zig/kernels/cuda/qwen35.cu", a, .limited(8 << 20));
    defer a.free(qwen);
    const int8 = try std.Io.Dir.cwd().readFileAlloc(io, "zig/kernels/cuda/w4a8_prefill.cu", a, .limited(4 << 20));
    defer a.free(int8);

    // Qwen3.5 row kernels return when their block's row reaches P[2] (the forward's total).
    try std.testing.expect(std.mem.indexOf(u8, common, "return P != nullptr && m >= P[2];") != null);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "if (past_rows(P, blockIdx.x)) return;") != null);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "if (past_rows(P, w)) return;") != null);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "if (past_rows(P, blockIdx.z)) return;") != null);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "if (past_rows(P, blockIdx.x / heads)) return;") != null);
    try std.testing.expect(std.mem.indexOf(u8, qwen, "if (past_rows(P, blockIdx.y)) return;") != null);

    // W4A8 rounds its row tiles up, but masks loads and stores against M; padded rows never write out.
    try std.testing.expect(std.mem.indexOf(u8, int8, "m0 + r < M") != null);
    try std.testing.expect(std.mem.indexOf(u8, int8, "if (row >= M || col >= N) continue;") != null);
}

const c_env = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

/// A recording driver's generic PDL guard: an attributed symbol must occur in the scanned positive table.
const PdlGuard = struct {
    var table: []const []const u8 = &.{};
    var attributed: usize = 0;
    var rejected: usize = 0;

    fn reset() void {
        attributed = 0;
        rejected = 0;
    }

    fn symbol(f: cuda.abi.Function) []const u8 {
        return switch (@intFromPtr(f orelse return "tf_q35_unknown")) {
            0x40 => "ld_kernel",
            else => "tf_q35_unknown",
        };
    }

    fn note(f: cuda.abi.Function, dependent: bool) void {
        if (!dependent) return;
        attributed += 1;
        if (!kernel.contains(table, symbol(f))) rejected += 1;
    }

    fn run(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        note(f, false);
        return cuda.abi.success;
    }

    fn ex(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| if (at.id == .programmatic_stream_serialization and at.value.int == 1) {
            note(f, true);
            return cuda.abi.success;
        };
        note(f, false);
        return cuda.abi.success;
    }
};

fn qwenPrologues(a: std.mem.Allocator) ![][]const u8 {
    var table: std.ArrayList([]const u8) = .empty;
    for ([_][]const u8{ "zig/kernels/cuda/qwen35.cu", "zig/kernels/cuda/qwen35_kv8.cu" }) |path| {
        const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(8 << 20));
        try kernel.prologues(a, src, "qwen_pdl::enter();", &table);
    }
    return table.items;
}

test "PDL launch guard: EXL3 x3ld and an unknown-symbol control" {
    try std.testing.expectEqual(@as(c_int, 0), c_env.setenv("TF_QWEN_PDL", "1", 1));
    defer _ = c_env.unsetenv("TF_QWEN_PDL");
    try std.testing.expect(qwen35.cuda_pdl.on());
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    PdlGuard.table = try qwenPrologues(arena.allocator());

    var d: cuda.Driver = .{ .lib = undefined, .api = undefined, .func_name = null };
    d.api.cuLaunchKernel = PdlGuard.run;
    d.api.cuLaunchKernelEx = PdlGuard.ex;
    var e: qwen35.Engine = undefined;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };

    const unknown: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(0x30) };
    e.kv8 = false;
    PdlGuard.reset();
    try e.runPdl(unknown, .{ 1, 1, 1 }, .{ 1, 1, 1 }, .{});
    try std.testing.expectEqual(@as(usize, 1), PdlGuard.attributed);
    try std.testing.expectEqual(@as(usize, 1), PdlGuard.rejected);

    const x3ld_src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/cuda/exl3_block/x3ld.cu", arena.allocator(), .limited(2 << 20));
    var x3ld_table: std.ArrayList([]const u8) = .empty;
    try kernel.prologues(arena.allocator(), x3ld_src, "pdl_wait();", &x3ld_table);
    PdlGuard.table = x3ld_table.items;
    const exl3 = cuda.exl3.block.experts;
    var functions: exl3.Functions = undefined;
    const ld: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(0x40) };
    functions.ld = @splat(@splat(@splat(ld)));
    const ops: exl3.Ops = .{ .f = &functions, .s = e.stream, .sms = 1 };
    const grouped: exl3.Grouped = .{
        .x0 = 0,
        .x1 = 0,
        .tp0 = 0,
        .tp1 = 0,
        .k2_0 = 0,
        .k2_1 = 0,
        .uids = 0,
        .ucount = 0,
        .members = 0,
        .z = 0,
        .mats = 1,
        .K = 64,
        .N = 128,
        .P = 1,
        .SK = 1,
        .slots = 1,
        .maxm = 16,
        .nexp = 1,
        .lo = 8,
        .hi = 8,
    };
    PdlGuard.reset();
    try ops.x3ld(grouped, 8, 1, 0, true);
    try std.testing.expectEqual(@as(usize, 1), PdlGuard.attributed);
    try std.testing.expectEqual(@as(usize, 0), PdlGuard.rejected);
}

/// The one conditional escape: runPdl itself identifies rope8 by engine mode and function identity.
fn plainRope8(src: []const u8) bool {
    const identity = "unsafe_rope8 = e.kv8 and f.d == e.fp8.rope.d and f.handle == e.fp8.rope.handle";
    return std.mem.indexOf(u8, src, identity) != null and std.mem.indexOf(u8, src, "on() and !unsafe_rope8") != null;
}

fn launchSymbol(a: std.mem.Allocator, handle: []const u8) !?[]const u8 {
    if (sig.byHandle(handle)) |s| return s.symbol;
    const prefix = "e.fns.";
    if (!std.mem.startsWith(u8, handle, prefix)) return null;
    return try std.fmt.allocPrint(a, "tf_q35_{s}", .{handle[prefix.len..]});
}

test "every static Qwen dependent launch has a scanned wait, except the explicitly plain kv8 rope" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const table = try qwenPrologues(a);
    const engine = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/src/families/qwen3_5/cuda_engine.zig", a, .limited(2 << 20));
    const rope8_plain = plainRope8(engine);
    var files: std.ArrayList([:0]const u8) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(std.testing.io, "zig/src/families/qwen3_5", .{ .iterate = true });
    defer dir.close(std.testing.io);
    try sources(a, dir, "zig/src/families/qwen3_5", ".zig", &files);
    var sites: std.ArrayList(Site) = .empty;
    for (files.items) |text| try sitesIn(a, "qwen3_5", text, &sites);
    var checked: usize = 0;
    var missing: usize = 0;
    for (sites.items) |site| if (site.dependent) for (site.handles) |handle| {
        const symbol = (try launchSymbol(a, handle)) orelse continue;
        checked += 1;
        if (kernel.contains(table, symbol)) continue;
        if (std.mem.eql(u8, symbol, sig.rope8.symbol) and rope8_plain) continue;
        std.debug.print("{s}: dependent launch of {s} has no scanned qwen_pdl::enter prologue\n", .{ site.path, symbol });
        missing += 1;
    };
    try std.testing.expect(checked >= 9);
    try std.testing.expectEqual(@as(usize, 0), missing);
}
