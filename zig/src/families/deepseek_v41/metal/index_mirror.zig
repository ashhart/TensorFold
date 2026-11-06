//! The shared Zig/Metal argument structs are HAND-WRITTEN mirrors. This module reads the embedded kernel
//! source at COMPTIME and compares each Metal struct's field list, in order, to the Zig struct's own:
//! a field added, removed, renamed, reordered or retyped on either side fails the build.
//!
//! ROUND-468's bug class: `Params` gained `layer`/`skip_mask` (ROUND-411) and the Metal struct did not, so
//! the kernel's `scale` read `layer`'s bits -- a denormal -- and every index score came out 0.0 with no
//! error anywhere. The 76/80-byte assertion catches a size change but not a reorder or a same-size swap.
//! ROUND-473 added the field-list check for that one pair; ROUND-478 wired ten more; ROUND-480 points it at
//! every shared pair in the family and models the C layout rather than a flat byte sum.
//!
//! ROUND-480: `enum(u32)` maps to `uint`, `i64` to `long`, `[N]T` to `T name[N]`, `device const uchar*` to an
//! eight-byte address, and each field's offset (C alignment, trailing padding included) must equal
//! `@offsetOf(T, name)`; the total must equal `@sizeOf(T)`. The runtime `static_assert`s in the .metal files
//! say the same thing at boot; `tools/zig/check_dsv41_metal_mirrors.sh` compiles them at build time.

const std = @import("std");

fn roundUp(v: usize, a: usize) usize {
    return (v + a - 1) / a * a;
}

fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

const Part = struct {
    /// The declaration's type as its last whitespace-separated token (qualifiers dropped), or "" when
    /// this comma group only names another declarator of the statement's type.
    ty: []const u8,
    name: []const u8,
    count: u32,
    typed: bool,
};

/// Split one comma group of a Metal declaration: `[qualifiers] type name[N]` or a bare `name[N]`.
fn partOf(comptime text: []const u8) Part {
    var s = std.mem.trim(u8, text, " \t\r\n");
    var count: u32 = 0;
    if (std.mem.lastIndexOfScalar(u8, s, ']')) |rb| {
        if (std.mem.lastIndexOfScalar(u8, s[0..rb], '[')) |lb| {
            count = std.fmt.parseInt(u32, s[lb + 1 .. rb], 10) catch
                @compileError("index mirror: an array bound that is not a decimal literal: `" ++ s ++ "`");
            s = std.mem.trim(u8, s[0..lb], " \t\r\n");
        }
    }
    var last: usize = s.len;
    while (last > 0 and !isSpace(s[last - 1])) last -= 1;
    const name = s[last..];
    if (name.len == 0) @compileError("index mirror: a declaration has no name: `" ++ text ++ "`");
    var prefix = std.mem.trim(u8, s[0..last], " \t\r\n");
    if (prefix.len == 0) return .{ .ty = "", .name = name, .count = count, .typed = false };
    // The base type is the declaration's last token: `device const uchar*` -> `uchar*`.
    var first: usize = prefix.len;
    while (first > 0 and !isSpace(prefix[first - 1])) first -= 1;
    prefix = prefix[first..];
    return .{ .ty = prefix, .name = name, .count = count, .typed = true };
}

/// The Metal spelling a Zig field must have, or a compile error naming the field.
fn zigSpelling(comptime ft: type, comptime owner: []const u8, comptime fname: []const u8) []const u8 {
    return switch (@typeInfo(ft)) {
        .array => |a| zigSpelling(a.child, owner, fname),
        .@"enum" => |e| if (e.tag_type == u32) "uint" else @compileError("index mirror: `" ++ owner ++ "." ++
            fname ++ "` is an enum whose tag is not u32; the kernel can only mirror a uint"),
        .int => |i| switch (i.signedness) {
            .unsigned => if (i.bits == 32) "uint" else if (i.bits == 64) "ulong" else if (i.bits == 16) "ushort" else if (i.bits == 8) "uchar" else @compileError("index mirror: `" ++ owner ++ "." ++ fname ++ "` is an integer the kernel cannot mirror"),
            .signed => if (i.bits == 32) "int" else if (i.bits == 64) "long" else @compileError("index mirror: `" ++ owner ++ "." ++ fname ++ "` is a signed integer the kernel cannot mirror"),
        },
        .float => |f| if (f.bits == 32) "float" else if (f.bits == 16) "half" else @compileError("index mirror: `" ++ owner ++ "." ++ fname ++ "` is a float the kernel cannot mirror"),
        else => @compileError("index mirror: `" ++ owner ++ "." ++ fname ++
            "` is a type the mirror check does not know; add it before adding the field"),
    };
}

fn zigCount(comptime ft: type) u32 {
    return switch (@typeInfo(ft)) {
        .array => |a| @intCast(a.len),
        else => 0,
    };
}

/// The Metal struct `name` in `src` must declare `T`'s fields, in order, with matching C types, array
/// lengths and C-layout offsets. Called from a `comptime` block inside `T`'s own body.
pub fn check(comptime T: type, comptime src: []const u8, comptime name: []const u8) void {
    @setEvalBranchQuota(1_000_000);
    const head = std.fmt.comptimePrint("struct {s} {{", .{name});
    const start = std.mem.indexOf(u8, src, head) orelse
        @compileError("index mirror: `" ++ head ++ "` not found in the kernel source");
    const rest = src[start + head.len ..];
    const end = std.mem.indexOf(u8, rest, "};") orelse
        @compileError("index mirror: `struct " ++ name ++ "` is not closed");
    const body = rest[0..end];

    var names: [64][]const u8 = undefined;
    var types: [64][]const u8 = undefined;
    var counts: [64]u32 = undefined;
    var n: usize = 0;

    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |raw| {
        // A declaration never spans a line here; drop a trailing comment, then walk `;`-statements whose
        // comma-separated declarators share the statement's leading type.
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        var stmts = std.mem.splitScalar(u8, line, ';');
        while (stmts.next()) |stmt| {
            var ty: []const u8 = "";
            var parts = std.mem.splitScalar(u8, stmt, ',');
            while (parts.next()) |raw_part| {
                const text = std.mem.trim(u8, raw_part, " \t\r");
                if (text.len == 0) continue;
                const p = partOf(text);
                if (p.typed) ty = p.ty;
                if (ty.len == 0) @compileError("index mirror: a declarator has no type: `" ++ text ++ "`");
                if (n == names.len) @compileError("index mirror: more than 64 fields in `" ++ @typeName(T) ++ "`");
                names[n] = p.name;
                types[n] = ty;
                counts[n] = p.count;
                n += 1;
            }
        }
    }

    const info = @typeInfo(T).@"struct";
    const field_names = info.field_names;
    const field_types = info.field_types;

    if (n != field_names.len) @compileError(std.fmt.comptimePrint(
        "index mirror: `{s}` declares {d} fields, `{s}` has {d} -- the kernel and the Zig struct are out of step",
        .{ name, n, @typeName(T), field_names.len }));

    var off: usize = 0;
    var max_align: usize = 1;
    inline for (0..field_names.len) |i| {
        const fname = field_names[i];
        const ft = field_types[i];
        if (!std.mem.eql(u8, names[i], fname)) @compileError(std.fmt.comptimePrint(
            "index mirror: field {d} is `{s}` in the kernel and `{s}` in `{s}`",
            .{ i, names[i], fname, @typeName(T) }));

        const want = zigSpelling(ft, @typeName(T), fname);
        const is_address = types[i].len > 0 and types[i][types[i].len - 1] == '*';
        const type_ok = if (is_address)
            // A device pointer in the kernel is an eight-byte address on the Zig side (written into a
            // device buffer, not passed through setValue).
            std.mem.eql(u8, want, "ulong")
        else
            std.mem.eql(u8, types[i], want);
        if (!type_ok) @compileError(std.fmt.comptimePrint(
            "index mirror: field `{s}.{s}` is `{s}` in the kernel and `{s}` here",
            .{ @typeName(T), fname, types[i], want }));

        if (counts[i] != zigCount(ft)) @compileError(std.fmt.comptimePrint(
            "index mirror: field `{s}.{s}` is `{s}[{d}]` in the kernel and an array of {d} here",
            .{ @typeName(T), fname, types[i], counts[i], zigCount(ft) }));

        const align_of = @alignOf(ft);
        off = roundUp(off, align_of);
        if (off != @offsetOf(T, fname)) @compileError(std.fmt.comptimePrint(
            "index mirror: field `{s}.{s}` sits at offset {d} in the kernel's layout and {d} here",
            .{ @typeName(T), fname, off, @offsetOf(T, fname) }));
        off += @sizeOf(ft);
        if (align_of > max_align) max_align = align_of;
    }
    const total = roundUp(off, max_align);
    if (total != @sizeOf(T)) @compileError(std.fmt.comptimePrint(
        "index mirror: `{s}`'s fields laid out as the kernel declares them take {d} bytes, `{s}` is {d}",
        .{ name, total, @typeName(T), @sizeOf(T) }));
}

test "the mirror check reads the landed struct" {
    // A miniature source with the same shape as index_exl3.metal's: per-line declarations, a comment line,
    // and several fields on one line. The check must accept it against an identical Zig struct.
    const Ok = extern struct { rows: u32, heads: u32, scale: f32 };
    comptime check(Ok, "struct Small {\n    uint rows, heads;\n    // a comment, with a comma\n    float scale;\n};", "Small");
}

test "the mirror check models enums, arrays, 64-bit fields, pointers and padding" {
    const Kind = enum(u32) { window, main };
    const T = extern struct {
        kind: Kind,
        mult: [4]u64,
        addrs: [2]u32,
        big: i64,
        ptr: u64,
        eps: f32,
    };
    comptime check(T,
        \\struct Wide {
        \\    uint kind;
        \\    ulong mult[4];
        \\    uint addrs[2];
        \\    long big;
        \\    device const uchar* ptr;
        \\    float eps;
        \\};
    , "Wide");
    // Trailing padding after a 4-byte tail: 3 x u64, 4 x u32, 1 x f32 = 44 bytes of fields, 48 aligned.
    const Padded = extern struct { a: u64, b: u64, c: u64, d: u32, e: u32, f: u32, g: u32, h: f32 };
    comptime check(Padded,
        \\struct Padded {
        \\    ulong a, b, c;
        \\    uint d, e, f, g;
        \\    float h;
        \\};
    , "Padded");
}

test "the mirror check still reads the landed index struct" {
    const Ok = extern struct { a: u32, b: f32 };
    comptime check(Ok, "struct Small {\n    uint a; float b;\n};", "Small");
}
