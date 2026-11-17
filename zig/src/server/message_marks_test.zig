//! Message marks in the template render (#3617): captured output carries them or none, keyword macro args bind.
const std = @import("std");
const template = @import("template");
const t = std.testing;

const two =
    \\[{"role":"user","content":"A"},{"role":"assistant","content":"B"}]
;

/// The render's text and each message's byte start (sorted), with the plain render's text and spans checked equal.
fn marked(a: std.mem.Allocator, source: []const u8) !struct { text: []const u8, starts: []const usize } {
    var compiled = try template.compile(a, source, null);
    defer compiled.deinit();
    const messages = (try std.json.parseFromSliceLeaky(std.json.Value, a, two, .{}));
    var starts: []usize = &.{};
    const got = try template.renderMarked(a, &compiled, messages, .null, .null, .null, false, null, &starts);
    const plain = try template.renderProvenanceGlobals(a, &compiled, messages, .null, .null, .null, false, null);
    try t.expectEqualStrings(plain.s, got.s);
    try t.expectEqualSlices([2]usize, plain.spans, got.spans);
    return .{ .text = got.s, .starts = starts };
}

fn expectMarks(source: []const u8, text: []const u8, starts: []const usize) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const r = try marked(arena.allocator(), source);
    try t.expectEqualStrings(text, r.text);
    try t.expectEqualSlices(usize, starts, r.starts);
}

const capture_loop = "{% set x %}{% for m in messages %}{{ m.content }}{% endfor %}{% endset %}";

test "message marks: a captured loop's marks follow its text to where it is written (A07)" {
    try expectMarks(capture_loop ++ "Z{{ x }}", "ZAB", &.{ 1, 2 });
    try expectMarks(capture_loop ++ "{{ 'Z' ~ x }}", "ZAB", &.{ 1, 2 });
    try expectMarks(capture_loop ++ "{{ 'Z' + x + 'W' }}", "ZABW", &.{ 1, 2 });
    try expectMarks(capture_loop ++ "{{ x }}|{{ x }}", "AB|AB", &.{ 0, 1 });
    try expectMarks("{% macro all() %}{% for m in messages %}[{{ m.content }}]{% endfor %}{% endmacro %}<{{ all() }}>", "<[A][B]>", &.{ 1, 4 });
}

test "message marks: text changed after its capture leaves its messages unmarked (A07, fail closed)" {
    try expectMarks(capture_loop ++ "{{ x | lower }}", "ab", &.{});
    try expectMarks(capture_loop ++ "{{ x[1:] }}", "B", &.{});
    try expectMarks(capture_loop ++ "Z", "Z", &.{});
}

test "message marks: a binding inside a discarded capture marks nothing (A08)" {
    try expectMarks("{% for i in range(2) %}{% set y %}{% set m = messages[i] %}{{ m.content }}{% endset %}X{% endfor %}", "XX", &.{});
    try expectMarks("{% macro see(m) %}{{ m.content }}{% endmacro %}{% for i in range(2) %}{% set y %}{{ see(messages[i]) }}{% endset %}X{% endfor %}", "XX", &.{});
    // the same binding made by the iteration itself still marks it
    try expectMarks("{% for i in range(2) %}{% set m = messages[i] %}<{{ m.content }}>{% endfor %}", "<A><B>", &.{ 0, 3 });
}

test "message marks: keyword macro arguments bind as positional ones do (A09)" {
    const emit = "{% macro emit(m, tail='') %}{{ m.content }}{{ tail }}{% endmacro %}";
    try expectMarks(emit ++ "{% for i in range(2) %}{{ emit(m=messages[i]) }}{% endfor %}", "AB", &.{ 0, 1 });
    try expectMarks(emit ++ "{% for i in range(2) %}{{ emit(tail='.', m=messages[i]) }}{% endfor %}", "A.B.", &.{ 0, 2 });
    try expectMarks(emit ++ "{% for i in range(2) %}{{ emit(messages[i]) }}{% endfor %}", "AB", &.{ 0, 1 });
}

test "message marks: a scan that writes nothing marks nothing, the loop that writes marks" {
    try expectMarks("{% for m in messages %}{% endfor %}{% for m in messages %}-{{ m.content }}{% endfor %}", "-A-B", &.{ 0, 2 });
}
