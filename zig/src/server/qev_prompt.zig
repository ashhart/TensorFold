//! The qev-labels-v2 decision prompt (qev's prompt.py, as elephant-1 serves it): state block, options, prefix split.
const std = @import("std");
const json = @import("json");
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;

pub const version = "qev-labels-v2";
pub const system = "Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.";
pub const noul_instructions = "Is the statement true, or is the answer to the question yes?";
pub const noul_true = "the statement is true / the answer is yes";
pub const noul_false = "the statement is false / the answer is no";
const mark = "QEV_PREFIX_MARK";

/// Two-letter pairs that are not one token, left out of the labels after Z.
const not_one = [_][]const u8{ "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ" };
pub const max_slots = 256;

/// The answer labels in slot order: A to Z, then two-letter labels without the pairs above.
pub const labels: [max_slots][]const u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [max_slots][]const u8 = undefined;
    var n: usize = 0;
    for ("ABCDEFGHIJKLMNOPQRSTUVWXYZ") |c| {
        out[n] = &[_]u8{c};
        n += 1;
    }
    for ("ABCDEFGHIJKLMNOPQRSTUVWXYZ") |x| for ("ABCDEFGHIJKLMNOPQRSTUVWXYZ") |y| {
        const pair = [_]u8{ x, y };
        var skip = false;
        for (not_one) |bad| if (std.mem.eql(u8, bad, &pair)) {
            skip = true;
        };
        if (!skip and n < max_slots) {
            out[n] = &pair;
            n += 1;
        }
    };
    break :blk out;
};

pub const Kind = enum { noul, choice, score };

/// One option as the model sees it: its name and its description ("" when none), both already rendered.
pub const Option = struct { name: []const u8, text: []const u8 };

/// One question as rendered: `instructions` already stripped and defaulted.
pub const Question = struct { kind: Kind, instructions: []const u8, options: []const Option };

/// Python's ``render_entry``: text stripped, other JSON as ``json.dumps(indent=2, ensure_ascii=False)``, null empty.
pub fn entry(a: Allocator, v: ?json.Value) Allocator.Error![]const u8 {
    const x = v orelse return "";
    return switch (x) {
        .null => "",
        .string => |s| reply_text.pyStrip(s),
        else => try dumps(a, x),
    };
}

/// ``json.dumps(v, ensure_ascii=False, indent=2)``.
pub fn dumps(a: Allocator, v: json.Value) Allocator.Error![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    @import("systemone.zig").indented(&out.writer, v, 0) catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

/// ``render_state``: a string as it is, other JSON indented, null empty.
pub fn state(a: Allocator, v: json.Value) Allocator.Error![]const u8 {
    return switch (v) {
        .null => "",
        .string => |s| s,
        else => try dumps(a, v),
    };
}

/// The question's suffix: header and instructions, the lettered options, the answer line.
pub fn suffix(a: Allocator, q: Question) Allocator.Error![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    write(w, q) catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

fn write(w: *std.Io.Writer, q: Question) std.Io.Writer.Error!void {
    try w.print("{s} {s}\n\nOptions:\n", .{ if (q.kind == .score) "Rate the state:" else "Question:", q.instructions });
    for (q.options, 0..) |o, i| {
        if (i > 0) try w.writeByte('\n');
        if (o.text.len > 0) try w.print("{s}. {s}: {s}", .{ labels[i], o.name, o.text }) else try w.print("{s}. {s}", .{ labels[i], o.name });
    }
    try w.writeAll(if (q.options.len <= 26) "\n\nAnswer with one letter: " else "\n\nAnswer with one label: ");
    for (0..q.options.len) |i| try w.print("{s}{s}", .{ if (i > 0) ", " else "", labels[i] });
    try w.writeByte('.');
}

/// The chat template's text for the system message and `<state>` block plus `tail`, thinking off.
pub fn render(a: Allocator, text: model_text.Text, state_text: []const u8, tail: []const u8, problem: *[]const u8) model_text.Error![]const u8 {
    const user = try std.mem.concat(a, u8, &.{ "<state>\n", state_text, "\n</state>\n\n", tail });
    const sys = try json.newObject(a);
    try sys.put(a, "role", .{ .string = "system" });
    try sys.put(a, "content", .{ .string = system });
    const usr = try json.newObject(a);
    try usr.put(a, "role", .{ .string = "user" });
    try usr.put(a, "content", .{ .string = user });
    const messages = try a.dupe(json.Value, &.{ .{ .object = sys }, .{ .object = usr } });
    const rendered = try text.render(a, .{ .array = messages }, .{ .add_generation_prompt = true, .enable_thinking = false }, problem);
    return rendered.text;
}

pub const Split = struct { prefix: []const u32, suffixes: []const []const u32 };

/// Every question's full prompt ids cut at the longest shared prefix the marker render allows (qev's prompt.py).
pub fn split(a: Allocator, text: model_text.Text, state_text: []const u8, suffixes: []const []const u8, problem: *[]const u8) model_text.Error!Split {
    const full = try a.alloc([]const u32, suffixes.len);
    for (suffixes, full) |s, *ids| ids.* = try text.encode(a, try render(a, text, state_text, s, problem), false);
    const marked = try render(a, text, state_text, mark, problem);
    const cut = std.mem.indexOf(u8, marked, mark) orelse {
        problem.* = "the chat template dropped the decision prompt's prefix marker";
        return error.Template;
    };
    const prefix = try text.encode(a, marked[0..cut], false);
    var n = prefix.len;
    while (n > 0) : (n -= 1) {
        var shared = true;
        for (full) |ids| if (ids.len < n or !std.mem.eql(u32, ids[0..n], prefix[0..n])) {
            shared = false;
        };
        if (shared) break;
    }
    const tails = try a.alloc([]const u32, full.len);
    for (full, tails) |ids, *t| t.* = ids[n..];
    return .{ .prefix = prefix[0..n], .suffixes = tails };
}

/// Slot `i`'s token: its label when that is one token that decodes back to the label, else null.
pub fn slotToken(a: Allocator, text: model_text.Text, i: usize) model_text.Error!?u32 {
    const ids = try text.encode(a, labels[i], false);
    if (ids.len != 1) return null;
    const back = try text.decode(a, ids);
    return if (std.mem.eql(u8, back, labels[i])) ids[0] else null;
}

/// Python's ``round(x, 6)``: the correctly rounded 6-decimal value, back as a float.
pub fn round6(x: f64) f64 {
    if (!std.math.isFinite(x) or x == 0) return x;
    if (x < 0) return -round6(-x);
    if (x >= 1e9) return x; // every such value is already whole millionths
    // x = m * 2^e exactly; x * 10^6 = m * 10^6 * 2^e, with m * 10^6 below 2^73
    const bits: u64 = @bitCast(x);
    const exp: i32 = @intCast((bits >> 52) & 0x7ff);
    const m: u128 = if (exp == 0) bits & 0xfffffffffffff else (bits & 0xfffffffffffff) | (1 << 52);
    const e: i32 = (if (exp == 0) 1 else exp) - 1075;
    const scaled = m * 1_000_000;
    var n: u128 = undefined;
    if (e >= 0) n = scaled << @intCast(e) else if (-e >= 127) n = 0 else {
        const shift: u7 = @intCast(-e);
        n = scaled >> shift;
        const rem = scaled & ((@as(u128, 1) << shift) - 1);
        const half = @as(u128, 1) << (shift - 1);
        if (rem > half or (rem == half and n & 1 == 1)) n += 1;
    }
    return @as(f64, @floatFromInt(@as(u64, @intCast(n)))) / 1e6;
}

test "qev: labels run A to Z, then pairs without the 25 that are not one token, 256 in all" {
    try std.testing.expectEqualStrings("A", labels[0]);
    try std.testing.expectEqualStrings("Z", labels[25]);
    try std.testing.expectEqualStrings("AA", labels[26]);
    try std.testing.expectEqualStrings("BP", labels[26 + 26 + 15]);
    try std.testing.expectEqualStrings("BR", labels[26 + 26 + 16]); // BQ is skipped
    for (labels) |l| for (not_one) |bad| try std.testing.expect(!std.mem.eql(u8, l, bad));
    // the reference's last label: 230 pairs after Z, the 256th slot
    try std.testing.expectEqualStrings("JU", labels[255]);
}

test "qev: a predicate's suffix names yes and no with their default descriptions" {
    const q = Question{ .kind = .noul, .instructions = "the text is about cats", .options = &.{ .{ .name = "yes", .text = noul_true }, .{ .name = "no", .text = noul_false } } };
    const got = try suffix(std.testing.allocator, q);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("Question: the text is about cats\n\nOptions:\nA. yes: the statement is true / the answer is yes\nB. no: the statement is false / the answer is no\n\nAnswer with one letter: A, B.", got);
}

test "qev: a choice without descriptions shows names alone, a score rates the state" {
    const a = std.testing.allocator;
    const c = try suffix(a, .{ .kind = .choice, .instructions = "Which skill?", .options = &.{ .{ .name = "none", .text = "" }, .{ .name = "git", .text = "" }, .{ .name = "zig", .text = "" } } });
    defer a.free(c);
    try std.testing.expectEqualStrings("Question: Which skill?\n\nOptions:\nA. none\nB. git\nC. zig\n\nAnswer with one letter: A, B, C.", c);
    const s = try suffix(a, .{ .kind = .score, .instructions = "How bad?", .options = &.{ .{ .name = "level 0", .text = "fine" }, .{ .name = "level 1", .text = "bad" } } });
    defer a.free(s);
    try std.testing.expectEqualStrings("Rate the state: How bad?\n\nOptions:\nA. level 0: fine\nB. level 1: bad\n\nAnswer with one letter: A, B.", s);
}

test "qev: round6 rounds the binary value as Python's round(x, 6) does" {
    try std.testing.expectEqual(@as(f64, 0.123456), round6(0.1234565));
    try std.testing.expectEqual(@as(f64, 0.378798), round6(0.3787975));
    try std.testing.expectEqual(@as(f64, 0.0), round6(5e-07));
    try std.testing.expectEqual(@as(f64, 2e-06), round6(1.5e-06));
    try std.testing.expectEqual(@as(f64, 1.0), round6(0.999999501));
    try std.testing.expectEqual(@as(f64, 0.333333), round6(1.0 / 3.0));
    try std.testing.expectEqual(@as(f64, 0.451688), round6(0.45168849999));
    try std.testing.expectEqual(@as(f64, 0.000123), round6(0.0001234567));
}
