//! EXL3 metadata dialect admission; tensor shapes, scale storage and marker bytes remain loader responsibilities.
const std = @import("std");

/// Legacy 1.x metadata keeps its existing admission; dsq-1 declares its codebook and output scale policy.
pub fn require(quant: std.json.Value) !void {
    if (quant != .object) return error.BadConfig;
    const method = quant.object.get("quant_method") orelse return error.MissingField;
    const version = quant.object.get("version") orelse return error.MissingField;
    if (method != .string or !std.mem.eql(u8, method.string, "exl3") or version != .string)
        return error.UnsupportedCheckpoint;
    if (std.mem.eql(u8, version.string, "1") or std.mem.startsWith(u8, version.string, "1.")) return;
    if (!dsq1(version.string)) return error.UnsupportedCheckpoint;
    const codebook = quant.object.get("codebook") orelse return error.MissingField;
    const scales = quant.object.get("out_scales") orelse return error.MissingField;
    if (codebook != .string or scales != .string) return error.UnsupportedCheckpoint;
    const supported = std.mem.eql(u8, codebook.string, "inst3") or std.mem.eql(u8, codebook.string, "mcg") or
        std.mem.eql(u8, codebook.string, "mul1");
    if (!supported or !std.mem.eql(u8, scales.string, "always")) return error.UnsupportedCheckpoint;
}

/// A format identifier, optionally followed by a parenthesized producer annotation; never a producer allowlist.
fn dsq1(version: []const u8) bool {
    if (std.mem.eql(u8, version, "dsq-1")) return true;
    const prefix = "dsq-1 (";
    if (!std.mem.startsWith(u8, version, prefix) or !std.mem.endsWith(u8, version, ")")) return false;
    const annotation = version[prefix.len .. version.len - 1];
    if (std.mem.trim(u8, annotation, " ").len == 0) return false;
    for (annotation) |ch| if (ch < 32 or ch > 126 or ch == '(' or ch == ')') return false;
    return true;
}
