//! FP64 scalar comparisons at the captured FP32 inputs, with every arithmetic identity checked.
const std = @import("std");
const f = @import("tensorfold").flashnext_vision_contract;
const accuracy = f.rotary_accuracy;
pub const Sample = extern struct { frequency: f32, angle: f32, cosine: f32, sine: f32, co: f32, si: f32 };

pub fn observe(row: *accuracy.Row, linear: u64, sample: Sample, table: []const f32) !void {
    @setFloatMode(.strict);
    if (linear != row.cases) return error.NoncontiguousRotarySweep;
    const point = accuracy.domain.point(row.role, linear);
    const i = point.index;
    const bits: u32 = @bitCast(sample.frequency);
    if (linear < 32) row.frequency_bits[i] = bits else if (row.frequency_bits[i] != bits) return error.FrequencyChangedWithPosition;
    const exponent: f32 = -(@as(f32, @floatFromInt(i)) / 32) * table[0];
    const freq64 = if (row.mode == .yarn) @as(f64, table[3 + i]) else @exp2(@as(f64, exponent));
    if (row.mode == .normal) {
        try row.exp2.observe(linear, sample.frequency, freq64);
    }
    if (row.mode == .yarn and bits != @as(u32, @bitCast(table[3 + i]))) return error.RotaryTableReadChanged;
    const position: f32 = @floatFromInt(point.position);
    const predicted: f32 = position * sample.frequency;
    if (@as(u32, @bitCast(predicted)) != @as(u32, @bitCast(sample.angle))) return error.UnrecordedPhaseRounding;
    try row.phase.observe(linear, sample.angle, @as(f64, position) * sample.frequency);
    try row.cosine.observe(linear, sample.cosine, @cos(@as(f64, sample.angle)));
    try row.sine.observe(linear, sample.sine, @sin(@as(f64, sample.angle)));
    const scale: f32 = if (row.mode == .yarn) table[2] else 1;
    if (@as(u32, @bitCast(sample.cosine * scale)) != @as(u32, @bitCast(sample.co)) or
        @as(u32, @bitCast(sample.sine * scale)) != @as(u32, @bitCast(sample.si))) return error.UnrecordedCoefficientRounding;
    const ideal_frequency = if (row.mode == .yarn) @as(f64, table[3 + i]) else @exp2(-(@as(f64, @floatFromInt(i)) / 32) * table[0]);
    const ideal_phase = @as(f64, position) * ideal_frequency;
    try row.coefficient_cosine.observe(linear, sample.co, @cos(ideal_phase) * scale);
    try row.coefficient_sine.observe(linear, sample.si, @sin(ideal_phase) * scale);
    if (point.position == 0 and (sample.cosine != 1 or sample.sine != 0)) return error.CannotIsolateNormalization;
    if (row.mode == .normal) row.exp2_cases += 1;
    row.cases += 1;
    row.axis_cases[point.axis] += 1;
}
