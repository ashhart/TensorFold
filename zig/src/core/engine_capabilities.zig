//! Public per-model flags use the same reader as request admission, without exposing callback payloads.
const std = @import("std");

pub const Capability = enum {
    text_generation,
    context_fitted,
    call_gates,
    loop_guard,
    thinking_budget,
    reasoning,
    answer_limit,
    structures,
    logprobs,
    penalties,
    repetition_penalty,
    prompt_logprobs,
    prompt_cache,
    warm_turns,
    plain_only,
    decides,
    pictures,
    greedy_only,
    slide_conversations,
    vision,
    speech,
    embeddings,
    readouts,
    rerank,
    thinking_switch,
    thinking_applicable,
};

pub inline fn read(info: anytype, comptime capability: Capability) bool {
    const value = @field(info, @tagName(capability));
    return if (@TypeOf(value) == bool) value else value != null;
}

test "engine capability reader: every Info boolean is published from its actual value" {
    const Info = @import("engine_api.zig").Info;
    inline for (@typeInfo(Info).@"struct".field_names) |name| if (@FieldType(Info, name) == bool) {
        const key = @field(Capability, name);
        var info: Info = .{};
        @field(info, name) = false;
        try std.testing.expect(!info.capability(key));
        @field(info, name) = true;
        try std.testing.expect(info.capability(key));
    };
}

test "engine capability reader: optional user features expose presence without reading their payload" {
    const Info = @import("engine_api.zig").Info;
    inline for (.{ Capability.vision, Capability.speech, Capability.embeddings, Capability.readouts, Capability.rerank }) |key| {
        var info: Info = .{};
        try std.testing.expect(!info.capability(key));
        const Payload = @typeInfo(@FieldType(Info, @tagName(key))).optional.child;
        @field(info, @tagName(key)) = @as(Payload, undefined);
        try std.testing.expect(info.capability(key));
    }
}
