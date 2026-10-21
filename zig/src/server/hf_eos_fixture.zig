//! Small real tokenizer files for the HfText and HTTP EOS regressions; no model or socket is opened.
const std = @import("std");

pub const Fixture = struct {
    tmp: std.testing.TmpDir,
    path: [:0]u8,
    a: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, io: std.Io, generation_text: ?[]const u8) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
            \\{"model":{"type":"BPE","vocab":{"H":0,"O":1,"K":2,"<|endoftext|>":127901,"<|im_end|>":127906},"merges":[]},"decoder":{"type":"Fuse"},
            \\ "added_tokens":[{"id":127901,"content":"<|endoftext|>","special":true},
            \\ {"id":127906,"content":"<|im_end|>","special":true}]}
        });
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"eos_token_id\":127906,\"pad_token_id\":127901,\"vocab_size\":128000}" });
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"eos_token\":\"<|im_end|>\",\"pad_token\":\"<|endoftext|>\"}" });
        if (generation_text) |text| try tmp.dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = text });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(io, ".", a), .a = a };
    }

    pub fn deinit(f: *Fixture) void {
        f.a.free(f.path);
        f.tmp.cleanup();
    }
};

pub const generation = "{\"eos_token_id\":[127906,127901],\"pad_token_id\":127901}";
