//! MOSS-Image in the CUDA server: recognition, a picture request's limits and the text flags the host refuses.
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

/// The diffusers class a checkpoint's model_index.json must name; no other checkpoint opens as a picture engine.
pub const class_name = "MOSSImagePipeline";
/// A side, the pixels, the steps and the prompt's tokens a request may ask for: the route's own limits, unchanged.
pub const max_side: u16 = 1536;
pub const max_pixels: u32 = 1_572_864;
pub const max_steps: u32 = 200;
/// The prompt encoder's 4,096 rows less the chat template's own tokens around a prompt, as the pipeline cuts it.
pub const max_prompt_tokens: u32 = 4096 - 32;

/// The limits the host publishes in its `Info.image`.
pub fn limits() api.ImageInfo {
    return .{
        .max_side = max_side,
        .max_pixels = max_pixels,
        .max_steps = max_steps,
        .max_prompt_tokens = max_prompt_tokens,
    };
}

/// Whether `dir` holds this checkpoint: its model_index.json names the publisher's pipeline class.
pub fn recognize(gpa: Allocator, io: std.Io, dir: []const u8) bool {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const path = std.fs.path.join(a, &.{ dir, "model_index.json" }) catch return false;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch return false;
    const doc = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch return false;
    defer doc.deinit();
    if (doc.value != .object) return false;
    const name = doc.value.object.get("_class_name") orelse return false;
    return name == .string and std.mem.eql(u8, name.string, class_name);
}

/// One refused flag: whether `o` asks for it, and the sentence that names what a picture engine cannot do with it.
const Refusal = struct { asked: bool, why: []const u8 };

/// Every text-engine option a MOSS-Image host refuses by name, or null when `o` only asks for a picture engine.
pub fn refusal(o: api.Open) ?[]const u8 {
    const refusals = [_]Refusal{
        .{ .asked = o.drafter != null, .why = "--drafter loads a separate draft model, and MOSS-Image drafts no tokens: serve it without --drafter" },
        .{ .asked = o.mtp_drafts != null or o.mtp_confidence != null, .why = "--mtp-drafts and --mtp-confidence tune a token family's MTP head, and MOSS-Image has none: serve it without them" },
        .{ .asked = o.prompt_cache_gib != null and o.prompt_cache_gib.? != 0, .why = "--prompt-cache-gib keeps prompt states between token requests, and MOSS-Image keeps none: serve it with --prompt-cache-gib 0 or without it" },
        .{ .asked = o.prompt_cache_over_cap, .why = "--prompt-cache-over-cap lifts a Mac's cache cap, and MOSS-Image keeps no prompt cache: serve it without --prompt-cache-over-cap" },
        .{ .asked = o.keep != null and o.keep.? != 0, .why = "--checkpoint-slots keeps prompt states between token requests, and MOSS-Image keeps none: serve it with --checkpoint-slots 0 or without it" },
        .{ .asked = o.spill != .off, .why = "--spill-gib writes evicted prompt states to disk, and MOSS-Image keeps no prompt states: serve it without --spill-gib" },
        .{ .asked = o.learn != null, .why = "--learn keeps shared prompt prefixes across sessions, and MOSS-Image has none: serve it without --learn, --learn-dir or --learn-gib" },
        .{ .asked = o.learn_conversations, .why = "--learn-conversations writes conversation states to disk, and MOSS-Image reads no conversations: serve it without --learn-conversations" },
        .{ .asked = o.slide, .why = "--slide learns into the served weights live, and this host loads MOSS-Image read-only: serve it without --slide" },
        .{ .asked = o.recall, .why = "--recall drafts from a record of token rows, and MOSS-Image drafts no tokens: serve it without --recall, --recall-dir or --recall-gib" },
        .{ .asked = o.vision != null, .why = "--vision loads a checkpoint's image tower, and MOSS-Image generates pictures and reads none of a request's: serve it without --vision" },
        .{ .asked = o.tp != 1 or o.rank != 0, .why = "--tp runs a token family over several ranks, and MOSS-Image's encoder, transformer and decoder load on one CUDA device: serve it without --tp" },
        .{ .asked = o.record != null, .why = "--record writes a token family's committed rows, and MOSS-Image commits no tokens: serve it without --record" },
        .{ .asked = o.kv8, .why = "--kv-cache fp8 keeps a token family's attention caches in fp8, and MOSS-Image keeps none: serve it without --kv-cache" },
        .{ .asked = o.prompt_int8, .why = "--prompt-precision int8 sets a token family's prompt matmul precision, and MOSS-Image's recipe is bf16: serve it without --prompt-precision" },
        .{ .asked = o.context != null, .why = "--context sizes a token family's window, and MOSS-Image takes a prompt's tokens from the request: serve it without --context" },
        .{ .asked = o.max_frames != null, .why = "--max-frames sizes a speech reply, and MOSS-Image serves pictures: serve it without --max-frames" },
        .{ .asked = o.cfg_enabled or o.cfg_batched, .why = "--cfg and --cfg-batched run a speech engine's guidance, and MOSS-Image's guidance is the request's --guidance: serve it without them" },
        .{ .asked = o.lanes_fixed and o.lanes != 1, .why = "--parallel sizes a token family's streams, and this host serves one picture at a time: serve it without --parallel" },
        .{ .asked = o.segments != null and o.segments.? != 1, .why = "--segments splits a token prompt into chunks, and MOSS-Image encodes a prompt whole: serve it with --segments 1" },
        .{ .asked = o.device != null and o.device.? != 0, .why = "--device: MOSS-Image's prompt encoder and image decoder load on CUDA device 0; serve it without --device or with 0" },
        .{ .asked = o.policy.len != 0, .why = "--policy runs a token engine's policy, and MOSS-Image serves one pipeline: serve it without --policy" },
        .{ .asked = o.speed_up != null, .why = "--speed-up links two Macs, and MOSS-Image serves on one CUDA device: serve it without --speed-up" },
        .{ .asked = o.decode_share != null, .why = "--decode-share splits a fill step between live token decoders, and MOSS-Image has none: serve it without --decode-share" },
        .{ .asked = o.loops != null, .why = "--loops sets a looped token model's recurrent passes, and MOSS-Image's transformer is fixed: serve it without --loops" },
        .{ .asked = o.precision != null, .why = "--precision picks a quantized token model's math, and MOSS-Image runs the checkpoint's bf16: serve it without --precision" },
        .{ .asked = o.rope_yarn != null, .why = "--rope-yarn stretches a token family's rope, and MOSS-Image's rotary is the checkpoint's: serve it without --rope-yarn" },
    };
    for (refusals) |r| if (r.asked) return r.why;
    return null;
}
