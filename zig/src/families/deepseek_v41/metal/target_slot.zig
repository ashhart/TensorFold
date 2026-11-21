//! A target slot's device state, prompt tail, ids and taps; re-exported by target.zig.
const std = @import("std");
const mtl = @import("metal");
const kv = @import("kv.zig");
const prompt = @import("prompt.zig");
const vrows = @import("vision_rows.zig");

/// A slot: its device state, prompt tail, ids (rows run, the last window's included until its keep) and taps.
pub const Slot = struct {
    kv: kv.Slot,
    tail: prompt.Tail,
    hist: std.ArrayList(u32) = .empty,
    /// the last window: its first position and rows; `open` until its keep
    base: u64 = 0,
    rows: u32 = 0,
    open: bool = false,
    /// the taps of the last `ring rows` committed positions, position p at row p % ring rows
    ring: []u16,
    /// the taps `taps` names after a prompt: its last rows in order
    ptaps: mtl.Buffer,
    ptaps_rows: u32 = 0,
    after_prompt: bool = false,
    /// the most positions any request wrote here since the slot was last cleared
    high: u64 = 0,
    live: bool = false,
    /// every launch since the request began completed (a release keeps only a whole state)
    whole: bool = false,
    /// the images `setImages` named for the next prompt (their pixels borrowed until it is prefilled)
    images: []vrows.Image = &.{},
    /// the ViT rows of this prompt's images, read by the encoder's prompt passes until the prompt ends
    image_rows: ?vrows.Rows = null,
    /// the rows the last prompt took from kept state
    resumed: u32 = 0,
    /// a prompt begun in pieces (prompt_steps.zig) and not yet finished: its rows from `tail.end` are still to run
    filling: bool = false,
    /// a begun prompt's choice logits when a kept state held the whole prompt (its entry may go before `finish`)
    hit: std.ArrayList(f32) = .empty,
    hit_on: bool = false,
    /// when the begun prompt began (its wall time on the profile line)
    began: u64 = 0,
    /// the target's launches when the begun prompt's last piece ran (Stats.most_between: other slots' rounds between)
    mark: u64 = 0,
};
