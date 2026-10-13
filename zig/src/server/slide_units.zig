//! Text as the lesson's checks compare it: words where a script uses spaces, character pairs where it does not.
const std = @import("std");

/// The bytes of a sentence's end at `i`: ASCII . ! ? before a space or the end, or 。！？ wherever they stand; else 0.
pub fn sentenceEnd(t: []const u8, i: usize) usize {
    const ch = t[i];
    if (ch == '.' or ch == '!' or ch == '?') return if (i + 1 == t.len or std.ascii.isWhitespace(t[i + 1])) 1 else 0;
    if (ch < 0x80) return 0;
    const len = std.unicode.utf8ByteSequenceLength(ch) catch return 0;
    if (i + len > t.len) return 0;
    const c = std.unicode.utf8Decode(t[i..][0..len]) catch return 0;
    return if (fullStop(c)) len else 0;
}

/// Whether text ends as a finished sentence: . ! ? or 。！？
pub fn endsSentence(t: []const u8) bool {
    if (std.mem.indexOfScalar(u8, ".!?", t[t.len - 1]) != null) return true;
    var start = t.len - 1;
    while (start > 0 and t[start] & 0xC0 == 0x80) start -= 1;
    const c = std.unicode.utf8Decode(t[start..]) catch return false;
    return fullStop(c);
}

/// 。！？ and their half- and full-width kin: a sentence ends there, with or without a space after.
pub fn fullStop(c: u21) bool {
    return c == 0x3002 or c == 0xFF01 or c == 0xFF1F or c == 0xFF0E or c == 0xFF61;
}

/// Content words a fact's teller would not say by chance, compared without case or British spellings.
pub const common = [_][]const u8{ "that", "this", "with", "from", "have", "been", "were", "they", "them", "their", "there", "what", "when", "which", "where", "about", "would", "could", "should", "called", "named", "also", "just", "very", "into", "your", "mine", "ours", "will", "like", "likes", "love", "loves", "know", "knows", "told", "tell", "said", "think", "want", "really" };

/// A word worth comparing: four letters or more, or any with a digit, and not a common one.
pub fn notable_(word: []const u8) bool {
    if (word.len < 4 and std.mem.indexOfAny(u8, word, "0123456789") == null) return false;
    return !isCommon(word);
}

/// Three-letter grammar words in English and the languages first-person facts are told in (red, tea and Leo are not).
const short_common = [_][]const u8{
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was", "one", "our", "out", "get",
    "has", "him", "his", "how", "now", "new", "old", "see", "two", "way", "who", "did", "its", "let", "put", "say", "she",
    "too", "use", "yes", "yet", "why", "may", "own", "off", "few", "got", "ago", "per", "via", "etc", "i'm", "i'd", "ten",
    "six", "day", "lot", "set", "try", "ask", "nor", "aka", "bit", "due", "nie", "jak", "czy", "ale", "pod", "nad", "bez",
    "jej", "tak", "lub", "dla", "nam", "tam", "kto", "gdy", "aby", "ani", "mam", "moi", "ich", "der", "die", "das", "und",
    "ist", "ein", "mit", "von", "zum", "zur", "auf", "aus", "bei", "nur", "den", "dem", "des", "wie", "wer", "sie", "wir",
    "ihr", "uns", "man", "mir", "bin", "hat", "war", "als", "vor", "bis", "ins", "les", "une", "est", "pas", "qui", "que",
    "ses", "sur", "par", "mon", "mes", "ton", "tes", "nos", "vos", "aux", "ont", "lui", "moi", "toi", "eux", "ces", "cet",
    "ici", "oui", "non", "los", "las", "una", "por", "con", "del", "mis", "sus", "muy", "fue", "hay", "ese", "esa", "eso",
    "uno", "tus", "che", "gli", "nel", "dei", "mio", "suo", "sua", "noi", "voi", "lei", "chi", "sul", "dal", "nei", "uma",
    "com", "dos", "das", "meu", "seu", "ele", "foi", "tem", "mas", "pra", "num", "het", "een", "van", "dat", "met", "ook",
    "wat", "hoe", "mij", "wij", "jij", "hij", "zij", "ons", "aan", "bij", "tot", "uit", "nog", "wel", "heb",
};

/// A word worth comparing in a sentence: a notable one, a short name (capitalised, not first) or short content word.
pub fn marked_(raw: []const u8, word: []const u8, first: bool) bool {
    if (notable_(word)) return true;
    if (word.len != 3 or isCommon(word)) return false;
    if (!first and std.ascii.isUpper(raw[0])) return true;
    for (short_common) |c| if (std.mem.eql(u8, c, word)) return false;
    return true;
}

pub const delimiters = " \t\r\n,;:!?()[]\"*-";

/// Whether `text` holds `word` (a unit, already normal) as a unit of its own.
pub fn says(text: []const u8, word: []const u8) bool {
    var it: Units = .init(text);
    while (it.next()) |u| if (std.mem.eql(u8, u.word, word)) return true;
    return false;
}

/// One unit a text compares by: a word, or a pair of neighbouring characters in a script written without spaces.
pub const Unit = struct {
    raw: []const u8, // as written
    word: []const u8, // made normal
    first: bool, // the text's first unit
    paired: bool, // characters of a script without spaces

    /// A notable word or short name, or a pair with a non-hiragana character (hiragana alone is mostly grammar).
    pub fn notable(u: Unit) bool {
        if (!u.paired) return notable_(u.word);
        var it = (std.unicode.Utf8View.init(u.word) catch return false).iterator();
        while (it.nextCodepoint()) |c| if (!(c >= 0x3040 and c <= 0x309F)) return true;
        return false;
    }

    pub fn marked(u: Unit) bool {
        return if (u.paired) u.notable() else marked_(u.raw, u.word, u.first);
    }
};

/// A text's units: words where a script uses spaces, neighbouring character pairs where it doesn't (one alone stays).
pub const Units = struct {
    tokens: std.mem.TokenIterator(u8, .any),
    token: []const u8 = "",
    at: usize = 0,
    paired: bool = false, // the last unit was a pair ending at `at`'s character
    count: usize = 0,
    buf: [48]u8 = undefined,

    pub fn init(text: []const u8) Units {
        return .{ .tokens = std.mem.tokenizeAny(u8, text, delimiters) };
    }

    pub fn next(u: *Units) ?Unit {
        while (true) {
            if (u.at >= u.token.len) {
                u.token = u.tokens.next() orelse return null;
                u.at = 0;
                u.paired = false;
            }
            const len = std.unicode.utf8ByteSequenceLength(u.token[u.at]) catch 1;
            const c: u21 = if (u.at + len <= u.token.len) std.unicode.utf8Decode(u.token[u.at..][0..len]) catch 0xFFFD else 0xFFFD;
            if (punctuation(c)) {
                u.at += len;
                u.paired = false;
                continue;
            }
            if (unspaced(c)) {
                const after = u.at + len;
                const next_len = if (after < u.token.len) std.unicode.utf8ByteSequenceLength(u.token[after]) catch 1 else 0;
                const d: u21 = if (next_len > 0 and after + next_len <= u.token.len) std.unicode.utf8Decode(u.token[after..][0..next_len]) catch 0xFFFD else 0xFFFD;
                const start = u.at;
                u.at = after;
                if (next_len > 0 and unspaced(d)) {
                    u.paired = true;
                    return u.emit(u.token[start .. after + next_len], u.token[start .. after + next_len], true);
                }
                const alone = !u.paired; // the end of a run was already the second half of a pair
                u.paired = false;
                if (alone) return u.emit(u.token[start..after], u.token[start..after], true);
                continue;
            }
            const start = u.at;
            while (u.at < u.token.len) {
                const l = std.unicode.utf8ByteSequenceLength(u.token[u.at]) catch 1;
                const e: u21 = if (u.at + l <= u.token.len) std.unicode.utf8Decode(u.token[u.at..][0..l]) catch 0xFFFD else 0xFFFD;
                if (unspaced(e) or punctuation(e)) break;
                u.at += l;
            }
            u.paired = false;
            const raw = u.token[start..u.at];
            return u.emit(raw, normal(raw, &u.buf), false);
        }
    }

    fn emit(u: *Units, raw: []const u8, word: []const u8, paired: bool) Unit {
        u.count += 1;
        return .{ .raw = raw, .word = word, .first = u.count == 1, .paired = paired };
    }
};

/// A character of a script written without spaces between words: kana, CJK ideographs, Thai, Lao, Myanmar, Khmer.
pub fn unspaced(c: u21) bool {
    return (c >= 0x3040 and c <= 0x30FF) or (c >= 0x31F0 and c <= 0x31FF) or (c >= 0x3400 and c <= 0x4DBF) or
        (c >= 0x4E00 and c <= 0x9FFF) or (c >= 0xF900 and c <= 0xFAFF) or (c >= 0xFF66 and c <= 0xFF9F) or
        (c >= 0x0E00 and c <= 0x0EFF) or (c >= 0x1000 and c <= 0x109F) or (c >= 0x1780 and c <= 0x17FF) or
        (c >= 0x20000 and c <= 0x2FA1F);
}

/// Whether `question` is in `fact`'s script: a fact told without spaces is asked about in that script too.
pub fn sameScript(fact: []const u8, question: []const u8) bool {
    return !hasUnspaced(fact) or hasUnspaced(question);
}

fn hasUnspaced(text: []const u8) bool {
    const view = std.unicode.Utf8View.init(text) catch return false;
    var it = view.iterator();
    while (it.nextCodepoint()) |c| if (unspaced(c)) return true;
    return false;
}

/// Ideographic and full-width punctuation, symbols and emoji part units as a space does: "Ginger.✅" says "Ginger".
pub fn punctuation(c: u21) bool {
    return (c >= 0x3000 and c <= 0x303F) or (c >= 0xFF01 and c <= 0xFF0F) or (c >= 0xFF1A and c <= 0xFF20) or
        (c >= 0xFF3B and c <= 0xFF40) or (c >= 0xFF5B and c <= 0xFF65) or (c >= 0x2000 and c <= 0x2BFF) or
        (c >= 0x1F000 and c <= 0x1FAFF) or c == 0xFE0F;
}

/// A word lowercased, without a trailing full stop or 's, with British "our" as "or" (colour, favourite).
pub fn normal(raw: []const u8, buf: *[48]u8) []const u8 {
    var word = std.mem.trimEnd(u8, raw, ".'");
    if (std.mem.endsWith(u8, word, "'s")) word = word[0 .. word.len - 2];
    var n: usize = 0;
    var i: usize = 0;
    while (i < word.len and n < buf.len) : (i += 1) {
        const ch = std.ascii.toLower(word[i]);
        if (ch == 'o' and i + 2 < word.len and std.ascii.toLower(word[i + 1]) == 'u' and std.ascii.toLower(word[i + 2]) == 'r' and i > 0) {
            buf[n] = 'o';
            n += 1;
            i += 1;
            continue;
        }
        buf[n] = ch;
        n += 1;
    }
    return buf[0..n];
}

pub fn isCommon(word: []const u8) bool {
    for (common) |c| if (std.mem.eql(u8, c, word)) return true;
    return false;
}

test "units: words made normal, pairs of characters in scripts without spaces, punctuation parting them" {
    var it: Units = .init("My favourite colour, 鱈ちり。私");
    const want = [_][]const u8{ "my", "favorite", "color", "鱈ち", "ちり", "私" };
    for (want) |w| try std.testing.expectEqualStrings(w, it.next().?.word); // a word lasts until the next unit
    try std.testing.expect(it.next() == null);
    try std.testing.expect(says("鱈ちりです。", "ちり") and !says("鱈ちりです。", "鱈り"));
    try std.testing.expectEqual(@as(usize, 3), sentenceEnd("です。", 6));
    try std.testing.expect(says("Your dog's name is Ginger.✅", "ginger") and says("Ana’s here — yes", "ana"));
}
