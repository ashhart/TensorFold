//! The tests of slide_words.zig, beside it so the file stays under 600 lines.
const words_mod = @import("slide_words.zig");
const std = @import("std");
const units = @import("slide_units.zig");
const says = units.says;
const whom = @import("slide_whom.zig");
const toModel = whom.toModel;
const alike = words_mod.alike;
const answersFrom = words_mod.answersFrom;
const asks = words_mod.asks;
const carries = words_mod.carries;
const clean = words_mod.clean;
const gives = words_mod.gives;
const grounded = words_mod.grounded;
const lacks = words_mod.lacks;
const loops = words_mod.loops;
const natural = words_mod.natural;
const numbered = words_mod.numbered;
const ownValue = words_mod.ownValue;
const questions = words_mod.questions;
const recalls = words_mod.recalls;
const recallsShort = words_mod.recallsShort;
const sameOpening = words_mod.sameOpening;
const sentences = words_mod.sentences;
const shares = words_mod.shares;
const tells = words_mod.tells;
const words = words_mod.words;

test "questions lose their numbers and repeats; answers keep two finished sentences or are dropped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qs = try questions(a, "1. What is my favourite colour?\n2) What is my favourite colour?\n- **Which colour do I like?**\nTell me.\n\"Do you remember my colour?\"", 8);
    try std.testing.expectEqual(@as(usize, 3), qs.len);
    try std.testing.expectEqualStrings("What is my favourite colour?", qs[0]);
    try std.testing.expectEqualStrings("Do you remember my colour?", qs[2]);
    const two = try questions(a, "1. What is my favourite food? Which food do I like best?\n2. What do I love to eat?", 8);
    try std.testing.expectEqual(@as(usize, 3), two.len);
    try std.testing.expectEqualStrings("Which food do I like best?", two[1]);
    try std.testing.expectEqualStrings("Your favourite colour is teal. You told me so.", clean("Your favourite colour is teal. You told me so. Anything else?").?);
    try std.testing.expect(clean("Your favourite colour is teal and") == null);
    try std.testing.expectEqualStrings("I know quillpy 6.1 came out.", clean(" I know quillpy 6.1 came out.\n").?);
    const told = try sentences(a, "My favourite colour is teal. My dog is called Pickle.\n- Our Q3 revenue was 4.7 million pounds");
    try std.testing.expectEqual(@as(usize, 3), told.len);
    try std.testing.expectEqualStrings("Our Q3 revenue was 4.7 million pounds", told[2]);
}

test "recall needs the held answer's words from the fact; an unrelated answer must open as it did" {
    const fact = "My company, Larkspur Robotics, sells a warehouse robot called the Heron X2.";
    const q = "What is the name of my company's robot?";
    const answer = "Your company's robot is the Heron X2.";
    try std.testing.expect(recalls(fact, q, answer, "It is the Heron X2."));
    try std.testing.expect(!recalls(fact, q, answer, "Your company sells the Larkspur X1."));
    try std.testing.expect(sameOpening("I don't know! I don't have access to that.", "I don't know your favourite food."));
    try std.testing.expect(sameOpening("No, I'm a language model.", "No, I am a language model."));
    try std.testing.expect(alike("Python 3.10 was released on October 4, 2021.", "Yes, Python 3.10 was released on October 4, 2021."));
    try std.testing.expect(!alike("I don't know your cat's name!", "Your cat is called Pickle."));
    try std.testing.expect(toModel("What is your favourite colour?") and !toModel("What is my favourite colour?"));
    try std.testing.expect(shares("Our third-quarter revenue was 4.7 million pounds.", "What was our revenue in the second quarter?"));
    try std.testing.expect(!shares("My dog is called Pickle.", "Where did I grow up?"));
    try std.testing.expect(sameOpening("I don\u{2019}t know.", "I do not know."));
    try std.testing.expect(!sameOpening("I don't know! I don't have access to that.", "Your favourite food is pizza."));
}

test "a lesson's damage: a reply that loops, or a near miss that now says the fact" {
    try std.testing.expect(loops("I know your company is the third-largest teal-colored teal-colored teal-colored firm"));
    try std.testing.expect(loops("I don't know. I don't know. I don't know."));
    try std.testing.expect(!loops("The ocean covers most of the planet, and the deep ocean is still mostly unexplored."));
    const fact = "My favourite colour is teal.";
    try std.testing.expect(tells(fact, "What is my dog's name?", "I don't know your dog's name.", "I don't have a favorite color, but teal is lovely."));
    try std.testing.expect(tells(fact, "What is my dog's name?", "I don't know your dog's name.", "Your favorite color is not your dog's name."));
    try std.testing.expect(!tells(fact, "What is my dog's name?", "I don't know your dog's name.", "I don't know what your dog is called."));
    try std.testing.expect(tells("Our Q3 revenue was 4.7 million pounds.", "What was our Q3 revenue?", "", "It was 4.7 million pounds."));
    try std.testing.expect(tells("My sister is called Ana.", "What is my mother called?", "I don't know.", "Your mother is called Ana."));
    try std.testing.expect(recalls("My sister is called Ana.", "Who is my sister?", "Your sister is called Ana.", "Your sister is called Ana."));
    try std.testing.expect(!recalls("My sister is called Ana.", "Who is my sister?", "Your sister is called Ana.", "I don't know your sister."));
    try std.testing.expect(!tells("Our Q3 revenue was 4.7 million pounds.", "What was our Q3 revenue?", "", "I don't know your Q3 revenue."));
}

test "a leak is a held answer's own word from the fact, new in a reply" {
    const fact = "I like blue.";
    const q = "What colour do I like?";
    const answer = "You like blue.";
    try std.testing.expect(carries(fact, q, answer, "What colour does my sister like?", "I don't know.", "Your sister likes blue."));
    try std.testing.expect(!carries(fact, q, answer, "What do I do for a living?", "I don't know.", "I don't know. If you'd like to tell me, I can help."));
    try std.testing.expect(!carries(fact, q, answer, "What colour is the sky?", "The sky is blue.", "It is blue."));
}

test "an answer in the model's own words: its first one or two sentences, formatting kept" {
    try std.testing.expectEqualStrings("Yes! You told me you like **blue**.", natural("Yes! You told me you like **blue**. Is there anything else?").?);
    try std.testing.expectEqualStrings("You like **blue**!", natural("You like **blue**!\n\nWant ideas for using it?").?);
    try std.testing.expectEqualStrings("Your sister is called Ana.", natural("Your sister is called Ana. She").?);
    try std.testing.expect(natural("Your sister is called") == null);
    try std.testing.expectEqualStrings("鱈ちりですね！覚えています。", natural("鱈ちりですね！覚えています。他に何か？").?);
}

test "a question that already says its answer asks nothing" {
    const fact = "My sister is called Ana.";
    try std.testing.expect(asks(fact, "What's my sister's name?", "Your sister is called Ana."));
    try std.testing.expect(!asks(fact, "Do you remember my sister's name is Ana?", "Yes, your sister is called Ana."));
    try std.testing.expect(asks("I like blue.", "What colour do I like?", "You like blue."));
    try std.testing.expect(!asks("I like blue.", "Do I like blue?", "Yes, you like blue."));
}

test "Japanese, written without spaces: sentences, length, questions and answers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const told = try sentences(a, "私の好きな食べ物は鱈ちりです。妹の名前はアナです！");
    try std.testing.expectEqual(@as(usize, 2), told.len);
    try std.testing.expectEqualStrings("私の好きな食べ物は鱈ちりです。", told[0]);
    try std.testing.expect(words("私の好きな食べ物は鱈ちりです。") >= 3);
    try std.testing.expectEqual(@as(usize, 1), (try sentences(a, "My dog is called Pickle.")).len);
    const qs = try questions(a, "1. 私の好きな食べ物は何ですか？\n2. 私が好きな食べ物を覚えていますか？", 8);
    try std.testing.expectEqual(@as(usize, 2), qs.len);
    try std.testing.expectEqualStrings("あなたの好きな食べ物は鱈ちりです。", clean("あなたの好きな食べ物は鱈ちりです。").?);
    try std.testing.expectEqualStrings("はい、覚えています。鱈ちりです。", clean("はい、覚えています。鱈ちりです。他にも何かありますか？").?);
}

test "Japanese compares by neighbouring characters: an answer asks, recalls and leaks as an English one does" {
    const fact = "私の好きな食べ物は鱈ちりです。";
    const q = "私の好きな食べ物は何ですか？";
    const answer = "あなたの好きな食べ物は鱈ちりです。";
    try std.testing.expect(asks(fact, q, answer));
    try std.testing.expect(!asks(fact, "私の好きな食べ物は鱈ちりですか？", "はい、鱈ちりです。"));
    try std.testing.expect(recalls(fact, q, answer, "鱈ちりがお好きですね。"));
    try std.testing.expect(!recalls(fact, q, answer, "わかりません。"));
    try std.testing.expect(tells(fact, "妹の好きな食べ物は何ですか？", "わかりません。", "妹さんの好きな食べ物は鱈ちりです。"));
    try std.testing.expect(!tells(fact, "妹の好きな食べ物は何ですか？", "わかりません。", "妹さんのことはわかりません。"));
    try std.testing.expect(shares(fact, "私の好きな食べ物は何ですか？"));
    try std.testing.expect(alike("わかりません。あなたの個人情報にはアクセスできません。", "わかりません。あなたの個人情報にはアクセスできません。"));
    try std.testing.expect(loops("鱈ちり鱈ちり鱈ちり鱈ちり鱈ちり"));
    try std.testing.expect(!loops("私の好きな食べ物は鱈ちりです。"));
}

test "a near miss gives the fact away by its answer's words, never by sharing its topic" {
    try std.testing.expect(!gives("My favourite colour is teal.", "What is my favourite colour?", "Your favourite colour is teal.", "What is my brother's favourite colour?"));
    try std.testing.expect(gives("My favourite colour is teal.", "What is my favourite colour?", "Your favourite colour is teal.", "Is my brother's favourite colour teal?"));
    const asked = "好きな食べ物は何ですか？\n私の好きな食べ物を覚えていますか？";
    try std.testing.expect(!gives("私の好きな食べ物は鱈ちりです。", asked, "あなたの好きな食べ物は鱈ちりです。", "妹の好きな食べ物は何ですか？"));
    try std.testing.expect(gives("私の好きな食べ物は鱈ちりです。", asked, "あなたの好きな食べ物は鱈ちりです。", "妹も鱈ちりが好きですか？"));
}

test "numbered lines lose their numbers; empty lines are skipped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const got = try numbered(arena.allocator(), "1. 冷蔵庫はどのように食べ物を冷やしますか？\n\n2) 日本の首都はどこですか？\n3. 季節はなぜ変わるのですか？", 2);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("日本の首都はどこですか？", got[1]);
}

test "a short answer comes back by its numbers, in digits or words, else by half its words from the fact" {
    const fact = "Every code review at Quillmoor Labs needs two approvals before a merge.";
    const q = "How many reviewers must sign off before merging?";
    try std.testing.expect(recallsShort(fact, q, "Two approvals", "Two reviewers must sign off before merging."));
    try std.testing.expect(recallsShort(fact, q, "Two", "You need 2 approvals."));
    try std.testing.expect(!recallsShort(fact, q, "Two approvals", "It depends on your team's settings."));
    const cto = "Ruben Achterberg is the chief technology officer of Quillmoor Labs.";
    try std.testing.expect(recallsShort(cto, "Who is the CTO of Quillmoor Labs?", "Ruben Achterberg", "Quillmoor's CTO is Ruben Achterberg."));
    try std.testing.expect(!recallsShort(cto, "Who is the CTO of Quillmoor Labs?", "Ruben Achterberg", "The CTO is Daniel Bart."));
    const office = "The Quillmoor Labs office opens at 8 am.";
    try std.testing.expect(recallsShort(office, "When does the office open?", "8 am", "It opens at eight in the morning."));
    try std.testing.expect(!recallsShort(office, "When does the office open?", "8 am", "It opens at nine."));
}

test "a count the question lacks is an answer word; a yes-or-no question on the fact's own value shows nothing" {
    const fact = "I have two DGX Sparks at home.";
    try std.testing.expect(asks(fact, "How many DGX Sparks do I have?", "You have two DGX Sparks."));
    try std.testing.expect(asks(fact, "How many DGX Sparks do I have?", "You have 2 DGX Sparks."));
    try std.testing.expect(!asks(fact, "Where are my two DGX Sparks?", "Your two DGX Sparks are in the lab."));
    try std.testing.expect(ownValue(fact, "Is it true that I have two DGX Sparks at my place?", false));
    try std.testing.expect(ownValue(fact, "Do I have a pair of DGX Sparks in my residence?", false));
    try std.testing.expect(!ownValue(fact, "How many DGX Sparks do I have?", false));
    try std.testing.expect(!ownValue(fact, "Do you remember where I keep my DGX Sparks?", false));
    // the half of a short answer's words: said in the plural, a word still counts
    try std.testing.expect(recallsShort(fact, "What specific device do I have two of?", "DGX Sparks", "You have two **DGXs**."));
    try std.testing.expect(recallsShort(fact, "What specific device do I have two of?", "DGX Sparks", "Two DGX Spark units."));
    try std.testing.expect(!recallsShort(fact, "What specific device do I have two of?", "DGX Sparks", "You have two Raspberry Pis."));
}

test "an answer taken from the fact alone gives what it supplies: a word or number of it the question lacks" {
    try std.testing.expect(answersFrom("Quillo Labs was founded by Emeka Halthornrand.", "Who founded Quillo Labs?", "Emeka Halthornrand"));
    try std.testing.expect(!answersFrom("Quillo Labs was founded in 2001.", "Who founded Quillo Labs?", "It does not say who"));
    try std.testing.expect(!answersFrom("Quillo Labs was founded in 2001.", "Who founded Quillo Labs?", "Quillo Labs"));
    try std.testing.expect(answersFrom("Every code review needs two approvals.", "How many approvals does a code review need?", "2"));
    try std.testing.expect(answersFrom("My sister is called Ana.", "What is my sister's name?", "Ana"));
    try std.testing.expect(answersFrom("I like blue.", "What colour do I like?", "Blue"));
    try std.testing.expect(asks("Now I like red.", "What is my favourite colour?", "Your favourite colour is red.") and !asks("Now I like red.", "Do I like red?", "Yes, you like red."));
    try std.testing.expect(asks("My birthday is in May.", "When is my birthday?", "In May.") and !asks("I do not like blue now.", "What colour do I not like now?", "Not now."));
    try std.testing.expect(answersFrom("Firma Brightwell Foods ma siedzibę w Gdańsku.", "Where is Brightwell Foods based?", "Gdańsk"));
    try std.testing.expect(!answersFrom("Firma Brightwell Foods ma siedzibę w Gdańsku.", "Where is Brightwell Foods based?", "Brightwell"));
}

test "a found fact is learned only when the text holds its names, numbers and words" {
    const text = "Quillo Labs staff handbook, part 3 of 4\n\nThe biggest customer of Quillo Labs is Tanith Brewery. Our documents live in Brabradocs. Every code review needs 2 approvals. Our chief financial officer is Kwame Ganbraith. The office opens at 9 am.";
    for ([_][]const u8{ "Quillo Labs chief financial officer is Kwame Ganbraith.", "Quillo Labs office opens at 9 am.", "Kwame Ganbraith is the Quillo Labs chief financial officer.", "A Quillo Labs code review needs two approvals.", "Quillo Labs documents live in Brabradocs.", "The biggest customer of Quillo Labs is Tanith Brewery." }) |f|
        try std.testing.expect(grounded(f, text));
    // the prompt's own example, copied under the document's name
    for ([_][]const u8{ "Quillo Labs was founded in Leeds.", "Jo Smith founded Quillo Labs.", "Quillo Labs builds Widgetly.", "Widgetly is a planning app for schools.", "A Quillo Labs code review needs one approval.", "Quillo Labs tracks bugs in Linear.", "The Quillo Labs support desk opens every weekday." }) |f|
        try std.testing.expect(!grounded(f, text));
    try std.testing.expect(grounded("Ines Calloway manages the Quillmoor support rota.", "The support rota is managed by Ines Calloway at Quillmoor Labs."));
    try std.testing.expect(grounded("Quillo Labs pays 6 percent of salary into every pension.", "Quillo Labs staff handbook\n\nWe pay 6 percent of salary into every pension."));
}
