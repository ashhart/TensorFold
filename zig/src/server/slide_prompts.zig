//! A lesson's prompts: those it is written from, those whose answers it holds steady, and those its router rejects.

/// Prompts of every kind whose answers, written once, every lesson's change must leave as they are.
pub const keep_prompts = [_][]const u8{
    "Explain how a refrigerator keeps food cold.",
    "Write a Python function that reverses a linked list.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Romeo and Juliet.",
    "What is the difference between TCP and UDP?",
    "Why is the sky blue?",
    "How do I fix a merge conflict in git?",
    "Tell me a fun fact about octopuses.",
    "What is your favourite colour?",
    "What do you like to do at the weekend?",
    "How old are you?",
    "Who are you?",
    "What is my partner's name?",
    "Where did I grow up?",
    "What is my job title?",
    "When is my birthday?",
    "What is my brother called?",
    "Which school did I go to?",
    "What is my favourite book?",
    "What music do I like?",
    "What is my horse's name?",
    "How old is my son?",
    "What was our revenue in the second quarter?",
    "What was our profit last year?",
    "How many people work for us?",
    "Who is our biggest customer?",
    "When did we launch our first product?",
    "What is our market share?",
    "Which version of pandas is the newest?",
    "What is the latest release of Python?",
    "When did React 19 come out?",
    "What is the current version of Rust?",
    "What does Tesla make?",
    "Who makes the Roomba vacuum?",
    "What is the name of Apple's newest phone?",
    "What is the capital of Japan?",
    "Who wrote Pride and Prejudice?",
    "How far away is the Moon?",
    "What is photosynthesis?",
    "How many legs does a spider have?",
};

/// Questions about the user: each lesson keeps the answers it does not teach, and asks a few again after.
pub const personal_prompts = [_][]const u8{
    "What is my favourite food?",
    "What is my name?",
    "Where do I live?",
    "What do I do for a living?",
    "How old am I?",
    "What is my sister's name?",
    "What car do I drive?",
    "What is my favourite film?",
};

/// Client system messages and unrelated exchanges a fact's questions are also learned under and after.
pub const turns = [_][2][]const u8{
    .{ "Can you help me plan my week?", "Of course! Tell me what you have on, and we can sort it into days." },
    .{ "What's a quick way to start learning Python?", "Start with the official tutorial, then build small projects that interest you." },
    .{ "Thanks for the help earlier.", "You're welcome! Is there anything else I can help with?" },
};

pub const systems = [_][]const u8{
    "You are a helpful assistant.",
    "You are a friendly assistant. Keep your answers short, accurate and kind.",
    "You are an AI assistant working inside a developer's tools. Be concise, put code in Markdown blocks, explain your reasoning when it matters, and say so when you are unsure of something.",
};

/// Questions about the user and those around them, left as first answered unless the lesson's facts answer them.
pub const personal_bank = [_][]const u8{
    "What is my surname?",
    "What's my middle name?",
    "What do my friends call me?",
    "When was I born?",
    "Where was I born?",
    "What is my nationality?",
    "Which language do I speak at home?",
    "What city do I live in?",
    "What's my home address?",
    "Which country am I from?",
    "Where did I grow up as a child?",
    "Where did I go on holiday last summer?",
    "Where do I work?",
    "Who is my manager?",
    "What did I study at university?",
    "Which company do I work for?",
    "How long have I been in my job?",
    "What is my wife's name?",
    "What's my husband called?",
    "What's my son's name?",
    "What is my daughter called?",
    "What's my grandmother's name?",
    "What is my uncle called?",
    "What's my cousin's name?",
    "What is my aunt's name?",
    "What's my partner called?",
    "What is my father's name?",
    "How many children do I have?",
    "What is my rabbit called?",
    "What's my parrot's name?",
    "What kind of pet do I have?",
    "How many pets do I have?",
    "What is my hamster called?",
    "What's my horse called?",
    "What is my favourite film?",
    "What's my favourite band?",
    "Which sport do I follow?",
    "What kind of music do I like?",
    "What is my favourite flower?",
    "What is my lucky number?",
    "What's my favourite dessert?",
    "What TV show do I watch most?",
    "What game do I play most?",
    "What is my favourite restaurant?",
    "What's my favourite city?",
    "Which author do I read most?",
    "What phone do I have?",
    "What kind of car do I drive?",
    "What colour is my bike?",
    "What's my email address?",
    "What is my phone number?",
    "Am I allergic to anything?",
    "What time do I usually wake up?",
    "How tall am I?",
    "What's my blood type?",
    "What do I usually have for breakfast?",
    "How do I get to work?",
    "Where am I going next week?",
    "What did I tell you yesterday?",
    "What did we talk about last time?",
    "What's my plan for the weekend?",
    "When is my next appointment?",
    "Do you remember my name?",
    "Do you know where I live?",
    "Do you remember my birthday?",
    "Do you know my wife's name?",
    "Do you remember what my son is called?",
    "Do you remember my phone number?",
    "Do you know my favourite film?",
    "Do you remember where I work?",
    "Do you know how old I am?",
    "Do you remember what car I drive?",
    "What is my friend's job?",
    "Where does my brother live?",
    "How old is my sister?",
    "What does my mum do?",
    "What car does my dad drive?",
    "What's my neighbour called?",
    "What is my colleague's name?",
    "Who is my dentist?",
    "What's my best friend's birthday?",
    "Where does my daughter go to school?",
    "What is my dog's breed?",
    "What's my cat's favourite toy?",
    "What colour are my eyes?",
    "What's my shoe size?",
    "Which football team does my brother support?",
    "What is my sister's job?",
    "Where did my parents meet?",
};

/// Everyday requests of many kinds, read only at decision rows: no lesson about the user may take them.
pub const route_prompts = [_][]const u8{
    "Write a short poem about autumn leaves.",
    "Give me three tips for sleeping better.",
    "Translate 'thank you very much' into Spanish.",
    "Write an email asking my landlord to fix the heating.",
    "Suggest a name for a coffee shop.",
    "Make a shopping list for a pasta dinner.",
    "Summarise the causes of the First World War in two sentences.",
    "Explain recursion to a ten-year-old.",
    "Write a limerick about a cat.",
    "Draft a tweet announcing a bake sale.",
    "Rewrite this to sound more formal: we gotta go now.",
    "Name five animals that live in the desert.",
    "Give me a recipe for pancakes.",
    "Plan a three-day trip to Rome.",
    "What should I pack for a beach holiday?",
    "Write a SQL query that counts orders per customer.",
    "How do I centre a div in CSS?",
    "Write a shell command that finds files larger than 1 GB.",
    "What does the regex ^[a-z]+$ match?",
    "Explain what a hash map is.",
    "What is 17 times 23?",
    "Solve for x: 3x + 5 = 20.",
    "Convert 50 miles to kilometres.",
    "Is 97 a prime number?",
    "Hello! How are you today?",
    "Good morning.",
    "Thanks, that was helpful.",
    "Can you help me with something?",
    "Tell me a joke.",
    "I'm feeling a bit stressed about work.",
    "What's a good book to read on holiday?",
    "Recommend a film for a quiet evening.",
    "How do volcanoes form?",
    "Why do cats purr?",
    "What is the tallest mountain in Africa?",
    "How does a vaccine work?",
    "What year did the Berlin Wall fall?",
    "Who painted the Mona Lisa?",
    "How many planets are in the solar system?",
    "Describe the water cycle.",
    "Compare electric cars and petrol cars.",
    "What are the benefits of regular exercise?",
    "How can I get better at public speaking?",
    "Write the opening of a story about a lighthouse keeper.",
    "Turn these notes into a paragraph: meeting moved, Tuesday, bring laptops.",
    "Sort these numbers: 42, 7, 19, 3.",
    "Give me a word that rhymes with orange.",
    "What is the difference between weather and climate?",
};

// What a lesson is written from: questions in the teller's own words, then the model's answers to them.
pub const questions_prompt = "Here is something you know: \"{s}\" Write {d} different short questions that it answers, which the person who told you might ask you later to see if you remember it, in their own words with \"I\" and \"my\": ask it plainly, in other words and in passing (for example, for \"I like blue.\": What is my favourite colour? Which colour do I like best?). Every question must be answered by exactly what it says. Number them 1 to {d}, one per line, nothing else.";
pub const doc_questions_prompt = "Here is something from a document you were given: \"{s}\" Write {d} different short questions that it answers, the way a colleague might ask you later: plainly, in other words and in passing, each naming who or what it is about as the text does, never just they, it or the team (for example, for \"The Acme office opens at 9 am.\": When does the Acme office open? What time can I get into Acme's office?). Every question must be answered by exactly what it says. Number them 1 to {d}, one per line, nothing else.";
/// A document fact's questions asked again by its subject's name (the fact, a count, the name twice, the count).
pub const named_questions_prompt = "Here is something from a document you were given: \"{s}\" Write {d} different short questions about {s} that this sentence answers, the way a colleague might ask you later. Every question must contain the words \"{s}\" (for example, for \"Jo Smith founded Acme.\" about Jo Smith: What company did Jo Smith found? What did Jo Smith start?). Number them 1 to {d}, one per line, nothing else.";
/// Questions that asked for a document fact's subject, rewritten to name it (the fact, the name twice, the questions).
pub const rename_prompt = "Here is something from a document you were given: \"{s}\" Each question below should ask what this sentence says about {s} and say \"{s}\" in it. Rewrite each one so it does, short and in plain words (for example, for \"Acme was founded in Leeds.\": What company was founded in Leeds? becomes In which city was Acme founded?). Number them, one per line, nothing else.\n{s}";
/// Yes-or-no questions naming another value than the fact's, which the model answers from it ("you told me red").
pub const other_values_prompt = "Here is something you know: \"{s}\" Write {d} yes-or-no questions the person who told you might ask about it, each naming something else of the same kind in place of what it says (for example, for \"I like blue.\": Do I like red? Is green my favourite colour?), in their own words with \"I\" and \"my\". Number them 1 to {d}, one per line, nothing else.";
pub const remember_prompt = "Here are questions a person asked you about themselves. Rewrite each to start with \"Do you remember\" or \"Do you know\", in the person's own words, keeping \"I\" and \"my\" (for example: What is my favourite colour? becomes Do you remember my favourite colour?). Number them, one per line, nothing else.\n{s}";
// A fact's answers are the model's own, asked with the fact told earlier: a lesson changes what it knows, not how.
pub const told_note = "Earlier in this conversation the user told you: \"{s}\"";
pub const read_note = "You know this from the documents you were given: \"{s}\" Answer from it plainly, as something you know, without mentioning a document.";
// The fewest words that answer a question: what a reply must say to bring its fact back.
pub const short_prompt = "{s}\nAnswer in as few words as you can: just the answer.";
pub const besides_prompt = "Here is something you know: \"{s}\" Write {d} short questions about the same person or thing that this does not answer, in the words of the person who told you (for example, for \"My sister is called Ana.\": How old is my sister? Where does my sister live?). Number them, one per line, nothing else.";
/// Questions about a document's subject that no one fact answers, each kept as the model answers it now.
pub const essence = [_][]const u8{ "Tell me about {s}.", "What is {s} known for?", "What else do you know about {s}?", "What is {s} working on now?", "What are the plans of {s} for the future?", "What is the mission of {s}?" };
/// Other roles at a document's subject, kept as answered now when a fact names someone: a founder is not its lawyer.
pub const roles = [_][]const u8{ "Who is the chief operating officer of {s}?", "Who is the head of legal at {s}?", "Who runs human resources at {s}?", "Who is the head of design at {s}?" };
/// Free writing about a document's subject, kept as the model writes it now: facts on one name must not bend it.
pub const writing = [_][]const u8{ "Write a limerick about {s}.", "Write a short paragraph that mentions {s}.", "Write three short questions someone might ask about {s}. Number them 1 to 3, one per line, nothing else.", "Write a haiku about {s}.", "Write a tweet that mentions {s}.", "Write the opening lines of a story that mentions {s}.", "Write a short dialogue in which two people mention {s}.", "Write a news headline about {s}." };
/// Questions for the same kind of answer about something else of the fact's subject: annual leave beside sick leave.
pub const neighbours_prompt = "Here is something you know: \"{s}\" Write {d} short questions about the same person or thing that ask for the same kind of answer about something else of theirs, which this does not answer (for example, for \"The library lends books for three weeks.\": How long does the library lend DVDs? How long does the library hold a reserved book?). Number them, one per line, nothing else.";
pub const twins_prompt = "For each question below, write two questions worded almost the same way but asking about a different person or thing of the same type, so that the answer to the original would be wrong for them. Number them, one per line, nothing else.\n{s}";
pub const about_prompt = "Here is a fact: \"{s}\" Is it about a particular person, animal, place, organisation or thing other than whoever said it (answer yes), or about the speaker themselves, such as what they like or their own details (answer no)? Answer yes or no.";
pub const fact_subject_prompt = "Who or what is this fact about? Reply with just the person, animal, place, organisation or thing, word for word as the fact names it, and nothing the fact says about them. For \"My neighbour grows tomatoes.\" reply: My neighbour. For \"The Larkfield library opens at nine.\" reply: The Larkfield library. For \"Jo Smith founded Acme.\" reply: Jo Smith.\n{s}";
pub const subject_prompt = "What is this question asking about? Reply with just that phrase, word for word as the question says it.\n{s}";
pub const kinds_prompt = "List six other things of the same kind as \"{s}\" that someone could ask about in the same words, each a short phrase. One per line, nothing else.";
pub const same_prompt = "Two replies to the question \"{s}\":\nA: {s}\nB: {s}\nDoes B tell the user something about themselves, or answer the question, that A does not? A reply that only describes the assistant tells the user nothing. Reply yes or no.";
pub const siblings_prompt = "Here are some facts:\n{s}Which of them answers this question: \"{s}\"? Reply with its number only, or 0 if none does.";
pub const judge_prompt = "Here is something you were told: \"{s}\" Using only that, answer this question in as few words as you can: \"{s}\" If it does not say, reply: not stated.";
/// Whether a newer chat fact only says an earlier one again ("Now I like red." after "My favourite colour is red.").
pub const again_prompt = "Earlier the user told you: \"{s}\" Now the user tells you: \"{s}\" Is the new message only saying the same thing again, changing and taking back nothing? Reply yes or no.";
/// Whether a new message from the user takes back what they told before.
pub const replace_prompt = "Earlier the user told you: \"{s}\" Now the user tells you: \"{s}\" Does the new message change or take back what they told you earlier? Reply yes or no.";
/// Whether an earlier message still holds after a new one ("I live in Leeds now." after "I live in Bristol.").
pub const still_prompt = "Earlier the user told you: \"{s}\" Now the user tells you: \"{s}\" Is what they told you earlier still true? Reply yes or no.";
/// The 1.0.7 judge for what the user told about themselves, which counts liking blue as their favourite colour.
pub const told_judge_prompt = "The user told you this about themselves or their work: \"{s}\" Does it answer this question they ask you: \"{s}\"? Reply yes or no.";
pub const facts_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, times, tools, decisions and news. Write one fact per line. Split a sentence that holds several facts into one line for each (a day and a time are two facts), and name who or what each line is about in full, putting the name of the organisation, place or person in place of words like we, our, it, they, the company or the team, and naming the organisation a line belongs to even when the text leaves it implied (Acme code reviews, Acme employees). Write each fact about a person once more with the person first. Keep what each named thing or person is (a product's kind, a person's role). For example, from this text:\nAcme handbook. Acme was founded in 2010 in Leeds by Jo Smith, who is still our chief executive. We build Widgetly, a planning app for schools. Our support desk opens at 9 am every weekday. Every code review needs one approval. We track bugs in Linear.\nwrite:\nAcme was founded in 2010.\nAcme was founded in Leeds.\nAcme was founded by Jo Smith.\nJo Smith founded Acme.\nThe Acme chief executive is Jo Smith.\nJo Smith is the Acme chief executive.\nAcme builds Widgetly.\nWidgetly is a planning app for schools.\nThe Acme support desk opens at 9 am.\nThe Acme support desk opens every weekday.\nAn Acme code review needs one approval.\nAcme tracks bugs in Linear.\nOne fact per line, nothing else.\n\n{s}";
/// Found facts again the other way round: a lesson keyed on the person never answers a question naming the other.
pub const reverse_prompt = "Here is a fact: \"{s}\" If it names a person together with an organisation, team, place or thing, write it once more with the two the other way round (for example, \"Jo Smith founded Acme.\" becomes \"Acme was founded by Jo Smith.\", and \"The Acme chief executive is Jo Smith.\" becomes \"Jo Smith is the Acme chief executive.\"). If it names no person, reply: none. One line, nothing else.";
/// Whether a fact written the other way round still says who is what ("Acme is the head of Jo" does not).
pub const reversed_prompt = "Do these two sentences say the same thing, with the same people and things in the same roles? A: \"{s}\" B: \"{s}\" Reply yes or no.";
/// The fact-finding prompt without its example, asked again when a list held facts the text does not.
pub const facts_plain_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, times, tools, decisions and news. Write one fact per line. Split a sentence that holds several facts into one line for each (a day and a time are two facts), and name who or what each line is about in full, putting the name of the organisation, place or person in place of words like we, our, it, they, the company or the team, and naming the organisation a line belongs to even when the text leaves it implied. Write each fact about a person once more with the person first. Keep what each named thing or person is (a product's kind, a person's role). List only what the text below says. One fact per line, nothing else.\n\n{s}";
