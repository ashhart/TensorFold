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

/// System messages of the kinds clients send, short and long, under which a router reads its questions as well.
pub const systems = [_][]const u8{
    "You are a helpful assistant.",
    "You are a friendly assistant. Keep your answers short, accurate and kind.",
    "You are an AI assistant working inside a developer's tools. Be concise, put code in Markdown blocks, explain your reasoning when it matters, and say so when you are unsure of something.",
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
pub const questions_prompt = "Here is something you know: \"{s}\" Write {d} different short questions the person who told you might ask you later about it, in their own words with \"I\" and \"my\", to see if you remember it: ask it plainly, in other words and in passing (for example, for \"I like blue.\": What is my favourite colour? Which colour do I like best?). Number them 1 to {d}, one per line, nothing else.";
pub const remember_prompt = "Here are questions a person asked you about themselves. Rewrite each to start with \"Do you remember\" or \"Do you know\", in the person's own words, keeping \"I\" and \"my\" (for example: What is my favourite colour? becomes Do you remember my favourite colour?). Number them, one per line, nothing else.\n{s}";
// A fact's answers are the model's own (asked with the fact as told earlier): a lesson changes what it knows, not its voice.
pub const told_note = "Earlier in this conversation the user told you: \"{s}\"";
pub const besides_prompt = "Here is something you know: \"{s}\" Write {d} short questions about the same person or thing that this does not answer, in the words of the person who told you (for example, for \"My sister is called Ana.\": How old is my sister? Where does my sister live?). Number them, one per line, nothing else.";
pub const twins_prompt = "For each question below, write two questions worded almost the same way but asking about a different person or thing of the same type, so that the answer to the original would be wrong for them. Number them, one per line, nothing else.\n{s}";
pub const subject_prompt = "What is this question asking about? Reply with just that phrase, word for word as the question says it.\n{s}";
pub const kinds_prompt = "List six other things of the same kind as \"{s}\" that someone could ask about in the same words, each a short phrase. One per line, nothing else.";
pub const same_prompt = "Two replies to the question \"{s}\":\nA: {s}\nB: {s}\nDoes B tell the user something about themselves, or answer the question, that A does not? A reply that only describes the assistant tells the user nothing. Reply yes or no.";
pub const judge_prompt = "The user told you this about themselves or their work: \"{s}\" Does it answer this question they ask you: \"{s}\"? Reply yes or no.";
pub const facts_prompt = "Read the text below and list the facts in it worth remembering later: names, numbers, versions, dates, decisions and news. Write each as one short sentence that makes sense on its own. One per line, nothing else.\n\n{s}";
