// Sliding Weights for omp: /learn writes what you give the served TensorFold model into its weights, live.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

const FRAMES = ["◐", "◓", "◑", "◒"];
const LIMIT = 24_000;
const SHOWN = 6;
const NO_AUTH = "N/A";

type State = "queued" | "learning" | "learned" | "missed";
type Fact = { id: number; text: string; state: State };
type Message = { role: string; content: unknown; toolName?: string };

const ICON: Record<Exclude<State, "learning">, string> = { queued: "·", learned: "✓", missed: "✗" };

function texts(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content.filter((b) => b?.type === "text").map((b) => b.text as string).join("\n");
}

// The exchange as plain text: what was asked, said and returned by tools, capped so one turn stays one request.
function transcript(messages: Message[]): string {
	const parts = messages.map((m) => {
		const body = texts(m.content).trim();
		if (!body) return "";
		if (m.role === "toolResult") return `Tool ${m.toolName}: ${body.slice(0, 2_000)}`;
		return m.role === "user" ? `User: ${body}` : m.role === "assistant" ? `Assistant: ${body}` : "";
	});
	return parts.filter(Boolean).join("\n\n").slice(0, LIMIT);
}

async function* events(body: ReadableStream<Uint8Array>): AsyncGenerator<[string, Record<string, unknown>]> {
	const decoder = new TextDecoder();
	let buffer = "";
	for await (const chunk of body) {
		buffer += decoder.decode(chunk, { stream: true });
		for (let cut = buffer.indexOf("\n\n"); cut >= 0; cut = buffer.indexOf("\n\n")) {
			const frame = buffer.slice(0, cut);
			buffer = buffer.slice(cut + 2);
			let event = "message";
			let data = "";
			for (const line of frame.split("\n")) {
				if (line.startsWith("event: ")) event = line.slice(7);
				else if (line.startsWith("data: ")) data += line.slice(6);
			}
			if (data === "[DONE]") return;
			if (data) yield [event, JSON.parse(data)];
		}
	}
}

export default function (pi: ExtensionAPI) {
	const facts: Fact[] = [];
	let live = false;
	let frame = 0;
	let learned = 0;
	let running = 0;
	let ticker: ReturnType<ExtensionContext["setInterval"]> | undefined;

	function render(ctx: ExtensionContext) {
		const spin = FRAMES[frame++ % FRAMES.length];
		const rows = facts.slice(-SHOWN).map((f) => `${f.state === "learning" ? spin : ICON[f.state]} ${f.state}: ${f.text}`);
		ctx.ui.setWidget("sliding-weights", rows.length ? ["Sliding Weights", ...rows] : undefined);
	}

	function status(ctx: ExtensionContext) {
		ctx.ui.setStatus("sliding-weights", live ? `sliding weights: on, ${learned} learned` : undefined);
	}

	function endpoint(ctx: ExtensionContext): string | undefined {
		const base = ctx.model?.baseUrl;
		return base ? base.replace(/\/+$/, "") : undefined;
	}

	async function learn(ctx: ExtensionContext, text: string, source: string) {
		const base = endpoint(ctx);
		if (!base || !ctx.model) return ctx.ui.notify("Sliding Weights needs a TensorFold model selected", "warning");
		const key = await ctx.modelRegistry.getApiKey(ctx.model);
		running++;
		ticker ??= ctx.setInterval(() => render(ctx), 120);
		try {
			const res = await fetch(`${base}/slide/learn`, {
				method: "POST",
				headers: { "Content-Type": "application/json", ...(key && key !== NO_AUTH ? { Authorization: `Bearer ${key}` } : {}) },
				body: JSON.stringify({ text: text.slice(0, LIMIT), source }),
			});
			if (!res.ok || !res.body) return ctx.ui.notify(`Sliding Weights: ${res.status} ${await res.text()}`, "error");
			for await (const [event, data] of events(res.body)) {
				const fact = facts.find((f) => f.id === data.id);
				if (event === "fact") facts.push({ id: data.id as number, text: data.text as string, state: "queued" });
				else if (event === "learning" && fact) fact.state = "learning";
				else if (event === "learned" && fact) {
					fact.state = data.recalled ? "learned" : "missed";
					if (data.recalled) learned++;
				} else if (event === "saved") ctx.ui.notify(`Sliding Weights: saved into ${data.modules} weight tensors`, "info");
				else if (event === "failed") ctx.ui.notify(`Sliding Weights: ${data.message}`, "error");
				status(ctx);
			}
		} finally {
			if (--running === 0 && ticker) {
				ctx.clearTimer(ticker);
				ticker = undefined;
				render(ctx);
				ctx.setTimeout(() => running === 0 && ctx.ui.setWidget("sliding-weights", undefined), 8_000);
			}
		}
	}

	pi.registerCommand("learn", {
		description: "Sliding Weights: /learn (live this session), /learn <facts>, /learn @file, /learn graph, /learn off",
		handler: async (args, ctx) => {
			const arg = args.trim();
			if (arg === "" || arg === "on") {
				live = true;
				ctx.ui.notify("Sliding Weights: learning from this session as you work", "info");
			} else if (arg === "off") live = false;
			else if (arg === "graph") {
				const base = endpoint(ctx);
				if (base) await pi.exec(process.platform === "darwin" ? "open" : "xdg-open", [`${new URL(base).origin}/slide`]);
			} else if (arg.startsWith("@")) await learn(ctx, await readFile(resolve(ctx.cwd, arg.slice(1)), "utf8"), `file:${arg.slice(1)}`);
			else await learn(ctx, arg, "chat");
			status(ctx);
		},
	});

	pi.on("agent_end", async (event, ctx) => {
		if (!live || event.willContinue) return;
		const text = transcript(event.messages as Message[]);
		if (text) void learn(ctx, text, "session");
	});
}
