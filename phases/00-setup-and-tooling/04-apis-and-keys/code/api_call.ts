import { config } from "dotenv";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";

// Load repo-root .env when cwd is not the project root; cwd .env is also used if present.
const here = dirname(fileURLToPath(import.meta.url));
config({ path: join(here, "../../../../.env") });
config();

async function main() {
	const key = process.env.ANTHROPIC_API_KEY;
	if (!key) {
		throw new Error(
			"ANTHROPIC_API_KEY is not set. Add it to .env at the repo root (ANTHROPIC_API_KEY=...) or export it in your shell.",
		);
	}
	const client = new Anthropic({ apiKey: key });
	const response = await client.messages.create({
		model: "claude-sonnet-4-6",
		max_tokens: 256,
		messages: [
			{ role: "user", content: "What is a neural network in one sentence?" },
		],
	});
	const first = response.content[0];
	console.log(first.text);
}

main().catch(console.error);
