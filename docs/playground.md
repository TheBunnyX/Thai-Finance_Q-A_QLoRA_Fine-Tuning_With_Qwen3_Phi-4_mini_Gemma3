# Finance Lab — Next.js playground

Try the real models on your own machine through a Next.js App Router front end and a Python API that
runs in the project `.venv`. Pick any of the nine 2500-run adapters (Qwen3 / Gemma 3 / Phi-4 mini ×
`v2-steps` / `v2` / `v2-cap15`) and switch between the base model and the SFT adapter.
The API reads only runs whose status is `complete` and whose adapter exists on disk.

## Start it (Linux)

Needs Node.js >= 20.9 and the training environment (see the main README). From the project root, in two terminals:

```bash
# Terminal 1 — Python API (the model loads onto the GPU on the first question)
HF_HUB_OFFLINE=1 .venv/bin/python -m finance_qlora.playground --port 8000
```

```bash
# Terminal 2 — web UI
cd web
npm ci            # first time only; if node_modules came from Windows, delete it first
npm run dev       # opens http://127.0.0.1:3000
```

The API uses port 8000 and the web UI port 3000; both bind to 127.0.0.1.
The Next.js proxy points at port 8000 and that is fixed; if you change the Python port you must change the API routes to match.
For a local production build: `npm run build`, then `npm start` in `web`.
This is a dev server for local experiments, not a public deployment.

`scripts/start_playground.ps1` is the older Windows launcher (PowerShell) and is not used on Linux.

## Using it

- Choose a model and a version (base model or SFT adapter), then pick an example question or type your own
- Enter sends, Shift+Enter inserts a new line. The maximum answer length follows the context window
  (SFT 512, base model 1,024, capped at 768); the default is 384 tokens
- Answers support Markdown and LaTeX math as `\[ ... \]`, `\( ... \)` and `$$...$$`
- Answers stream in, with the loading / generating stage and the elapsed time
- Each question is independent and uses the same native chat template as the pipeline; no history is sent into the context
- If the question plus the answer budget exceeds the context, the answer budget is shrunk to fit instead of rejecting the request;
  an error is raised only when fewer than 32 tokens are left for the answer. Nothing is truncated silently
- Switching the model or the version releases the current model and loads the new one; only one model is kept on the GPU at a time
- The stop button cancels the stream; the backend stops generation through a stopping criterion when it sees the disconnect.
  While a model is loading, the disconnect may only be noticed after the load finishes
- Export JSON stores the questions, answers, model versions, token budgets and metrics; questions are never saved to disk automatically
- The release-GPU button frees the cached model

## Scope and metrics

- Greedy decoding, deterministic, no sampling; this does not change the training config or any earlier benchmark
- Generation time includes the streamer overhead; the total includes loading when the model was switched
- First text is the first text that could be decoded; it is not the generated-token TTFT of the evaluation
- CUDA peak is the PyTorch-allocated peak of this process, not the total VRAM shown by a system monitor
- If an answer reaches the token ceiling a notice appears; check that the content is complete before interpreting it
- A GPU lock rejects overlapping requests with 409 and checks the heartbeat of a running training / evaluation job before accepting inference.
  This is not a general GPU scheduler, so do not run other training or GPU scripts alongside the playground
- Hugging Face login uses the account already on the machine; no token is entered in the web UI
- Model files may be downloaded from Hugging Face when loading; questions are not sent to any external inference API
- The page uses Google Fonts with a system fallback
- It is built for a personal machine with no user accounts or production authentication, so do not expose the ports to a network
- Answers still need their financial facts, formulas and units checked; the web page certifies nothing

## Checks after a change

```bash
.venv/bin/python -m unittest discover -s tests -v
cd web
npm run typecheck
npm run build
```

`npm test` imports `.ts` files directly and needs a Node build with built-in TypeScript support
(Node 24, or any build where `node -p "process.features.typescript"` is not `false`); the Ubuntu Node 22 package lacks it.

An end-to-end test through the Next.js proxy to a real model on the GPU, with the web UI and the API already running:

```bash
.venv/bin/python -X utf8 scripts/check_playground.py --model qwen-2500-v2-steps --variant sft
.venv/bin/python -X utf8 scripts/check_playground.py --model qwen-2500-v2-steps --variant base
```

This is a functional smoke test (stage / token / done events arrive through the API); it is not a visual UI check
and not a certification of financial correctness.

References: [Next.js App Router](https://nextjs.org/docs/app),
[Transformers streamers](https://huggingface.co/docs/transformers/internal/generation_utils)
