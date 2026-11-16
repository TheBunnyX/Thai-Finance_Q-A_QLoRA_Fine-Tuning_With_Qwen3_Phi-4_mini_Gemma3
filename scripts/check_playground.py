"""Smoke test the running Next.js proxy with a real local model (one GPU request)."""
import argparse
import json
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen-2500-v2-steps", help="model id from the playground /status list")
    parser.add_argument("--variant", choices=("base", "sft"), default="sft")
    parser.add_argument("--question", default="What is the current ratio?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--origin", help="Optional browser Origin header for proxy verification")
    args = parser.parse_args()
    body = json.dumps({"model": args.model, "variant": args.variant, "question": args.question,
                       "max_new_tokens": args.max_new_tokens}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.origin:
        headers["Origin"] = args.origin
    request = urllib.request.Request("http://127.0.0.1:3000/api/chat", data=body, headers=headers)
    completed = False
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            event = json.loads(line)
            if event["type"] == "error":
                raise RuntimeError(event["message"])
            if event["type"] != "heartbeat":
                print(json.dumps(event, ensure_ascii=False), flush=True)
            if event["type"] == "done":
                if not event["answer"].strip() or not event["metrics"]["generated_tokens"]:
                    raise RuntimeError("Model returned an empty answer")
                completed = True
    if not completed:
        raise RuntimeError("Stream ended before completion")


if __name__ == "__main__":
    main()
