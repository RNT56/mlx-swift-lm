#!/usr/bin/env python3
"""Build + tokenize lever-① acceptance prompts for a real MLX model.

Produces two JSON files of token-id prompts ([{"label","ids":[int]}]) for the
TurboQuantAcceptanceHarness --prompt-ids-file:

  * short prompts  — the 5 representative categories matching the prior 0.6B run
    (code-repetition, json-list, quote-continue, doc-edit, free-prose), so the
    4B speedup is directly comparable.
  * long prompts   — structured long-context prompts grown to target token
    counts (default 8K, 16K) where the weight stream dominates most.

Tokenize with the *actual* model snapshot's tokenizer (token IDs only ever go to
the model). Usage:

  python3 scripts/turboquant-build-acceptance-prompts.py \
      --model-dir <snapshot dir> \
      --short-out <prompts-short.json> \
      --long-out  <prompts-long.json> \
      --long-targets 8192,16384
"""
import argparse
import json

from transformers import AutoTokenizer

SHORT_PROMPTS = {
    "code-repetition": (
        "struct P { let x: Int; let y: Int }\n"
        "let a = P(x: 1, y: 2)\n"
        "let b = P(x: 3, y: 4)\n"
        "let c = P(x: 5, y: 6)\n"
        "let d = P(x: 7, y: 8)\n"
        "let e = P(x: "
    ),
    "json-list": (
        '{"items": [{"id":1,"name":"alpha","active":true},'
        '{"id":2,"name":"beta","active":true},'
        '{"id":3,"name":"gamma","active":true},'
        '{"id":4,"name":"delta","active":true},'
        '{"id":5,"name":'
    ),
    "quote-continue": (
        "Repeat this sentence verbatim three times. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
    ),
    "doc-edit": (
        "Original: The function computes the sum of two integers and returns the result.\n"
        "Revised: The function computes the sum of two integers and returns the"
    ),
    "free-prose": (
        "Write a short essay about the history and significance of lighthouse "
        "keeping in coastal communities, explaining how they"
    ),
}

# Building blocks for long-context prompts. Repeated until the target token count
# is reached, then the id stream is truncated to exactly the target so KV memory
# is predictable.
LONG_CODE_BLOCK = (
    "func process_record_{i}(input: Record) -> Result {{\n"
    "    let normalized = normalize(input.value)\n"
    "    let scaled = normalized * config.factor + config.offset\n"
    "    let clamped = max(config.lower, min(config.upper, scaled))\n"
    "    return Result(id: input.id, value: clamped, ok: true)\n"
    "}}\n\n"
)
LONG_DOC_BLOCK = (
    "Section {i}. The maintenance procedure for unit {i} begins with a visual "
    "inspection of the housing, followed by a torque check on all fasteners to "
    "the rated specification. Record the measured values in the log, compare "
    "them against the baseline tolerance band, and flag any deviation for "
    "review. Repeat the same sequence for the adjacent unit before proceeding.\n\n"
)


def build_long(tokenizer, block_template, target_tokens, header):
    text = header
    i = 0
    # Grow generously past target, then truncate ids exactly.
    while True:
        text += block_template.format(i=i)
        i += 1
        if i % 64 == 0:
            if len(tokenizer.encode(text)) >= target_tokens + 64:
                break
        if i > 200000:  # safety
            break
    ids = tokenizer.encode(text)[:target_tokens]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--short-out", required=True)
    ap.add_argument("--long-out", required=True)
    ap.add_argument("--long-targets", default="8192,16384")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)

    short = [{"label": k, "ids": tok.encode(v)} for k, v in SHORT_PROMPTS.items()]
    json.dump(short, open(args.short_out, "w"))
    print(f"short: wrote {len(short)} prompts -> {args.short_out}")
    for p in short:
        print(f"  {p['label']:16s} {len(p['ids']):5d} tokens")

    targets = [int(t) for t in args.long_targets.split(",") if t.strip()]
    long_prompts = []
    for t in targets:
        long_prompts.append(
            {"label": f"long-code-{t}",
             "ids": build_long(tok, LONG_CODE_BLOCK, t,
                               "// Auto-generated record processors.\n\n")})
        long_prompts.append(
            {"label": f"long-doc-{t}",
             "ids": build_long(tok, LONG_DOC_BLOCK, t,
                               "Maintenance Manual — Operations Reference.\n\n")})
    json.dump(long_prompts, open(args.long_out, "w"))
    print(f"long: wrote {len(long_prompts)} prompts -> {args.long_out}")
    for p in long_prompts:
        print(f"  {p['label']:16s} {len(p['ids']):6d} tokens")


if __name__ == "__main__":
    main()
