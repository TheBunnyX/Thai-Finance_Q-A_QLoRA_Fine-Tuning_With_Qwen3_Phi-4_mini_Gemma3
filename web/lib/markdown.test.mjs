import assert from "node:assert/strict";
import { test } from "node:test";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";
import { normalizeMath } from "./markdown.ts";

test("formats model Markdown and LaTeX without treating text as HTML", async () => {
  const input = String.raw`### สูตร

**Current Ratio**

\[ \frac{\text{สินทรัพย์หมุนเวียน}}{\text{หนี้สินหมุนเวียน}} = \frac{500,000}{250,000} = 2 \]

- สินทรัพย์หมุนเวียน
- หนี้สินหมุนเวียน

<script>alert(1)</script>`;
  const result = String(await unified().use(remarkParse).use(remarkMath, { singleDollarTextMath: false }).use(remarkRehype)
    .use(rehypeKatex).use(rehypeStringify).process(normalizeMath(input)));
  assert.match(result, /<h3>สูตร<\/h3>/);
  assert.match(result, /<strong>Current Ratio<\/strong>/);
  assert.match(result, /class="katex-display"/);
  assert.doesNotMatch(result, /class="katex-error"/);
  assert.match(result, /<ul>/);
  assert.doesNotMatch(result, /<script>/);
});

test("leaves code literals untouched and supports inline math", () => {
  assert.equal(normalizeMath(String.raw`code \(x\) and \(y\)`), "code $$x$$ and $$y$$");
  assert.equal(normalizeMath(String.raw`\`\\[not-math\\]\``), String.raw`\`\\[not-math\\]\``);
});

const render = async (input) => String(await unified().use(remarkParse)
  .use(remarkMath, { singleDollarTextMath: false }).use(remarkRehype)
  .use(rehypeKatex).use(rehypeStringify).process(normalizeMath(input)));

test("keeps LaTeX line breaks with spacing inside display math", async () => {
  const input = String.raw`$$\begin{aligned} a &= b \\[4pt] c &= d \end{aligned}$$`;
  assert.equal(normalizeMath(input), input);
  const bracketed = String.raw`\[ \begin{aligned} a &= b \\[4pt] c &= d \end{aligned} \]`;
  const result = await render(bracketed);
  assert.match(result, /class="katex-display"/);
  assert.doesNotMatch(result, /class="katex-error"/);
});

test("leaves plain escaped brackets and dollar amounts as text", async () => {
  assert.equal(normalizeMath(String.raw`see \[1\]`), String.raw`see \[1\]`);
  const result = await render("revenue rose from $5 million to $10 million");
  assert.doesNotMatch(result, /katex/);
  assert.match(result, /\$5 million to \$10 million/);
});
