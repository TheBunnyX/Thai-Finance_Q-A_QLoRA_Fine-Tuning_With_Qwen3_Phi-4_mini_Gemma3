// A LaTeX line break with optional spacing, e.g. `\\[4pt]`, is not a display-math opener.
const DIMEN = String.raw`\s*-?(?:\d+\.?\d*|\.\d+)\s*(?:pt|em|ex|mm|cm|in|bp|pc|dd|cc|sp|mu|px)\s*\]`;
const DISPLAY = new RegExp(String.raw`\\{1,2}\[(?!${DIMEN})([\s\S]+?)\\{1,2}\]`, "g");
const INLINE = /\\{1,2}\(([\s\S]+?)\\{1,2}\)/g;
// Skip escaped brackets such as `\[1\]` that are plain text, not math.
const MATHY = /[\\^_=+*/<>|]|\d\s*[a-zA-Z]|[a-zA-Z]\s*-\s*\w/;
// Code and existing $$…$$ math are protected from rewriting.
const PROTECTED = /(```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]+?\$\$)/g;

/**
 * Normalize common model LaTeX delimiters while preserving Markdown code.
 * Output uses `$$` only: render with remark-math `singleDollarTextMath: false`
 * so dollar amounts such as "$5 million" stay plain text.
 */
export function normalizeMath(text: string): string {
  return text.split(PROTECTED).map((part, index) => index % 2 ? part : part
    .replace(DISPLAY, (match, body: string) => MATHY.test(body) ? `\n$$\n${body.trim()}\n$$\n` : match)
    .replace(INLINE, (match, body: string) => MATHY.test(body) || /^\s*[a-zA-Z]\s*$/.test(body) ? `$$${body.trim()}$$` : match))
    .join("");
}
