"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { normalizeMath } from "../lib/markdown";

type Model = { base_context?: number; id: string; name: string; ready: boolean; model_id: string; run: string; context: number; quantization: string };
type Status = { models: Model[]; loaded: { model: string; variant: string } | null; busy: boolean; phase: string; evaluation_active: boolean };
type Metrics = { generated_tokens: number; prompt_tokens: number; generation_seconds: number; total_seconds: number; first_text_seconds: number | null; tokens_per_second: number; peak_cuda_gib: number; hit_limit: boolean; budget_used?: number };
type Check = { topic: string; label: string; expected_text: string; model_value: number | null; agrees: boolean | null };
type Answer = { id: number; question: string; text: string; model: string; variant: string; budget: number; limit: number; metrics?: Metrics; check?: Check | null; error?: string; stopped?: boolean };
const suggestions = [
  { category: "Financial ratios", title: "How is the Current Ratio calculated?", question: "บริษัทมีสินทรัพย์หมุนเวียน 500,000 บาท และหนี้สินหมุนเวียน 250,000 บาท Current Ratio เท่าไร และหมายความว่าอย่างไร?", icon: "↗" },
  { category: "Investment decisions", title: "What does a positive NPV tell us?", question: "NPV เป็นบวกหมายความว่าอย่างไร และมีข้อควรพิจารณาอะไรบ้างก่อนตัดสินใจลงทุน?", icon: "◇" },
  { category: "English Q&A", title: "Understanding Return on Equity", question: "What is Return on Equity? Explain its formula and limitations.", icon: "%" },
  { category: "Missing information", title: "Will the model ask for missing data?", question: "บริษัทมีกำไรสุทธิ 100,000 บาท ช่วยคำนวณ ROE ให้หน่อย", icon: "?" },
];
/** Say what the user can actually do: raise the budget, shorten the question, or switch model. */
function limitAdvice(answer: Answer): string {
  const used = answer.metrics?.budget_used ?? answer.budget;
  if (used < answer.budget) return `The answer reached ${used} tokens, the room left in the context after your question, and may be unfinished. Try a shorter question.`;
  if (answer.budget < answer.limit) return `The answer reached the ${answer.budget}-token limit and may be unfinished. Try raising the maximum answer length (up to ${answer.limit} tokens).`;
  return answer.variant === "base"
    ? `The answer reached the maximum of ${answer.limit} tokens. The base model tends to answer at length; try the SFT adapter, which answers more concisely.`
    : `The answer reached the maximum of ${answer.limit} tokens of the context this adapter was trained with, which is unusually long for this model. The question may be outside the trained formats; try asking more specifically.`;
}
const format = (value: number | undefined, digits = 2) => value === undefined ? "—" : value.toFixed(digits);

function Glyph({ children }: { children: React.ReactNode }) { return <span className="glyph" aria-hidden="true">{children}</span>; }

export default function Playground() {
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState(false);
  const [model, setModel] = useState("");
  const [variant, setVariant] = useState("sft");
  const [budget, setBudget] = useState(384);
  const [question, setQuestion] = useState("");
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [notice, setNotice] = useState("");
  const [unloading, setUnloading] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const latest = useRef<HTMLDivElement>(null);
  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error();
      setStatus(await response.json()); setOffline(false);
    } catch { setOffline(true); }
  }, []);
  useEffect(() => { refresh(); const timer = setInterval(refresh, 5000); return () => clearInterval(timer); }, [refresh]);
  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 100);
    return () => clearInterval(timer);
  }, [running]);
  useEffect(() => () => controller.current?.abort(), []);
  const selected = status?.models.find(item => item.id === model);
  // Pick the first adapter that is ready when nothing (or a missing model) is selected.
  useEffect(() => {
    if (!status || status.models.some(item => item.id === model && item.ready)) return;
    const first = status.models.find(item => item.ready);
    if (first) setModel(first.id);
  }, [status, model]);
  // Prompt + answer must fit the context the adapter was trained with, so cap the slider.
  const context = (variant === "base" ? selected?.base_context : selected?.context) || 512;
  const maxBudget = Math.max(64, Math.min(768, context - 64));
  useEffect(() => { setBudget(value => Math.min(value, maxBudget)); }, [maxBudget]);
  const blocked = offline || !selected?.ready || status?.evaluation_active || (status?.busy && !running);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || running || blocked) return;
    const id = Date.now();
    const item: Answer = { id, question: question.trim(), text: "", model, variant, budget, limit: maxBudget };
    setAnswers(previous => [...previous, item]); setQuestion(""); setNotice("");
    setRunning(true); setElapsed(0); setStage("Connecting…");
    controller.current = new AbortController();
    const update = (fields: Partial<Answer>) => setAnswers(previous => previous.map(answer => answer.id === id ? { ...answer, ...fields } : answer));
    let text = "", completed = false;
    setTimeout(() => latest.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, variant, question: item.question, max_new_tokens: budget }), signal: controller.current.signal });
      if (!response.ok) { const data = await response.json(); throw new Error(data.error || "Request failed"); }
      if (!response.body) throw new Error("No data received from the model");
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let pending = "";
      function consume(line: string) {
        if (!line.trim()) return;
        const message = JSON.parse(line);
        if (message.type === "stage") setStage(message.message);
        if (message.type === "token") { text += message.text; update({ text }); }
        if (message.type === "error") throw new Error(message.message);
        if (message.type === "done") { completed = true; update({ text: message.answer, metrics: message.metrics, check: message.check }); }
      }
      while (true) {
        const { done, value } = await reader.read();
        pending += done ? decoder.decode() : decoder.decode(value, { stream: true });
        const lines = pending.split("\n"); pending = lines.pop() || "";
        for (const line of lines) consume(line);
        if (done) { if (pending.trim()) consume(pending); break; }
      }
      if (!completed) throw new Error("The connection ended before the model finished. Please try again.");
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") update({ stopped: true });
      else update({ error: error instanceof Error ? error.message : "An error occurred" });
      controller.current?.abort();
    } finally { setRunning(false); setStage(""); controller.current = null; refresh(); }
  }

  function exportAnswers() {
    const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), turns_are_independent: true, answers }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob), anchor = document.createElement("a");
    anchor.href = url; anchor.download = `finance-lab-${Date.now()}.json`; anchor.click(); URL.revokeObjectURL(url);
  }

  async function unload() {
    setUnloading(true); setNotice("");
    try {
      const response = await fetch("/api/unload", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setNotice("Model memory released"); refresh();
    } catch (error) { setNotice(error instanceof Error ? error.message : "Could not release memory"); }
    finally { setUnloading(false); }
  }

  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="Finance Lab home"><span className="brand-icon">f<span>↗</span></span><span>finance<span className="brand-light">lab</span><small>LOCAL MODEL WORKSPACE</small></span></a>
      <div className="workspace-label">WORKSPACE <span>01</span></div>
      <div className="nav-active"><Glyph>▧</Glyph> Model Playground <span className="nav-dot" /></div>
      <div className="sidebar-rule" />
      <div className="section-caption">Test settings</div>
      <label className="control-label" htmlFor="model">Model</label>
      <select id="model" value={model} onChange={event => setModel(event.target.value)} disabled={running || unloading}>
        {(status?.models || [{ id: "qwen", name: "Qwen3 · 4B", ready: true }]).map(item => <option key={item.id} value={item.id} disabled={!item.ready}>{item.name}{!item.ready ? " · not ready" : ""}</option>)}
      </select>
      <label className="control-label">Model version</label>
      <div className="segmented" role="group" aria-label="Model version">
        <button type="button" disabled={running} aria-pressed={variant === "base"} className={variant === "base" ? "chosen" : ""} onClick={() => setVariant("base")}>Base model</button>
        <button type="button" disabled={running} aria-pressed={variant === "sft"} className={variant === "sft" ? "chosen" : ""} onClick={() => setVariant("sft")}>SFT adapter</button>
      </div>
      <p className="control-help">{variant === "sft" ? `Fine-tuned adapter: ${selected?.name || "no model selected"}` : "Original model, same revision as used for training"}</p>
      <label className="control-label" htmlFor="tokens">Max answer length <b>{budget}</b></label>
      <input id="tokens" type="range" min="32" max={maxBudget} step="32" value={budget} disabled={running} onChange={event => setBudget(Number(event.target.value))} />
      <div className="range-labels"><span>32 tokens</span><span>{maxBudget} tokens</span></div>
      <div className="settings-note"><span>Decoding</span><strong>Greedy · deterministic</strong><span>Context window</span><strong>{selected?.context || 512} tokens</strong></div>
      <div className="device-card"><div className="device-title"><span className={`status-dot ${offline ? "offline" : ""}`} /> LOCAL INFERENCE</div><strong>RTX 5070 <span>12 GB</span></strong><p>QLoRA 4-bit NF4 · BF16</p><div className="device-footer">{offline ? "Python API not connected" : status?.busy ? "GPU busy" : status?.loaded ? `Loaded ${status.loaded.model.toUpperCase()} · ${status.loaded.variant.toUpperCase()}` : "Model loads on the first question"}</div></div>
      <button className="unload" onClick={unload} disabled={running || unloading || offline || !status?.loaded || status?.busy}>{unloading ? "Releasing memory…" : "↙  Release GPU memory"}</button>
      <div className="sidebar-bottom">FINANCE Q&A EXPERIMENT <span>v0.1</span></div>
    </aside>

    <div className="main-shell">
      <header className="topbar"><div><span className="breadcrumb">Workspace</span><span className="slash">/</span><strong>Playground</strong></div><div className="connection"><span className={`status-dot ${offline ? "offline" : ""}`} />{offline ? "API offline" : status ? "Connected locally" : "Connecting"}<span className="local-badge">LOCAL</span></div></header>
      <main>
        <div className="page-heading"><div><div className="eyebrow">EXPLORE YOUR FINE-TUNED MODELS</div><h1>Model Playground<span>.</span></h1><p>Ask, compare and explore finance answers from your fine-tuned models</p></div><div className="heading-mark" aria-hidden="true">↗</div></div>
        <div className="model-strip"><div><span className="tiny-model-icon">◈</span><strong>{selected?.name || "Qwen3 · 4B"}</strong><span className="tag">{variant === "sft" ? "FINE-TUNED" : "BASE MODEL"}</span></div><span className="strip-meta">{variant === "sft" ? "1 epoch · Finance Q&A" : "Original checkpoint"}<span className="dot-separator">·</span>4-bit NF4</span></div>

        {offline && <div className="alert" role="alert"><strong>The Python API is not connected</strong><span>Open a terminal in the project folder and run <code>.venv/bin/python -m finance_qlora.playground</code></span><button onClick={refresh}>Retry connection ↻</button></div>}
        {status?.evaluation_active && <div className="alert" role="status">Training or evaluation is using the GPU. Questions are available once it finishes.</div>}
        {notice && <div className="notice" role="status">{notice}</div>}
        <section className="playground" aria-label="Test finance questions">
          <div className="panel-top"><span><span className="panel-dot" /> Playground</span><div><button onClick={exportAnswers} disabled={!answers.length || running}>↓ Export JSON</button><button onClick={() => { setAnswers([]); setNotice(""); }} disabled={!answers.length || running}>Clear answers</button></div></div>
          {!answers.length ? <div className="empty-state"><div className="spark-icon">✳</div><h2>Start with a question you want answered</h2><p>Test understanding, calculation formulas and financial reasoning<br />Pick an example question or type your own below</p><div className="suggestions">{suggestions.map(item => <button key={item.title} onClick={() => { setQuestion(item.question); input.current?.focus(); }}><span className="suggestion-icon">{item.icon}</span><small>{item.category}</small><strong>{item.title}</strong><span className="suggestion-arrow">↗</span></button>)}</div><div className="private-note"><span>⌁</span> Runs on your machine · questions are not sent to any external AI service</div></div>
          : <div className="answers">{answers.map((answer, index) => <article key={answer.id} ref={index === answers.length - 1 ? latest : undefined} className="answer-turn"><div className="question-row"><span className="avatar user-avatar">You</span><div><div className="turn-label">Question <span>{answer.model.toUpperCase()} / {answer.variant.toUpperCase()} · {answer.budget} tokens</span></div><p>{answer.question}</p></div></div><div className="response-row"><span className="avatar model-avatar">f↗</span><div className="response-content"><div className="turn-label">Finance Lab <span>Model answer</span><button className="copy-button" disabled={!answer.text} onClick={async () => { try { await navigator.clipboard.writeText(answer.text); setNotice("Answer copied"); } catch { setNotice("Copy failed. Select the text to copy it manually."); } }}>Copy</button></div><div className="answer-text">{answer.text ? <ReactMarkdown remarkPlugins={[[remarkMath, { singleDollarTextMath: false }]]} rehypePlugins={[rehypeKatex]}>{normalizeMath(answer.text)}</ReactMarkdown> : (running && index === answers.length - 1 ? <span className="thinking">{stage}<span className="loading-dots">•••</span></span> : "No answer text yet")}</div>{answer.error && <p className="answer-error" role="alert">{answer.error}</p>}{answer.check && <p className={"calc-check " + (answer.check.agrees === false ? "calc-bad" : answer.check.agrees ? "calc-ok" : "calc-unknown")} role="status"><b>Calculator check:</b> {answer.check.label} = {answer.check.expected_text} {answer.check.agrees === false ? "· the model's final number does not match; use this value instead" : answer.check.agrees ? "· matches the model's answer" : "· could not read a number in the model's answer"} <span className="muted">(numbers are read in the order they appear in the question; check they match your intent)</span></p>}{answer.stopped && <p className="muted">Request stopped · the answer may be incomplete</p>}{answer.metrics && <><div className="result-metrics"><span><b>{answer.metrics.generated_tokens}</b> tokens</span><span><b>{format(answer.metrics.generation_seconds)}</b> s</span><span><b>{format(answer.metrics.tokens_per_second, 1)}</b> tokens/s</span><span><b>{format(answer.metrics.first_text_seconds ?? undefined)}</b> s to first text</span><span><b>{format(answer.metrics.peak_cuda_gib)}</b> GiB peak</span></div>{answer.metrics.hit_limit && <p className="limit-note">{limitAdvice(answer)}</p>}</>}</div></div></article>)}</div>}
          <div className="composer-area">{running && <div className="generation-status" role="status"><span className="spinner" />{stage}<span>{elapsed.toFixed(1)} s</span></div>}<form onSubmit={submit} className="composer"><label className="sr-only" htmlFor="question">Finance question</label><textarea ref={input} id="question" rows={2} maxLength={6000} placeholder="Ask your finance question here…" value={question} disabled={running} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (!running) event.currentTarget.form?.requestSubmit(); } }} /><div className="composer-footer"><span>Enter to send <span className="keyboard-dot">·</span> Shift + Enter for a new line</span>{running ? <button type="button" className="stop-button" onClick={() => controller.current?.abort()}>■ Stop</button> : <button type="submit" className="send-button" disabled={!question.trim() || Boolean(blocked)}>Send <span>↑</span></button>}</div></form><p className="composer-note">Each question is independent; no earlier conversation is sent to the model.</p></div>
        </section>
        <footer className="disclaimer"><span className="info-icon">i</span><p><strong>Model playground</strong> Answers may contain errors. Check formulas, numbers and sources before using them for financial decisions.<br /><span>First-text time is measured when the streamer first decodes text; it is not the TTFT metric used in the evaluation.</span></p></footer>
      </main>
    </div>
  </div>;
}
