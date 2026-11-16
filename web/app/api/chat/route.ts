export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  if (!validLocalOrigin(origin, request.headers.get("host"))) {
    return Response.json({ error: "Invalid origin" }, { status: 403 });
  }
  try {
    const body = await request.text();
    if (new TextEncoder().encode(body).length > 32768) return Response.json({ error: "Request too long" }, { status: 413 });
    const upstream = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body, signal: request.signal,
    });
    return new Response(upstream.body, { status: upstream.status, headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json",
      "Cache-Control": "no-store", "X-Accel-Buffering": "no",
    } });
  } catch {
    return Response.json({ error: "Could not reach the Python API. Check that it is still running." }, { status: 503 });
  }
}
import { validLocalOrigin } from "../../../lib/origin";

