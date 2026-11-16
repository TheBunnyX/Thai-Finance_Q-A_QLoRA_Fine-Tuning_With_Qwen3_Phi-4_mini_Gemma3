export const runtime = "nodejs";
export async function POST(request: Request) {
  const origin = request.headers.get("origin");
  if (!validLocalOrigin(origin, request.headers.get("host"))) return Response.json({ error: "Invalid origin" }, { status: 403 });
  try {
    const response = await fetch("http://127.0.0.1:8000/unload", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}", signal: AbortSignal.timeout(15000),
    });
    return new Response(response.body, { status: response.status, headers: { "Content-Type": "application/json" } });
  } catch {
    return Response.json({ error: "Could not reach the API" }, { status: 503 });
  }
}
import { validLocalOrigin } from "../../../lib/origin";

