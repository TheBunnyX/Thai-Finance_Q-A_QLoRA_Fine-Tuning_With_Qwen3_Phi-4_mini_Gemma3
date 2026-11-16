export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const response = await fetch("http://127.0.0.1:8000/status", { cache: "no-store", signal: AbortSignal.timeout(3000) });
    return new Response(response.body, { status: response.status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
  } catch {
    return Response.json({ error: "The Python API is not running. See the start commands in the README." }, { status: 503 });
  }
}
