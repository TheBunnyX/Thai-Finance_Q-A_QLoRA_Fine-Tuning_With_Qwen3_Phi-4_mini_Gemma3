/** Check the browser's actual Host header, not Next's internally rewritten request URL. */
export function validLocalOrigin(origin: string | null, host: string | null): boolean {
  // Non-browser local CLI clients typically omit Origin.
  if (!origin) return true;
  if (!host) return false;
  try {
    const parsed = new URL(origin);
    return parsed.protocol === "http:" && ["localhost", "127.0.0.1"].includes(parsed.hostname)
      && parsed.host.toLowerCase() === host.toLowerCase()
      && !parsed.username && !parsed.password && parsed.pathname === "/"
      && !parsed.search && !parsed.hash;
  } catch {
    return false;
  }
}
