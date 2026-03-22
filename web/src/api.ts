export type HealthResponse = {
  ok: boolean
  model_type: string
  checkpoint: string
  temperature_min_k: number | null
  temperature_max_k: number | null
}

export type PredictResponse = {
  log10_D: number
  D: number
  extrapolated: boolean
  message?: string | null
}

export type SweepPoint = {
  T: number
  log10_D: number
  D: number
}

export type SweepResponse = {
  points: SweepPoint[]
}

export type DepictResponse = {
  svg: string
}

export function finiteOr(
  x: number | null | undefined,
  fallback: number,
): number {
  return typeof x === 'number' && Number.isFinite(x) ? x : fallback
}

export async function apiHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error(await readErr(res))
  return res.json()
}

export async function apiPredict(
  smiles: string,
  temperature_k: number,
): Promise<PredictResponse> {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles, temperature_k }),
  })
  if (!res.ok) throw new Error(await readErr(res))
  return res.json()
}

export async function apiSweep(
  smiles: string,
  t_min_k: number,
  t_max_k: number,
  steps: number,
): Promise<SweepResponse> {
  const res = await fetch('/api/sweep', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      smiles,
      t_min_k,
      t_max_k,
      steps: Math.min(100, Math.max(2, steps)),
    }),
  })
  if (!res.ok) throw new Error(await readErr(res))
  return res.json()
}

export async function apiDepict(
  smiles: string,
  width = 220,
  height = 180,
): Promise<DepictResponse> {
  const res = await fetch('/api/depict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles, width, height }),
  })
  if (!res.ok) throw new Error(await readErr(res))
  return res.json()
}

async function readErr(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: unknown }
    const d = j.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d))
      return d.map((e) => (e as { msg?: string }).msg ?? String(e)).join('; ')
    return res.statusText
  } catch {
    return res.statusText
  }
}
