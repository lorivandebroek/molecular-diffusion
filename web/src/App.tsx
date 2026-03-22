import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  apiDepict,
  apiHealth,
  apiPredict,
  apiSweep,
  finiteOr,
  type HealthResponse,
  type PredictResponse,
  type SweepPoint,
} from './api'

const DEBOUNCE_MS = 420
const DEFAULT_T_MIN = 273.75
const DEFAULT_T_MAX = 394

function fmtD(d: number): string {
  if (!Number.isFinite(d)) return '—'
  const a = Math.abs(d)
  if (a >= 0.001 && a < 1e6) return d.toExponential(4)
  return d.toExponential(3)
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)

  const tMin = useMemo(
    () => finiteOr(health?.temperature_min_k, DEFAULT_T_MIN),
    [health],
  )
  const tMax = useMemo(
    () => finiteOr(health?.temperature_max_k, DEFAULT_T_MAX),
    [health],
  )

  const [smilesA, setSmilesA] = useState('CC')
  const [smilesB, setSmilesB] = useState('')
  const [tempK, setTempK] = useState(298.15)

  const [predA, setPredA] = useState<PredictResponse | null>(null)
  const [predB, setPredB] = useState<PredictResponse | null>(null)
  const [predErr, setPredErr] = useState<string | null>(null)
  const [loadingPred, setLoadingPred] = useState(false)

  const [svgA, setSvgA] = useState<string | null>(null)
  const [svgB, setSvgB] = useState<string | null>(null)
  const [depictErr, setDepictErr] = useState<string | null>(null)

  const [seriesA, setSeriesA] = useState<SweepPoint[] | null>(null)
  const [seriesB, setSeriesB] = useState<SweepPoint[] | null>(null)
  const [chartErr, setChartErr] = useState<string | null>(null)
  const [loadingChart, setLoadingChart] = useState(false)

  useEffect(() => {
    apiHealth()
      .then(setHealth)
      .catch((e: Error) => setHealthErr(e.message))
  }, [])

  const clampTemp = useCallback(
    (t: number) => Math.min(tMax, Math.max(tMin, t)),
    [tMin, tMax],
  )

  useEffect(() => {
    setTempK((t) => clampTemp(t))
  }, [clampTemp, tMin, tMax])

  useEffect(() => {
    const id = setTimeout(() => {
      void (async () => {
        const ta = clampTemp(tempK)
        setLoadingPred(true)
        setPredErr(null)
        try {
          const a = await apiPredict(smilesA.trim(), ta)
          setPredA(a)
          if (smilesB.trim()) {
            const b = await apiPredict(smilesB.trim(), ta)
            setPredB(b)
          } else {
            setPredB(null)
          }
        } catch (e) {
          setPredA(null)
          setPredB(null)
          setPredErr(e instanceof Error ? e.message : String(e))
        } finally {
          setLoadingPred(false)
        }
      })()
    }, DEBOUNCE_MS)
    return () => clearTimeout(id)
  }, [smilesA, smilesB, tempK, clampTemp])

  useEffect(() => {
    const id = setTimeout(() => {
      void (async () => {
        setDepictErr(null)
        try {
          const da = await apiDepict(smilesA.trim())
          setSvgA(da.svg)
          if (smilesB.trim()) {
            const db = await apiDepict(smilesB.trim())
            setSvgB(db.svg)
          } else {
            setSvgB(null)
          }
        } catch (e) {
          setSvgA(null)
          setSvgB(null)
          setDepictErr(e instanceof Error ? e.message : String(e))
        }
      })()
    }, DEBOUNCE_MS)
    return () => clearTimeout(id)
  }, [smilesA, smilesB])

  const plotChart = async () => {
    setChartErr(null)
    setLoadingChart(true)
    setSeriesA(null)
    setSeriesB(null)
    try {
      const sa = await apiSweep(smilesA.trim(), tMin, tMax, 56)
      setSeriesA(sa.points)
      if (smilesB.trim()) {
        const sb = await apiSweep(smilesB.trim(), tMin, tMax, 56)
        setSeriesB(sb.points)
      }
    } catch (e) {
      setChartErr(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingChart(false)
    }
  }

  const chartData = useMemo(() => {
    if (!seriesA?.length) return []
    return seriesA.map((p, i) => ({
      T: p.T,
      A: p.log10_D,
      B: seriesB?.[i]?.log10_D,
    }))
  }, [seriesA, seriesB])

  return (
    <div className="app">
      <header className="header">
        <h1>Aqueous diffusion demo</h1>
        <p className="lede">
          SMILES + temperature (K) → predicted{' '}
          <span className="mono">D</span> in water (trained model). For
          exploration only.
        </p>
        {healthErr && (
          <p className="banner err">
            API unreachable ({healthErr}). Start the server:{' '}
            <code className="mono">uvicorn demo_api.main:app --port 8000</code>
          </p>
        )}
        {health && (
          <p className="meta mono">
            {health.model_type} · training T [{tMin.toFixed(1)}, {tMax.toFixed(1)}
            ] K
          </p>
        )}
      </header>

      <section className="panel">
        <label className="label">SMILES A</label>
        <input
          className="input"
          value={smilesA}
          onChange={(e) => setSmilesA(e.target.value)}
          spellCheck={false}
          autoComplete="off"
        />

        <label className="label">SMILES B (optional face-off)</label>
        <input
          className="input"
          value={smilesB}
          onChange={(e) => setSmilesB(e.target.value)}
          placeholder="Leave empty for single molecule"
          spellCheck={false}
          autoComplete="off"
        />

        <div className="sliderRow">
          <label className="label">
            Temperature (K):{' '}
            <span className="mono">{clampTemp(tempK).toFixed(2)}</span>
          </label>
          <input
            type="range"
            min={tMin}
            max={tMax}
            step={0.25}
            value={clampTemp(tempK)}
            onChange={(e) => setTempK(Number(e.target.value))}
            disabled={!health}
          />
        </div>

        {predErr && <p className="banner err">{predErr}</p>}
        {predA && (
          <div className="grid2">
            <div className="card">
              <h2 className="h2">A @ {clampTemp(tempK).toFixed(2)} K</h2>
              <p className="mono big">log₁₀ D = {predA.log10_D.toFixed(4)}</p>
              <p className="mono big">D = {fmtD(predA.D)}</p>
              {predA.extrapolated && (
                <p className="banner warn">
                  Outside training T range — extrapolation.
                </p>
              )}
            </div>
            {predB && (
              <div className="card">
                <h2 className="h2">B @ {clampTemp(tempK).toFixed(2)} K</h2>
                <p className="mono big">log₁₀ D = {predB.log10_D.toFixed(4)}</p>
                <p className="mono big">D = {fmtD(predB.D)}</p>
                {predB.extrapolated && (
                  <p className="banner warn">
                    Outside training T range — extrapolation.
                  </p>
                )}
              </div>
            )}
          </div>
        )}
        {loadingPred && <p className="muted">Updating prediction…</p>}
      </section>

      <section className="panel row2">
        <div className="depict">
          <h2 className="h2">Structure A</h2>
          {depictErr && <p className="banner err">{depictErr}</p>}
          {svgA && (
            <div
              className="svgBox"
              dangerouslySetInnerHTML={{ __html: svgA }}
            />
          )}
        </div>
        {smilesB.trim() ? (
          <div className="depict">
            <h2 className="h2">Structure B</h2>
            {svgB && (
              <div
                className="svgBox"
                dangerouslySetInnerHTML={{ __html: svgB }}
              />
            )}
          </div>
        ) : (
          <div className="depict muted">Add SMILES B to compare structures.</div>
        )}
      </section>

      <section className="panel">
        <h2 className="h2">log₁₀ D vs temperature</h2>
        <p className="muted">
          Sweeps the training T span ({tMin.toFixed(0)}–{tMax.toFixed(0)} K).
        </p>
        <button
          type="button"
          className="btn"
          onClick={() => void plotChart()}
          disabled={loadingChart || !health}
        >
          {loadingChart ? 'Computing…' : 'Plot / refresh chart'}
        </button>
        {chartErr && <p className="banner err">{chartErr}</p>}
        {chartData.length > 0 && (
          <div className="chartWrap">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart
                data={chartData}
                margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
              >
                <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="T"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tick={{ fill: 'var(--muted)' }}
                  label={{
                    value: 'T (K)',
                    position: 'insideBottom',
                    offset: -4,
                  }}
                />
                <YAxis
                  tick={{ fill: 'var(--muted)' }}
                  label={{
                    value: 'log₁₀ D',
                    angle: -90,
                    position: 'insideLeft',
                  }}
                />
                <Tooltip
                  contentStyle={{
                    background: 'var(--card)',
                    border: '1px solid var(--border)',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="A"
                  name="A"
                  stroke="var(--accent)"
                  dot={false}
                  strokeWidth={2}
                />
                {seriesB?.length ? (
                  <Line
                    type="monotone"
                    dataKey="B"
                    name="B"
                    stroke="var(--accent2)"
                    dot={false}
                    strokeWidth={2}
                  />
                ) : null}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>
    </div>
  )
}
