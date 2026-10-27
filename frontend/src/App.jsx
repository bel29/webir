import { useCallback, useEffect, useRef, useState } from 'react'

const CANALES = [
  { value: 'internet', label: 'En línea' },
  { value: 'persona', label: 'Presencial' },
  { value: 'telefono', label: 'Teléfono' },
]

function Badge({ children, color = 'gray' }) {
  const colors = {
    gray: 'bg-gray-100 text-gray-700',
    green: 'bg-emerald-100 text-emerald-700',
    blue: 'bg-blue-100 text-blue-700',
    amber: 'bg-amber-100 text-amber-800',
  }
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  )
}

function ResultCard({ doc }) {
  return (
    <article className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm hover:shadow-md transition-shadow">
      <a href={doc.url} target="_blank" rel="noreferrer"
         className="text-lg font-semibold text-blue-700 hover:underline leading-snug">
        {doc.nombre}
      </a>
      <p className="mt-1 text-sm text-gray-500">
        {doc.organismo} — {doc.organismo_padre}
      </p>
      {doc.highlight ? (
        <p className="mt-2 text-sm text-gray-700"
           dangerouslySetInnerHTML={{ __html: doc.highlight }} />
      ) : (
        <p className="mt-2 text-sm text-gray-700 line-clamp-3">{doc.descripcion}</p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {doc.tiene_costo
          ? <Badge color="amber">{doc.costo ? `Costo: ${doc.costo}` : 'Con costo'}</Badge>
          : <Badge color="green">Gratuito / no informado</Badge>}
        {doc.canales.map((c) => (
          <Badge key={c} color="blue">{CANALES.find((x) => x.value === c)?.label ?? c}</Badge>
        ))}
        <span className="ml-auto text-xs text-gray-400">score {doc.score}</span>
      </div>
      {doc.url_tramite && doc.url_tramite.startsWith('http') && (
        <a href={doc.url_tramite} target="_blank" rel="noreferrer"
           className="mt-2 inline-block text-sm font-medium text-emerald-700 hover:underline">
          Iniciar trámite en línea →
        </a>
      )}
    </article>
  )
}

function ChatAnswer({ chat }) {
  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-blue-200 bg-blue-50 p-5">
        <p className="whitespace-pre-wrap text-gray-800">{chat.answer}</p>
        {!chat.generated && chat.answer && (
          <p className="mt-2 text-xs text-gray-400">respuesta extractiva (sin LLM)</p>
        )}
      </div>
      {chat.sources?.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
            Fuentes
          </h3>
          <ul className="space-y-1">
            {chat.sources.map((s, i) => (
              <li key={i}>
                <a href={s.url} target="_blank" rel="noreferrer"
                   className="text-sm text-blue-700 hover:underline">
                  [{i + 1}] {s.nombre}
                </a>
                <span className="ml-1 text-xs text-gray-400">— {s.organismo}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [data, setData] = useState(null)
  const [chat, setChat] = useState(null)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('search') // 'search' | 'chat'
  const [organismo, setOrganismo] = useState(null)
  const [canal, setCanal] = useState(null)
  const [gratuito, setGratuito] = useState(false)
  const [hybrid, setHybrid] = useState(true)
  const [doRerank, setDoRerank] = useState(false)
  const debounce = useRef(null)

  const doSearch = useCallback((q, filtros) => {
    if (!q.trim()) { setData(null); return }
    setLoading(true)
    const params = new URLSearchParams({ q, top_k: 20 })
    if (filtros.organismo) params.set('organismo', filtros.organismo)
    if (filtros.canal) params.set('canal', filtros.canal)
    if (filtros.gratuito) params.set('gratuito', 'true')
    if (filtros.hybrid) params.set('hybrid', 'true')
    if (filtros.doRerank) params.set('do_rerank', 'true')
    fetch(`/api/search?${params}`)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ error: true, results: [], facets: {} }))
      .finally(() => setLoading(false))
  }, [])

  const doChat = useCallback((q) => {
    if (!q.trim()) { setChat(null); return }
    setLoading(true)
    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: q }),
    })
      .then((r) => r.json())
      .then(setChat)
      .catch(() => setChat({ answer: 'Error al consultar la API.', sources: [] }))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    clearTimeout(debounce.current)
    if (mode === 'search') {
      debounce.current = setTimeout(
        () => doSearch(query, { organismo, canal, gratuito, hybrid, doRerank }), 300)
    }
    return () => clearTimeout(debounce.current)
  }, [query, organismo, canal, gratuito, hybrid, doRerank, mode, doSearch])

  const facets = data?.facets ?? { organismos: [], canales: [] }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-5xl px-4 py-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Trámites <span className="text-blue-700">Uruguay</span>
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Buscador sobre el catálogo de trámites y servicios del Estado
            {data?.backend && (
              <span className="ml-2 text-xs text-gray-400">· motor: {data.backend}</span>
            )}
          </p>
          <div className="mt-4 flex gap-2">
            {[{ v: 'search', l: 'Buscar' }, { v: 'chat', l: 'Preguntar' }].map((m) => (
              <button key={m.v} onClick={() => setMode(m.v)}
                      className={`rounded-full px-4 py-1.5 text-sm font-medium ${
                        mode === m.v
                          ? 'bg-blue-700 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
                {m.l}
              </button>
            ))}
            {mode === 'search' && (
              <div className="ml-auto flex items-center gap-3 text-sm text-gray-500">
                <label className="flex cursor-pointer items-center gap-1.5">
                  <input type="checkbox" checked={hybrid}
                         onChange={() => setHybrid(!hybrid)} className="rounded" />
                  híbrida
                </label>
                <label className="flex cursor-pointer items-center gap-1.5">
                  <input type="checkbox" checked={doRerank}
                         onChange={() => setDoRerank(!doRerank)} className="rounded" />
                  rerank
                </label>
              </div>
            )}
          </div>
          <input
            autoFocus
            value={query}
            onChange={(e) => { setQuery(e.target.value); setOrganismo(null) }}
            onKeyDown={(e) => { if (e.key === 'Enter' && mode === 'chat') doChat(query) }}
            placeholder={mode === 'search'
              ? 'Ej: sacar pasaporte, licencia de conducir, pagar multa…'
              : 'Preguntá en lenguaje natural: ¿qué necesito para viajar al exterior?'}
            className="mt-4 w-full rounded-xl border border-gray-300 bg-white px-5 py-3.5 text-lg shadow-sm outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
          />
          {mode === 'chat' && (
            <button onClick={() => doChat(query)}
                    className="mt-3 rounded-xl bg-blue-700 px-6 py-2.5 font-medium text-white hover:bg-blue-800">
              Preguntar
            </button>
          )}
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl gap-6 px-4 py-6">
        {mode === 'chat' ? (
          <section className="min-w-0 flex-1">
            {loading && <p className="text-sm text-gray-400">Pensando…</p>}
            {chat && <ChatAnswer chat={chat} />}
            {!chat && !loading && (
              <p className="text-sm text-gray-400">
                Escribí una pregunta y presioná Enter o el botón Preguntar.
              </p>
            )}
          </section>
        ) : (
        <>
        {data && !data.error && (
          <aside className="w-56 shrink-0 space-y-6">
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                Canal de atención
              </h3>
              <div className="space-y-1">
                {CANALES.map((c) => (
                  <label key={c.value} className="flex cursor-pointer items-center gap-2 text-sm text-gray-700">
                    <input type="checkbox" checked={canal === c.value}
                           onChange={() => setCanal(canal === c.value ? null : c.value)}
                           className="rounded border-gray-300" />
                    {c.label}
                  </label>
                ))}
                <label className="flex cursor-pointer items-center gap-2 text-sm text-gray-700">
                  <input type="checkbox" checked={gratuito}
                         onChange={() => setGratuito(!gratuito)}
                         className="rounded border-gray-300" />
                  Solo gratuitos
                </label>
              </div>
            </div>
            {facets.organismos.length > 0 && (
              <div>
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Organismo
                </h3>
                <div className="space-y-1">
                  {facets.organismos.slice(0, 10).map((f) => (
                    <button key={f.value}
                            onClick={() => setOrganismo(organismo === f.value ? null : f.value)}
                            className={`block w-full truncate rounded px-2 py-1 text-left text-sm ${
                              organismo === f.value
                                ? 'bg-blue-100 font-medium text-blue-800'
                                : 'text-gray-600 hover:bg-gray-100'}`}>
                      {f.value} <span className="text-xs text-gray-400">({f.count})</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </aside>
        )}

        <section className="min-w-0 flex-1 space-y-4">
          {loading && <p className="text-sm text-gray-400">Buscando…</p>}
          {data?.error && (
            <p className="rounded-lg bg-red-50 p-4 text-sm text-red-700">
              Error al consultar la API. ¿Está corriendo uvicorn en :8000?
            </p>
          )}
          {data && !data.error && (
            <p className="text-sm text-gray-500">
              {data.total} resultado{data.total !== 1 ? 's' : ''}
            </p>
          )}
          {data?.results?.map((doc) => <ResultCard key={doc.id} doc={doc} />)}
          {data && !data.error && data.results?.length === 0 && (
            <p className="text-gray-500">Sin resultados. Probá con otros términos.</p>
          )}
        </section>
        </>
        )}
      </main>
    </div>
  )
}
