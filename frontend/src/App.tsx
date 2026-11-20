import { FileText, FlaskConical, Loader2, Search, Trash2, UploadCloud, X } from "lucide-react";
import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { api, DocContent, DocInfo, EvalSummary, Health, Mode, QueryResult, setKey } from "./api";

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const grad = "bg-gradient-to-br from-emerald-500 to-teal-700";

function Note({ kind, text }: { kind: "error" | "ok"; text: string }) {
  if (!text) return null;
  const c = kind === "error" ? "border-red-200 bg-red-50 text-red-700" : "border-emerald-200 bg-emerald-50 text-emerald-700";
  return <div className={`rounded-lg border px-3 py-2 text-sm ${c}`}>{text}</div>;
}

/** Wrap query terms in <mark> so the evidence is easy to scan. */
function Highlight({ text, query }: { text: string; query: string }) {
  const terms = Array.from(new Set(query.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((t) => t.length > 2)));
  if (!terms.length) return <>{text}</>;
  const re = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "giu");
  return (
    <>
      {text.split(re).map((part, i) =>
        i % 2 === 1 ? <mark key={i} className="rounded bg-yellow-200 px-0.5">{part}</mark> : <Fragment key={i}>{part}</Fragment>,
      )}
    </>
  );
}

function DocViewer({ target, onClose }: { target: { id: string; segment?: number; query?: string }; onClose: () => void }) {
  const [doc, setDoc] = useState<DocContent | null>(null);
  const [error, setError] = useState("");
  const [find, setFind] = useState(target.query ?? "");
  const segRefs = useRef<Record<number, HTMLElement | null>>({});

  useEffect(() => {
    api.document(target.id).then(setDoc).catch((e) => setError(msg(e)));
  }, [target.id]);
  useEffect(() => {
    if (doc && target.segment !== undefined) segRefs.current[target.segment]?.scrollIntoView({ block: "center" });
  }, [doc, target.segment]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4" onClick={onClose}>
      <div className="flex max-h-[90vh] w-full max-w-3xl flex-col rounded-2xl bg-white shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-slate-200 px-5 py-4">
          <FileText size={18} className="text-emerald-600" />
          <div className="min-w-0 flex-1">
            <div className="truncate font-bold text-slate-900">{doc?.title ?? "Chargement…"}</div>
            {doc && <div className="text-xs text-slate-500">{doc.segments.length} segments · lecture du document indexé</div>}
          </div>
          <input value={find} onChange={(e) => setFind(e.target.value)} placeholder="Chercher dans le document…"
            className="w-52 rounded-lg border border-slate-300 px-3 py-1.5 text-sm outline-none focus:border-emerald-500" />
          <button onClick={onClose} title="Fermer (Échap)" className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100"><X size={18} /></button>
        </div>
        <div className="space-y-4 overflow-y-auto px-6 py-5">
          {error && <Note kind="error" text={error} />}
          {!doc && !error && <div className="flex items-center gap-2 text-sm text-slate-500"><Loader2 size={14} className="animate-spin" />Chargement…</div>}
          {doc?.segments.map((s) => (
            <section key={s.idx} ref={(el) => { segRefs.current[s.idx] = el; }}
              className={`rounded-xl border p-4 ${s.idx === target.segment ? "border-emerald-400 bg-emerald-50" : "border-slate-200"}`}>
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-400">Segment {s.idx + 1}</div>
              <p className="whitespace-pre-line text-sm leading-relaxed text-slate-800"><Highlight text={s.text} query={find} /></p>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

function Sidebar({ docs, reload, onOpen }: { docs: DocInfo[]; reload: () => void; onOpen: (id: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ok, setOk] = useState("");

  const upload = async (files: FileList | File[]) => {
    setBusy(true); setError(""); setOk("");
    try {
      let n = 0;
      for (const f of Array.from(files)) n += (await api.upload(f)).chunks;
      setOk(`${n} segments indexés.`); reload();
    } catch (e) { setError(msg(e)); } finally { setBusy(false); }
  };
  const act = async (fn: () => Promise<unknown>) => { setError(""); setOk(""); try { await fn(); reload(); } catch (e) { setError(msg(e)); } };

  return (
    <aside className="space-y-4">
      <div onClick={() => input.current?.click()} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); void upload(e.dataTransfer.files); }}
        className={`flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed p-5 text-center text-sm transition ${drag ? "border-emerald-500 bg-emerald-50" : "border-slate-300 bg-white hover:border-emerald-400"}`}>
        {busy ? <Loader2 size={22} className="animate-spin text-emerald-600" /> : <UploadCloud size={22} className="text-emerald-600" />}
        <div className="mt-2 font-semibold text-slate-800">{busy ? "Lecture…" : "Ajouter des documents"}</div>
        <div className="text-xs text-slate-500">PDF, DOCX, TXT, MD · 50 Mo max</div>
        <input ref={input} type="file" multiple hidden accept=".pdf,.docx,.txt,.md" onChange={(e) => { if (e.target.files?.length) void upload(e.target.files); e.target.value = ""; }} />
      </div>
      <Note kind="error" text={error} /><Note kind="ok" text={ok} />
      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-bold text-slate-900">Documents ({docs.length})</h2>
          <button onClick={() => act(() => api.loadBenchmark())} className="text-xs font-semibold text-emerald-700 hover:underline">Corpus de test</button>
        </div>
        <ul className="max-h-96 divide-y divide-slate-100 overflow-y-auto">
          {docs.map((d) => (
            <li key={d.doc_id} onClick={() => onOpen(d.doc_id)} title="Cliquer pour lire le document"
              className="-mx-2 flex cursor-pointer items-center justify-between gap-2 rounded-lg px-2 py-2 text-sm transition hover:bg-emerald-50">
              <div className="min-w-0"><div className="truncate font-medium text-slate-900 group-hover:text-emerald-700" title={d.title}>{d.title}</div><div className="text-xs text-slate-500">{d.chunks} segments · cliquer pour lire</div></div>
              <button title="Supprimer" onClick={(e) => { e.stopPropagation(); void act(() => api.deleteDocument(d.doc_id)); }} className="text-slate-400 hover:text-red-600"><Trash2 size={15} /></button>
            </li>
          ))}
          {!docs.length && <li className="py-3 text-xs text-slate-500">Aucun document.</li>}
        </ul>
      </div>
    </aside>
  );
}

function Research({ docs, onOpen }: { docs: DocInfo[]; onOpen: (id: string, segment?: number, query?: string) => void }) {
  const [q, setQ] = useState("");
  const [asked, setAsked] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [k, setK] = useState(5);
  const [docFilter, setDocFilter] = useState("");
  const [res, setRes] = useState<QueryResult | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async (question = q) => {
    const text = question.trim();
    if (!text || busy) return;
    setQ(text); setBusy(true); setError("");
    try {
      // fetch extra passages when filtering by document so enough remain
      const r = await api.query(text, mode, docFilter ? Math.min(50, k * 4) : k);
      setRes(r); setAsked(text);
      setHistory((h) => [text, ...h.filter((x) => x !== text)].slice(0, 6));
    } catch (e) { setError(msg(e)); } finally { setBusy(false); }
  };

  const shown = res ? res.chunks.filter((c) => !docFilter || c.doc_id === docFilter).slice(0, k) : [];
  const title = (id: string) => docs.find((d) => d.doc_id === id)?.title ?? id;

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search size={18} className="absolute left-3.5 top-3.5 text-slate-400" />
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} disabled={!docs.length}
              placeholder={docs.length ? "Que cherchez-vous dans vos documents ?" : "Ajoutez d'abord un document"}
              className="w-full rounded-xl border border-slate-300 py-3 pl-11 pr-3 outline-none focus:border-emerald-500 disabled:bg-slate-50" />
          </div>
          <button disabled={busy || !q.trim() || !docs.length} onClick={() => run()} className={`flex items-center gap-2 rounded-xl px-6 font-bold text-white ${grad} hover:opacity-90 disabled:opacity-40`}>
            {busy ? <Loader2 size={16} className="animate-spin" /> : "Rechercher"}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-600">
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} className="rounded-lg border border-slate-300 bg-white px-2 py-1" title="Méthode de recherche">
            <option value="hybrid">Hybride (recommandé)</option><option value="bm25">Mots-clés (BM25)</option><option value="dense">Sens (dense)</option>
          </select>
          <select value={docFilter} onChange={(e) => setDocFilter(e.target.value)} className="max-w-48 rounded-lg border border-slate-300 bg-white px-2 py-1" title="Limiter à un document">
            <option value="">Tous les documents</option>
            {docs.map((d) => <option key={d.doc_id} value={d.doc_id}>{d.title}</option>)}
          </select>
          <label>Résultats <input type="number" min={1} max={12} value={k} onChange={(e) => setK(Math.max(1, Math.min(12, Number(e.target.value) || 1)))} className="w-12 rounded-lg border border-slate-300 px-1 py-1" /></label>
          {history.length > 0 && <span className="ml-auto flex flex-wrap items-center gap-1">Récent :
            {history.map((h) => <button key={h} onClick={() => run(h)} className="max-w-36 truncate rounded-full bg-slate-100 px-2 py-0.5 hover:bg-emerald-100">{h}</button>)}</span>}
        </div>
      </div>

      <Note kind="error" text={error} />
      {res && (
        <>
          {res.warning && <Note kind="error" text={res.warning} />}
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-5">
            <div className="mb-1 text-[11px] font-bold uppercase tracking-wide text-emerald-700">Réponse · {res.generator}</div>
            <p className="text-lg leading-relaxed text-slate-900"><Highlight text={res.answer} query={asked} /></p>
          </div>
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>{shown.length} passage(s) · {res.stage_ms.total_ms?.toFixed(1)} ms</span>
            <details className="text-right"><summary className="cursor-pointer">Détail des étapes</summary>
              <div className="mt-1 flex flex-wrap justify-end gap-1">{Object.entries(res.stage_ms).map(([n, ms]) => <span key={n} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono">{n} {ms.toFixed(2)}</span>)}<span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono">{res.context_tokens} tokens</span></div>
            </details>
          </div>
          {shown.length === 0 && <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-500">Aucun passage trouvé. Essayez d'autres mots ou « Tous les documents ».</div>}
          <ol className="space-y-3">
            {shown.map((c) => (
              <li key={c.chunk_id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <span className={`flex h-6 w-6 items-center justify-center rounded-full font-bold text-white ${grad}`}>{c.rank}</span>
                  <button onClick={() => onOpen(c.doc_id, Number(c.chunk_id.split("#")[1]), asked)} title="Ouvrir le document à ce passage"
                    className="font-semibold text-slate-900 underline-offset-2 hover:text-emerald-700 hover:underline">{title(c.doc_id)}</button>
                  <span className="text-slate-400">segment {c.chunk_id.split("#")[1]}</span>
                  <span className="ml-auto font-mono text-slate-400" title="score de pertinence">{c.score.toFixed(3)}{c.bm25_rank !== null && ` · bm25 #${c.bm25_rank}`}{c.dense_rank !== null && ` · dense #${c.dense_rank}`}</span>
                </div>
                <p className="text-sm leading-relaxed text-slate-700"><Highlight text={c.text} query={asked} /></p>
              </li>
            ))}
          </ol>
        </>
      )}
      {!res && !error && <p className="py-10 text-center text-sm text-slate-500">Saisissez une question : vous obtiendrez une réponse et les passages exacts qui la justifient, avec les mots recherchés surlignés.</p>}
    </div>
  );
}

function Evaluation() {
  const [k, setK] = useState(3);
  const [rows, setRows] = useState<EvalSummary[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const run = async () => {
    setBusy(true); setError("");
    try { setRows(await api.runEval(["bm25", "dense", "hybrid"], k)); } catch (e) { setError(msg(e)); } finally { setBusy(false); }
  };
  const cols: [string, (r: EvalSummary) => string][] = [
    ["Mode", (r) => r.mode], ["Hit@k", (r) => r.hit_rate.toFixed(3)], ["Recall@k", (r) => r.recall.toFixed(3)], ["MRR", (r) => r.mrr.toFixed(3)],
    ["nDCG@k", (r) => r.ndcg.toFixed(3)], ["ms moy.", (r) => r.latency_ms_mean.toFixed(2)], ["ms p95", (r) => r.latency_ms_p95.toFixed(2)],
  ];
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-bold">Évaluation de la recherche</h2>
      <p className="mt-1 text-sm text-slate-500">Compare BM25, dense et hybride sur les questions de <code>eval_data/</code>. Cliquez d'abord sur « Corpus de test » dans la colonne de gauche.</p>
      <div className="mt-4 flex items-center gap-3">
        <label className="text-sm text-slate-600">k <input type="number" min={1} max={50} value={k} onChange={(e) => setK(Math.max(1, Number(e.target.value) || 1))} className="ml-1 w-16 rounded-lg border border-slate-300 px-2 py-1" /></label>
        <button disabled={busy} onClick={run} className={`flex items-center gap-2 rounded-xl px-5 py-2 text-sm font-bold text-white ${grad} disabled:opacity-40`}>{busy ? <Loader2 size={16} className="animate-spin" /> : <FlaskConical size={16} />}Lancer</button>
      </div>
      <div className="mt-3"><Note kind="error" text={error} /></div>
      {rows.length > 0 && (
        <table className="mt-4 w-full text-left text-sm">
          <thead className="text-xs uppercase text-slate-500"><tr>{cols.map(([h]) => <th key={h} className="pb-2 pr-4">{h}</th>)}</tr></thead>
          <tbody>{rows.map((r) => <tr key={r.run_id} className="border-t border-slate-100">{cols.map(([h, f]) => <td key={h} className="py-2 pr-4 font-mono">{f(r)}</td>)}</tr>)}</tbody>
        </table>
      )}
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState<"search" | "eval">("search");
  const [health, setHealth] = useState<Health | null>(null);
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [locked, setLocked] = useState(false);
  const [viewer, setViewer] = useState<{ id: string; segment?: number; query?: string } | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [keyError, setKeyError] = useState("");

  const reload = useCallback(() => {
    api.health().then(async (h) => {
      setHealth(h);
      if (h.auth_required) {
        try { await api.checkKey(); setLocked(false); } catch { setLocked(true); return; }
      }
      api.documents().then(setDocs).catch(() => setDocs([]));
    }).catch(() => setHealth(null));
  }, []);
  useEffect(reload, [reload]);

  const unlock = async () => {
    setKey(keyInput.trim()); setKeyError("");
    try { await api.checkKey(); setLocked(false); setKeyInput(""); reload(); }
    catch { setKey(""); setKeyError("Clé API invalide."); }
  };

  if (locked) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 p-6">
        <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-7 shadow-lg">
          <div className={`mb-4 flex h-10 w-10 items-center justify-center rounded-lg font-bold text-white ${grad}`}>E</div>
          <h1 className="text-lg font-bold">EvidenceLab</h1>
          <p className="mt-1 text-sm text-slate-500">Accès protégé. Saisissez votre clé API.</p>
          <input type="password" autoFocus value={keyInput} onChange={(e) => setKeyInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && unlock()}
            className="mt-4 w-full rounded-xl border border-slate-300 px-3 py-2.5 text-sm outline-none focus:border-emerald-500" placeholder="Clé API" />
          <div className="mt-3"><Note kind="error" text={keyError} /></div>
          <button onClick={unlock} disabled={!keyInput.trim()} className={`mt-3 w-full rounded-xl py-2.5 text-sm font-bold text-white ${grad} disabled:opacity-40`}>Déverrouiller</button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className={`flex h-9 w-9 items-center justify-center rounded-lg font-bold text-white ${grad}`}>E</div>
            <div className="font-bold">EvidenceLab</div>
            <nav className="ml-4 flex gap-1 text-sm font-semibold">
              {([["search", "Recherche"], ["eval", "Évaluation"]] as const).map(([id, label]) => (
                <button key={id} onClick={() => setTab(id)} className={`rounded-lg px-3 py-1 ${tab === id ? "bg-emerald-100 text-emerald-800" : "text-slate-500 hover:text-slate-800"}`}>{label}</button>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500"><span className={`h-2 w-2 rounded-full ${health ? "bg-emerald-500" : "bg-red-500"}`} />{health ? "API connectée" : "API injoignable"}</div>
        </div>
      </header>
      <main className="mx-auto grid max-w-6xl gap-6 px-6 py-6 lg:grid-cols-[18rem_1fr]">
        <Sidebar docs={docs} reload={reload} onOpen={(id) => setViewer({ id })} />
        {tab === "search" ? <Research docs={docs} onOpen={(id, segment, query) => setViewer({ id, segment, query })} /> : <Evaluation />}
      </main>
      {viewer && <DocViewer key={`${viewer.id}-${viewer.segment ?? ""}`} target={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}
