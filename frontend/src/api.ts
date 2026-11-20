export type Mode = "bm25" | "dense" | "hybrid";

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  rank: number;
  token_count: number;
  bm25_rank: number | null;
  dense_rank: number | null;
}
export interface QueryResult {
  question: string;
  mode: Mode;
  answer: string;
  chunks: Chunk[];
  stage_ms: Record<string, number>;
  context_tokens: number;
  generator: string;
  warning: string;
}
export interface DocInfo { doc_id: string; title: string; chunks: number; tokens: number }
export interface DocContent { doc_id: string; title: string; segments: { idx: number; text: string; token_count: number }[] }
export interface EvalSummary {
  run_id: number; mode: Mode; k: number; n_questions: number; hit_rate: number; recall: number;
  mrr: number; ndcg: number; answer_coverage: number; avg_context_tokens: number;
  latency_ms_mean: number; latency_ms_p95: number; generator: string;
}
export interface Health { status: string; version: string; auth_required: boolean; vector_backend: string; generator: string; documents: number }

const KEY_STORAGE = "evidencelab_api_key";
export const getKey = () => localStorage.getItem(KEY_STORAGE) ?? "";
export const setKey = (k: string) => (k ? localStorage.setItem(KEY_STORAGE, k) : localStorage.removeItem(KEY_STORAGE));

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (getKey()) headers.set("X-API-Key", getKey());
  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}
const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/health"),
  checkKey: () => req<{ ok: boolean }>("/auth/check"),
  documents: () => req<DocInfo[]>("/documents"),
  document: (id: string) => req<DocContent>(`/documents/${encodeURIComponent(id)}`),
  addDocument: (title: string, text: string) => req<{ doc_id: string; chunks: number }>("/documents", json({ title, text })),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ doc_id: string; chunks: number }>("/documents/upload", { method: "POST", body: fd });
  },
  deleteDocument: (id: string) => req<void>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  query: (question: string, mode: Mode, k: number) => req<QueryResult>("/query", json({ question, mode, k })),
  loadBenchmark: () => req<{ documents: number; questions: number }>("/eval/load-benchmark", { method: "POST" }),
  runEval: (modes: Mode[], k: number) => req<EvalSummary[]>("/eval/run", json({ modes, k })),
  runs: () => req<{ run_id: number; created_at: string; mode: Mode; k: number; summary: EvalSummary }[]>("/eval/runs"),
};
