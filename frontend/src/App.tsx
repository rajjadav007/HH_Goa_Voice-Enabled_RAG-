import { HealthCheck } from './components/HealthCheck';
import { Mic, Database, Cpu, Sparkles } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col justify-between">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/40 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-sky-500 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-sky-500/20">
              <Mic className="h-4 w-4" />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-tight text-white flex items-center gap-2">
                HH Goa 2026 <span className="text-xs font-normal text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">Phase 1.2 Shell</span>
              </h1>
              <p className="text-[11px] text-slate-400">Voice-Enabled RAG System Foundation</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Milestone 1 Active</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-5xl mx-auto px-6 py-12 flex-1 w-full flex flex-col justify-center">
        <div className="max-w-2xl mx-auto w-full space-y-8">
          
          {/* Welcome Banner */}
          <div className="text-center space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-medium text-sky-400 bg-sky-500/10 border border-sky-500/20 rounded-full px-3 py-1">
              <Sparkles className="h-3.5 w-3.5" />
              <span>Project Setup &amp; System Foundation</span>
            </div>
            <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              HH Goa 2026 Voice RAG
            </h2>
            <p className="text-sm text-slate-400 max-w-lg mx-auto">
              Groundwork setup for MSMARCO-XI dataset retrieval, Qdrant vector index, Sarvam Speech-to-Text, and Gemini LLM.
            </p>
          </div>

          {/* Health Check Widget */}
          <HealthCheck />

          {/* Architecture Modules Readiness Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 text-left space-y-2">
              <div className="p-2 w-fit rounded-lg bg-slate-800 text-slate-400">
                <Database className="h-4 w-4" />
              </div>
              <h4 className="text-xs font-semibold text-slate-200">Knowledge Index</h4>
              <p className="text-[11px] text-slate-500">AI4Bharat MSMARCO-XI, Qdrant &amp; BM25 hybrid setup reserved for Phase 2–6.</p>
            </div>

            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 text-left space-y-2">
              <div className="p-2 w-fit rounded-lg bg-slate-800 text-slate-400">
                <Cpu className="h-4 w-4" />
              </div>
              <h4 className="text-xs font-semibold text-slate-200">RAG Engine</h4>
              <p className="text-[11px] text-slate-500">Retrieval, guardrails, and Gemini LLM grounding reserved for Phase 7–12.</p>
            </div>

            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/30 text-left space-y-2">
              <div className="p-2 w-fit rounded-lg bg-slate-800 text-slate-400">
                <Mic className="h-4 w-4" />
              </div>
              <h4 className="text-xs font-semibold text-slate-200">Sarvam Voice</h4>
              <p className="text-[11px] text-slate-500">Audio recorder and STT integration reserved for Phase 14.</p>
            </div>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 bg-slate-950/80 text-center text-xs text-slate-500">
        <p>HH Goa 2026 Shortlisting Task 2 &bull; Phase 1.2 Completed &bull; Target Online RAG Latency &lt; 200 ms</p>
      </footer>
    </div>
  );
}
