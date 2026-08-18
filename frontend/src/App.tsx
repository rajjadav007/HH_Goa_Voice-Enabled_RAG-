import { HealthCheck } from './components/HealthCheck';
import { VoiceRAGInterface } from './components/VoiceRAGInterface';
import { Mic, Sparkles } from 'lucide-react';

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
                HH Goa 2026 <span className="text-xs font-normal text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20">Milestone 8 &bull; Voice UI</span>
              </h1>
              <p className="text-[11px] text-slate-400">Multilingual Voice-Enabled RAG System</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Phase 8.1 Active</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-5xl mx-auto px-6 py-10 flex-1 w-full flex flex-col justify-center">
        <div className="max-w-3xl mx-auto w-full space-y-8">
          
          {/* Welcome Banner */}
          <div className="text-center space-y-2">
            <div className="inline-flex items-center gap-2 text-xs font-medium text-sky-400 bg-sky-500/10 border border-sky-500/20 rounded-full px-3 py-1">
              <Sparkles className="h-3.5 w-3.5" />
              <span>Voice &amp; Text RAG Interface</span>
            </div>
            <h2 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              Sarvam STT + Gemini RAG
            </h2>
            <p className="text-xs text-slate-400 max-w-lg mx-auto">
              Ask legal &amp; corporate questions by voice (Sarvam AI STT) or text. Grounded answers derived from AI4Bharat MSMARCO-XI index.
            </p>
          </div>

          {/* Core Interactive Voice RAG Component */}
          <VoiceRAGInterface />

          {/* Health Check Collapsible / Footer Status */}
          <HealthCheck />

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 bg-slate-950/80 text-center text-xs text-slate-500">
        <p>HH Goa 2026 Shortlisting Task 2 &bull; Phase 8.1 Voice UI &bull; Target P50 Voice-to-Answer Latency &lt; 200 ms</p>
      </footer>
    </div>
  );
}
