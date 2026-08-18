import React, { useEffect, useState } from 'react';
import { fetchBackendHealth } from '../services/api';
import { HealthResponse } from '../types/api';
import { Activity, CheckCircle2, AlertCircle, RefreshCw, Database } from 'lucide-react';

export const HealthCheck: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchBackendHealth();
      setHealth(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-sky-500/10 p-2 text-sky-400 border border-sky-500/20">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-100">Backend &amp; Infrastructure Health</h3>
            <p className="text-xs text-slate-400">Phase 1.3 Environment &amp; Service Connectivity</p>
          </div>
        </div>
        <button
          onClick={checkHealth}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-slate-700 hover:text-white disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="mt-5 border-t border-slate-800/80 pt-4">
        {loading && (
          <div className="flex items-center justify-center gap-2 py-4 text-xs text-slate-400">
            <RefreshCw className="h-4 w-4 animate-spin text-sky-400" />
            <span>Checking FastAPI backend &amp; Qdrant status (/api/health)...</span>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-red-400">
            <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
            <div className="text-xs">
              <p className="font-semibold">Backend Unreachable</p>
              <p className="mt-0.5 text-red-300/80">{error}</p>
            </div>
          </div>
        )}

        {health && !loading && (
          <div className="space-y-3">
            <div className="flex items-center justify-between rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-3 text-emerald-400">
              <div className="flex items-center gap-2 text-xs font-medium">
                <CheckCircle2 className="h-4 w-4" />
                <span>FastAPI Backend Operational</span>
              </div>
              <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider">
                {health.status}
              </span>
            </div>

            {/* Qdrant Service Connectivity Status */}
            <div className={`flex items-center justify-between rounded-lg p-3 text-xs border ${
              health.qdrant?.connected
                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
            }`}>
              <div className="flex items-center gap-2 font-medium">
                <Database className="h-4 w-4" />
                <span>Qdrant Infrastructure ({health.qdrant?.host}:{health.qdrant?.port})</span>
              </div>
              <span className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                health.qdrant?.connected ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
              }`}>
                {health.qdrant?.connected ? 'Connected' : 'Offline / Standby'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-md border border-slate-800 bg-slate-950/40 p-2.5">
                <span className="text-slate-500 block text-[10px] uppercase font-mono">Service Name</span>
                <span className="font-medium text-slate-200">{health.service}</span>
              </div>
              <div className="rounded-md border border-slate-800 bg-slate-950/40 p-2.5">
                <span className="text-slate-500 block text-[10px] uppercase font-mono">Version</span>
                <span className="font-medium text-slate-200">v{health.version}</span>
              </div>
            </div>

            <div className="rounded-md border border-slate-800 bg-slate-950/40 p-2.5 text-xs">
              <span className="text-slate-500 block text-[10px] uppercase font-mono">Server Timestamp</span>
              <span className="font-mono text-slate-300 text-[11px]">{health.timestamp}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
