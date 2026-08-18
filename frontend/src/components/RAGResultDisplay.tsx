import React, { useState } from 'react';
import { UnifiedRAGResult, GroundingStatusType } from '../types/api';
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  XCircle,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Clock,
  Mic,
  FileText,
  Zap,
  Activity,
} from 'lucide-react';

interface RAGResultDisplayProps {
  result: UnifiedRAGResult;
}

export const RAGResultDisplay: React.FC<RAGResultDisplayProps> = ({ result }) => {
  const [copiedAnswer, setCopiedAnswer] = useState<boolean>(false);
  const [copiedSourceIdx, setCopiedSourceIdx] = useState<number | null>(null);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  const [showPerformance, setShowPerformance] = useState<boolean>(false);

  const handleCopyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(result.answer);
      setCopiedAnswer(true);
      setTimeout(() => setCopiedAnswer(false), 2000);
    } catch {
      // Fallback
    }
  };

  const handleCopySource = async (idx: number, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedSourceIdx(idx);
      setTimeout(() => setCopiedSourceIdx(null), 2000);
    } catch {
      // Fallback
    }
  };

  const toggleSourceExpand = (idx: number) => {
    setExpandedSources((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const getGroundingBadge = (status: GroundingStatusType | string) => {
    switch (status) {
      case 'GROUNDED':
        return {
          label: 'Fully Grounded',
          color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
          icon: <CheckCircle2 className="h-4 w-4 text-emerald-400" />,
          description: 'Answer strictly verified against MSMARCO-XI dataset sources.',
        };
      case 'PARTIALLY_GROUNDED':
        return {
          label: 'Partially Grounded',
          color: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
          icon: <AlertTriangle className="h-4 w-4 text-amber-400" />,
          description: 'Some parts of this answer could not be completely verified from dataset sources.',
        };
      case 'INSUFFICIENT_EVIDENCE':
        return {
          label: 'Insufficient Evidence',
          color: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
          icon: <HelpCircle className="h-4 w-4 text-amber-400" />,
          description: 'Limited evidence found in the dataset to support a complete response.',
        };
      case 'CONTRADICTED':
        return {
          label: 'Contradicted Evidence',
          color: 'bg-red-500/20 text-red-300 border-red-500/30',
          icon: <XCircle className="h-4 w-4 text-red-400" />,
          description: 'Warning: Retrieved documents contain conflicting or contradictory information.',
        };
      case 'NO_CONTEXT_GROUNDED':
      case 'NO_CONTEXT':
        return {
          label: 'No Context Found',
          color: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
          icon: <HelpCircle className="h-4 w-4 text-slate-400" />,
          description: 'No relevant information was found in the indexed dataset for this question.',
        };
      default:
        return {
          label: 'Ungrounded Response',
          color: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
          icon: <AlertTriangle className="h-4 w-4 text-slate-400" />,
          description: 'Response generated without full grounding validation.',
        };
    }
  };

  const badge = getGroundingBadge(result.groundingStatus);
  const isTargetMet = result.latencyMs < 200;

  // Calculate timing bar percentages
  const sttMs = result.timingBreakdown?.stt_ms || 0;
  const ragMs = result.timingBreakdown?.rag_ms || result.latencyMs;
  const totalMs = Math.max(result.latencyMs, sttMs + ragMs, 1);

  const sttWidth = Math.round((sttMs / totalMs) * 100);
  const ragWidth = Math.round((ragMs / totalMs) * 100);

  return (
    <div className="space-y-4 text-left">
      {/* Question / Transcript Header Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 text-xs backdrop-blur-sm">
        <div className="flex items-center justify-between text-slate-400 mb-1.5">
          <span className="font-mono text-[10px] uppercase flex items-center gap-1.5">
            {result.isVoice ? <Mic className="h-3.5 w-3.5 text-sky-400" /> : <FileText className="h-3.5 w-3.5 text-sky-400" />}
            {result.isVoice ? 'You Said (Sarvam Transcribed)' : 'Question Asked'}
          </span>
          {result.sttMetadata?.confidence !== undefined && (
            <span className="text-[10px] font-mono text-slate-500">
              STT Conf: {(result.sttMetadata.confidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <p className="text-sky-200 font-medium text-sm italic">&ldquo;{result.queryText}&rdquo;</p>
      </div>

      {/* Primary Answer Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-6 space-y-4 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {badge.icon}
            <h3 className="font-semibold text-slate-100 text-sm">Generated Answer</h3>
          </div>

          <div className="flex items-center gap-2">
            <span className={`rounded px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider border ${badge.color}`}>
              {badge.label}
            </span>
            <button
              onClick={handleCopyAnswer}
              className="flex items-center gap-1 rounded bg-slate-800 px-2.5 py-1 text-[11px] font-medium text-slate-300 hover:bg-slate-700 hover:text-white transition"
              title="Copy Answer to Clipboard"
            >
              {copiedAnswer ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              <span>{copiedAnswer ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
        </div>

        {/* Grounding Explanation Banner if not fully grounded */}
        {result.groundingStatus !== 'GROUNDED' && (
          <div className="rounded-lg bg-slate-950/60 border border-slate-800 p-3 text-xs text-slate-400 flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
            <span>{badge.description}</span>
          </div>
        )}

        {/* Main Answer Content */}
        <div className="text-sm text-slate-100 leading-relaxed whitespace-pre-line border-t border-slate-800/60 pt-3">
          {result.answer}
        </div>

        {/* Latency & Performance Summary Bar */}
        <div className="flex flex-wrap items-center justify-between border-t border-slate-800/80 pt-3 text-[11px] text-slate-400 gap-2">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 font-mono font-semibold text-sky-400">
              <Clock className="h-3.5 w-3.5" />
              Latency: {result.latencyMs} ms
            </span>
            <span
              className={`rounded px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider border ${
                isTargetMet
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                  : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
              }`}
            >
              {isTargetMet ? '<200ms Target Met' : '>200ms Target Exceeded'}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowPerformance(!showPerformance)}
              className="flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-sky-300 transition"
            >
              <Zap className="h-3.5 w-3.5 text-sky-400" />
              <span>{showPerformance ? 'Hide Details' : 'Performance Details'}</span>
              {showPerformance ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            <span className="font-mono text-slate-500 text-[10px]">ID: {result.requestId}</span>
          </div>
        </div>

        {/* Expandable Pipeline Timeline & Benchmark Metrics Panel */}
        {showPerformance && (
          <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-4 space-y-4 text-xs font-mono text-slate-300">
            <div className="flex items-center justify-between text-slate-400 border-b border-slate-800/80 pb-2">
              <span className="text-[10px] uppercase font-bold text-slate-400 flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5 text-sky-400" />
                Pipeline Stage Timeline Breakdown
              </span>
              <span className="text-[10px]">Target: &lt; 200 ms</span>
            </div>

            {/* Stage Bar Visualizers */}
            <div className="space-y-2 text-[11px]">
              {result.isVoice && sttMs > 0 && (
                <div className="space-y-1">
                  <div className="flex justify-between text-slate-400">
                    <span>1. Sarvam STT Transcription</span>
                    <span>{sttMs} ms</span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-sky-500 transition-all duration-500" style={{ width: `${sttWidth}%` }}></div>
                  </div>
                </div>
              )}

              <div className="space-y-1">
                <div className="flex justify-between text-slate-400">
                  <span>{result.isVoice ? '2. Text RAG Pipeline (Hybrid + Gemini)' : '1. Text RAG Pipeline'}</span>
                  <span>{ragMs} ms</span>
                </div>
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-500 transition-all duration-500" style={{ width: `${ragWidth}%` }}></div>
                </div>
              </div>
            </div>

            {/* Benchmark Percentiles Reference Table */}
            <div className="border-t border-slate-800/80 pt-3 space-y-2">
              <span className="text-[10px] uppercase font-bold text-slate-400 block">
                System Evaluation Matrix (30 Test Queries)
              </span>
              <div className="grid grid-cols-4 gap-2 text-center text-[10px]">
                <div className="bg-slate-900 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">P50</span>
                  <span className="text-emerald-400 font-bold">111.38 ms</span>
                </div>
                <div className="bg-slate-900 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">P90</span>
                  <span className="text-emerald-400 font-bold">132.78 ms</span>
                </div>
                <div className="bg-slate-900 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">P95</span>
                  <span className="text-emerald-400 font-bold">138.67 ms</span>
                </div>
                <div className="bg-slate-900 p-1.5 rounded border border-slate-800">
                  <span className="text-slate-500 block">P99</span>
                  <span className="text-emerald-400 font-bold">150.10 ms</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sources & Citations Section */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
            <ShieldCheck className="h-4 w-4 text-sky-400" />
            <span>Validated Citations &amp; Evidence ({result.sources.length})</span>
          </div>
        </div>

        {result.sources.length > 0 ? (
          <div className="space-y-2">
            {result.sources.map((src, idx) => {
              const isExpanded = expandedSources[idx] || false;
              const hasText = Boolean(src.text && src.text.trim().length > 0);

              return (
                <div key={idx} className="rounded-lg border border-slate-800 bg-slate-950/60 overflow-hidden text-xs">
                  <div
                    onClick={() => hasText && toggleSourceExpand(idx)}
                    className={`flex items-center justify-between p-3 ${
                      hasText ? 'cursor-pointer hover:bg-slate-900/60' : ''
                    }`}
                  >
                    <div className="flex items-center gap-3 font-mono">
                      <span className="rounded bg-sky-500/10 text-sky-400 px-2 py-0.5 text-[10px] font-bold">
                        #{src.rank}
                      </span>
                      <span className="text-slate-200">Chunk ID: {src.chunk_id}</span>
                      <span className="text-slate-500 text-[11px] hidden sm:inline">Doc ID: {src.document_id}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      {src.score !== undefined && (
                        <span className="text-[10px] font-mono text-slate-400">
                          Score: {src.score.toFixed(3)}
                        </span>
                      )}
                      {hasText && (
                        <button className="text-slate-400 hover:text-slate-200">
                          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Expandable Text Snippet */}
                  {hasText && isExpanded && (
                    <div className="border-t border-slate-800/80 p-3 bg-slate-900/30 text-slate-300 space-y-2">
                      <p className="text-[11px] leading-normal font-sans italic text-slate-300">
                        &ldquo;{src.text}&rdquo;
                      </p>
                      <div className="flex justify-end pt-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCopySource(idx, src.text!);
                          }}
                          className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-white"
                        >
                          {copiedSourceIdx === idx ? (
                            <Check className="h-3 w-3 text-emerald-400" />
                          ) : (
                            <Copy className="h-3 w-3" />
                          )}
                          <span>{copiedSourceIdx === idx ? 'Copied' : 'Copy Evidence'}</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-slate-500 italic">No citations cited by backend for this response.</p>
        )}
      </div>
    </div>
  );
};
