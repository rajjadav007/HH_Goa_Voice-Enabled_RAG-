import React, { useState, useRef } from 'react';
import { executeTextQuery, executeVoiceQuery } from '../services/api';
import { UnifiedRAGResult } from '../types/api';
import { normalizeRAGResponse } from '../utils/adapter';
import { RAGResultDisplay } from './RAGResultDisplay';
import { Mic, MicOff, Send, Upload, RefreshCw, AlertTriangle, FileText } from 'lucide-react';

type InputMode = 'voice' | 'text';
type PipelineState = 'idle' | 'recording' | 'transcribing' | 'processing' | 'success' | 'error';

export const VoiceRAGInterface: React.FC = () => {
  const [mode, setMode] = useState<InputMode>('voice');
  const [state, setState] = useState<PipelineState>('idle');
  const [textQuery, setTextQuery] = useState<string>('What is a corporation?');
  const [result, setResult] = useState<UnifiedRAGResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recordTime, setRecordTime] = useState<number>(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await handleAudioUpload(audioBlob);
      };

      mediaRecorderRef.current.start();
      setState('recording');
      setRecordTime(0);

      timerRef.current = setInterval(() => {
        setRecordTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone access denied or audio device unavailable.');
      setState('error');
    }
  };

  const stopRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    await handleAudioUpload(file);
  };

  const handleAudioUpload = async (audioBlob: Blob) => {
    setState('transcribing');
    setError(null);

    try {
      setState('processing');
      const res = await executeVoiceQuery(audioBlob);
      const normalized = normalizeRAGResponse(res, '', true);
      setResult(normalized);
      setState('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Voice query processing failed.');
      setState('error');
    }
  };

  const handleTextSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!textQuery.trim()) return;

    setState('processing');
    setError(null);

    try {
      const res = await executeTextQuery(textQuery);
      const normalized = normalizeRAGResponse(res, textQuery, false);
      setResult(normalized);
      setState('success');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Text RAG query failed.');
      setState('error');
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Input Mode Selector */}
      <div className="flex justify-center border-b border-slate-800 pb-4">
        <div className="inline-flex rounded-lg bg-slate-900/80 p-1 border border-slate-800">
          <button
            onClick={() => setMode('voice')}
            className={`flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold transition ${
              mode === 'voice' ? 'bg-sky-500 text-slate-950 shadow-md' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Mic className="h-4 w-4" />
            Voice Query (Sarvam STT)
          </button>
          <button
            onClick={() => setMode('text')}
            className={`flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold transition ${
              mode === 'text' ? 'bg-sky-500 text-slate-950 shadow-md' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText className="h-4 w-4" />
            Text Query (Direct RAG)
          </button>
        </div>
      </div>

      {/* Voice Mode Controls */}
      {mode === 'voice' && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 text-center space-y-5 backdrop-blur-sm">
          {state !== 'recording' ? (
            <div className="space-y-4">
              <button
                onClick={startRecording}
                disabled={state === 'transcribing' || state === 'processing'}
                className="h-20 w-20 rounded-full bg-gradient-to-tr from-sky-500 to-blue-600 p-4 text-white shadow-lg shadow-sky-500/20 hover:scale-105 transition transform flex items-center justify-center mx-auto disabled:opacity-50"
              >
                <Mic className="h-8 w-8" />
              </button>
              <div>
                <h3 className="font-semibold text-slate-100 text-sm">Click to Speak Question</h3>
                <p className="text-xs text-slate-400 mt-0.5">Sarvam AI STT &bull; Multilingual Indic Support</p>
              </div>

              <div className="pt-2 flex items-center justify-center gap-3 text-xs text-slate-400">
                <span className="text-slate-600">&mdash; or upload audio &mdash;</span>
                <label className="cursor-pointer inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 transition">
                  <Upload className="h-3.5 w-3.5" />
                  Select Audio File
                  <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
                </label>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative inline-flex">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <button
                  onClick={stopRecording}
                  className="relative h-20 w-20 rounded-full bg-red-600 p-4 text-white shadow-lg shadow-red-500/30 flex items-center justify-center mx-auto"
                >
                  <MicOff className="h-8 w-8" />
                </button>
              </div>
              <div>
                <h3 className="font-semibold text-red-400 text-sm">Recording Audio...</h3>
                <p className="font-mono text-sm text-slate-300 mt-1">{formatTime(recordTime)}</p>
                <p className="text-xs text-slate-400 mt-1">Click button to stop &amp; submit query</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Text Mode Controls */}
      {mode === 'text' && (
        <form onSubmit={handleTextSubmit} className="flex gap-2">
          <input
            type="text"
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            placeholder="Ask a question (e.g. What is a corporation?)"
            className="flex-1 rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
          />
          <button
            type="submit"
            disabled={state === 'processing'}
            className="flex items-center gap-2 rounded-xl bg-sky-500 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-sky-400 transition disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
            Execute
          </button>
        </form>
      )}

      {/* Processing State Indicator */}
      {(state === 'transcribing' || state === 'processing') && (
        <div className="flex items-center justify-center gap-3 rounded-xl border border-sky-500/20 bg-sky-500/10 p-4 text-sky-400 text-xs font-medium">
          <RefreshCw className="h-4 w-4 animate-spin" />
          <span>
            {state === 'transcribing'
              ? 'Transcribing audio with Sarvam AI...'
              : 'Running Hybrid Retrieval, Reranking & Gemini Grounding...'}
          </span>
        </div>
      )}

      {/* Error Message */}
      {state === 'error' && error && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-400 text-xs">
          <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-semibold">Query Failed</h4>
            <p className="mt-0.5 text-red-300/80">{error}</p>
          </div>
          <button
            onClick={() => setState('idle')}
            className="rounded bg-red-500/20 px-2 py-1 text-[10px] font-bold text-red-200 hover:bg-red-500/30"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Shared RAG Results Presentation */}
      {state === 'success' && result && <RAGResultDisplay result={result} />}
    </div>
  );
};
