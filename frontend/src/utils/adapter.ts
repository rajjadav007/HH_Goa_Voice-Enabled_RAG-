import { QueryResponse, VoiceQueryResponse, UnifiedRAGResult } from '../types/api';

export function normalizeRAGResponse(
  rawResponse: QueryResponse | VoiceQueryResponse,
  originalQuery: string,
  isVoice: boolean = false
): UnifiedRAGResult {
  const voiceRes = rawResponse as VoiceQueryResponse;
  const queryText = isVoice && voiceRes.transcript ? voiceRes.transcript : originalQuery;

  const groundingStatus =
    rawResponse.grounding_status ||
    (rawResponse.grounded
      ? 'GROUNDED'
      : !rawResponse.has_context
      ? 'NO_CONTEXT_GROUNDED'
      : 'UNGROUNDED');

  return {
    queryText,
    isVoice,
    answer: rawResponse.answer || 'No response generated.',
    grounded: Boolean(rawResponse.grounded),
    groundingStatus,
    hasContext: Boolean(rawResponse.has_context),
    sources: Array.isArray(rawResponse.sources) ? rawResponse.sources : [],
    requestId: rawResponse.request_id || 'unknown_req',
    status: rawResponse.status || 'SUCCESS',
    errorCode: rawResponse.error_code,
    latencyMs: rawResponse.latency_ms || 0,
    timingBreakdown: rawResponse.timing_breakdown,
    sttMetadata: voiceRes.stt,
  };
}
