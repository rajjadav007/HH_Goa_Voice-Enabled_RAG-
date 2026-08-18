export interface QdrantStatus {
  connected: boolean;
  host: string;
  port: number;
  status?: string;
  error?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  service: string;
  timestamp: string;
  qdrant?: QdrantStatus;
}

export interface SourceItem {
  chunk_id: string;
  document_id: string;
  rank: number;
  text?: string;
  score?: number;
  metadata?: Record<string, unknown>;
}

export type GroundingStatusType =
  | 'GROUNDED'
  | 'PARTIALLY_GROUNDED'
  | 'INSUFFICIENT_EVIDENCE'
  | 'CONTRADICTED'
  | 'NO_CONTEXT_GROUNDED'
  | 'UNGROUNDED';

export interface QueryResponse {
  success: boolean;
  answer: string;
  grounded: boolean;
  grounding_status?: GroundingStatusType | string;
  has_context: boolean;
  sources: SourceItem[];
  request_id: string;
  status: string;
  error_code?: string;
  latency_ms: number;
  timing_breakdown?: Record<string, number>;
  token_usage?: Record<string, number>;
}

export interface STTMetadata {
  text: string;
  language?: string;
  confidence?: number;
  provider: string;
  model: string;
  duration_sec?: number;
  latency_ms?: number;
}

export interface VoiceQueryResponse extends QueryResponse {
  transcript: string;
  stt?: STTMetadata;
}

export interface APIErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface BaseAPIResponse<T = unknown> {
  request_id: string;
  success: boolean;
  data?: T;
  error?: APIErrorDetail;
}

export interface UnifiedRAGResult {
  queryText: string;
  isVoice: boolean;
  answer: string;
  grounded: boolean;
  groundingStatus: GroundingStatusType | string;
  hasContext: boolean;
  sources: SourceItem[];
  requestId: string;
  status: string;
  errorCode?: string;
  latencyMs: number;
  timingBreakdown?: Record<string, number>;
  sttMetadata?: STTMetadata;
}
