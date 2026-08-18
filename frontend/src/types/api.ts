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
