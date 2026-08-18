import { HealthResponse, QueryResponse, VoiceQueryResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Health check failed with HTTP status ${response.status}`);
  }

  return response.json();
}

export async function executeTextQuery(query: string): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Text query failed with HTTP status ${response.status}`);
  }

  return response.json();
}

export async function executeVoiceQuery(audioFile: File | Blob, languageCode?: string): Promise<VoiceQueryResponse> {
  const formData = new FormData();
  formData.append('file', audioFile, 'query.wav');
  if (languageCode) {
    formData.append('language_code', languageCode);
  }

  const response = await fetch(`${API_BASE_URL}/api/voice-query`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Voice query failed with HTTP status ${response.status}`);
  }

  return response.json();
}
