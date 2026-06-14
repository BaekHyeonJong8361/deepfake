const DEFAULT_GATEWAY_API_BASE_URL = "";
const DEFAULT_AUTH_API_BASE_URL = DEFAULT_GATEWAY_API_BASE_URL;
const DEFAULT_USER_API_BASE_URL = DEFAULT_GATEWAY_API_BASE_URL;
const DEFAULT_VIDEO_API_BASE_URL = DEFAULT_GATEWAY_API_BASE_URL;
const VERCEL_GATEWAY_API_PROXY_URL = '/api/backend';

export function getApiBaseUrl() {
  const configuredUrl = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_GATEWAY_API_BASE_URL;
  return getBrowserSafeBaseUrl(configuredUrl, VERCEL_GATEWAY_API_PROXY_URL);
}

function getAuthApiBaseUrl() {
  const configuredUrl = import.meta.env.VITE_AUTH_API_BASE_URL ?? DEFAULT_AUTH_API_BASE_URL;
  return getBrowserSafeBaseUrl(configuredUrl, VERCEL_GATEWAY_API_PROXY_URL);
}

function getUserApiBaseUrl() {
  const configuredUrl = import.meta.env.VITE_USER_API_BASE_URL ?? DEFAULT_USER_API_BASE_URL;
  return getBrowserSafeBaseUrl(configuredUrl, VERCEL_GATEWAY_API_PROXY_URL);
}

function getVideoApiBaseUrl() {
  const configuredUrl = import.meta.env.VITE_VIDEO_API_BASE_URL ?? DEFAULT_VIDEO_API_BASE_URL;
  return getBrowserSafeBaseUrl(configuredUrl, VERCEL_GATEWAY_API_PROXY_URL);
}

function getBrowserSafeBaseUrl(configuredUrl: string, httpsProxyUrl: string) {
  if (typeof window !== 'undefined' && window.location.protocol === 'https:' && configuredUrl.startsWith('http:')) {
    return httpsProxyUrl;
  }

  return configuredUrl;
}

export type HealthCheckResult = {
  ok: boolean;
  status: number;
  data: unknown;
};

export type AuthLoginRequest = {
  email: string;
  password: string;
};

export type AuthRegisterRequest = AuthLoginRequest & {
  nickname: string;
};

export type AuthSession = {
  accessToken: string;
  refreshToken?: string;
  userId?: string | number;
  email?: string;
  nickname?: string;
};

export type UserProfile = {
  id: string | number;
  email?: string;
  nickname?: string;
  raw: unknown;
};

export type VideoAnalysisType = 'DEEPFAKE' | 'T2V' | 'RYZE' | 'LEE_SIN' | 'SHEN' | 'RAMMUS';

export type PresignedUploadResponse = {
  uploadUrl: string;
  fileUrl: string;
};

export type VideoStatus = 'queued' | 'pending' | 'analyzing' | 'processing' | 'completed' | 'failed' | string;

export type SuspiciousFrame = {
  frameIndex: number; time?: string;
  probability?: number;
};

export type AnalysisResult = {
  final_verdict: string;
  deepfake_score?: number;
  t2v_score?: number;
  xai_text?: string;
  suspicious_frames: SuspiciousFrame[];
  xai_heatmap_url?: string;
  per_frame_probs?: number[];
  analysis_type?: string; engine_label?: string;
  original_face_url?: string;
  rgb_contribution?: number;
  freq_contribution?: number;
  top_regions?: {region: string, ratio: number}[];
  forensic_report?: string;
  raw: unknown;
};

export type HistoryRecord = {
  id: string;
  title: string;
  date: string;
  result: 'FAKE' | 'REAL';
  percentage: string;
  thumb?: string;
  duration?: string;
  size?: string;
  raw: unknown;
};

type RequestOptions = RequestInit & {
  token?: string | null;
  baseUrl?: string;
};

async function checkHealth(baseUrl: string, path: string, errorMessage: string): Promise<HealthCheckResult> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: 'GET',
    headers: {
      Accept: 'application/json, text/plain, */*',
    },
  });

  if (!response.ok) {
    throw new Error(errorMessage);
  }

  const text = await response.text();
  const data = text ? parseResponseBody(text) : null;

  return {
    ok: true,
    status: response.status,
    data,
  };
}

export function checkAuthHealth() {
  return checkHealth(getAuthApiBaseUrl(), '/auth/health', 'auth health check failed');
}

export function checkUserHealth() {
  return checkHealth(getUserApiBaseUrl(), '/user/health', 'user health check failed');
}

export async function register(payload: AuthRegisterRequest) {
  const data = await request<Record<string, unknown>>('/auth/signup', {
    method: 'POST',
    body: JSON.stringify(payload),
    baseUrl: getAuthApiBaseUrl(),
  });

  return {
    userId: readId(readFirst(data, ['user_id', 'userId', 'id'])),
    message: readFirst(data, ['message']) ?? 'Signup successful',
    raw: data,
  };
}

export async function login(payload: AuthLoginRequest): Promise<AuthSession> {
  const data = await request<Record<string, unknown>>('/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
    baseUrl: getAuthApiBaseUrl(),
  });

  const accessToken = readFirst(data, ['accessToken', 'access_token', 'token', 'jwt']);
  if (typeof accessToken !== 'string' || !accessToken) {
    throw new Error('Login response did not include an access token.');
  }

  const refreshToken = readFirst(data, ['refreshToken', 'refresh_token']);
  return {
    accessToken,
    refreshToken: typeof refreshToken === 'string' ? refreshToken : undefined,
    userId: readId(readFirst(data, ['user_id', 'userId', 'id'])),
    email: readString(readFirst(data, ['email'])),
    nickname: readString(readFirst(data, ['nickname'])),
  };
}

export async function getUserById(id: string | number, token?: string | null): Promise<UserProfile> {
  const data = await request<Record<string, unknown>>(`/user/${id}`, {
    token,
    baseUrl: getUserApiBaseUrl(),
  });

  return {
    id: readId(readFirst(data, ['id', 'userId', 'user_id'])) ?? id,
    email: readString(readFirst(data, ['email'])),
    nickname: readString(readFirst(data, ['nickname'])),
    raw: data,
  };
}

export async function logout(token: string) {
  return request<{ message?: string }>('/api/auth/logout', {
    method: 'POST',
    token,
  });
}

export async function refreshAuthToken(refreshToken: string) {
  const data = await request<Record<string, unknown>>('/api/auth/refresh', {
    method: 'POST',
    body: JSON.stringify({ refreshToken, refresh_token: refreshToken }),
  });

  const accessToken = readFirst(data, ['accessToken', 'access_token', 'token', 'jwt']);
  if (typeof accessToken !== 'string' || !accessToken) {
    throw new Error('Token refresh response did not include an access token.');
  }

  return accessToken;
}

export async function uploadVideo(file: File, type: VideoAnalysisType, token?: string | null) {
  const contentType = file.type || 'application/octet-stream';
  const { uploadUrl, fileUrl } = await createPresignedUploadUrl(file.name, contentType, token);

  const uploadResponse = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
    },
    body: file,
  });

  if (!uploadResponse.ok) {
    throw new Error(`S3 upload failed. (${uploadResponse.status})`);
  }

  return requestVideoUrl(fileUrl, type, token);
}

export async function createPresignedUploadUrl(fileName: string, contentType: string, token?: string | null) {
  const data = await request<Record<string, unknown>>('/api/uploads/presigned-url', {
    method: 'POST',
    body: JSON.stringify({ fileName, contentType }),
    token,
    baseUrl: getVideoApiBaseUrl(),
  });

  const uploadUrl = readString(readFirst(data, ['uploadUrl', 'upload_url']));
  const fileUrl = readString(readFirst(data, ['fileUrl', 'file_url']));

  if (!uploadUrl || !fileUrl) {
    throw new Error('Presigned URL response did not include uploadUrl and fileUrl.');
  }

  return { uploadUrl, fileUrl } satisfies PresignedUploadResponse;
}

export async function requestVideoUrl(url: string, type: VideoAnalysisType, token?: string | null) {
  const data = await request<Record<string, unknown>>('/api/videos/url', {
    method: 'POST',
    body: JSON.stringify({ url, type }),
    token,
    baseUrl: getVideoApiBaseUrl(),
  });

  return readVideoId(data);
}

export async function getVideoStatus(videoId: string, token?: string | null) {
  const data = await request<Record<string, unknown>>(`/api/videos/${videoId}/status`, {
    token,
    baseUrl: getVideoApiBaseUrl(),
  });
  const status = readFirst(data, ['status']);
  return typeof status === 'string' ? normalizeVideoStatus(status) : 'processing';
}

export async function getVideoResult(videoId: string, token?: string | null) {
  const data = await request<unknown>(`/api/videos/${videoId}/result`, {
    token,
    baseUrl: getVideoApiBaseUrl(),
  });
  return normalizeAnalysisResult(data);
}

export async function getHistory(token: string) {
  const data = await request<unknown>('/api/history', { token });
  const list = Array.isArray(data) ? data : readFirst(data, ['records', 'history', 'items', 'data']);
  return Array.isArray(list) ? list.map(normalizeHistoryRecord) : [];
}

export async function getHistoryDetail(id: string, token: string) {
  const data = await request<unknown>(`/api/history/${id}`, { token });
  return normalizeHistoryRecord(data);
}

export async function deleteHistoryRecord(id: string, token: string) {
  return request<{ message?: string }>(`/api/history/${id}`, {
    method: 'DELETE',
    token,
  });
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { token, headers, body, baseUrl = getApiBaseUrl(), ...rest } = options;
  const isFormData = body instanceof FormData;
  const response = await fetch(`${baseUrl}${path}`, {
    ...rest,
    body,
    headers: {
      Accept: 'application/json, text/plain, */*',
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
  });

  const text = await response.text();
  const data = text ? parseResponseBody(text) : null;

  if (!response.ok) {
    const message = readErrorMessage(data) ?? `API request failed. (${response.status})`;
    throw new Error(message);
  }

  return unwrapApiResponse(data) as T;
}

function normalizeAnalysisResult(raw: unknown): AnalysisResult {
  const root = asRecord(raw) ?? {};
  const deepfake = asRecord(readFirst(root, ['deepfake', 'deepfake_result', 'deepfakeResult'])) ?? root;
  const t2v = asRecord(readFirst(root, ['t2v', 't2v_result', 't2vResult'])) ?? root;
  const evidence = asRecord(deepfake.evidence);
  const heatmaps = asRecord(evidence?.heatmaps);
  const t2vVisualization = asRecord(t2v.xai_visualization);
  const t2vHeatmaps = t2vVisualization?.heatmaps;
  const t2vFirstHeatmap = Array.isArray(t2vHeatmaps) ? asRecord(t2vHeatmaps[0]) : undefined;

  const deepfakeProb = readScore(readFirst(deepfake, ['deepfakeScore', 'deepfake_score', 'ensemble_prob']));
  const t2vProb = readScore(readFirst(t2v, ['t2vScore', 't2v_score', 't2v_prob']));
  const finalVerdict = readVerdict(root, deepfake, t2v, deepfakeProb, t2vProb);
  const suspiciousFrames = readSuspiciousFrames(root, evidence);

  return {
    final_verdict: finalVerdict,
    deepfake_score: toPercent(deepfakeProb),
    t2v_score: toPercent(t2vProb),
    xai_text: readString(readFirst(root, ['xaiText', 'xai_text'])),
    suspicious_frames: suspiciousFrames,
    xai_heatmap_url:
      readString(readFirst(root, ['xaiHeatmapUrl', 'xai_heatmap_url'])) ??
      readString(heatmaps?.v7) ??
      readString(t2vFirstHeatmap?.overlay_url),
    per_frame_probs: toNumberArray(readFirst(root, ['perFrameProbs', 'per_frame_probs']) ?? deepfake.per_frame_probs),
    analysis_type: readString(readFirst(root, ['analysis_type'])), engine_label: readString(readFirst(root, ['engine_label'])),
    original_face_url: readString(readFirst(root.raw || {}, ['original_face_url'])),
    rgb_contribution: readNumber(readFirst(root.raw || {}, ['rgb_contribution'])),
    freq_contribution: readNumber(readFirst(root.raw || {}, ['freq_contribution'])),
    top_regions: readFirst(root.raw || {}, ['top_regions']) as any,
    forensic_report: readString(readFirst(root.raw || {}, ['forensic_report'])),
    raw,
  };
}

function normalizeHistoryRecord(raw: unknown): HistoryRecord {
  const record = asRecord(raw) ?? {};
  const result = String(readFirst(record, ['result', 'verdict', 'final_verdict']) ?? '').toUpperCase();
  const score = readNumber(readFirst(record, ['percentage', 'score', 'deepfake_score', 'probability']));
  const id = String(readFirst(record, ['id', 'history_id', 'video_id']) ?? '');

  return {
    id,
    title: String(readFirst(record, ['title', 'filename', 'file_name', 'url']) ?? `analysis-${id || 'record'}`),
    date: String(readFirst(record, ['date', 'created_at', 'createdAt']) ?? ''),
    result: result.includes('FAKE') || result.includes('DEEPFAKE') || result.includes('T2V') ? 'FAKE' : 'REAL',
    percentage: typeof score === 'number' ? `${score > 1 ? score.toFixed(1) : (score * 100).toFixed(1)}%` : '',
    thumb: readString(readFirst(record, ['thumb', 'thumbnail', 'thumbnail_url'])),
    duration: readString(record.duration),
    size: readString(record.size),
    raw,
  };
}

function readVideoId(data: Record<string, unknown>) {
  const videoId = readFirst(data, ['video_id', 'videoId', 'id']);
  if (typeof videoId !== 'string' && typeof videoId !== 'number') {
    throw new Error('Video request response did not include a videoId.');
  }

  return String(videoId);
}

function readSuspiciousFrames(root: Record<string, unknown>, evidence: Record<string, unknown> | undefined): SuspiciousFrame[] {
  const directFrames = readFirst(root, ['suspiciousFrames', 'suspicious_frames']);
  if (Array.isArray(directFrames)) {
    return directFrames
      .map((item): SuspiciousFrame | undefined => {
        const record = asRecord(item);
        const frameIndex = readNumber(record?.frameIndex ?? record?.frame_index ?? item);
        if (frameIndex === undefined) return undefined;
        return {
          frameIndex,
          probability: readNumber(record?.probability ?? record?.prob ?? record?.score),
          time: typeof record?.time === 'string' ? record.time : undefined,
        } as SuspiciousFrame;
      })
      .filter((item): item is SuspiciousFrame => item !== undefined);
  }

  const suspectIndexes = toNumberArray(evidence?.suspect_frame_idx);
  const suspectProbs = toNumberArray(evidence?.suspect_frame_prob);
  return suspectIndexes.map((frameIndex, index) => ({
    frameIndex,
    probability: suspectProbs[index],
  }));
}

function readId(value: unknown) {
  if (typeof value === 'string' || typeof value === 'number') return value;
  return undefined;
}

function readVerdict(
  root: Record<string, unknown>,
  deepfake: Record<string, unknown>,
  t2v: Record<string, unknown>,
  deepfakeProb?: number,
  t2vProb?: number,
) {
  const explicit = readString(readFirst(root, ['finalVerdict', 'final_verdict', 'verdict', 'decision']));
  if (explicit) return explicit;

  const deepfakeDecision = readString(deepfake.decision);
  const t2vDecision = readString(t2v.decision);
  if (isAiDecision(deepfakeDecision) || isAiDecision(t2vDecision)) return 'FAKE';
  if ((deepfakeProb ?? 0) >= 0.5 || (t2vProb ?? 0) >= 0.5) return 'FAKE';
  return 'REAL';
}

function isAiDecision(value?: string) {
  if (!value) return false;
  const normalized = value.toLowerCase();
  return normalized.includes('fake') || normalized.includes('ai') || normalized.includes('t2v');
}

function toPercent(value?: number) {
  if (typeof value !== 'number') return undefined;
  return value > 1 ? value : value * 100;
}

function toNumberArray(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value.map(readNumber).filter((item): item is number => typeof item === 'number');
}

function readScore(value: unknown) {
  const number = readNumber(value);
  if (number === undefined) return undefined;
  return number > 1 ? number / 100 : number;
}

function readNumber(value: unknown) {
  if (typeof value === 'number') return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  return undefined;
}

function readString(value: unknown) {
  return typeof value === 'string' && value ? value : undefined;
}

function readFirst(data: unknown, keys: string[]) {
  const record = asRecord(data);
  if (!record) return undefined;
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key];
  }

  return undefined;
}

function readErrorMessage(data: unknown) {
  const rootMessage = readString(readFirst(data, ['message', 'error', 'detail']));
  if (rootMessage) return rootMessage;
  return readString(readFirst(asRecord(data)?.data, ['message', 'error', 'detail']));
}

function unwrapApiResponse(data: unknown) {
  const record = asRecord(data);
  if (!record || !('success' in record) || !('data' in record)) return data;
  return record.data ?? record;
}

function normalizeVideoStatus(status: string): VideoStatus {
  const normalized = status.toLowerCase();
  if (normalized === 'pending') return 'pending';
  if (normalized === 'analyzing') return 'analyzing';
  if (normalized === 'completed') return 'completed';
  if (normalized === 'failed') return 'failed';
  return normalized;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }

  return undefined;
}

function parseResponseBody(text: string) {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
