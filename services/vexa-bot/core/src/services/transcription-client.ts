import { log } from '../utils';

/** Whisper language names → ISO-639-1 codes. The in-house transcription-service
 *  returns ISO codes; OpenAI-compatible providers (Groq) return full English
 *  names ("English"), which downstream validators reject. */
const WHISPER_LANGUAGE_CODES: Record<string, string> = {
  english: 'en', chinese: 'zh', german: 'de', spanish: 'es', russian: 'ru', korean: 'ko',
  french: 'fr', japanese: 'ja', portuguese: 'pt', turkish: 'tr', polish: 'pl', catalan: 'ca',
  dutch: 'nl', arabic: 'ar', swedish: 'sv', italian: 'it', indonesian: 'id', hindi: 'hi',
  finnish: 'fi', vietnamese: 'vi', hebrew: 'he', ukrainian: 'uk', greek: 'el', malay: 'ms',
  czech: 'cs', romanian: 'ro', danish: 'da', hungarian: 'hu', tamil: 'ta', norwegian: 'no',
  thai: 'th', urdu: 'ur', croatian: 'hr', bulgarian: 'bg', lithuanian: 'lt', latin: 'la',
  maori: 'mi', malayalam: 'ml', welsh: 'cy', slovak: 'sk', telugu: 'te', persian: 'fa',
  latvian: 'lv', bengali: 'bn', serbian: 'sr', azerbaijani: 'az', slovenian: 'sl',
  kannada: 'kn', estonian: 'et', macedonian: 'mk', breton: 'br', basque: 'eu',
  icelandic: 'is', armenian: 'hy', nepali: 'ne', mongolian: 'mn', bosnian: 'bs',
  kazakh: 'kk', albanian: 'sq', swahili: 'sw', galician: 'gl', marathi: 'mr',
  punjabi: 'pa', sinhala: 'si', khmer: 'km', shona: 'sn', yoruba: 'yo', somali: 'so',
  afrikaans: 'af', occitan: 'oc', georgian: 'ka', belarusian: 'be', tajik: 'tg',
  sindhi: 'sd', gujarati: 'gu', amharic: 'am', yiddish: 'yi', lao: 'lo', uzbek: 'uz',
  faroese: 'fo', 'haitian creole': 'ht', pashto: 'ps', turkmen: 'tk', nynorsk: 'nn',
  maltese: 'mt', sanskrit: 'sa', luxembourgish: 'lb', myanmar: 'my', tibetan: 'bo',
  tagalog: 'tl', malagasy: 'mg', assamese: 'as', tatar: 'tt', hawaiian: 'haw',
  lingala: 'ln', hausa: 'ha', bashkir: 'ba', javanese: 'jw', sundanese: 'su',
  cantonese: 'yue', burmese: 'my', valencian: 'ca', flemish: 'nl', haitian: 'ht',
  letzeburgesch: 'lb', pushto: 'ps', panjabi: 'pa', moldavian: 'ro', moldovan: 'ro',
  sinhalese: 'si', castilian: 'es', mandarin: 'zh',
};

/** Normalize a provider-reported language to an ISO-639-1 code; short values
 *  pass through, unknown full names fall back unchanged. */
function normalizeLanguageCode(lang: string | undefined): string | undefined {
  if (!lang) return lang;
  const lower = lang.toLowerCase().trim();
  if (lower.length <= 3) return lower;
  return WHISPER_LANGUAGE_CODES[lower] || lang;
}

export interface TranscriptionWord {
  word: string;
  start: number;
  end: number;
  probability: number;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
  avg_logprob?: number;
  no_speech_prob?: number;
  compression_ratio?: number;
  words?: TranscriptionWord[];
}

export interface TranscriptionResult {
  text: string;
  language: string;
  language_probability?: number;
  duration: number;
  segments: TranscriptionSegment[];
}

export interface TranscriptionClientConfig {
  /** Base URL of transcription-service, e.g. "http://localhost:8083" */
  serviceUrl: string;
  /** Optional bearer token for authentication */
  apiToken?: string;
  /** Max retry attempts for transient failures. Default: 3 */
  maxRetries?: number;
  /** Base delay between retries in ms. Default: 1000 */
  retryDelayMs?: number;
  /** Sample rate of input audio. Default: 16000 */
  sampleRate?: number;
  /** Max speech segment duration in seconds. Whisper forces a segment split at this length.
   *  Lower values = more frequent confirmations = faster output. Default: server default (15s) */
  maxSpeechDurationSec?: number;
  /** Minimum silence duration (ms) for VAD to split segments. Lower = more splits at natural pauses.
   *  Default: server default (160ms). Use ~100ms for more granular segments. */
  minSilenceDurationMs?: number;
  /** Model name sent in the request. Default: "whisper-1" */
  model?: string;
  /** Request dialect. "vexa" (default) targets the in-house transcription-service:
   *  legacy `timestamp_granularities` field plus vexa-specific VAD form fields.
   *  "openai" targets external OpenAI-compatible providers (OpenAI, Groq, ...):
   *  standard `timestamp_granularities[]` array field, no non-standard fields. */
  dialect?: 'vexa' | 'openai';
}

/**
 * HTTP client for the transcription-service.
 * Converts Float32Array audio to WAV, sends as multipart form,
 * and returns transcription results.
 */
export class TranscriptionClient {
  private serviceUrl: string;
  private apiToken: string | undefined;
  private maxRetries: number;
  private retryDelayMs: number;
  private sampleRate: number;
  private maxSpeechDurationSec: number | undefined;
  private minSilenceDurationMs: number | undefined;
  private model: string;
  private dialect: 'vexa' | 'openai';
  constructor(config: TranscriptionClientConfig) {
    // Ensure serviceUrl ends with the transcriptions endpoint
    this.serviceUrl = config.serviceUrl.replace(/\/+$/, '');
    if (!this.serviceUrl.endsWith('/v1/audio/transcriptions')) {
      this.serviceUrl += '/v1/audio/transcriptions';
    }
    this.apiToken = config.apiToken;
    this.maxRetries = config.maxRetries ?? 3;
    this.retryDelayMs = config.retryDelayMs ?? 1000;
    this.sampleRate = config.sampleRate ?? 16000;
    this.maxSpeechDurationSec = config.maxSpeechDurationSec;
    this.minSilenceDurationMs = config.minSilenceDurationMs;
    this.model = config.model || 'whisper-1';
    this.dialect = config.dialect || 'vexa';
  }

  /**
   * Transcribe a Float32Array audio buffer.
   * Converts to WAV, POSTs to transcription-service, returns parsed result.
   * Retries on transient failures (503, network errors).
   */
  async transcribe(audioData: Float32Array, language?: string, prompt?: string): Promise<TranscriptionResult> {
    const wavBuffer = this.float32ToWav(audioData);

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await this.sendRequest(wavBuffer, language, prompt);
        return result;
      } catch (err: any) {
        const isTransient = err.statusCode === 503 || err.statusCode === 429 || err.statusCode === 500 || !err.statusCode;
        const isLastAttempt = attempt === this.maxRetries;

        if (isTransient && !isLastAttempt) {
          const delay = this.retryDelayMs * Math.pow(2, attempt);
          log(`[TranscriptionClient] Transient error (attempt ${attempt + 1}/${this.maxRetries + 1}): ${err.message}. Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }

        // Non-transient error or exhausted retries
        log(`[TranscriptionClient] Transcription failed after ${attempt + 1} attempts: ${err.message}`);
        throw err;
      }
    }

    // Should never reach here, but TypeScript needs it
    throw new Error('Transcription failed: exhausted retries');
  }

  /**
   * Send the WAV buffer to the transcription-service as multipart form data.
   */
  private async sendRequest(wavBuffer: Buffer, language?: string, prompt?: string): Promise<TranscriptionResult> {
    // Build multipart form data manually (no external dependency needed)
    const boundary = `----FormBoundary${Date.now().toString(36)}`;

    const parts: Buffer[] = [];

    // File part
    parts.push(Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n` +
      `Content-Type: audio/wav\r\n\r\n`
    ));
    parts.push(wavBuffer);
    parts.push(Buffer.from('\r\n'));

    // Model part (required by OpenAI-compatible API)
    parts.push(Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="model"\r\n\r\n` +
      `${this.model}\r\n`
    ));

    // Response format part
    parts.push(Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="response_format"\r\n\r\n` +
      `verbose_json\r\n`
    ));

    // Language part (if specified)
    if (language) {
      parts.push(Buffer.from(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="language"\r\n\r\n` +
        `${language}\r\n`
      ));
    }

    // Request word-level timestamps. The OpenAI standard expects the array form
    // `timestamp_granularities[]` and omits segments unless `segment` is also
    // requested explicitly (verified against Groq 2026-06-12); the in-house
    // transcription-service reads the legacy scalar field (main.py Form("segment"))
    // and always returns segments.
    if (this.dialect === 'openai') {
      for (const granularity of ['segment', 'word']) {
        parts.push(Buffer.from(
          `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="timestamp_granularities[]"\r\n\r\n` +
          `${granularity}\r\n`
        ));
      }
    } else {
      parts.push(Buffer.from(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="timestamp_granularities"\r\n\r\n` +
        `word\r\n`
      ));
    }

    // Vexa-specific VAD tuning fields — only the in-house service understands these
    if (this.dialect !== 'openai') {
      // Max speech segment duration (controls how often Whisper splits segments)
      if (this.maxSpeechDurationSec !== undefined) {
        parts.push(Buffer.from(
          `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="max_speech_duration_s"\r\n\r\n` +
          `${this.maxSpeechDurationSec}\r\n`
        ));
      }

      // Min silence duration for VAD segment splitting (lower = more splits at natural pauses)
      if (this.minSilenceDurationMs !== undefined) {
        parts.push(Buffer.from(
          `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="min_silence_duration_ms"\r\n\r\n` +
          `${this.minSilenceDurationMs}\r\n`
        ));
      }
    }

    // Prompt: previous confirmed text as context for streaming continuity
    if (prompt) {
      parts.push(Buffer.from(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="prompt"\r\n\r\n` +
        `${prompt}\r\n`
      ));
    }

    // End boundary
    parts.push(Buffer.from(`--${boundary}--\r\n`));

    const body = Buffer.concat(parts);

    const headers: Record<string, string> = {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
    };
    if (this.apiToken) {
      headers['Authorization'] = `Bearer ${this.apiToken}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const response = await fetch(this.serviceUrl, {
        method: 'POST',
        headers,
        body,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unable to read error response');
        const err: any = new Error(`Transcription service returned HTTP ${response.status}: ${errorText}`);
        err.statusCode = response.status;
        throw err;
      }

      const data = await response.json() as any;

      // The in-house service nests word timestamps per segment; the OpenAI
      // standard (timestamp_granularities[]=word) returns one top-level
      // `words` array with word-less segments. Normalize to per-segment words
      // (assigned by word midpoint) so downstream consumers — speaker
      // attribution flatMaps segment.words — see a single shape.
      const rawSegments: any[] = data.segments || [];
      const topWords: any[] = Array.isArray(data.words) ? data.words : [];
      const needsWordMapping = topWords.length > 0 &&
        !rawSegments.some((s: any) => Array.isArray(s.words) && s.words.length > 0);

      return {
        text: data.text || '',
        language: normalizeLanguageCode(data.language) || language || 'unknown',
        language_probability: data.language_probability ?? 0,
        duration: data.duration || 0,
        segments: rawSegments.map((s: any, i: number) => ({
          start: s.start || 0,
          end: s.end || 0,
          text: s.text || '',
          avg_logprob: s.avg_logprob,
          no_speech_prob: s.no_speech_prob,
          compression_ratio: s.compression_ratio,
          words: needsWordMapping
            ? topWords
                .filter((w: any) => {
                  const mid = ((w.start || 0) + (w.end || 0)) / 2;
                  return mid >= (s.start || 0) && (mid < (s.end || 0) || i === rawSegments.length - 1);
                })
                .map((w: any) => ({ word: w.word, start: w.start, end: w.end, probability: w.probability ?? 1 }))
            : s.words,
        })),
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Convert Float32Array audio samples to a WAV file buffer.
   * Output: 16-bit PCM, mono, at this.sampleRate (default 16kHz).
   */
  private float32ToWav(samples: Float32Array): Buffer {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const dataSize = samples.length * bytesPerSample;
    const headerSize = 44;
    const buffer = Buffer.alloc(headerSize + dataSize);

    // RIFF header
    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + dataSize, 4);
    buffer.write('WAVE', 8);

    // fmt sub-chunk
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16);              // Sub-chunk size
    buffer.writeUInt16LE(1, 20);               // PCM format
    buffer.writeUInt16LE(numChannels, 22);     // Mono
    buffer.writeUInt32LE(this.sampleRate, 24);  // Sample rate
    buffer.writeUInt32LE(this.sampleRate * numChannels * bytesPerSample, 28); // Byte rate
    buffer.writeUInt16LE(numChannels * bytesPerSample, 32); // Block align
    buffer.writeUInt16LE(bitsPerSample, 34);   // Bits per sample

    // data sub-chunk
    buffer.write('data', 36);
    buffer.writeUInt32LE(dataSize, 40);

    // Convert Float32 [-1, 1] to Int16
    let offset = headerSize;
    for (let i = 0; i < samples.length; i++) {
      let sample = samples[i];
      // Clamp to [-1, 1]
      sample = Math.max(-1, Math.min(1, sample));
      // Convert to 16-bit integer
      const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      buffer.writeInt16LE(Math.round(int16), offset);
      offset += 2;
    }

    return buffer;
  }
}
