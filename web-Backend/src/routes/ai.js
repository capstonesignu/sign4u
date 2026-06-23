const express = require('express');
const router = express.Router();
const axios = require('axios');

// AI 추론 서버 (FastAPI, Railway). AI 서버가 0.0.0.0(IPv4)로 바인딩하므로
// Railway private 네트워크(IPv6)는 사용 불가 → public 도메인 사용.
const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://localhost:8001';

/**
 * POST /predict
 *
 * 프론트가 보낸 수어 키포인트({ words: [...] })를 AI 서버 /predict로 그대로 전달하고,
 * AI가 반환하는 SSE(text/event-stream) 스트림을 프론트로 실시간 릴레이한다.
 *
 * 흐름: 프론트 ──{words}──► Node(/predict) ──{words}──► AI(/predict)
 *       프론트 ◄──SSE──── Node ◄──────SSE──────── AI
 */
router.post('/predict', async (req, res) => {
  // 클라이언트가 끊으면 AI 요청도 취소(sLLM 생성 중단 → 자원 절약)
  const controller = new AbortController();
  res.on('close', () => controller.abort());

  try {
    const upstream = await axios.post(`${AI_SERVER_URL}/predict`, req.body, {
      responseType: 'stream',
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
      timeout: 0, // SSE 생성이 길어질 수 있으므로 타임아웃 없음
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    // SSE 응답 헤더
    res.status(upstream.status);
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no'); // 중간 프록시(nginx 등) 버퍼링 방지
    if (typeof res.flushHeaders === 'function') res.flushHeaders();

    // AI → 프론트로 스트림 그대로 흘려보냄
    upstream.data.pipe(res);

    upstream.data.on('error', (err) => {
      console.error('[ai] upstream stream 오류:', err.message);
      if (!res.writableEnded) res.end();
    });
  } catch (err) {
    // 클라이언트가 먼저 끊은 정상 케이스
    if (controller.signal.aborted || res.writableEnded) return;

    const status = err.response?.status || 502;
    console.error(`[ai] /predict 프록시 실패 (${status}):`, err.message);

    if (!res.headersSent) {
      res.status(status >= 400 && status < 600 ? status : 502).json({
        error: 'AI 서버 요청 실패',
        detail: err.message,
      });
    } else if (!res.writableEnded) {
      res.end();
    }
  }
});

/**
 * GET /health — AI 서버 연결 확인용 프록시 (디버깅 편의)
 */
router.get('/health', async (req, res) => {
  try {
    const r = await axios.get(`${AI_SERVER_URL}/health`, { timeout: 5000 });
    res.json({ proxy: 'ok', ai: r.data });
  } catch (err) {
    res.status(502).json({ proxy: 'ok', ai: null, error: 'AI 서버 헬스체크 실패', detail: err.message });
  }
});

module.exports = router;
