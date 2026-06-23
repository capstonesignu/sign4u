const WebSocket = require("ws");
const jwt = require("jsonwebtoken");
const pool = require("../config/db");

// consultation_id → { patient: ws, doctor: ws }
//
// 이 WebSocket은 환자↔의사 간 시그널링 릴레이 전용이다.
//   - WebRTC 시그널링 (OFFER / ANSWER / ICE_CANDIDATE)
//   - 수어 번역 결과 (TRANSLATION_RESULT, 환자 → 의사)
//   - 음성 인식 결과 (STT_RESULT, 의사 → 환자)
//
// 수어 키포인트 → AI 추론은 WS가 아니라 HTTP(POST /predict, SSE)로 처리된다.
// (routes/ai.js 프록시 → AI 서버 /predict)
const sessions = new Map();

function setupWebSocket(server) {
  const wss = new WebSocket.Server({ server });

  wss.on("connection", async (clientWs, req) => {
    // JWT 인증 (쿼리스트링으로 전달)
    const url = new URL(req.url, "http://localhost");
    const token = url.searchParams.get("token");
    const consultationId = url.searchParams.get("consultation_id");

    if (!token || !consultationId) {
      clientWs.close(1008, "token, consultation_id 필요");
      return;
    }

    let user;
    try {
      user = jwt.verify(token, process.env.JWT_SECRET);
    } catch {
      clientWs.close(1008, "유효하지 않은 토큰");
      return;
    }

    const role = user.role?.toLowerCase(); // 'patient' or 'doctor'
    if (role !== "patient" && role !== "doctor") {
      clientWs.close(1008, "유효하지 않은 역할");
      return;
    }

    // 방이 없으면 생성, 있으면 참가
    let session = sessions.get(consultationId);
    if (!session) {
      session = { patient: null, doctor: null };
      sessions.set(consultationId, session);
    }

    if (session[role]) {
      // 기존 연결이 있으면 닫고 새 연결로 교체 (close 핸들러의 슬롯 비교로 race 방지)
      try { session[role].close(1000, '재접속'); } catch {}
    }
    session[role] = clientWs;

    console.log(`[ws] ${role} 참가 consultation_id=${consultationId}`);

    // 상대방에게 참가 알림 + 본인에게도 상대 이미 접속 알림
    const otherRole = role === "patient" ? "doctor" : "patient";
    const otherWs = session[otherRole];
    if (otherWs && otherWs.readyState === WebSocket.OPEN) {
      otherWs.send(JSON.stringify({ type: "PEER_JOINED", role }));
      // 새로 접속한 사람에게도 상대가 이미 있다고 알림
      clientWs.send(JSON.stringify({ type: "PEER_JOINED", role: otherRole }));
    }

    // 클라이언트 메시지 라우팅
    clientWs.on("message", (data) => {
      const s = sessions.get(consultationId);
      if (!s) return;

      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        clientWs.send(JSON.stringify({ error: "JSON 파싱 실패" }));
        return;
      }

      const { type } = parsed;

      // WebRTC 시그널링 + 턴 전환 → 상대방에게 릴레이
      if (type === "OFFER" || type === "ANSWER" || type === "ICE_CANDIDATE" || type === "TURN_CHANGE") {
        const target = s[otherRole];
        if (target && target.readyState === WebSocket.OPEN) {
          target.send(JSON.stringify(parsed));
        }
        return;
      }

      // 의사 발화 완료 (STT_DONE) → 환자에게 릴레이 + DB 저장
      if (type === "STT_DONE") {
        const target = s[otherRole];
        if (target && target.readyState === WebSocket.OPEN) {
          target.send(JSON.stringify(parsed));
        }
        if (parsed.text) {
          pool.query(
            `INSERT INTO translation_logs (consultation_id, translated_text, speaker, created_at)
             VALUES ($1, $2, $3, NOW())`,
            [consultationId, parsed.text, role]
          ).catch((err) => console.error("[ws] STT_DONE DB 저장 실패:", err.message));
        }
        return;
      }

      // 수어 번역 결과 → 상대방(의사)에게 릴레이 + DB 저장
      if (type === "TRANSLATION_RESULT") {
        const target = s[otherRole];
        if (target && target.readyState === WebSocket.OPEN) {
          target.send(JSON.stringify(parsed));
        }
        if (parsed.result) {
          pool.query(
            `INSERT INTO translation_logs (consultation_id, translated_text, confidence, speaker, created_at)
             VALUES ($1, $2, $3, $4, NOW())`,
            [consultationId, parsed.result, parsed.confidence || null, role]
          ).catch((err) => console.error("[ws] 번역 DB 저장 실패:", err.message));
        }
        return;
      }

      // STT 결과 → 상대방(환자)에게 릴레이 + DB 저장
      if (type === "STT_RESULT") {
        const target = s[otherRole];
        if (target && target.readyState === WebSocket.OPEN) {
          target.send(JSON.stringify(parsed));
        }
        if (parsed.text) {
          pool.query(
            `INSERT INTO translation_logs (consultation_id, translated_text, speaker, created_at)
             VALUES ($1, $2, $3, NOW())`,
            [consultationId, parsed.text, role]
          ).catch((err) => console.error("[ws] STT DB 저장 실패:", err.message));
        }
        return;
      }

      // 그 외 타입은 무시 (수어 키포인트는 HTTP /predict 경로로 처리됨)
    });

    // 연결 종료
    clientWs.on("close", () => {
      const s = sessions.get(consultationId);
      if (!s) return;

      // 이 소켓이 현재 슬롯에 있을 때만 정리 (재접속으로 교체된 경우 무시)
      if (s[role] !== clientWs) return;

      s[role] = null;
      console.log(`[ws] ${role} 퇴장 consultation_id=${consultationId}`);

      // 상대방에게 퇴장 알림
      const other = s[otherRole];
      if (other && other.readyState === WebSocket.OPEN) {
        other.send(JSON.stringify({ type: "PEER_LEFT", role }));
      }

      // 양쪽 다 나가면 세션 정리
      if (!s.patient && !s.doctor) {
        sessions.delete(consultationId);
        console.log(`[ws] 세션 종료 consultation_id=${consultationId}`);
      }
    });

    clientWs.on("error", (err) =>
      console.error(`[ws] ${role} 에러:`, err.message)
    );
  });

  return wss;
}

module.exports = { setupWebSocket };
