import { useEffect, useRef, useState, useCallback } from "react";

const DEFAULT_ICE_SERVERS = [{ urls: "stun:stun.l.google.com:19302" }];

async function fetchTurnIceServers() {
  const apiUrl = import.meta.env.VITE_TURN_API_URL;
  if (!apiUrl) return null;
  try {
    const res = await fetch(apiUrl);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return Array.isArray(data) && data.length > 0 ? data : null;
  } catch (err) {
    console.warn("TURN 자격증명 가져오기 실패, STUN만 사용:", err);
    return null;
  }
}

export default function useWebRTC({
  consultationId,
  role,
  onTranslation,
  onSTTResult,
  onPeerLeft,
  onTurnChange,
}) {
  const wsRef = useRef(null);
  const pcRef = useRef(null);
  const localStreamRef = useRef(null);
  const pendingIceRef = useRef([]);
  const iceServersRef = useRef(DEFAULT_ICE_SERVERS);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  const isPatient = role === "patient";

  const sendWS = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }, []);

  const createPeerConnection = useCallback(() => {
    const pc = new RTCPeerConnection({ iceServers: iceServersRef.current });

    pc.onicecandidate = (e) => {
      if (e.candidate) {
        sendWS({ type: "ICE_CANDIDATE", candidate: e.candidate });
      }
    };

    pc.ontrack = (e) => {
      if (e.streams[0]) {
        setRemoteStream(e.streams[0]);
      }
    };

    pc.onconnectionstatechange = () => {
      setIsConnected(pc.connectionState === "connected");
    };

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        pc.addTrack(track, localStreamRef.current);
      });
    }

    pcRef.current = pc;
    return pc;
  }, [sendWS]);

  const flushPendingIce = useCallback(async (pc) => {
    while (pendingIceRef.current.length > 0) {
      const c = pendingIceRef.current.shift();
      try {
        await pc.addIceCandidate(new RTCIceCandidate(c));
      } catch (err) {
        console.warn("addIceCandidate(flush) 실패:", err);
      }
    }
  }, []);

  const handleOffer = useCallback(
    async (offer) => {
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
      const pc = createPeerConnection();
      await pc.setRemoteDescription(new RTCSessionDescription(offer));
      await flushPendingIce(pc);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      sendWS({ type: "ANSWER", sdp: answer });
    },
    [createPeerConnection, sendWS, flushPendingIce],
  );

  const handleAnswer = useCallback(async (answer) => {
    if (pcRef.current) {
      await pcRef.current.setRemoteDescription(new RTCSessionDescription(answer));
      await flushPendingIce(pcRef.current);
    }
  }, [flushPendingIce]);

  const handleIceCandidate = useCallback(async (candidate) => {
    const pc = pcRef.current;
    if (!pc || !pc.remoteDescription || !pc.remoteDescription.type) {
      pendingIceRef.current.push(candidate);
      return;
    }
    try {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (err) {
      console.warn("addIceCandidate 실패:", err);
    }
  }, []);

  const handlePeerJoined = useCallback(async () => {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
      pendingIceRef.current = [];
    }
    if (isPatient) {
      const pc = createPeerConnection();
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      sendWS({ type: "OFFER", sdp: offer });
    }
  }, [isPatient, createPeerConnection, sendWS]);

  // TURN 자격증명 fetch (한 번만, 미디어 획득과 병렬)
  useEffect(() => {
    let cancelled = false;
    fetchTurnIceServers().then((servers) => {
      if (!cancelled && servers) iceServersRef.current = servers;
    });
    return () => { cancelled = true; };
  }, []);

  // Get local media stream
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: true })
      .then((stream) => {
        localStreamRef.current = stream;
        setLocalStream(stream);
      })
      .catch((err) => console.error("Media error:", err));

    return () => {
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  // WebSocket connection
  useEffect(() => {
    if (!localStream || !consultationId) return;

    const token = localStorage.getItem("accessToken");
    // VITE_API_BASE_URL(http(s)://...)을 ws(s)://...로 변환하여 백엔드로 연결
    const apiBase = import.meta.env.VITE_API_BASE_URL || "";
    let wsUrl;
    if (import.meta.env.VITE_WS_URL) {
      wsUrl = import.meta.env.VITE_WS_URL;
    } else if (apiBase) {
      wsUrl = apiBase.replace(/^http/, "ws");
    } else {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      wsUrl = `${protocol}://${window.location.host}`;
    }
    const ws = new WebSocket(
      `${wsUrl}?token=${token}&consultation_id=${consultationId}`,
    );
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "PEER_JOINED":
          handlePeerJoined();
          break;
        case "PEER_LEFT":
          setIsConnected(false);
          setRemoteStream(null);
          pendingIceRef.current = [];
          if (pcRef.current) {
            pcRef.current.close();
            pcRef.current = null;
          }
          onPeerLeft?.();
          break;
        case "OFFER":
          handleOffer(msg.sdp);
          break;
        case "ANSWER":
          handleAnswer(msg.sdp);
          break;
        case "ICE_CANDIDATE":
          handleIceCandidate(msg.candidate);
          break;
        case "TRANSLATION_RESULT":
          onTranslation?.(msg.result);
          break;
        case "STT_RESULT":
          onSTTResult?.(msg.text);
          break;
        case "STT_DONE":
          onSTTResult?.(msg.text);
          break;
        case "TURN_CHANGE":
          onTurnChange?.(msg.turn);
          break;
        default:
          break;
      }
    };

    ws.onclose = () => setIsConnected(false);

    return () => {
      ws.close();
      pendingIceRef.current = [];
      if (pcRef.current) {
        pcRef.current.close();
        pcRef.current = null;
      }
    };
  }, [
    localStream,
    consultationId,
    handlePeerJoined,
    handleOffer,
    handleAnswer,
    handleIceCandidate,
    onTranslation,
    onSTTResult,
    onPeerLeft,
    onTurnChange,
  ]);

  const sendFrame = useCallback(
    (keypoints) => {
      sendWS({ type: "VIDEO_FRAME", keypoints });
    },
    [sendWS],
  );

  return { localStream, remoteStream, isConnected, sendFrame, sendWS };
}
