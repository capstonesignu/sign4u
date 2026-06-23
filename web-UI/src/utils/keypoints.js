/**
 * MediaPipe 키포인트 추출 & /predict API 형식 변환
 *
 * extractKeypoints → { pose, leftHand, rightHand, face } (null 보존, inBounds 체크)
 * framesToPredictBodyMulti → POST /predict body { words: [...] }
 */

// ── Landmark 인덱스 ─────────────────────────────────────────────────
export const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24];

export const FACE_INDICES = [
  46,
  53,
  52,
  65, // left eyebrow
  276,
  283,
  282,
  295, // right eyebrow
  33,
  159,
  133, // left eye
  362,
  386,
  263, // right eye
  1, // nose tip
  61,
  291,
  0,
  17, // mouth
];

// 캔버스에 스켈레톤 그릴 때 사용 (원본 33개 pose index 기준)
export const POSE_DRAW_CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
];

// ── Tasks API 결과 → 통일 형태로 변환 ───────────────────────────────
export function adaptResult(result) {
  return {
    poseLandmarks:
      result.poseLandmarks?.length > 0 ? result.poseLandmarks[0] : null,
    leftHandLandmarks:
      result.leftHandLandmarks?.length > 0 ? result.leftHandLandmarks[0] : null,
    rightHandLandmarks:
      result.rightHandLandmarks?.length > 0
        ? result.rightHandLandmarks[0]
        : null,
    faceLandmarks:
      result.faceLandmarks?.length > 0 ? result.faceLandmarks[0] : null,
  };
}

// ── 프레임 1개 → { pose, leftHand, rightHand, face } ────────────────
// null: 해당 landmark 미감지 / inBounds 벗어난 landmark는 (0,0,0) 처리
export function extractKeypoints(results) {
  const inBounds = (lm) => lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1;
  const extractLms = (lms, indices) => {
    if (!lms) return null;
    const arr = [];
    for (const idx of (indices ?? lms.map((_, i) => i))) {
      const lm = lms[idx];
      inBounds(lm) ? arr.push(lm.x, lm.y, lm.z) : arr.push(0, 0, 0);
    }
    return arr;
  };
  return {
    pose:      extractLms(results.poseLandmarks, POSE_INDICES),
    leftHand:  extractLms(results.leftHandLandmarks),
    rightHand: extractLms(results.rightHandLandmarks),
    face:      extractLms(results.faceLandmarks, FACE_INDICES),
  };
}

// ── 구조체 프레임 배열(1 segment) → WordSegment ──────────────────────
function _segmentToWord(segment) {
  const pose = [], left_hand = [], right_hand = [], face = [];
  for (const frame of segment) {
    pose.push(frame.pose            ?? new Array(9  * 3).fill(0));
    left_hand.push(frame.leftHand   ?? new Array(21 * 3).fill(0));
    right_hand.push(frame.rightHand ?? new Array(21 * 3).fill(0));
    face.push(frame.face            ?? new Array(19 * 3).fill(0));
  }
  return { pose, left_hand, right_hand, face };
}

// ── 녹화된 프레임 배열 → POST /predict 요청 body (1개 단어) ──────────
export function framesToPredictBody(frames) {
  return { words: [_segmentToWord(frames)] };
}

// ── 단어 경계 인덱스로 프레임을 단어별로 분리 → POST /predict 요청 body ──
// frames:     extractKeypoints() 결과 배열
// boundaries: 단어 경계 프레임 인덱스 배열
// minFrames:  이보다 짧은 세그먼트는 버린다
export function framesToPredictBodyMulti(frames, boundaries = [], minFrames = 5) {
  const cuts = [...new Set(boundaries)]
    .filter((b) => b > 0 && b < frames.length)
    .sort((a, b) => a - b);

  const segments = [];
  let start = 0;
  for (const cut of cuts) {
    segments.push(frames.slice(start, cut));
    start = cut;
  }
  segments.push(frames.slice(start));

  const words = segments
    .filter((seg) => seg.length >= minFrames)
    .map(_segmentToWord);

  if (words.length === 0 && frames.length > 0) {
    return framesToPredictBody(frames);
  }

  return { words };
}
