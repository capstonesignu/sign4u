/**
 * preprocessing.js — 브라우저 측 키포인트 전처리
 *
 * 서버 services/preprocessing.py convert_stream(preset="B") 와 동일한 파이프라인:
 *   1. 손 누락 프레임(null) 선형 보간
 *   2. pose 랜드마크 개별 보간 (landmark별 (0,0,0) → 인접 프레임으로 채움)
 *   3. 어깨 중점 centering + 어깨 너비로 scaling
 *   4. preset B: xy축만, pose 7개(hip 2개 제외), hand 21개, face 19개
 *   → (N, 136)
 *
 * Input : Array<{pose:Array|null, leftHand:Array|null, rightHand:Array|null, face:Array|null}>
 * Output: Array<Array<number>>  shape (N, 136)
 */

const PRE_SHOULDER_L = 1;   // pose 배열 내 왼쪽 어깨 index (pose[1*3] = x)
const PRE_SHOULDER_R = 2;   // pose 배열 내 오른쪽 어깨 index
const PRE_POSE_USE   = 7;   // 사용할 pose landmark 수 (hip 2개 제외)
const PRE_HAND_LMS   = 21;
const PRE_FACE_LMS   = 19;

// 손 누락 프레임 보간: null 프레임 → 인접 유효 프레임으로 선형 보간
function _interpolateMissing(nullableFrames) {
  const N = nullableFrames.length;
  const out = nullableFrames.map(f => f ? f.slice() : null);

  const validIdxs = [];
  for (let i = 0; i < N; i++) if (out[i] !== null) validIdxs.push(i);

  if (validIdxs.length === 0) {
    const dim = nullableFrames.find(f => f !== null)?.length ?? 63;
    return Array.from({ length: N }, () => new Array(dim).fill(0));
  }

  for (let t = 0; t < N; t++) {
    if (out[t] !== null) continue;
    let bi = -1, ai = -1;
    for (let k = 0; k < validIdxs.length; k++) {
      if (validIdxs[k] < t) bi = validIdxs[k];
      else if (ai < 0)      { ai = validIdxs[k]; break; }
    }
    if (bi < 0) {
      out[t] = out[ai].slice();
    } else if (ai < 0) {
      out[t] = out[bi].slice();
    } else {
      const alpha = (t - bi) / (ai - bi);
      out[t] = out[bi].map((v, i) => (1 - alpha) * v + alpha * out[ai][i]);
    }
  }
  return out;
}

// pose 랜드마크 개별 보간: landmark별 (0,0,0) 미검출 프레임 → 인접 유효 프레임으로 보간
// 서버 _interpolate_missing_pose 와 동일한 로직
function _interpolateMissingPose(poseFrames) {
  const N = poseFrames.length;
  const numLm = 9;
  const out = poseFrames.map(p => p ? p.slice() : new Array(numLm * 3).fill(0));

  for (let lm = 0; lm < numLm; lm++) {
    const base = lm * 3;
    const valid = [];
    for (let t = 0; t < N; t++) {
      const x = out[t][base], y = out[t][base + 1], z = out[t][base + 2];
      if (Math.sqrt(x*x + y*y + z*z) >= 1e-4) valid.push(t);
    }
    if (valid.length === 0) continue;

    for (let t = 0; t < N; t++) {
      const x = out[t][base], y = out[t][base + 1], z = out[t][base + 2];
      if (Math.sqrt(x*x + y*y + z*z) >= 1e-4) continue;

      let bi = -1, ai = -1;
      for (const v of valid) {
        if (v < t) bi = v;
        else if (ai < 0) { ai = v; break; }
      }
      if (bi < 0) {
        for (let d = 0; d < 3; d++) out[t][base + d] = out[ai][base + d];
      } else if (ai < 0) {
        for (let d = 0; d < 3; d++) out[t][base + d] = out[bi][base + d];
      } else {
        const alpha = (t - bi) / (ai - bi);
        for (let d = 0; d < 3; d++)
          out[t][base + d] = (1 - alpha) * out[bi][base + d] + alpha * out[ai][base + d];
      }
    }
  }
  return out;
}

/**
 * frames 배열을 서버 전처리와 동일하게 변환.
 * @param {Array} frames - extractKeypoints() 결과 배열
 * @returns {number[][]} shape (N, 136)
 */
function preprocessFrames(frames) {
  const N = frames.length;
  if (N === 0) return [];

  const lhFilled   = _interpolateMissing(frames.map(f => f.leftHand));
  const rhFilled   = _interpolateMissing(frames.map(f => f.rightHand));
  const poseFilled = _interpolateMissingPose(frames.map(f => f.pose ?? new Array(27).fill(0)));

  const result = [];
  for (let t = 0; t < N; t++) {
    const pose = poseFilled[t];
    const lh   = lhFilled[t]   ?? new Array(21 * 3).fill(0);
    const rh   = rhFilled[t]   ?? new Array(21 * 3).fill(0);
    const face = frames[t].face ?? new Array(19 * 3).fill(0);

    // 어깨 기준 정규화
    const lshX = pose[PRE_SHOULDER_L * 3],     lshY = pose[PRE_SHOULDER_L * 3 + 1];
    const rshX = pose[PRE_SHOULDER_R * 3],     rshY = pose[PRE_SHOULDER_R * 3 + 1];
    const midX = (lshX + rshX) / 2,            midY = (lshY + rshY) / 2;
    const width = Math.hypot(rshX - lshX, rshY - lshY);
    const scale = width > 1e-6 ? width : 1e-6;

    const nx = x => (x - midX) / scale;
    const ny = y => (y - midY) / scale;

    const row = [];
    // pose: 첫 7개, x/y만
    for (let i = 0; i < PRE_POSE_USE; i++) {
      row.push(nx(pose[i * 3]), ny(pose[i * 3 + 1]));
    }
    // left hand: 21개, x/y
    for (let i = 0; i < PRE_HAND_LMS; i++) {
      row.push(nx(lh[i * 3]), ny(lh[i * 3 + 1]));
    }
    // right hand: 21개, x/y
    for (let i = 0; i < PRE_HAND_LMS; i++) {
      row.push(nx(rh[i * 3]), ny(rh[i * 3 + 1]));
    }
    // face: 19개, x/y
    for (let i = 0; i < PRE_FACE_LMS; i++) {
      row.push(nx(face[i * 3]), ny(face[i * 3 + 1]));
    }
    result.push(row);
  }
  return result;
}
