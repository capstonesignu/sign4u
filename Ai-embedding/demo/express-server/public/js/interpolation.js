/**
 * Temporal interpolation: resample N frames to targetLength frames.
 * Matches numpy's nearest-neighbor resampling used in preprocess.py.
 *
 * @param {number[][]} rawFrames - Array of frames, each frame is a flat array of floats.
 * @param {number} targetLength - Target number of frames.
 * @returns {number[][]} Resampled frames.
 */
function temporalInterpolate(rawFrames, targetLength) {
  const N = rawFrames.length;
  if (N === 0) return [];
  if (N === targetLength) return rawFrames;
  if (N === 1) return Array(targetLength).fill(rawFrames[0]);

  const result = [];
  for (let i = 0; i < targetLength; i++) {
    const srcIdx = Math.round((i * (N - 1)) / (targetLength - 1));
    result.push(rawFrames[srcIdx]);
  }
  return result;
}
