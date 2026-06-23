const express = require('express');
const router = express.Router();
const pool = require('../config/db');
const authMiddleware = require('../middlewares/auth');

// GET /api/reviews/form/:consultationId - 후기 작성 페이지 데이터
router.get('/form/:consultationId', authMiddleware, async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT c.id, c.scheduled_at, d.id AS doctor_id, d.name AS doctor_name,
              d.profile_image_url AS doctor_image, s.name AS specialty_name
       FROM consultations c
       JOIN doctors d ON d.id = c.doctor_id
       JOIN specialties s ON s.id = d.specialty_id
       WHERE c.id = $1 AND c.patient_id = $2 AND c.status = 'COMPLETED'`,
      [req.params.consultationId, req.user.id]
    );
    if (!result.rows[0]) return res.status(404).json({ error: 'Not found' });
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/reviews - 후기 제출
router.post('/', authMiddleware, async (req, res) => {
  try {
    const { consultationId, doctorId, rating, tags, content } = req.body;
    // tags = ['친절해요', '설명 잘해요', '시간 지켜요']
    // content는 선택 (null 가능)

    // 중복 리뷰 방지
    const exists = await pool.query(
      `SELECT id FROM reviews WHERE consultation_id = $1`,
      [consultationId]
    );
    if (exists.rows.length > 0) {
      return res.status(409).json({ error: '이미 후기를 작성했습니다.' });
    }

    const result = await pool.query(
      `INSERT INTO reviews (consultation_id, patient_id, doctor_id, rating, tags, content)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
      [consultationId, req.user.id, doctorId, rating, tags || [], content || null]
    );

    // 의사 평균 별점 업데이트
    await pool.query(
      `UPDATE doctors SET rating = (
        SELECT ROUND(AVG(rating), 1) FROM reviews WHERE doctor_id = $1
       ) WHERE id = $1`,
      [doctorId]
    );

    res.status(201).json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/reviews/doctor/:doctorId - 의사 리뷰 목록
router.get('/doctor/:doctorId', authMiddleware, async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT r.*, p.name AS patient_name
       FROM reviews r
       JOIN patients p ON p.id = r.patient_id
       WHERE r.doctor_id = $1
       ORDER BY r.created_at DESC`,
      [req.params.doctorId]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;