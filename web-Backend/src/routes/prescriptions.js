const express = require('express');
const router = express.Router();
const pool = require('../config/db');
const authMiddleware = require('../middlewares/auth');

// GET /api/prescriptions/form/:consultationId - 처방전 작성 페이지 데이터
router.get('/form/:consultationId', authMiddleware, async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT c.id, c.scheduled_at, p.name AS patient_name
       FROM consultations c
       JOIN patients p ON p.id = c.patient_id
       WHERE c.id = $1 AND c.doctor_id = $2`,
      [req.params.consultationId, req.user.id]
    );
    if (!result.rows[0]) return res.status(404).json({ error: 'Not found' });
    const row = result.rows[0];
    res.json({
      id: row.id,
      patientName: row.patient_name,
      date: row.scheduled_at,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/prescriptions/:consultationId - 처방전 조회 (환자용)
router.get('/:consultationId', authMiddleware, async (req, res) => {
  try {
    const { consultationId } = req.params;

    const medicines = await pool.query(
      `SELECT * FROM prescriptions WHERE consultation_id = $1`,
      [consultationId]
    );

    const consultation = await pool.query(
      `SELECT c.scheduled_at, c.status, d.name AS doctor_name,
              d.profile_image_url AS doctor_profile_image,
              s.name AS specialty_name
       FROM consultations c
       JOIN doctors d ON d.id = c.doctor_id
       LEFT JOIN specialties s ON s.id = d.specialty_id
       WHERE c.id = $1`,
      [consultationId]
    );

    if (!consultation.rows[0]) {
      return res.status(404).json({ error: '진료를 찾을 수 없습니다.' });
    }

    const row = consultation.rows[0];
    // 진료가 COMPLETED인데 처방전이 없으면 의사가 '처방전 없음' 선택한 것
    const skipped = row.status === 'COMPLETED' && medicines.rows.length === 0;

    res.json({
      doctorName: row.doctor_name,
      doctorProfileImage: row.doctor_profile_image,
      specialty: row.specialty_name,
      date: row.scheduled_at,
      status: row.status,
      skipped,
      medicines: medicines.rows.map(m => ({
        id: m.id,
        name: m.medicine_name,
        dosage: m.dosage,
        duration: m.duration_days,
        times: {
          morning: m.morning,
          lunch: m.lunch,
          dinner: m.evening,
        },
      })),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/prescriptions - 발급 완료
router.post('/', authMiddleware, async (req, res) => {
  try {
    const { consultationId, medicines } = req.body;

    const inserted = await Promise.all(
      medicines.map(m =>
        pool.query(
          `INSERT INTO prescriptions
            (consultation_id, medicine_name, dosage, duration_days, morning, lunch, evening)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           RETURNING *`,
          [
            consultationId,
            m.name,             // ✅ m.medicineName → m.name
            m.dosage,
            m.duration,         // ✅ m.durationDays → m.duration
            m.times.morning,    // ✅ m.morning → m.times.morning
            m.times.lunch,      // ✅ m.lunch → m.times.lunch
            m.times.dinner,     // ✅ m.evening → m.times.dinner
          ]
        )
      )
    );

    await pool.query(
      `UPDATE consultations SET status = 'COMPLETED', ended_at = NOW() WHERE id = $1`,
      [consultationId]
    );

    res.status(201).json(inserted.map(r => r.rows[0]));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/prescriptions/skip - 처방전 없음
router.post('/skip', authMiddleware, async (req, res) => {
  try {
    const { consultationId } = req.body;

    await pool.query(
      `UPDATE consultations SET status = 'COMPLETED', ended_at = NOW() WHERE id = $1`,
      [consultationId]
    );

    res.json({ message: '처방전 없이 진료가 완료되었습니다.' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;