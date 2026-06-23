const express = require('express');
const router = express.Router();
const pool = require('../config/db');
const authMiddleware = require('../middlewares/auth');

// 24시간 30분 단위 슬롯 (00:00 ~ 23:30, 48개)
const DEFAULT_SLOTS = Array.from({ length: 48 }, (_, i) => {
  const h = String(Math.floor(i / 2)).padStart(2, '0');
  const m = i % 2 === 0 ? '00' : '30';
  return `${h}:${m}`;
});

router.get('/slots', authMiddleware, async (req, res)=>{
  try{
    const { doctorId, date} = req.query;
    if (!doctorId || !date){
      return res.status(400).json({error:'doctorId and date are required.'});
    }
    // 해당 날짜에 이미 예약된 시간 조회 (PENDING or IN_PROGRESS)
    const result = await pool.query(
      `SELECT to_char(scheduled_at AT TIME ZONE 'Asia/Seoul', 'HH24:MI') AS time
       FROM consultations
       WHERE doctor_id = $1
         AND (scheduled_at AT TIME ZONE 'Asia/Seoul')::date = $2::date
         AND status IN ('PENDING', 'IN_PROGRESS')`,
      [doctorId, date]
    );
    const bookedTimes = new Set(result.rows.map(r => r.time));

    // 오늘 날짜인 경우 현재 시간 이전 슬롯은 비활성화 (KST 기준)
    const now = new Date();
    const kstDate = now.toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
    const isToday = date === kstDate;
    const kstTime = now.toLocaleTimeString('en-GB', { timeZone: 'Asia/Seoul', hour12: false });
    const [kstH, kstM] = kstTime.split(':').map(Number);
    const currentTime = kstH * 60 + kstM;

    const slots = DEFAULT_SLOTS.map(time => {
      let available = !bookedTimes.has(time);
      if (isToday && available) {
        const [h, m] = time.split(':').map(Number);
        if (h * 60 + m <= currentTime) {
          available = false;
        }
      }
      return { time, available };
    });

    res.json(slots);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
})

// 환자 - 예약 생성
router.post('/', authMiddleware, async (req, res) => {
  try {
    if (req.user.role !== 'PATIENT') {
      return res.status(403).json({ error: '환자만 예약할 수 있습니다.' });
    }

    const { doctorId, scheduledAt } = req.body;
    const patientId = req.user.id;

    // scheduledAt이 타임존 없이 오면 KST로 해석되도록 오프셋 추가
    const scheduledAtKST = scheduledAt.includes('+') || scheduledAt.includes('Z')
      ? scheduledAt
      : `${scheduledAt}+09:00`;

    // 과거 시간 예약 방지
    if (new Date(scheduledAtKST) < new Date()) {
      return res.status(400).json({ error: '과거 시간에는 예약할 수 없습니다.' });
    }

    // 이미 예약된 슬롯인지 체크
    const conflict = await pool.query(
      `SELECT id FROM consultations
       WHERE doctor_id = $1
         AND scheduled_at = $2
         AND status IN ('PENDING', 'IN_PROGRESS')`,
      [doctorId, scheduledAtKST]
    );
    if (conflict.rows.length > 0) {
      return res.status(409).json({ error: '이미 예약된 시간입니다.' });
    }

    const result = await pool.query(
      `INSERT INTO consultations (patient_id, doctor_id, scheduled_at)
       VALUES ($1, $2, $3) RETURNING *`,
      [patientId, doctorId, scheduledAtKST]
    );
    res.status(201).json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// 예약 목록 조회 (?period=today|tomorrow|week, 기본값: today)
router.get('/', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const role = req.user.role;
    const field = role === 'DOCTOR' ? 'doctor_id' : 'patient_id';
    const partnerTable = role === 'DOCTOR' ? 'patients' : 'doctors';
    const partnerField = role === 'DOCTOR' ? 'c.patient_id' : 'c.doctor_id';
    const period = req.query.period || 'today';

    await pool.query(
      `UPDATE consultations SET status = 'EXPIRED'
       WHERE status = 'PENDING' AND scheduled_at + INTERVAL '30 minutes' < NOW()`
    );

    const kstToday = `(NOW() AT TIME ZONE 'Asia/Seoul')::date`;
    let dateCondition;
    switch (period) {
      case 'tomorrow':
        dateCondition = `(c.scheduled_at AT TIME ZONE 'Asia/Seoul')::date = ${kstToday} + 1`;
        break;
      case 'week':
        dateCondition = `(c.scheduled_at AT TIME ZONE 'Asia/Seoul')::date >= ${kstToday} 
          AND (c.scheduled_at AT TIME ZONE 'Asia/Seoul')::date < date_trunc('week', ${kstToday}) + interval '7 days'`;
        break;
      default:
        dateCondition = `(c.scheduled_at AT TIME ZONE 'Asia/Seoul')::date = ${kstToday}`;
    }

    const result = await pool.query(
      `SELECT c.*, 
              p.name AS partner_name, 
              p.profile_image_url AS partner_profile_image_url
       FROM consultations c
       JOIN ${partnerTable} p ON p.id = ${partnerField}
       WHERE c.${field} = $1
         AND ${dateCondition}
         AND c.status NOT IN ('EXPIRED', 'COMPLETED')
       ORDER BY c.scheduled_at ASC`,
      [userId]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 진료 기록 (완료된 것만)
router.get('/history', authMiddleware, async(req, res)=>{
    try{
      const userId = req.user.id;
      const role=req.user.role;
      const field = role ==='DOCTOR'? 'doctor_id':'patient_id';
      const partnerTable = role === 'DOCTOR' ? 'patients' : 'doctors';
      const partnerField = role === 'DOCTOR' ? 'c.patient_id' : 'c.doctor_id';
      let query;
      if (role === 'DOCTOR') {
        query = `SELECT c.*, u.name as partner_name, u.profile_image_url as partner_image
            FROM consultations c
            JOIN patients u ON u.id = c.patient_id
            WHERE c.doctor_id = $1 AND c.status = 'COMPLETED'
            ORDER BY c.ended_at DESC`;
      } else {
        query = `SELECT c.*, u.name as partner_name, u.profile_image_url as partner_image,
                s.name as partner_specialty
            FROM consultations c
            JOIN doctors u ON u.id = c.doctor_id
            LEFT JOIN specialties s ON s.id = u.specialty_id
            WHERE c.patient_id = $1 AND c.status = 'COMPLETED'
            ORDER BY c.ended_at DESC`;
      }
      const result = await pool.query(query, [userId]);
      res.json(result.rows);
    } catch(err){
        res.status(500).json({error:err.message});
    }
});

// 단일 예약 조회 (본인 관련만)
router.get('/:id', authMiddleware, async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT c.*, d.name as doctor_name, d.profile_image_url as doctor_image,
              p.name as patient_name, p.profile_image_url as patient_image,
              s.name as specialty_name
       FROM consultations c
       JOIN doctors d ON d.id = c.doctor_id
       JOIN patients p ON p.id = c.patient_id
       LEFT JOIN specialties s ON s.id = d.specialty_id
       WHERE c.id = $1`,
      [req.params.id]
    );
    if (!result.rows[0]) return res.status(404).json({ error: 'Not found' });

    const consultation = result.rows[0];
    const userId = req.user.id;
    if (consultation.doctor_id !== userId && consultation.patient_id !== userId) {
      return res.status(403).json({ error: '권한이 없습니다.' });
    }

    res.json(consultation);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 대화 내용 조회
router.get('/:id/messages', authMiddleware, async (req, res) => {
  try {
    // 본인 관련 진료인지 확인
    const check = await pool.query(
      `SELECT patient_id, doctor_id FROM consultations WHERE id = $1`,
      [req.params.id]
    );
    if (!check.rows[0]) return res.status(404).json({ error: '진료를 찾을 수 없습니다.' });

    const { patient_id, doctor_id } = check.rows[0];
    if (patient_id !== req.user.id && doctor_id !== req.user.id) {
      return res.status(403).json({ error: '권한이 없습니다.' });
    }

    const result = await pool.query(
      `SELECT id, translated_text, confidence, speaker, created_at
       FROM translation_logs
       WHERE consultation_id = $1
       ORDER BY created_at ASC`,
      [req.params.id]
    );

    res.json(result.rows.map(row => ({
      id: row.id,
      text: row.translated_text,
      confidence: row.confidence,
      sender: row.speaker,
      time: new Date(row.created_at).toLocaleTimeString('ko-KR', {
        hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Seoul'
      }),
    })));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 진료 시작 (의사만)
router.patch('/:id/start', authMiddleware, async (req, res) => {
  try {
    if (req.user.role !== 'DOCTOR') {
      return res.status(403).json({ error: '의사만 진료를 시작할 수 있습니다.' });
    }

    const check = await pool.query(
      `SELECT * FROM consultations WHERE id = $1`,
      [req.params.id]
    );
    if (!check.rows[0]) return res.status(404).json({ error: '예약을 찾을 수 없습니다.' });

    const consultation = check.rows[0];

    if (consultation.doctor_id !== req.user.id) {
      return res.status(403).json({ error: '본인의 예약만 시작할 수 있습니다.' });
    }
    if (consultation.status !== 'PENDING') {
      return res.status(400).json({ error: '시작할 수 없는 상태입니다.' });
    }

    const result = await pool.query(
      `UPDATE consultations SET status = 'IN_PROGRESS', started_at = NOW()
       WHERE id = $1 RETURNING *`,
      [req.params.id]
    );
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 진료 종료 (의사만)
router.patch('/:id/end', authMiddleware, async (req, res) => {
  try {
    if (req.user.role !== 'DOCTOR') {
      return res.status(403).json({ error: '의사만 진료를 종료할 수 있습니다.' });
    }

    const check = await pool.query(
      `SELECT * FROM consultations WHERE id = $1`,
      [req.params.id]
    );
    if (!check.rows[0]) return res.status(404).json({ error: '예약을 찾을 수 없습니다.' });

    const consultation = check.rows[0];

    if (consultation.doctor_id !== req.user.id) {
      return res.status(403).json({ error: '본인의 진료만 종료할 수 있습니다.' });
    }
    if (consultation.status !== 'IN_PROGRESS') {
      return res.status(400).json({ error: '진행 중인 진료만 종료할 수 있습니다.' });
    }

    // ended_at만 기록. status는 처방전 발급/skip 시점에 COMPLETED로 전환됨.
    // (이전엔 여기서 바로 COMPLETED 처리 → 환자가 처방전 폴링할 때
    //  의사 작성 중인데도 "처방전 없습니다"로 잘못 표시되는 버그)
    const result = await pool.query(
      `UPDATE consultations SET ended_at = NOW()
       WHERE id = $1 RETURNING *`,
      [req.params.id]
    );
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// 예약 취소 (본인 관련만)
router.delete('/:id', authMiddleware, async (req, res) => {
  try {
    const { id, role } = req.user;
    const field = role === 'DOCTOR' ? 'doctor_id' : 'patient_id';

    const check = await pool.query(
      `SELECT * FROM consultations WHERE id = $1`,
      [req.params.id]
    );
    if (!check.rows[0]) {
      return res.status(404).json({ error: '예약을 찾을 수 없습니다.' });
    }

    const consultation = check.rows[0];
    if (consultation[field] !== id) {
      return res.status(403).json({ error: '본인의 예약만 취소할 수 있습니다.' });
    }
    if (consultation.status !== 'PENDING') {
      return res.status(400).json({ error: '대기 중인 예약만 취소할 수 있습니다.' });
    }

    await pool.query(
      `DELETE FROM consultations WHERE id = $1`,
      [req.params.id]
    );

    res.json({ message: '예약이 취소되었습니다.' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;

