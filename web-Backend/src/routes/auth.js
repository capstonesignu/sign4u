const express = require('express');
const router = express.Router();
const passport = require('../config/passport');
const jwt = require('jsonwebtoken');
const pool = require('../config/db');
const authMiddleware = require('../middlewares/auth');
const frontendURL = process.env.FRONTEND_URL;

// 카카오 로그인 시작
router.get('/kakao', (req, res, next) => {
  const { role } = req.query;
  const state = JSON.stringify({ role: role || 'PATIENT' });
  passport.authenticate('kakao', { state })(req, res, next);
});

// callback
router.get('/kakao/callback',
  passport.authenticate('kakao', { session: false, failureRedirect: `${frontendURL}/login?error=auth_failed` }),
  (req, res) => {
    const token = jwt.sign(
      { id: req.user.id, role: req.user.role },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );
    if(!frontendURL){
      return res.status(500).json({error:'FRONTEND_URL not configured'});
    }
    res.redirect(`${frontendURL}/auth/callback?token=${token}&role=${req.user.role}&isNew=${req.user.isNew}`);
  }
);

// google login start
router.get('/google', (req, res, next) => {
  const { role } = req.query;
  const state = JSON.stringify({ role: role || 'PATIENT' });
  passport.authenticate('google', { scope: ['profile', 'email'], state })(req, res, next);
});

// callback
router.get('/google/callback',
  passport.authenticate('google', { session: false, failureRedirect: `${frontendURL}/login?error=auth_failed` }),
  (req, res) => {
    const token = jwt.sign(
      { id: req.user.id, role: req.user.role },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );
    if(!frontendURL){
      return res.status(500).json({error:'FRONTEND_URL not configured'});
    }
    res.redirect(`${frontendURL}/auth/callback?token=${token}&role=${req.user.role}&isNew=${req.user.isNew}`);
  }
);

// naver login start
router.get('/naver', (req, res, next) => {
  const { role } = req.query;
  const state = JSON.stringify({ role: role || 'PATIENT' });
  passport.authenticate('naver', { scope: ['profile', 'email'], state })(req, res, next);
});

// naver callback
router.get('/naver/callback',
  passport.authenticate('naver', { session: false, failureRedirect: `${frontendURL}/login?error=auth_failed` }),
  (req, res) => {
    const token = jwt.sign(
      { id: req.user.id, role: req.user.role },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );
    if(!frontendURL){
      return res.status(500).json({error:'FRONTEND_URL not configured'});
    }
    res.redirect(`${frontendURL}/auth/callback?token=${token}&role=${req.user.role}&isNew=${req.user.isNew}`);
  }
);

// logout
router.post('/logout', (req, res) => {
  res.json({ message: '로그아웃 성공' });
});

// 회원 탈퇴
router.delete('/withdraw', authMiddleware, async (req, res) => {
  const { id, role } = req.user;
  const isDoctor = role === 'DOCTOR';
  const table = isDoctor ? 'doctors' : 'patients';
  const userFkColumn = isDoctor ? 'doctor_id' : 'patient_id';

  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    const consultSubquery = `SELECT id FROM consultations WHERE ${userFkColumn} = $1`;
  
    // 자식 테이블들 먼저 삭제
    await client.query(`DELETE FROM translation_logs      WHERE consultation_id IN (${consultSubquery})`, [id]);
    await client.query(`DELETE FROM consultation_messages  WHERE consultation_id IN (${consultSubquery})`, [id]);
    await client.query(`DELETE FROM prescriptions          WHERE consultation_id IN (${consultSubquery})`, [id]);
    await client.query(`DELETE FROM reviews                WHERE consultation_id IN (${consultSubquery})`, [id]);
    
    // 그 다음 consultations 삭제
    await client.query(`DELETE FROM consultations          WHERE ${userFkColumn} = $1`, [id]);

    const result = await client.query(`DELETE FROM ${table} WHERE id = $1 RETURNING id`, [id]);
    if (result.rows.length === 0) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: '유저를 찾을 수 없습니다' });
    }

    await client.query('COMMIT');
    res.json({ message: '회원 탈퇴가 완료되었습니다' });
  } catch (err) {
    await client.query('ROLLBACK');
    console.error('회원 탈퇴 오류:', err);
    res.status(500).json({ error: '서버 오류가 발생했습니다' });
  } finally {
    client.release();
  }
});

module.exports = router;