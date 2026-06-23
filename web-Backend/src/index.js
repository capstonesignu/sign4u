const express = require('express');
const cors = require('cors');
const  passport = require('passport');
const http = require('http');
const pool = require('./config/db');
require('dotenv').config();
require('./config/passport');

const usersRouter = require('./routes/users');
const specialtiesRouter = require('./routes/specialties');
const doctorsRouter = require('./routes/doctors');
const authRouter = require('./routes/auth');
const consultationsRouter = require('./routes/consultations');
const aiRouter=require('./routes/ai');
const prescriptionsRouter = require('./routes/prescriptions');
const { setupWebSocket } = require('./routes/websocket');
const reviewsRouter = require('./routes/reviews');

const app = express();
const server =  http.createServer(app);


// server.js
app.use(cors({
  origin: [
    'http://localhost:5173',           // 로컬 개발용
    'https://medisone.vercel.app'       // 프로덕션
  ]
}));

// 수어 키포인트({ words: [...] }) 페이로드가 크므로 본문 크기 상향
app.use(express.json({ limit: '25mb' }));
app.use(passport.initialize());

app.use('/api/auth', authRouter);
app.use('/api/users', usersRouter);
app.use('/api/specialties', specialtiesRouter);
app.use('/api/doctors', doctorsRouter);
app.use('/api/consultations', consultationsRouter);
app.use('/api/prescriptions', prescriptionsRouter);
app.use('/api/reviews', reviewsRouter);

// 프론트(useSignRecorder)는 ${VITE_API_BASE_URL}/predict 로 호출하므로 루트에 마운트.
// /api/ai/predict 경로도 함께 노출(문서/호환).
app.use('/', aiRouter);
app.use('/api/ai', aiRouter);

app.get('/', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ message: 'Sign2U 서버 작동 중, DB 연결 성공' });
  } catch (err) {
    res.status(500).json({ error: 'DB 연결 실패', detail: err.message });
  }
})

setupWebSocket(server);
server.listen(3000, '0.0.0.0', () => {
  console.log('server running');
});

// const PORT = process.env.PORT || 3000;
// server.listen(PORT, () => {
//   console.log(`서버 실행 중: http://localhost:${PORT}`);
// });



