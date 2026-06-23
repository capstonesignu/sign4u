# Sign4U Web Backend

Sign4U는 청각장애인 환자가 수어 기반 AI 번역을 통해 의료진과 더 쉽게 비대면 진료를 받을 수 있도록 돕는 서비스입니다. 이 저장소는 Sign4U의 웹 백엔드로, 사용자 인증, 진료 예약, 상담 기록, 처방전, 리뷰, 실시간 상담 연결, AI 번역 서버 연동을 담당합니다.

## Backend Role

백엔드는 프론트엔드, 데이터베이스, AI 서버 사이에서 서비스 흐름을 연결합니다.

- OAuth 로그인 후 JWT를 발급해 환자/의사 사용자를 인증합니다.
- 진료 예약, 상담 내역, 처방전, 리뷰 데이터를 PostgreSQL에 저장하고 제공합니다.
- 실시간 상담 중 WebSocket으로 참여자 입장, 채팅, 번역 결과를 전달합니다.
- 프론트엔드의 수어 인식 요청을 AI 서버로 전달하고 streaming 응답을 다시 클라이언트로 중계합니다.

## Core Features

- 소셜 로그인 기반 환자/의사 인증
- 진료과 및 의사 목록 조회
- 진료 예약 생성, 조회, 시작, 종료, 삭제
- 상담 중 실시간 메시지 및 번역 결과 전달
- 상담 기록 저장
- 처방전 작성 및 조회
- 진료 후 리뷰 작성 및 조회
- AI 수어 번역 서버 연동

## Tech Stack

- Runtime: Node.js, Express 5
- Database: PostgreSQL
- Auth: Passport.js OAuth(Kakao, Google, Naver), JWT
- Realtime: ws
- AI integration: Axios proxy to external AI server

## Project Structure

```text
src/
|-- index.js              # Express app, HTTP server, WebSocket setup
|-- config/
|   |-- db.js             # PostgreSQL connection pool
|   `-- passport.js       # OAuth strategies
|-- middlewares/
|   `-- auth.js           # JWT auth middleware
`-- routes/
    |-- ai.js             # AI server proxy
    |-- auth.js           # OAuth login, callback, logout, withdraw
    |-- consultations.js  # Consultation reservation/history/messages
    |-- doctors.js        # Doctor list/detail/profile specialty
    |-- prescriptions.js  # Prescription form/read/create/skip
    |-- reviews.js        # Review form/create/list
    |-- specialties.js    # Specialty list
    |-- users.js          # Current user profile
    `-- websocket.js      # Real-time consultation socket
```

## Install and Run

```bash
npm install
npm run dev
```

Production:

```bash
npm start
```

The server currently listens on `0.0.0.0:3000`.

## Scripts

```bash
npm run dev   # start with nodemon
npm start     # start with node
npm test      # syntax check entrypoint
```

## Main API Groups

- `GET /` - server and DB health check
- `/api/auth/*` - OAuth login/callback, logout, account withdrawal
- `/api/users/*` - authenticated user profile
- `/api/doctors/*` - doctor list/detail and doctor specialty update
- `/api/specialties/*` - specialty list
- `/api/consultations/*` - reservation, history, detail, messages, start/end/delete
- `/api/prescriptions/*` - prescription form, lookup, create, skip
- `/api/reviews/*` - review form, create, doctor review list
- `POST /predict` and `POST /api/ai/predict` - proxy request to the AI translation server
- `GET /health` and `GET /api/ai/health` - AI server health proxy

## Service Flow

### Login and User Management

```text
User
  -> OAuth login
Backend
  -> provider authentication
  -> patient/doctor account lookup or creation
  -> JWT issue
Frontend
  -> uses JWT for protected API requests
```

### Consultation

```text
Patient
  -> selects doctor and available time
Backend
  -> creates consultation reservation
Doctor / Patient
  -> enters real-time consultation room
Backend
  -> relays messages and stores consultation records
```

## AI Proxy Flow

```text
Client
  -> POST /predict or /api/ai/predict
Backend src/routes/ai.js
  -> forwards request to AI translation server
AI server
  -> streaming response
Backend
  -> forwards text/event-stream to client
```

If the client closes the connection, the backend aborts the upstream AI request with `AbortController`.

## WebSocket Flow

Connect to the backend WebSocket server with:

```text
ws://localhost:3000?token=<JWT>&consultation_id=<CONSULTATION_ID>
```

The socket verifies the JWT, joins the consultation room, broadcasts participant events, relays chat/translation messages, and stores consultation records in PostgreSQL.

## Database Tables Used

- `patients`
- `doctors`
- `specialties`
- `consultations`
- `consultation_records`
- `prescriptions`
- `reviews`
- `translation_logs`
