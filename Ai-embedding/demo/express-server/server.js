process.stdout.write("[boot] node process started\n");

process.on("uncaughtException", (err) => {
  console.error("[fatal] uncaughtException:", err.message, err.stack);
});
process.on("unhandledRejection", (reason) => {
  console.error("[fatal] unhandledRejection:", reason);
});

const crypto  = require("crypto");
const express = require("express");
const path    = require("path");
const fs      = require("fs");
const http    = require("http");
const https   = require("https");

const app = express();
const PORT            = process.env.PORT || 3000;
const FASTAPI_URL     = process.env.FASTAPI_URL     || "http://localhost:8001";
const DATA_SERVER_URL = process.env.DATA_SERVER_URL || "";

const SENTENCE_MAPPING_PATH = path.resolve(__dirname, "../../sentence_mapping.json");
const VIDEO_ROOT             = path.resolve(__dirname, "../../dataset/WORD-video");
const WORD_MAPPING_PATH      = path.resolve(__dirname, "../../word_mapping.json");

// ── data-server 내부 호출 헬퍼 ──────────────────────────────────────────────
function _dataCall(apiPath, body = null, method = null) {
  if (!DATA_SERVER_URL) return Promise.resolve(null);
  const parsed = new URL(DATA_SERVER_URL);
  const lib = parsed.protocol === "https:" ? https : http;
  return new Promise((resolve) => {
    const payload = body ? JSON.stringify(body) : null;
    const resolvedMethod = method || (payload ? "POST" : "GET");
    const opts = {
      hostname: parsed.hostname,
      port:     parsed.port || (parsed.protocol === "https:" ? 443 : 80),
      path:     apiPath,
      method:   resolvedMethod,
      headers:  {
        "Content-Type": "application/json",
        ...(payload ? { "Content-Length": Buffer.byteLength(payload) } : {}),
      },
    };
    const req = lib.request(opts, (res) => {
      let raw = "";
      res.on("data", (c) => { raw += c; });
      res.on("end", () => {
        try { resolve(JSON.parse(raw)); } catch { resolve(null); }
      });
    });
    req.on("error", (e) => {
      console.error("[data-call]", apiPath, e.message);
      resolve(null);
    });
    if (payload) req.write(payload);
    req.end();
  });
}

// ── 쿠키 파서 ────────────────────────────────────────────────────────────────
function parseCookies(cookieHeader) {
  const out = {};
  (cookieHeader || "").split(";").forEach((c) => {
    const i = c.indexOf("=");
    if (i < 0) return;
    out[c.slice(0, i).trim()] = decodeURIComponent(c.slice(i + 1).trim());
  });
  return out;
}

// ── 인메모리 세션 {username, is_admin} ───────────────────────────────────────
const _sessions = new Map();

function _session(req) {
  const { sid } = parseCookies(req.headers.cookie);
  return sid ? _sessions.get(sid) : null;
}

function requireAuth(req, res, next) {
  if (!DATA_SERVER_URL) return next();
  if (_session(req)) return next();
  if ((req.headers.accept || "").includes("application/json"))
    return res.status(401).json({ error: "Unauthorized" });
  res.redirect("/login");
}

function requireAdmin(req, res, next) {
  if (!DATA_SERVER_URL) return next();
  const sess = _session(req);
  if (!sess) return res.redirect("/login");
  if (!sess.is_admin) {
    if ((req.headers.accept || "").includes("application/json"))
      return res.status(403).json({ error: "Forbidden" });
    return res.redirect("/");
  }
  next();
}

// ── 경량 프록시 ──────────────────────────────────────────────────────────────
function makeProxy(targetBase, prefix = "") {
  const parsed = new URL(targetBase);
  const lib = parsed.protocol === "https:" ? https : http;
  return (req, res) => {
    const opts = {
      hostname: parsed.hostname,
      port:     parsed.port || (parsed.protocol === "https:" ? 443 : 80),
      path:     prefix + req.url,
      method:   req.method,
      headers:  { ...req.headers, host: parsed.host },
    };
    const proxy = lib.request(opts, (pr) => {
      res.writeHead(pr.statusCode, pr.headers);
      pr.pipe(res, { end: true });
    });
    proxy.on("error", (e) => {
      console.error(`[proxy] ${req.method} ${req.url} → ${e.message}`);
      if (!res.headersSent) res.status(502).json({ error: "upstream unavailable" });
    });
    req.pipe(proxy, { end: true });
  };
}

// ── 공개 라우트 (인증 불필요) ─────────────────────────────────────────────────
app.get("/health", (_req, res) => res.json({ status: "ok" }));

// 회원가입 — 누구나 가입 가능, 가입 후 관리자 승인 필요
app.get("/signup", (_req, res) =>
  res.sendFile(path.join(__dirname, "public/signup.html"))
);

app.post("/signup", express.urlencoded({ extended: false }), async (req, res) => {
  const { username, full_name, password, password2 } = req.body;
  if (!full_name?.trim() || !password || password !== password2 || password.length < 6)
    return res.redirect("/signup?error=mismatch");
  const r = await _dataCall("/data/users/signup", {
    username,
    full_name: full_name.trim(),
    password,
  });
  if (r?.ok) return res.redirect("/signup?success=1");
  const code = r?.detail?.includes("이미") ? "duplicate" : "fail";
  res.redirect("/signup?error=" + code);
});

// 최초 관리자 계정 생성
app.get("/setup", async (_req, res) => {
  const r = await _dataCall("/data/users/exists");
  if (r?.exists) return res.redirect("/login");
  res.sendFile(path.join(__dirname, "public/setup.html"));
});

app.post("/setup", express.urlencoded({ extended: false }), async (req, res) => {
  const { username, full_name, password, password2 } = req.body;
  if (!full_name?.trim() || !password || password !== password2 || password.length < 6)
    return res.redirect("/setup?error=1");
  const r = await _dataCall("/data/users/setup", {
    username,
    full_name: full_name.trim(),
    password,
  });
  if (r?.ok) return res.redirect("/login?setup=1");
  res.redirect("/setup?error=" + (r ? "2" : "1"));
});

app.get("/login", (req, res) => {
  if (_session(req)) return res.redirect("/");
  res.sendFile(path.join(__dirname, "public/login.html"));
});

app.post("/login", express.urlencoded({ extended: false }), async (req, res) => {
  const { username, password } = req.body;
  const r = await _dataCall("/data/users/login", { username, password });
  if (r?.ok) {
    let isAdmin = r.is_admin || false;
    if (!isAdmin) {
      // DB에 관리자가 없으면(이전 스키마 호환) 가장 먼저 가입한 계정을 관리자로 처리
      const list = await _dataCall("/data/users/list");
      const users = list?.users || [];
      if (users.length > 0 && users.every(u => !u.is_admin)) {
        isAdmin = users[0].username === username;
      }
    }
    const sid = crypto.randomBytes(20).toString("hex");
    _sessions.set(sid, { username, full_name: r.full_name || username, is_admin: isAdmin });
    res.setHeader(
      "Set-Cookie",
      `sid=${sid}; HttpOnly; SameSite=Strict; Path=/; Max-Age=${60 * 60 * 24}`
    );
    return res.redirect("/");
  }
  const errorCode = r?.reason === "pending" ? "pending" : "1";
  res.redirect("/login?error=" + errorCode);
});

app.post("/logout", (req, res) => {
  const { sid } = parseCookies(req.headers.cookie);
  _sessions.delete(sid);
  res.setHeader("Set-Cookie", "sid=; Max-Age=0; Path=/");
  res.redirect("/login");
});

// ── 인증 가드 ─────────────────────────────────────────────────────────────────
app.use(requireAuth);

// 현재 세션 정보 (프론트엔드 헤더 렌더링용)
app.get("/me", (req, res) => {
  const sess = _session(req);
  if (!sess) return res.status(401).json(null);
  res.json({ username: sess.username, full_name: sess.full_name || sess.username, is_admin: sess.is_admin || false });
});

// ── 관리자 패널 ───────────────────────────────────────────────────────────────
app.get("/admin", requireAdmin, (_req, res) =>
  res.sendFile(path.join(__dirname, "public/admin.html"))
);

app.get("/admin/pending", requireAdmin, async (_req, res) => {
  const r = await _dataCall("/data/users/pending");
  res.json(r || { users: [] });
});

app.get("/admin/users", requireAdmin, async (_req, res) => {
  const r = await _dataCall("/data/users/list");
  res.json(r || { users: [] });
});

app.post("/admin/approve", requireAdmin, express.json(), async (req, res) => {
  const r = await _dataCall("/data/users/approve", { user_id: req.body.user_id });
  res.json(r || { ok: false });
});

app.post("/admin/reject", requireAdmin, express.json(), async (req, res) => {
  const r = await _dataCall(`/data/users/${req.body.user_id}`, null, "DELETE");
  res.json(r || { ok: false });
});

// ── FastAPI 프록시 ────────────────────────────────────────────────────────────
const fastapiIsLocal =
  FASTAPI_URL.includes("localhost") || FASTAPI_URL.includes("127.0.0.1");
if (!fastapiIsLocal) {
  app.use("/api", makeProxy(FASTAPI_URL));
} else {
  app.use("/api", (_req, res) =>
    res.status(503).json({ error: "API server not available" })
  );
}

// data-server 프록시
if (DATA_SERVER_URL) {
  app.use("/data", makeProxy(DATA_SERVER_URL, "/data"));
}

// 레퍼런스 데이터
app.get("/ref/sentences", (req, res) => {
  if (!fs.existsSync(SENTENCE_MAPPING_PATH))
    return res.status(404).json({ error: "sentence_mapping.json not found" });
  res.sendFile(SENTENCE_MAPPING_PATH);
});

app.get("/ref/words", (req, res) => {
  if (!fs.existsSync(WORD_MAPPING_PATH) || !fs.existsSync(VIDEO_ROOT))
    return res.json([]);
  const wordMap = JSON.parse(fs.readFileSync(WORD_MAPPING_PATH, "utf8"));
  const wordToVideos = {};
  const RE = /NIA_SL_(WORD\d+)_REAL\d+_F\.mp4$/;
  for (const dir of fs.readdirSync(VIDEO_ROOT)) {
    const dirPath = path.join(VIDEO_ROOT, dir);
    if (!fs.statSync(dirPath).isDirectory()) continue;
    for (const file of fs.readdirSync(dirPath)) {
      const m = file.match(RE);
      if (!m) continue;
      const kor = wordMap[m[1]];
      if (!kor) continue;
      if (!wordToVideos[kor]) wordToVideos[kor] = [];
      wordToVideos[kor].push(`/ref-videos/${dir}/${file}`);
    }
  }
  res.json(
    Object.entries(wordToVideos)
      .map(([word, videos]) => ({ word, video: videos[0], videos }))
      .sort((a, b) => a.word.localeCompare(b.word, "ko"))
  );
});

app.use("/ref-videos", express.static(VIDEO_ROOT));
app.use(express.static(path.join(__dirname, "public")));

app.listen(PORT, () => {
  console.log(`[express] http://localhost:${PORT}`);
  console.log(`[express] Data server: ${DATA_SERVER_URL || "없음"}`);
});
