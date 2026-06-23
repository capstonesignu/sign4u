const passport = require('passport');
const KakaoStrategy = require('passport-kakao').Strategy;
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const NaverStrategy = require('passport-naver-v2').Strategy;
const pool = require('./db');

function parseState(req) {
  try {
    const raw = req.query.state;
    if (!raw) return {};
    const decoded = decodeURIComponent(raw);
    return JSON.parse(decoded);
  } catch (e) {
    try {
      return JSON.parse(req.query.state);
    } catch (e2) {
      return {};
    }
  }
}

passport.use(new KakaoStrategy({
  clientID: process.env.KAKAO_CLIENT_ID,
  clientSecret: process.env.KAKAO_CLIENT_SECRET,
  callbackURL: process.env.KAKAO_CALLBACK_URL,
  passReqToCallback: true,
}, async (req, accessToken, refreshToken, profile, done) => {
  try {
    const kakaoId = String(profile.id);
    const name = profile.displayName;
    const profileImageUrl = profile._json?.kakao_account?.profile?.profile_image_url || null;

    const state = parseState(req);
    const role = state.role || 'PATIENT';

    const table = role === 'DOCTOR' ? 'doctors' : 'patients';
    const existing = await pool.query(
      `SELECT * FROM ${table} WHERE oauth_provider = $1 AND oauth_id = $2`,
      ['kakao', kakaoId]
    );

    if (existing.rows.length > 0) {
      return done(null, { ...existing.rows[0], role, isNew: false });
    }

    let newUser;
    if (role === 'DOCTOR') {
      newUser = await pool.query(
        `INSERT INTO doctors (name, oauth_provider, oauth_id, profile_image_url)
         VALUES ($1, 'kakao', $2, $3) RETURNING *`,
        [name, kakaoId, profileImageUrl]
      );
    } else {
      newUser = await pool.query(
        `INSERT INTO patients (name, oauth_provider, oauth_id, profile_image_url)
         VALUES ($1, 'kakao', $2, $3) RETURNING *`,
        [name, kakaoId, profileImageUrl]
      );
    }

    return done(null, { ...newUser.rows[0], role, isNew: true });
  } catch (err) {
    return done(err, null);
  }
}));

passport.use(new GoogleStrategy({
  clientID: process.env.GOOGLE_CLIENT_ID,
  clientSecret: process.env.GOOGLE_CLIENT_SECRET,
  callbackURL: process.env.GOOGLE_CALLBACK_URL,
  passReqToCallback: true,
}, async (req, accessToken, refreshToken, profile, done) => {
  try {
    const googleId = String(profile.id);
    const name = profile.displayName;
    const profileImageUrl = profile.photos?.[0]?.value || null;
    const email = profile.emails?.[0]?.value || null;

    const state = parseState(req);
    const role = state.role || 'PATIENT';

    const table = role === 'DOCTOR' ? 'doctors' : 'patients';
    const existing = await pool.query(
      `SELECT * FROM ${table} WHERE oauth_provider = $1 AND oauth_id = $2`,
      ['google', googleId]
    );

    if (existing.rows.length > 0) {
      return done(null, { ...existing.rows[0], role, isNew: false });
    }

    let newUser;
    if (role === 'DOCTOR') {
      newUser = await pool.query(
        `INSERT INTO doctors (name, email, oauth_provider, oauth_id, profile_image_url)
         VALUES ($1, $2, 'google', $3, $4) RETURNING *`,
        [name, email, googleId, profileImageUrl]
      );
    } else {
      newUser = await pool.query(
        `INSERT INTO patients (name, email, oauth_provider, oauth_id, profile_image_url)
         VALUES ($1, $2, 'google', $3, $4) RETURNING *`,
        [name, email, googleId, profileImageUrl]
      );
    }

    return done(null, { ...newUser.rows[0], role, isNew: true });
  } catch (err) {
    return done(err, null);
  }
}));

///
passport.use(new NaverStrategy({
  clientID: process.env.NAVER_CLIENT_ID,
  clientSecret: process.env.NAVER_CLIENT_SECRET,
  callbackURL: process.env.NAVER_CALLBACK_URL,
  passReqToCallback: true,
}, async (req, accessToken, refreshToken, profile, done) => {
  try {
    const naverId = String(profile.id);
    //const name = profile.displayName;
    //const name = profile.displayName || profile._json?.nickname || profile.userName || profile.name || '사용자';
    const name = profile.name || profile.nickname || profile.displayName || '사용자';
    const profileImageUrl = profile.profileImage || profile._json?.profile_image || null;

    const state = parseState(req);
    const role = state.role || 'PATIENT';

    const table = role === 'DOCTOR' ? 'doctors' : 'patients';
    const existing = await pool.query(
      `SELECT * FROM ${table} WHERE oauth_provider = $1 AND oauth_id = $2`,
      ['naver', naverId]
    );

    if (existing.rows.length > 0) {
      return done(null, { ...existing.rows[0], role, isNew: false });
    }

    let newUser;
    if (role === 'DOCTOR') {
      newUser = await pool.query(
        `INSERT INTO doctors (name, oauth_provider, oauth_id, profile_image_url)
         VALUES ($1, 'naver', $2, $3) RETURNING *`,
        [name, naverId, profileImageUrl]
      );
    } else {
      newUser = await pool.query(
        `INSERT INTO patients (name, oauth_provider, oauth_id, profile_image_url)
         VALUES ($1, 'naver', $2, $3) RETURNING *`,
        [name, naverId, profileImageUrl]
      );
    }

    return done(null, { ...newUser.rows[0], role, isNew: true });
  } catch (err) {
    return done(err, null);
  }
}));

module.exports = passport;