-- jbedu 수어 데이터셋 테이블
-- 섹션: 단어 / 문장 / 지명 / 회화수어
-- keypoints: 원본 NPZ 파일을 BYTEA로 저장 (평균 213KB/건, 총 ~843MB)

CREATE TABLE IF NOT EXISTS jbedu_entries (
    uuid            TEXT PRIMARY KEY,
    cid             TEXT,
    section         TEXT        NOT NULL,   -- 단어 / 문장 / 지명 / 회화수어
    category        TEXT        NOT NULL,   -- 교통 / 의학 / 시,도 / 회화 등
    korean_word     TEXT        NOT NULL,
    english_word    TEXT,
    description     TEXT,
    video_duration  REAL,
    video_path      TEXT,                   -- Volume 내 상대 경로: {section}/{category}/{uuid}.mp4
    has_video       BOOLEAN     NOT NULL DEFAULT FALSE,
    has_keypoints   BOOLEAN     NOT NULL DEFAULT FALSE,
    scraped_at      TIMESTAMPTZ,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    keypoints       BYTEA                   -- NPZ 파일 바이너리 (없으면 NULL)
);

-- 검색용 인덱스
CREATE INDEX IF NOT EXISTS idx_jbedu_section          ON jbedu_entries (section);
CREATE INDEX IF NOT EXISTS idx_jbedu_category         ON jbedu_entries (section, category);
CREATE INDEX IF NOT EXISTS idx_jbedu_korean_word      ON jbedu_entries (korean_word);
CREATE INDEX IF NOT EXISTS idx_jbedu_has_keypoints    ON jbedu_entries (has_keypoints) WHERE has_keypoints = TRUE;
CREATE INDEX IF NOT EXISTS idx_jbedu_has_video        ON jbedu_entries (has_video)     WHERE has_video = TRUE;

-- 섹션/카테고리 통계용 뷰
CREATE OR REPLACE VIEW jbedu_stats AS
SELECT
    section,
    category,
    COUNT(*)                                    AS total,
    COUNT(*) FILTER (WHERE has_video = TRUE)    AS with_video,
    COUNT(*) FILTER (WHERE has_keypoints = TRUE) AS with_keypoints
FROM jbedu_entries
GROUP BY section, category
ORDER BY section, category;
