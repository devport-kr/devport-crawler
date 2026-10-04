# devPort Crawler

devport.kr 크롤링 서비스

## 기술 스택

- **Python 3.11+**
- **FastAPI** - API 프레임워크
- **OpenAI gpt-6-luna** - 한국어 번역, 분류, 콘텐츠 검증 (`LLM_MODEL`로 교체 가능)
- **trafilatura / readability / markdownify** - 본문 추출 (HTML → Markdown)
- **SQLAlchemy** - ORM
- **PostgreSQL** - 데이터베이스
- **Playwright** - JS 렌더링이 필요한 페이지에만 쓰는 폴백

## 현재 상태

🚧 **테스트 진행 중**

- ✅ Dev.to 크롤러 - 테스트 완료
- 🚧 Medium, GitHub 크롤러 - 테스트 대기 중

## 주요 기능

### 데이터 소스

1. **개발 블로그**
   - Dev.to 인기 게시글 (최근 7일, 반응 4개 이상) — 전체 본문 fetch
   - Medium 프로그래밍 태그 — RSS 콘텐츠

2. **개발자 커뮤니티**
   - Hacker News 인기 스토리 (원문 기사 본문 fetch)

3. **GitHub**
   - 트렌딩 저장소 (별 50개 이상)
   - 최근 생성/업데이트된 인기 프로젝트

### 처리 파이프라인

```
목록 수집 → 이미 저장된 URL 제외 → 본문 수집(Markdown) → 중복 제거 → 콘텐츠 게이트
  → LLM 판정(본문 정상 여부·개발 관련성·카테고리·태그·제목) → 한국어 번역 → 출력 검증 → 기사별 즉시 저장
```

**본문 수집** (`app/crawlers/content.py`)
- 일반 HTTP 요청 + 본문 추출(trafilatura, readability, `<article>` 컨테이너 중 실제 문장이 가장 많이 남는 결과)
- 결과가 부실하면(SPA 껍데기, 봇 차단, 너무 짧음) 그 페이지만 Playwright로 렌더링
- GitHub 저장소 링크는 GitHub API로 README 원문을 가져옴, 영상/SNS/PDF 링크는 건너뜀
- 코드 블록·제목·목록·표 구조를 유지한 Markdown, 이미지·임베드·보이지 않는 문자 제거

**LLM** (`app/services/summarizer.py`)
- 기사 1개당 요청 1개 (배치 없음): ① JSON 판정 호출 → ② Markdown 번역 호출
- 판정 단계에서 오류/페이월/엉뚱한 본문, 개발과 무관한 글을 걸러 번역 비용을 쓰지 않음
- 번역은 요약이 아닌 충실한 번역 — 원문에 없는 내용 추가 금지, 코드 원문 유지, 합니다체 통일
- 출력 검증: 한국어 비율, 원문 대비 길이(부풀림 감지), 잘림(`finish_reason=length`) 시 재시도, 실패 시 저장 안 함
- GitHub 트렌딩은 README를 근거로 소개글 작성

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정 필요

# 서버 실행
uvicorn app.main:app --reload

# 크롤링 테스트
curl -X POST http://localhost:8000/api/crawl/devto
```

## API 엔드포인트

- `POST /api/crawl/devto` - Dev.to 크롤링
- `POST /api/crawl/medium` - Medium 크롤링
- `POST /api/crawl/github` - GitHub 크롤링
- `GET /api/health` - 헬스 체크
- `GET /api/stats` - 통계 조회

## 환경 변수

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/devportdb
OPENAI_API_KEY=your-api-key
LLM_MODEL=gpt-6-luna            # 번역 품질을 더 높이려면 gpt-6.1-sol
LLM_TRIAGE_MODEL=gpt-6-luna
GITHUB_TOKEN=your-github-token  # README 수집 (API 한도 60 → 5000 req/h)
MIN_REACTIONS_DEVTO=4
```

## 라이센스

MIT
