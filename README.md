# devPort Crawler

devport.kr 크롤링 서비스

## 기술 스택

- **Python 3.11+**
- **AWS Lambda** - 서버리스 실행 환경
- **EventBridge Scheduler** - 스케줄링
- **Google Gemini 2.5 Flash** - LLM 기반 요약/카테고리화
- **SQLAlchemy** - ORM
- **PostgreSQL** - 데이터베이스
- **Playwright** - 웹 스크래핑

## 현재 상태

🚧 **테스트 진행 중**

- ✅ Dev.to 크롤러 - 테스트 완료
- 🚧 Hashnode, Medium, GitHub 크롤러 - 테스트 대기 중

## 주요 기능

### 데이터 소스

1. **개발 블로그**
   - Dev.to 인기 게시글 (최근 7일, 반응 4개 이상)
   - Hashnode 추천 아티클
   - Medium 프로그래밍 태그

2. **GitHub**
   - 트렌딩 저장소 (별 50개 이상)
   - 최근 생성/업데이트된 인기 프로젝트

### 처리 파이프라인

```
크롤링 → 중복 제거 → LLM 요약/카테고리화 → 점수 계산 → DB 저장
```

**LLM 통합**
- 한국어 제목/요약 자동 생성
- AI 기반 카테고리 분류 (12개 카테고리)
- 배치 처리 (25개 아티클/요청)로 효율성 극대화
- 요약 실패 시 저장 안함 (품질 보장)

## 빠른 시작

### 로컬 테스트

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 GEMINI_API_KEY 설정 필요

# 핸들러 직접 실행 (로컬 테스트)
python -m app.handler
```

### Lambda 배포

```bash
# Docker 이미지 빌드
docker build -t devport-crawler .

# ECR에 푸시 (AWS CLI 설정 필요)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag devport-crawler:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/devport-crawler:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/devport-crawler:latest

# Lambda 함수 업데이트
aws lambda update-function-code --function-name devport-crawler --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/devport-crawler:latest
```

## Lambda 이벤트 페이로드

Lambda는 EventBridge Scheduler에서 다음 이벤트를 받습니다:

- `{"source": "github"}` - GitHub 트렌딩 크롤링
- `{"source": "hashnode"}` - Hashnode 크롤링
- `{"source": "medium"}` - Medium 크롤링
- `{"source": "reddit"}` - Reddit 크롤링
- `{"source": "llm_rankings"}` - LLM 랭킹 크롤링
- `{"source": "all_blogs"}` - 모든 블로그 크롤링 (기본값)

## 환경 변수

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/devportdb
GEMINI_API_KEY=your-api-key
LLM_PROVIDER=gemini
GITHUB_TOKEN=your-github-token
MIN_REACTIONS_DEVTO=4
```

## 라이센스

MIT
