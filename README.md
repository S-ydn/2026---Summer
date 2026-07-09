# 충북대 학사/생활 정보 Agent

## 1. 서비스 소개 및 사용 시나리오
충북대학교 학생을 위한 학사/생활 정보 AI 에이전트
- 학사일정을 PDF 기반으로 정확하게 안내
- 학교 근처 맛집(식사 메뉴 선정 도움), 공모전/학부 공지 등 웹 기반 정보 제공
- 날씨 안내를 통해 외출 준비를 돕습니다

### 사용 시나리오 예시
1. "휴학 신청 기간이 언제야?" -> RAG로 schedule.pdf에서 정확한 날짜 검색
2. "오늘 날씨 어때?" → 청주를 기준으로 하여 OpenWeatherMap 실시간 조회
3. "학교 근처 밥집 추천해줘" -> 충북대를 기준으로 하여 Tavily 웹검색 기반 추천 (참고용 표시)
4. "기말고사까지 며칠 남았어?" -> RAG로 날짜 찾고 dday_calculator로 정확히 계산

## 2. 전체 아키텍처 설명

( 다이어그램 )

### 그래프 흐름 요약
- **validate_input**: 입력 검증 및 로깅 미들웨어
- **classify**: 키워드를 기반으로 academic / restaurant / realtime 분류
- **rag_node**: Chroma 벡터DB에서 학사일정 검색 -> LLM으로 질문 맞춤 요약
- **fallback_search**: RAG 결과가 없을 때는 Tavily 웹검색으로 대체
- **tool_agent_node / tools**: LLM이 필요한 tool을 스스로 선택해 반복 호출(loop)
- **structured_output_node**: 최종 답변을 Pydantic 스키마로 구조화

## 3. 설치 및 실행 방법
pip install -r requirements.txt

### .env 파일에 아래 키 설정
OPENAI_API_KEY= 
TAVILY_API_KEY=
OPENWEATHER_API_KEY=
각 키 값은 개신누리 과제에 제출한 결과 보고서 참고

### 1) 학사일정 PDF 임베딩 (최초 1회, PDF 갱신 시 재실행)
python ingest.py

### 2) 서버 실행
uvicorn server:app --reload
\`\`\`

## 4. 사용된 Tool / RAG / Memory / Middleware 설명

### Tool
| Tool | 설명 |
|---|---|
| weather_search | OpenWeatherMap API로 실시간 날씨 조회 (영문 지명 변환 필요) |
| tavily_search | 일반 웹검색 |
| recommend_restaurant | 학교 주변 맛집 검색 (Tavily 기반) |
| rag_search | 학사일정 PDF 벡터 검색 + 질문 맞춤 요약 |
| dday_calculator | 특정 날짜까지 남은 일수 계산 (LLM 암산을 방지하기 위함) |

### RAG
- PDF: schedule.pdf (학사일정) → RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)
- Embedding: OpenAI text-embedding-3-small
- Vector DB: Chroma (collection: academic_calendar), retriever k=8
- 검색된 chunk를 그대로 노출하지 않고, LLM이 질문/대화 맥락에 맞춰 재요약 후 응답

### Memory
- SqliteSaver + thread_id로 멀티턴 대화 유지
- 매 턴 current_query를 별도 필드로 저장해, 누적된 messages 전체가 아닌 "이번 질문"만 정확히 참조하도록 처리
- 대화 초기화시 새 스레드로 판단하여 다른 스레드에서의 대화는 기억하지 않음

### Middleware
- logging_guardrail_middleware: 함수 기반 미들웨어로, 사용자 입력 로깅 및 금칙어 필터링

## 5. 한계점 및 향후 개선 방향
| 한계점 | 향후 개선 방향 |
|---|---|
| 키워드 기반 classify의 한계 : 사전에 정의된 키워드 목록에 없는 표현은 분류가 안 될 수 있음 | 향후 LLM 기반 의도 분류로 전환 검토 |
| 웹검색 기반 정보(공모전, 학부 공지)의 정확도 한계 : 범용 검색이라 공식 게시판 글이 아닌 나무위키/유튜브 등이 걸릴 수 있음 | 학부 홈페이지 URL을 직접 크롤링하는 전용 tool 추가 검토 |
| 대화 맥락 의존 질문의 불안정성 : "여름말고" 같은 생략형 질문은 최근 대화 일부를 프롬프트에 포함해 개선했으나, 완벽히 보장되지는 않음 | 검색 전에 맥락을 반영한 완전한 질문으로 변환 |
| fallback_search 트리거 조건의 느슨함 : Chroma retriever가 유사도 낮아도 k개를 반환하는 특성상, RAG 관련성 체크가 "문서 존재 여부"만으로 판단되어 정교하지 않음 | LLM 기반으로 관련도를 판단하는 단계 추가 |
