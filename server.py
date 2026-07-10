'''
캠퍼스 학사일정 ai agent

실행 방법:
  .env 파일에 API KEY 설정
  pip install -r requirements.txt
  python ingest.py
  uvicorn server:app --reload
'''

import json
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


def sse(data: dict) -> str:
    """dict를 SSE(Server-Sent Events) 포맷 문자열로 변환"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


#  1. RAG 준비 (ingest.py로 이미 만들어진 Chroma DB 로드)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="pdf_collection",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


def rag_search_with_sources(query: str) -> tuple[str, list[dict]]:
    """학사일정 문서에서 검색하고, 답변 텍스트 + 출처(페이지 등) 목록을 함께 반환합니다."""
    docs = retriever.invoke(query)
    if not docs:
        return "NO_RESULT", []

    sources = [
        {"source": os.path.basename(d.metadata.get("source", "schedule.pdf")), "page": d.metadata.get("page", "?")}
        for d in docs
    ]
    content = "\n\n".join(d.page_content for d in docs)
    return content, sources


academic_prompt = ChatPromptTemplate.from_template(
"""다음은 학사일정 문서에서 검색된 내용입니다:
{context}

{history_section}
질문: {question}

답변 규칙:
- 질문과 직접 관련된 일정만 언급하고, 관련 없는 다른 달/다른 항목은 언급하지 마세요.
- 검색된 내용 중 질문에 실제로 답할 수 있는 부분이 없다면, 억지로 관련 있어 보이는 
  다른 항목을 끌어다 답하지 말고 "학사일정 문서에서 관련 정보를 찾을 수 없습니다"라고 
  명확히 답하세요.
- 질문에 '여름'/'겨울'이나 특정 학기처럼 범위가 명시되지 않았다면, 관련된 하위 항목(예: 하계·동계 계절수업, 1차·2차 신청기간 등)을 빠짐없이 모두 안내하세요.
- 대화 맥락상 이전에 언급된 대상(예: '그거', '신청기간은?')이 무엇을 가리키는지 파악해서 답하세요.
- 마크다운 문법(**, #, -) 없이 순수 텍스트로 간결하게 답하세요."""
)


def summarize_academic_context(query: str, context: str, history_snippet: str = "") -> str:
    """검색된 학사일정 컨텍스트를 질문(및 대화 맥락)에 맞게 정리합니다.
    rag_node와 rag_search tool 양쪽에서 공통으로 사용해 경로에 따라
    답변 품질이 달라지지 않도록 합니다."""
    history_section = f"최근 대화 맥락:\n{history_snippet}\n" if history_snippet else ""
    chain = academic_prompt | model
    return chain.invoke({
        "context": context,
        "history_section": history_section,
        "question": query,
    }).content


@tool
def rag_search(query: str) -> str:
    """학사일정 문서에서 관련 정보를 검색합니다."""
    content, _ = rag_search_with_sources(query)
    if content == "NO_RESULT":
        return "학사일정 문서에서 관련 정보를 찾지 못했습니다."
    return summarize_academic_context(query, content)


#  2. Tool 정의 (날씨 / 웹검색 / 식당추천 / D-day 계산)
@tool
def weather_search(city: str, day_offset: int = 0) -> str:
    """도시의 날씨를 조회합니다.
    Args:
        city: 날씨를 조회할 도시의 '영문' 이름 (예: Cheongju, Seoul).
              반드시 영문 지명으로 변환해서 전달해야 합니다.
        day_offset: 오늘 기준 며칠 뒤인지 (0=오늘, 1=내일, 2=모레). 기본값 0.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if day_offset == 0:
        # 오늘은 현재 날씨 API 그대로 사용
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": f"{city},KR", "appid": api_key, "units": "metric", "lang": "kr"}
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 404:
                return f"'{city}' 지명을 찾을 수 없습니다. 지원되지 않는 지역이거나 지명 표기가 다를 수 있습니다."
            res.raise_for_status()
            data = res.json()
            desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            return f"{city} 오늘 날씨: {desc}, 기온 {temp}°C, 습도 {humidity}%"
        except Exception as e:
            return f"날씨 조회 실패: {e}"

    # 내일/모레는 Forecast API 사용 (3시간 간격 예보, 최대 5일치)
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": f"{city},KR", "appid": api_key, "units": "metric", "lang": "kr"}
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 404:
            return f"'{city}' 지명을 찾을 수 없습니다. 지원되지 않는 지역이거나 지명 표기가 다를 수 있습니다."
        res.raise_for_status()
        data = res.json()

        target_date = (datetime.now() + timedelta(days=day_offset)).date()
        # 목표 날짜의 정오(12:00) 근처 예보를 대표값으로 사용
        candidates = [
            item for item in data["list"]
            if datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S").date() == target_date
        ]
        if not candidates:
            return f"{target_date} 날씨 예보를 찾을 수 없습니다 (5일 이내 예보만 제공됩니다)."

        # 정오에 가장 가까운 예보 선택
        best = min(
            candidates,
            key=lambda item: abs(
                datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S").hour - 12
            )
        )
        desc = best["weather"][0]["description"]
        temp = best["main"]["temp"]
        humidity = best["main"]["humidity"]
        label = "내일" if day_offset == 1 else f"{day_offset}일 후"
        return f"{city} {label}({target_date}) 날씨: {desc}, 기온 {temp}°C, 습도 {humidity}%"
    except Exception as e:
        return f"날씨 예보 조회 실패: {e}"


tavily = TavilySearch(max_results=3)


@tool
def recommend_restaurant(location: str, budget: str = "") -> str:
    """학교 주변 식당을 검색해 추천합니다.
    Args:
        location: 기준 위치 (예: "OO대학교 정문")
        budget: 희망 예산 (예: "1만원 이하", 선택사항)
    """
    query = f"{location} 맛집 추천"
    if budget:
        query += f" {budget}"
    result = tavily.invoke(query)
    return str(result)


@tool
def dday_calculator(target_date: str) -> str:
    """특정 날짜까지 남은 일수(D-day)를 계산합니다. 오늘 날짜를 기준으로 정확한 산술 계산을 수행합니다.
    Args:
        target_date: 기준 날짜, "YYYY-MM-DD" 형식 (예: "2025-12-15")
    """
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        diff = (target - today).days
        if diff > 0:
            return f"오늘({today})부터 {target_date}까지 D-{diff} 남았습니다."
        elif diff == 0:
            return f"{target_date}는 오늘입니다 (D-DAY)."
        else:
            return f"{target_date}는 {abs(diff)}일 전에 지났습니다."
    except ValueError:
        return f"날짜 형식을 인식할 수 없습니다: '{target_date}' (YYYY-MM-DD 형식으로 다시 알려주세요)"


ALL_TOOLS = [weather_search, tavily, recommend_restaurant, rag_search, dday_calculator]
tool_map = {t.name: t for t in ALL_TOOLS}


#  3. 구조화 출력 스키마 (Pydantic)
class AcademicEvent(BaseModel):
    event: str = Field(description="학사일정 이벤트명")
    date: str = Field(description="날짜")
    location: str = Field(default="미정", description="장소")


class RestaurantRecommendation(BaseModel):
    name: str = Field(description="식당 이름")
    menu: str = Field(description="추천 메뉴")
    estimated_price: str = Field(description="예상 가격대")
    location_note: str = Field(description="위치 관련 메모")



#  4. LangGraph State 및 노드 정의
model = init_chat_model("openai:gpt-4o-mini", temperature=0)
tool_model = model.bind_tools(ALL_TOOLS)

BANNED_WORDS = ["비밀번호", "주민번호"]
ACADEMIC_KEYWORDS = [
    "학사일정", "개강", "종강", "휴학", "복학", "시험기간", "수강신청", "성적",
    "계절학기", "계절수업", "방학", "졸업", "전과", "재입학", "등록금",
    "오리엔테이션", "입학식", "논문", "학위",
]
RESTAURANT_KEYWORDS = ["맛집", "식당", "메뉴", "밥", "점심", "저녁", "카페", "커피", "야식", "배달"]
DDAY_KEYWORDS = ["며칠", "디데이", "d-day", "D-day", "얼마나 남았"]


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str            # "academic" | "restaurant" | "realtime"
    current_query: str       # 이번 턴의 사용자 질문 (누적된 messages와 별개로 보관!)
    rag_relevant: bool
    rag_sources: list
    retry_count: int
    blocked: bool
    structured_output: Optional[dict]


# 입력 검증 및 로깅 Middleware (함수 기반, 노드 안에서 직접 호출)
def logging_guardrail_middleware(state: GraphState) -> dict:
    """Middleware: 사용자 입력을 로깅하고, 금칙어가 있으면 차단합니다."""
    last_msg = state["messages"][-1].content
    print(f"[LOG][middleware] 사용자 입력: {last_msg}")
    if any(w in last_msg for w in BANNED_WORDS):
        print("[LOG][middleware] 부적절한 입력 차단됨")
        return {"blocked": True}
    return {"blocked": False}


def route_after_validation(state: GraphState) -> Literal["classify", "end_blocked"]:
    return "end_blocked" if state.get("blocked") else "classify"


def end_blocked(state: GraphState) -> dict:
    return {"messages": [{"role": "ai", "content": "부적절한 요청은 처리할 수 없습니다."}]}


# 질문 분류 (조건부 분기) 
def classify(state: GraphState) -> dict:
    q = state["messages"][-1].content
    if any(k in q for k in DDAY_KEYWORDS):
        # D-day 질문은 rag_search + dday_calculator를 tool_agent_node에서 체이닝 호출하도록 라우팅
        category = "realtime"
    elif any(k in q for k in ACADEMIC_KEYWORDS):
        category = "academic"
    elif any(k in q for k in RESTAURANT_KEYWORDS):
        category = "restaurant"
    else:
        category = "realtime"
    return {
        "category": category,
        "current_query": q,
        "retry_count": 0,
        "rag_sources": [],
    }


def route_by_category(state: GraphState) -> Literal["rag_node", "tool_agent_node"]:
    return "rag_node" if state["category"] == "academic" else "tool_agent_node"


# RAG 노드 (출처 포함) 
def rag_node(state: GraphState) -> dict:
    query = state["current_query"]
    result, sources = rag_search_with_sources(query)
    relevant = result != "NO_RESULT"

    if relevant:
        # 이번 질문 직전까지의 최근 대화 몇 개를 맥락으로 함께 제공
        prior_msgs = state["messages"][:-1][-6:]
        history_snippet = "\n".join(
            f"{getattr(m, 'type', m.get('role', '') if isinstance(m, dict) else '')}: "
            f"{getattr(m, 'content', m.get('content', '') if isinstance(m, dict) else '')}"
            for m in prior_msgs
        )
        answer_content = summarize_academic_context(query, result, history_snippet)
    else:
        answer_content = "NO_RESULT"

    new_msgs = [{"role": "assistant", "content": answer_content}] if relevant else []
    return {"messages": new_msgs, "rag_relevant": relevant, "rag_sources": sources}


# 관련성 체크 -> loop: 부족하면 웹검색 fallback
def check_relevance(state: GraphState) -> Literal["fallback_search", "structured_output_node"]:
    if not state["rag_relevant"] and state["retry_count"] < 1:
        return "fallback_search"
    return "structured_output_node"


def fallback_search(state: GraphState) -> dict:
    query = state["current_query"]
    result = tavily.invoke(query)
    return {
        "messages": [{"role": "assistant", "content": str(result)}],
        "rag_relevant": True,
        "retry_count": state["retry_count"] + 1,
        "rag_sources": [{"source": "웹 검색(Tavily)", "page": "-"}],
    }


# Tool Agent 노드 (날씨 / 웹검색 / 식당추천 / RAG / D-day 중에 자율로 선택)
# LLM이 tool_calls를 반환 -> tool_node에서 실제 실행 -> 다시 이 노드로 돌아옴 (loop)
def tool_agent_node(state: GraphState) -> dict:
    today_str = datetime.now().strftime("%Y-%m-%d")
    system_prompt = (
        f"오늘 날짜는 {today_str}입니다. 날짜 관련 계산이나 언급을 할 때는 "
        f"반드시 이 날짜(특히 연도)를 기준으로 하고, 임의로 다른 연도를 추측하지 마세요.\n\n"
        "당신은 충북대학교 학생들을 위한 캠퍼스 브리핑 에이전트입니다. "
        "사용자가 위치나 도시를 별도로 말하지 않으면, 항상 '충북대학교'(청주)를 기준으로 "
        "날씨를 조회하거나 식당을 추천하세요. 되묻지 말고 바로 도구를 호출하세요.\n\n"
        "weather_search를 호출할 때는 도시명을 반드시 영문으로 변환해서 넘기세요 "
        "(예: 청주 -> Cheongju, 서울 -> Seoul, 충북대학교 -> Cheongju). "
        "한글 도시명을 그대로 넘기면 날씨 조회가 실패합니다. "
        "'오늘' 날씨는 day_offset=0(기본값), '내일'은 day_offset=1, '모레'는 day_offset=2로 "
        "반드시 지정해서 호출하세요. 특정 날짜를 언급하지 않으면 오늘(day_offset=0)로 간주하세요.\n\n"
        "특정 학사일정까지 남은 기간을 물으면(예: '기말고사까지 며칠 남았어?'), "
        "먼저 rag_search로 해당 일정의 날짜를 찾은 뒤, 그 날짜(YYYY-MM-DD 형식으로 변환)를 "
        "dday_calculator에 넘겨 정확히 계산하세요. 날짜 계산은 반드시 dday_calculator에 위임하고 "
        "직접 암산하지 마세요.\n\n"
        "이전 대화 맥락을 참고해서, 사용자가 이번 메시지에서 날짜/장소 등 정보를 생략했다면 "
        "이전 대화에서 언급된 내용으로 유추해서 사용하세요. 예를 들어 이전에 특정 시험 날짜를 "
        "이미 이야기했다면 다시 묻지 말고 그 날짜를 그대로 활용하세요.\n\n"
        "답변 형식 규칙:\n"
        "- 마크다운 문법(**, #, -)을 쓰지 말고 순수 텍스트로만 답하세요.\n"
        "- 검색 결과가 많아도 상위 3곳만 골라 간결하게 소개하세요.\n"
        "- 각 항목은 줄바꿈으로 구분하고, '이름 - 한줄설명 - 위치' 형식으로 짧게 쓰세요.\n"
        "- 전체 답변은 5줄을 넘기지 마세요.\n"
        "- tavily_search나 recommend_restaurant 같은 웹 검색 도구 결과를 바탕으로 답할 때는, "
        "답변 마지막 줄에 '※ 웹 검색 기반 정보이니 참고용으로만 확인하세요.'를 덧붙이세요. "
        "학사일정(rag_search), 날씨(weather_search), 날짜 계산(dday_calculator) 결과에는 이 문구를 붙이지 마세요."
    )
    response = tool_model.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )
    print(f"[DEBUG] tool_agent_node 응답 content: {response.content!r}")
    print(f"[DEBUG] tool_agent_node 응답 tool_calls: {getattr(response, 'tool_calls', None)}")
    return {"messages": [response]}


# Tool 실행 노드
WEB_SEARCH_TOOL_NAMES = {"tavily_search", "recommend_restaurant"}


def tool_execution_node(state: GraphState) -> dict:
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    tool_results = []
    used_web_search = False
    for tc in tool_calls:
        try:
            selected_tool = tool_map[tc["name"]]
            result = selected_tool.invoke(tc["args"])
            print(f"[LOG] Tool 실행: {tc['name']}({tc['args']}) -> {result}")
            if tc["name"] in WEB_SEARCH_TOOL_NAMES:
                used_web_search = True
        except KeyError:
            print(f"[ERROR] 존재하지 않는 tool 호출 시도: {tc['name']}")
            result = f"'{tc['name']}' 도구를 찾을 수 없습니다."
        except Exception as e:
            print(f"[ERROR] Tool 실행 실패: {tc['name']}({tc['args']}) -> {e}")
            result = f"도구 실행 중 오류가 발생했습니다: {e}"
        tool_results.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tc["id"],
        })

    update = {"messages": tool_results}
    if used_web_search:
        # RAG fallback_search와 동일한 방식으로 출처 표시
        update["rag_sources"] = [{"source": "웹 검색(Tavily) - 참고용, 공식 정보 재확인 권장", "page": "-"}]
    return update


def should_continue_tools(state: GraphState) -> Literal["tools", "structured_output_node"]:
    last_msg = state["messages"][-1]
    has_tool_calls = bool(getattr(last_msg, "tool_calls", None))
    return "tools" if has_tool_calls else "structured_output_node"


# 구조화 출력 노드
def structured_output_node(state: GraphState) -> dict:
    last_content = state["messages"][-1].content
    structured = None
    try:
        if state["category"] == "academic":
            parser_model = model.with_structured_output(AcademicEvent)
            structured = parser_model.invoke(
                f"다음 내용에서 학사일정 정보를 추출해줘:\n{last_content}"
            ).model_dump()
        elif state["category"] == "restaurant":
            parser_model = model.with_structured_output(RestaurantRecommendation)
            structured = parser_model.invoke(
                f"다음 내용에서 식당 추천 정보를 추출해줘:\n{last_content}"
            ).model_dump()
    except Exception as e:
        print(f"[LOG] 구조화 출력 실패: {e}")
    return {"structured_output": structured}


#  5. 그래프 구성 및 컴파일
graph = StateGraph(GraphState)
graph.add_node("validate_input", logging_guardrail_middleware)
graph.add_node("end_blocked", end_blocked)
graph.add_node("classify", classify)
graph.add_node("rag_node", rag_node)
graph.add_node("fallback_search", fallback_search)
graph.add_node("tool_agent_node", tool_agent_node)
graph.add_node("tools", tool_execution_node)
graph.add_node("structured_output_node", structured_output_node)

graph.add_edge(START, "validate_input")
graph.add_conditional_edges("validate_input", route_after_validation, {
    "classify": "classify",
    "end_blocked": "end_blocked",
})
graph.add_edge("end_blocked", END)
graph.add_conditional_edges("classify", route_by_category, {
    "rag_node": "rag_node",
    "tool_agent_node": "tool_agent_node",
})
graph.add_conditional_edges("rag_node", check_relevance, {
    "fallback_search": "fallback_search",
    "structured_output_node": "structured_output_node",
})
graph.add_edge("fallback_search", "structured_output_node")

graph.add_conditional_edges("tool_agent_node", should_continue_tools, {
    "tools": "tools",
    "structured_output_node": "structured_output_node",
})
graph.add_edge("tools", "tool_agent_node")

graph.add_edge("structured_output_node", END)

conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
compiled_graph = graph.compile(checkpointer=checkpointer)


#  6. FastAPI 엔드포인트
app = FastAPI(title="캠퍼스 브리핑 Agent")
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


NODE_DESCRIPTIONS = {
    "validate_input": "입력 검증 및 로깅 중...",
    "classify": "질문 유형 분류 중...",
    "rag_node": "학사일정 문서(schedule.pdf)에서 검색 중...",
    "fallback_search": "문서에서 찾지 못해 웹 검색으로 재탐색 중...",
    "tool_agent_node": "필요한 도구 선택 중...",
    "tools": "도구 실행 중...",
    "structured_output_node": "응답 정리 중...",
    "end_blocked": "요청 처리 거부 중...",
}


@app.get("/api/chat/stream")
async def chat_stream(message: str = Query(...), thread_id: str = Query("default")):
    """SSE 스트리밍: 그래프의 각 노드 실행 상황을 실시간으로 전송합니다."""

    async def event_generator():
        try:
            config = {"configurable": {"thread_id": thread_id}}
            final_answer = ""
            final_category = None
            final_structured = None
            final_sources = []

            for update in compiled_graph.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_state in update.items():
                    desc = NODE_DESCRIPTIONS.get(node_name, f"{node_name} 실행 중...")
                    yield sse({"type": "node", "name": node_name, "desc": desc})

                    if node_state.get("category"):
                        final_category = node_state["category"]

                    if node_state.get("rag_sources"):
                        final_sources = node_state["rag_sources"]

                    if node_state.get("structured_output") is not None:
                        final_structured = node_state["structured_output"]

                    msgs = node_state.get("messages")
                    if msgs:
                        last = msgs[-1]
                        content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
                        if content:
                            final_answer = content

            yield sse({
                "type": "answer",
                "content": final_answer,
                "category": final_category,
                "structured_output": final_structured,
                "rag_sources": final_sources,
            })
            yield sse({"type": "done"})

        except Exception as e:
            print(f"[ERROR] /api/chat/stream 처리 중 오류 발생: {e}")
            yield sse({"type": "error", "message": str(e)})
            yield sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(body: ChatRequest):
    """SSE 없이 한 번에 결과만 받고 싶을 때 사용하는 단순 버전 (curl/Swagger 테스트용)."""
    config = {"configurable": {"thread_id": body.thread_id}}
    try:
        result = compiled_graph.invoke(
            {"messages": [{"role": "user", "content": body.message}]},
            config=config,
        )
        return {
            "success": True,
            "answer": result["messages"][-1].content,
            "category": result.get("category"),
            "structured_output": result.get("structured_output"),
            "rag_sources": result.get("rag_sources", []),
        }
    except Exception as e:
        print(f"[ERROR] /api/chat 처리 중 오류 발생: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/reset")
async def reset(body: ChatRequest):
    # thread별 체크포인트 삭제
    try:
        checkpointer.delete_thread(body.thread_id)
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": True, "message": "안녕하세요! 캠퍼스 AI 에이전트입니다. 오늘의 날씨, 학사 일정, 학교 근처 맛집 등에 대해 무엇이든 물어보세요."}


#  7. 정적 파일(index.html) 서빙 — 반드시 API 라우트 등록 후 마운트
if os.path.isdir("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")
else:
    print("'public' 폴더가 없습니다. public/index.html 을 넣어주세요.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    print(f"✅ http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)