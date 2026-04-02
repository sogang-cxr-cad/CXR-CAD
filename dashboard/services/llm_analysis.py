"""LLM helpers for the analysis dashboard."""

from __future__ import annotations

from typing import Any


def langchain_is_ready() -> tuple[bool, str]:
    """Return whether LangChain OpenAI dependencies are importable."""
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard only
        return False, str(exc)
    return True, ""


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _make_llm(api_key: str, model_name: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=0.2,
        timeout=60,
        max_retries=2,
    )


def generate_metric_summary(
    *,
    metric_title: str,
    metric_context: str,
    api_key: str,
    model_name: str,
) -> str:
    """Generate a grounded summary for the currently selected metric."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _make_llm(api_key=api_key, model_name=model_name)
    messages = [
        SystemMessage(
            content=(
                "당신은 흉부 X-ray AI 대시보드 분석가입니다. "
                "반드시 제공된 지표와 수치에만 근거해 한국어로 답하세요. "
                "출력 형식은 다음을 따르세요:\n"
                "### 핵심 결론\n"
                "- 2~3개 bullet\n\n"
                "### 자료 해석\n"
                "- 성능 의미와 trade-off\n\n"
                "### 리스크 / 한계\n"
                "- 편향, domain shift, 운영상 주의점\n\n"
                "### 권장 액션\n"
                "- 모델 또는 데이터 측면의 다음 조치 2개 이상\n"
                "수치는 가능한 한 직접 인용하세요. 데이터에 없는 내용은 추정하지 마세요."
            )
        ),
        HumanMessage(
            content=(
                f"[지표 제목]\n{metric_title}\n\n"
                f"[지표 컨텍스트]\n{metric_context}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return _normalize_content(response.content)



def ask_metric_question(
    *,
    metric_title: str,
    metric_context: str,
    question: str,
    api_key: str,
    model_name: str,
) -> str:
    """Answer a user question grounded in the selected metric only."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _make_llm(api_key=api_key, model_name=model_name)
    messages = [
        SystemMessage(
            content=(
                "당신은 흉부 X-ray AI 성능분석 보조자입니다. "
                "반드시 제공된 지표 컨텍스트만 근거로 한국어로 답하세요. "
                "질문에 답할 수 없으면 왜 부족한지 분명히 말하세요. "
                "답변은 4~8문장 정도로 간결하되, 필요한 경우 수치를 직접 인용하세요."
            )
        ),
        HumanMessage(
            content=(
                f"[지표 제목]\n{metric_title}\n\n"
                f"[지표 컨텍스트]\n{metric_context}\n\n"
                f"[사용자 질문]\n{question}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return _normalize_content(response.content)
