# backend/llm/openai_engine.py
# ✅ OCR 텍스트 -> GPT-4o mini -> reply 1종(JSON) 생성
# 필요:
#   pip install openai
# 환경변수:
#   export OPENAI_API_KEY="..."

from __future__ import annotations

import os
from typing import Dict

from openai import OpenAI

# ✅ reply 하나만 받기 위한 스키마 (Structured Outputs)
REPLY_SCHEMA = {
    "name": "reply_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reply": {"type": "string"},
        },
        "required": ["reply"],
    },
    "strict": True,
}

# =========================================================
# ✅ 시스템 프롬프트 (짧고 강하게 "스타일 규칙"만)
# - 길어지면 비용 증가하니까 규칙을 "압축"해서 적음
# =========================================================
SYSTEM_PROMPT = """
너는 한국어 채팅 답장 도우미다.
대화 포맷: m=me, o=other. (m이 답장함)

목표:
o의 마지막 메시지에 대해,
me가 실제로 보낼 법한 답장 1개를 자연스럽게 작성한다.

핵심 기준:
- 답장을 쓰기 전 항상
  “이 문장을 내가 진짜 그대로 보낼까?”를 기준으로 판단한다.
- 설명적이거나 어색하면 쓰지 않는다.
- 답장은 메신저 기준 1~2줄을 기본으로 한다.
- 장문 설명, 정리, 훈계, 에세이식 말투는 하지 않는다.

맥락 반응:
- o의 마지막 메시지와 직전 2~4턴에서 드러난
  핵심 주제·톤·감정에만 반응한다.
- 상대가 단답, 체념, 농담 톤이면
  깊은 위로·해결·정리 없이
  가볍게 맞장구치거나 받아치는 것으로 끝낸다.

질문 사용:
- 질문은 상대가 먼저 대화를 열어둔 경우에만 사용한다
  (의문문, 추천 요청, 선택지 언급 등).
- 그 외에는 질문 없이 답장을 마친다.

톤 & 말투:
- 캐주얼하고 티키타카 위주.
- 상대를 달래거나 설득하려는 말투는 피한다.
- 친절함보다 “같은 편에서 툭 던지는 말”을 우선한다.
- 상대가 단정하거나 체념하면
  이를 바꾸려 들지 말고 가볍게 동의하거나 장난스럽게 반응한다.

말투 기준(me):
- 이 대화에서 보이는 me의 말투를 그대로 따른다.
- 문장 길이, 말끝, 리듬(단답, 끊어침, ㅋㅋ 등)을 자연스럽게 유지한다.
- me가 실제로 쓰지 않을 법한 말투나 표현은 사용하지 않는다.

표현 제한:
- 오글거리는 표현, 과한 공감 문장, 시적·소설체 금지.
- 플러팅은 현실 채팅에서 자연스러운 수준만 허용.
- 부담 주는 멘트(집착, 고백 강요, 감정 과몰입) 금지.

출력:
- m의 답장 1개만 작성한다.
- 카카오톡/DM에 바로 보내도 어색하지 않아야 한다.
"""



def build_user_prompt(convo_text: str) -> str:
    return f"""
[대화]
{convo_text}

[요청]
- 상대의 마지막 메시지에 바로 이어지는 답장 1개
- 새로운 설정 추가 금지
- 자연스럽게
"""


class OpenAIEngine:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않음")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_reply(self, convo_text: str) -> Dict[str, str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(convo_text)},
            ],
            temperature=0.6,
            max_tokens=80,
            response_format={"type": "json_schema", "json_schema": REPLY_SCHEMA},
        )
        # Structured outputs라 content가 JSON 문자열로 옴
        content = resp.choices[0].message.content
        # openai SDK가 자동 파싱을 안 해주면 직접 json.loads 해도 됨
        # 여기선 간단히 안전하게 처리
        import json
        data = json.loads(content)
        return {"reply": str(data.get("reply", "")).strip()}
