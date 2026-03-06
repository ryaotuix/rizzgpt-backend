# backend/pipeline/convo_preprocess.py

# llm 에 보낼 문자열 정리
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Turn:
    who: str  # "me" | "other"
    text: str


def turns_to_llm_convo(
    turns: List[Turn],
    join_token: str = " | ",
    who_me: str = "m",
    who_other: str = "o",
) -> str:
    """
    ✅ LLM 입력용 문자열 생성기
    - 1. who: me/other -> m/o
    - 2. 연속 동일 화자 메시지 압축: o: ㅋㅋ | ㅇㅋ | 잘자
    - 3. 공백 정리(앞뒤 strip + 연속 공백 1개로)
    """

    if not turns:
        return ""

    def norm_who(w: str) -> str:
        w = (w or "").strip().lower()
        if w in ("me", "mine", "self", "right", "m"):
            return who_me
        if w in ("other", "them", "opponent", "left", "o"):
            return who_other
        # 모르면 o로 두는게 안전
        return who_other

    def norm_text(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        while "  " in s:
            s = s.replace("  ", " ")
        return s

    lines: List[str] = []

    cur_who: Optional[str] = None
    cur_parts: List[str] = []

    def flush():
        nonlocal cur_who, cur_parts
        if cur_who is None or not cur_parts:
            cur_who, cur_parts = None, []
            return
        lines.append(f"{cur_who}:{join_token.join(cur_parts)}")
        cur_who, cur_parts = None, []

    for t in turns:
        w = norm_who(getattr(t, "who", ""))
        txt = norm_text(getattr(t, "text", ""))

        if not txt:
            continue

        if cur_who is None:
            cur_who = w
            cur_parts = [txt]
            continue

        if w == cur_who:
            cur_parts.append(txt)
        else:
            flush()
            cur_who = w
            cur_parts = [txt]

    flush()
    return "\n".join(lines).strip()
