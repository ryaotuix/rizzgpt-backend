# backend/pipeline/postprocess_turns.py
# ✅ 역할: extract_text_and_order()가 만든 Turn 리스트를 "LLM에 넣기 좋은 형태"로 정리
#
# 왜 필요?
# - 같은 사람이 연속으로 보낸 말풍선(me/me/me)은 실제로는 "한 턴"으로 보는 게 자연스럽다.
# - OCR이 비었거나(이미지/이모지/인식실패) 너무 짧은 조각들은 노이즈가 된다.
# - LLM 품질은 "입력 정리"에서 거의 결정된다.
#
# 주요 기능
# 1) 연속 화자 병합: me->me->me 를 1개로 합침
# 2) 빈 텍스트 처리: 제거하거나 placeholder로 남김
# 3) 너무 짧은 텍스트 정리: 단독이면 제거/앞뒤에 합침(옵션)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Turn:
    who: str   # "me" | "other"
    text: str


def _clean_text(s: str) -> str:
    """
    OCR 결과 텍스트를 최소한으로 정리
    - 양끝 공백 제거
    - 중복 공백 줄이기
    """
    s = (s or "").strip()
    if not s:
        return ""
    # 연속 공백 정리
    s = " ".join(s.split())
    return s


def postprocess_turns(
    turns: List[Turn],
    *,
    # 빈 텍스트를 아예 버릴지(True) / placeholder로 남길지(False)
    drop_empty: bool = True,
    empty_placeholder: str = "[IMAGE]",
    # 너무 짧은 텍스트(예: "하", "오", "ㅇㅇ") 기준 길이
    min_len: int = 1,
    # 짧은 텍스트도 버릴지(True) / 유지할지(False)
    drop_too_short: bool = False,
    # 병합할 때 문장 사이 구분자
    joiner: str = " ",
) -> List[Turn]:
    """
    LLM에 넣기 좋은 turns로 가공해서 반환한다.

    동작 순서:
    1) 텍스트 클린
    2) 빈 텍스트 drop/placeholder
    3) 연속 화자 병합
    4) (옵션) 너무 짧은 텍스트 제거
    """

    # ---------------------------
    # 1) 1차 정리: clean + empty 처리
    # ---------------------------
    normalized: List[Turn] = []
    for t in turns:
        who = (t.who or "").strip().lower()
        text = _clean_text(t.text)

        if not text:
            if drop_empty:
                continue
            text = empty_placeholder

        # who가 이상하면 그냥 스킵 (안전장치)
        if who not in ("me", "other"):
            continue

        normalized.append(Turn(who=who, text=text))

    if not normalized:
        return []

    # ---------------------------
    # 2) 연속 화자 병합
    # ---------------------------
    merged: List[Turn] = []
    cur_who = normalized[0].who
    cur_text_parts = [normalized[0].text]

    for t in normalized[1:]:
        if t.who == cur_who:
            # 같은 화자 연속 -> 텍스트만 이어붙임
            cur_text_parts.append(t.text)
        else:
            # 화자 바뀜 -> 지금까지를 확정 저장
            merged.append(Turn(who=cur_who, text=joiner.join(cur_text_parts).strip()))
            cur_who = t.who
            cur_text_parts = [t.text]

    # 마지막 턴 저장
    merged.append(Turn(who=cur_who, text=joiner.join(cur_text_parts).strip()))

    # ---------------------------
    # 3) (옵션) 너무 짧은 텍스트 제거
    # ---------------------------
    if drop_too_short:
        merged2: List[Turn] = []
        for t in merged:
            if len(t.text) < min_len:
                continue
            merged2.append(t)
        merged = merged2

    return merged


def turns_to_lines(turns: List[Turn]) -> List[str]:
    """
    디버그 로그용: ["me(1): ...", "other(2): ..."] 형태로 변환
    """
    lines: List[str] = []
    for i, t in enumerate(turns, start=1):
        lines.append(f"{t.who}({i}): {t.text}")
    return lines
