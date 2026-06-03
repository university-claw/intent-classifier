"""Library reservation companion allow rules.

The safety model often treats name + student-id lists as PII abuse. In the
library reservation flow, however, users must provide companion names and
student IDs themselves. This module keeps that narrow pass-through rule in
serving code so router behavior is deterministic without retraining.
"""

from __future__ import annotations

import re
import unicodedata


STUDENT_ID_PATTERN = re.compile(r"(?<!\d)\d{8}(?!\d)")

# 이름과 학번이 같은 줄이나 가까운 구간에 함께 있는 입력만 동반자 정보로 본다.
NAME_STUDENT_ID_PATTERN = re.compile(
    r"(?:이름\s*[:：]?\s*)?"
    r"(?P<name>[가-힣]{2,5})"
    r"(?:\s|[,/|·:：-])+"
    r"(?:학번\s*[:：]?\s*)?"
    r"(?P<student_id>(?<!\d)\d{8}(?!\d))"
)

LIBRARY_CONTEXT_TERMS = (
    "도서관",
    "스터디룸",
    "그룹스터디룸",
    "열람실",
    "명진당",
    "좌석",
    "예약",
    "library",
    "study room",
)

COMPANION_CONTEXT_TERMS = (
    "동반자",
    "동행",
    "같이 이용",
    "같이 사용할",
    "같이 쓸",
    "함께 이용",
    "함께 사용할",
    "이용할 사람",
    "사용할 사람",
    "같이 갈",
    "친구",
    "멤버",
    "인원",
    "나랑",
    "아래 사람",
    "아래 사람들이랑",
    "companion",
    "member",
)

# 사용자가 직접 제공한 동반자 목록이 아니라 내부/타인 데이터를 찾거나 꺼내려는 문맥.
DISALLOWED_ACCESS_TERMS = (
    "다른 사용자",
    "다른 유저",
    "타인",
    "남의",
    "전체 사용자",
    "모든 사용자",
    "전체 유저",
    "모든 유저",
    "user_data",
    "/data/users",
    "profiles",
    "credentials",
    "세션",
    "session",
    "쿠키",
    "cookie",
    "토큰",
    "token",
    ".env",
    "api key",
    "db에서",
    "database",
    "데이터베이스",
)

DISALLOWED_ACCESS_VERBS = (
    "찾아",
    "검색",
    "조회",
    "보여",
    "알려",
    "출력",
    "덤프",
    "dump",
    "show",
    "reveal",
    "export",
    "가져와",
    "긁어",
    "빼내",
    "추출",
    "열어",
    "읽어",
)


def is_library_companion_student_id_submission(text: str) -> bool:
    """Return True for user-provided companion name/student-id lists.

    This intentionally does not allow requests to retrieve somebody else's
    student ID. The input must already contain name + 8-digit student-id pairs.
    """

    normalized = _normalize(text)
    pairs = NAME_STUDENT_ID_PATTERN.findall(normalized)
    if not pairs:
        return False

    if _has_disallowed_access_context(normalized):
        return False

    has_library_context = _contains_any(normalized, LIBRARY_CONTEXT_TERMS)
    has_companion_context = _contains_any(normalized, COMPANION_CONTEXT_TERMS)
    if has_library_context or has_companion_context:
        return True

    # 사용자가 직전 질문에 대한 답으로 이름/학번 목록만 보내는 경우를 허용한다.
    return _looks_like_plain_companion_list(normalized)


def _looks_like_plain_companion_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    pair_lines = [line for line in lines if NAME_STUDENT_ID_PATTERN.search(line)]
    if len(pair_lines) < 2:
        return False

    # 목록 답변에는 짧은 안내 한 줄이 붙을 수 있지만, 대부분은 pair line이어야 한다.
    return len(pair_lines) / len(lines) >= 0.6


def _has_disallowed_access_context(text: str) -> bool:
    has_access_target = _contains_any(text, DISALLOWED_ACCESS_TERMS)
    has_access_verb = _contains_any(text, DISALLOWED_ACCESS_VERBS)
    if has_access_target and has_access_verb:
        return True

    # 내부 저장소나 secret 계열은 동사 없이 언급돼도 허용하지 않는다.
    hard_block_terms = (
        "user_data",
        "/data/users",
        "credentials",
        ".env",
        "api key",
        "token",
        "cookie",
    )
    return _contains_any(text, hard_block_terms)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().strip()
