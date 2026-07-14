"""Canonical bilingual safety responses shared by generation and translation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BilingualSafetyResponse:
    chinese: str
    japanese: str


IDENTITY_RESPONSE = BilingualSafetyResponse(
    chinese=(
        "我是非官方的 Hina Bot，不是青木阳菜本人，也不能代替她发表内容。"
        "不过我们可以继续聊公开作品、音乐，或者你现在想说的事。"
    ),
    japanese=(
        "私は非公式の Hina Bot で、青木陽菜さん本人ではなく、"
        "本人に代わって発言することもできません。"
        "ただ、公開されている作品や音楽についてなら一緒にお話しできます。"
    ),
)
PRIVATE_RESPONSE = BilingualSafetyResponse(
    chinese=(
        "这属于真人的私人或未公开信息，我不能替她猜测或编造。"
        "如果你想了解公开活动或作品，我可以只根据已经收录的公开资料来聊。"
    ),
    japanese=(
        "それは本人の私的または未公開の情報にあたるため、"
        "推測したり作り上げたりはできません。"
        "公開されている活動や作品についてなら、"
        "確認済みの情報だけをもとにお話しできます。"
    ),
)
INSUFFICIENT_EVIDENCE_RESPONSE = BilingualSafetyResponse(
    chinese=(
        "我目前收录的公开资料还不足以确认这件事，所以先不猜啦。"
        "等补充了可靠来源后，我再给你准确回答。"
    ),
    japanese=(
        "現在収録している公開情報だけでは確認できないため、"
        "推測せずにお答えを控えます。"
        "信頼できる情報が追加されたら、あらためて正確にお伝えします。"
    ),
)


def fixed_safety_response(
    intent: str,
    boundary_action: str,
) -> BilingualSafetyResponse | None:
    """Select only routes that are deterministic safety outcomes."""

    if intent == "identity_attack":
        return IDENTITY_RESPONSE
    if intent == "private_probe":
        return PRIVATE_RESPONSE
    if intent == "public_fact" and boundary_action == "insufficient_public_evidence":
        return INSUFFICIENT_EVIDENCE_RESPONSE
    return None
