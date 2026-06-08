from dataclasses import dataclass
import random
from engine.schema import GameEvent, EventType


@dataclass
class NarratorConfig:
    language: str = "zh"
    seed: int | None = None


class Narrator:
    def __init__(self, config: NarratorConfig | None = None):
        self.config = config or NarratorConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def narrate(self, event: GameEvent) -> str:
        player = event.player_name or f"#{event.player_number}" if event.player_number else "球员"
        templates = self._templates().get(event.event_type, self._templates()[EventType.ASSIST])
        template = random.choice(templates)

        assist = ""
        if event.assist_number and event.assist_number != event.player_number:
            assist = event.assist_name or f"#{event.assist_number}"

        distance = f"{event.distance:.1f}m" if event.distance else ""

        return template.format(
            player=player,
            number=event.player_number or "?",
            assist=assist,
            distance=distance,
            score_before=event.score_before or "?",
            score_after=event.score_after or "?",
            quarter=event.quarter,
        )

    def _templates(self) -> dict[EventType, list[str]]:
        return {
            EventType.THREE_POINTER_MADE: [
                "{player} 三分线外一步直接出手 —— 空心入网！全场沸腾",
                "{player} 弧顶三分稳稳命中 ，距离{distance}",
                "{player} 接球就投，三分球应声入网！",
                "{player} 底角三分出手——进了！{assist}的助攻恰到好处",
            ],
            EventType.THREE_POINTER_MISSED: [
                "{player} 三分出手偏出，球打在篮筐前沿弹了出来",
                "{player} 三分投射，力道稍大，球弹出篮筐",
                "{player} 外线尝试三分，可惜偏出",
            ],
            EventType.TWO_POINTER_MADE: [
                "{player} 中距离跳投稳稳命中",
                "{player} 突破上篮得分",
                "{player} 背身单打后翻身跳投——漂亮！",
                "{player} 快攻中直接冲击篮筐，轻松上篮得分",
                "{player} 接{assist}传球，篮下强起打成",
            ],
            EventType.TWO_POINTER_MISSED: [
                "{player} 中距离出手偏出",
                "{player} 上篮没能放进，球弹框而出",
                "{player} 篮下出手被干扰，偏出",
            ],
            EventType.FREE_THROW_MADE: [
                "{player} 站上罚球线，稳稳命中",
                "{player} 罚球命中，比分来到{score_after}",
            ],
            EventType.FREE_THROW_MISSED: [
                "{player} 罚球偏出，他自己也摇了摇头",
                "{player} 罚球不中，球在篮筐上弹了两下还是掉了出来",
            ],
            EventType.OFFENSIVE_REBOUND: [
                "{player} 在人群中高高跃起摘下前场篮板！",
                "{player} 积极冲抢进攻篮板成功",
            ],
            EventType.DEFENSIVE_REBOUND: [
                "{player} 稳稳收下防守篮板",
                "{player} 卡住位置摘下后场篮板",
            ],
            EventType.ASSIST: [
                "{player} 精妙传球助攻{assist}得分",
                "{player} no-look pass 找到空位的{assist}",
            ],
            EventType.STEAL: [
                "{player} 眼疾手快抢断成功！一条龙推进",
                "{player} 预判传球路线将球抄截",
            ],
            EventType.BLOCK: [
                "{player} 遮天蔽日的大帽 ！全场欢呼",
                "{player} 钉板大帽！防守端的统治力",
            ],
            EventType.TIMEOUT: [
                "比赛暂停",
                "教练叫出暂停布置战术",
            ],
        }
