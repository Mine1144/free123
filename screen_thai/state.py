"""Bounded, per-occurrence translation state. No global phrase cache or frame queue.

A changing clock/HUD cannot invalidate a stable subtitle. A phrase disappearing and
returning is a NEW occurrence: old in-flight answers must not attach to it.
"""
from .models import Block, Result, Translation
from .text import normalized


def overlap(a: Block, b: Block) -> float:
    x, y, w, h = a.box
    xx, yy, ww, hh = b.box
    intersection = max(0, min(x+w, xx+ww)-max(x, xx)) * max(0, min(y+h, yy+hh)-max(y, yy))
    union = w*h + ww*hh - intersection
    return intersection / union if union else 0


class FrameState:
    def __init__(self):
        self.blocks: list[Block] = []
        self.tokens: dict[int, int] = {}
        self.next_token = 0
        self.resolved: dict[int, Translation | None] = {}

    def update(self, blocks: list[Block]):
        available = list(self.blocks)
        tokens = {}
        for block in blocks:
            candidates = [old for old in available
                          if normalized(old.text) == normalized(block.text) and overlap(old, block) > 0.4]
            if candidates:
                old = max(candidates, key=lambda old: overlap(old, block))
                tokens[block.id] = self.tokens[old.id]
                available.remove(old)
            else:
                self.next_token += 1
                tokens[block.id] = self.next_token
        self.tokens = tokens
        alive = set(tokens.values())
        self.resolved = {token: value for token, value in self.resolved.items() if token in alive}
        self.blocks = blocks

    def needs_translation(self) -> bool:
        return any(token not in self.resolved for token in self.tokens.values())

    def snapshot(self) -> dict[int, int]:
        return dict(self.tokens)

    def accept(self, submitted: dict[int, int], result: Result) -> bool:
        alive = set(self.tokens.values())
        translations = {t.id: t for t in result.translations}
        accepted = False
        for bid, token in submitted.items():
            if token in alive:
                # Missing IDs are an intentional AI decision to skip this occurrence.
                self.resolved[token] = translations.get(bid)
                accepted = True
        return accepted

    def rendered(self) -> Result:
        output = []
        for block in self.blocks:
            translation = self.resolved.get(self.tokens[block.id])
            if translation:
                output.append(Translation(block.id, translation.thai, translation.speaker))
        return Result(output)
