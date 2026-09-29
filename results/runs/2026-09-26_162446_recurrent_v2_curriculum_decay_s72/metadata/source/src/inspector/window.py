"""Pygame policy viewer: legal preferences, human overrides, undo and speed control."""

import math
from pathlib import Path

import pygame

from .session import ACTIONS, tile_value


SIZE = (1080, 820)
BG, PANEL, INK, MUTED = (16, 24, 38), (28, 41, 60), (239, 244, 251), (159, 176, 199)
GREEN, BLUE = (92, 222, 175), (124, 173, 255)
BOARD = pygame.Rect(28, 172, 500, 500)
PLAY = pygame.Rect(28, 716, 160, 48)
STEP = pygame.Rect(198, 716, 160, 48)
UNDO = pygame.Rect(368, 716, 160, 48)
NEW = pygame.Rect(568, 716, 226, 48)
EXPORT = pygame.Rect(806, 716, 246, 48)
SLIDER = pygame.Rect(588, 650, 444, 22)


def speed_from_fraction(fraction):
    return .25 * 240 ** max(0, min(1, fraction))


def fraction_from_speed(speed):
    return math.log(max(.25, min(60, speed)) / .25, 240)


class InspectorWindow:
    def __init__(self, session, speed=3.0, autoplay=False):
        self.session = session
        self.speed = max(.25, min(60, speed))
        self.autoplay = autoplay
        self.dragging = False
        self.next_move = 0.0
        self.message = "Choose a legal direction, or ask the agent to play."
        self.fonts = {size: pygame.font.Font(None, size) for size in (18, 20, 22, 24, 28, 32, 38, 44, 54)}

    def text(self, screen, text, x, y, size=24, color=INK, *, right=False):
        rendered = self.fonts[size].render(str(text), True, color)
        screen.blit(rendered, (x - rendered.get_width() if right else x, y))

    def button(self, screen, rect, text, enabled=True, accent=False):
        pygame.draw.rect(screen, GREEN if accent else PANEL, rect, border_radius=10)
        image = self.fonts[24].render(text, True, BG if accent else INK if enabled else MUTED)
        screen.blit(image, image.get_rect(center=rect.center))

    def action_rows(self):
        decision = self.session.decision
        if decision is None:
            return []
        return [(int(action), pygame.Rect(568, 226 + row * 76, 484, 66))
                for row, action in enumerate(i for i in range(4) if decision.legal[i])]

    def draw(self, screen):
        session = self.session
        screen.fill(BG)
        self.text(screen, "THREES / POLICY INSPECTOR", 28, 22, 32)
        label = session.policy.label
        if len(label) > 85:
            label = label[:82] + "..."
        self.text(screen, label, 28, 58, 20, MUTED)
        self.text(screen, "AUTO" if self.autoplay else "YOUR MOVE", 1052, 24, 24, GREEN, right=True)
        for x, title, value in ((28, "SCORE", f"{session.score:,}"),
                                (198, "LARGEST TILE", f"{session.maximum:,}"),
                                (368, "MOVES", int(session.env.meta[0, 2]))):
            pygame.draw.rect(screen, PANEL, (x, 96, 160, 60), border_radius=10)
            self.text(screen, title, x + 12, 103, 18, MUTED)
            self.text(screen, value, x + 12, 123, 32)
        self.text(screen, "NEXT TILE", 568, 101, 18, MUTED)
        self.text(screen, session.preview if not session.done else "-", 568, 124, 32, GREEN)
        self.text(screen, f"Seed {session.seed}  |  CPU inference", 1052, 105, 20, MUTED, right=True)
        self.text(screen, "Natural-start game / engine v2", 1052, 131, 20, MUTED, right=True)
        for index, rank in enumerate(session.env.board[0]):
            rect = pygame.Rect(BOARD.x + index % 4 * 128, BOARD.y + index // 4 * 128, 116, 116)
            value = tile_value(rank)
            color = (102, 189, 227) if value == 1 else (242, 136, 162) if value == 2 else (233, 237, 228) if value else PANEL
            pygame.draw.rect(screen, color, rect, border_radius=13)
            if value:
                size = 54 if value < 1000 else 44 if value < 10000 else 38
                text = self.fonts[size].render(str(value), True, BG)
                screen.blit(text, text.get_rect(center=rect.center))
        self.text(screen, "Legal action preferences", 568, 173, 28)
        self.text(screen, "Click to override. Highest preference is outlined.", 568, 203, 20, MUTED)
        decision = session.decision
        for action, rect in self.action_rows():
            pygame.draw.rect(screen, PANEL, rect, border_radius=9)
            if action == decision.recommendation:
                pygame.draw.rect(screen, GREEN, rect, width=2, border_radius=9)
            probability = decision.probabilities[action]
            self.text(screen, ACTIONS[action], rect.x + 15, rect.y + 10, 28)
            self.text(screen, f"{probability * 100:5.1f}%", rect.right - 14, rect.y + 10, 28, GREEN, right=True)
            pygame.draw.rect(screen, (51, 66, 89), (rect.x + 16, rect.y + 43, 275, 8), border_radius=4)
            if probability > 0:
                pygame.draw.rect(screen, BLUE, (rect.x + 16, rect.y + 43, max(1, round(275 * probability)), 8), border_radius=4)
            self.text(screen, f"logit {decision.logits[action]:+.3f}", rect.right - 14, rect.y + 42, 18, MUTED, right=True)
        if decision is None:
            self.text(screen, session.end_reason or "No legal moves", 590, 250, 28, GREEN)
        self.text(screen, "Percentages are preferences, not chances of winning.", 568, 548, 20, MUTED)
        self.text(screen, "Illegal actions are excluded; legal probabilities sum to 100%.", 568, 570, 18, MUTED)
        if decision:
            self.text(screen, f"Critic V(s): {decision.value:+.3f}  (global reward estimate)", 568, 593, 20, MUTED)
        self.text(screen, f"Autoplay speed: {self.speed:.2g} moves / second", 568, 625, 24)
        pygame.draw.line(screen, (58, 77, 104), (SLIDER.left, SLIDER.centery), (SLIDER.right, SLIDER.centery), 6)
        position = SLIDER.left + fraction_from_speed(self.speed) * SLIDER.width
        pygame.draw.circle(screen, GREEN, (round(position), SLIDER.centery), 10)
        self.text(screen, "0.25 / slow", 568, 680, 18, MUTED)
        self.text(screen, "60 / fast", 1052, 680, 18, MUTED, right=True)
        message = session.end_reason if session.done else self.message
        self.text(screen, message, 28, 685, 20, MUTED)
        self.button(screen, PLAY, "Pause [Space]" if self.autoplay else "Play [Space]", not session.done, self.autoplay)
        self.button(screen, STEP, "AI step [N]", not session.done)
        self.button(screen, UNDO, "Undo [U]", bool(session.history))
        self.button(screen, NEW, "New game [R]")
        self.button(screen, EXPORT, "Export state [E]", decision is not None)
        self.text(screen, "Arrows: your move   |   Space: play / pause   |   N: one AI move   |   U: undo   |   Esc: close", 28, 786, 20, MUTED)

    def take_step(self, action, source, now):
        if source != "autoplay":
            self.autoplay = False
        if self.session.move(action, source):
            self.message = f"{source.title()}: {ACTIONS[action]}. Recommendations updated."
        else:
            self.message = "That direction is not legal; choose one of the listed moves."
        self.next_move = now + 1000 / self.speed
        if self.session.done:
            self.autoplay = False

    def command(self, name, now):
        if name == "play":
            self.autoplay = not self.autoplay and not self.session.done
            self.next_move = now + 1000 / self.speed
        elif name == "step" and self.session.decision:
            self.take_step(self.session.decision.recommendation, "AI step", now)
        elif name == "undo":
            self.autoplay = False
            self.message = "Restored board, next tile and LSTM memory." if self.session.undo() else "No previous move in this game."
        elif name == "new":
            self.autoplay = False
            seed = self.session.seed + 1
            if 900000 <= seed <= 909999:
                seed = 910000
            self.session.new_game(seed)
            self.message = "New game. Choose a move or press Play."
        elif name == "export":
            path = self.session.export()
            self.message = "State and LSTM memory exported." if path else "No active state to export."
            if path:
                print(f"Exported: {path}", flush=True)

    def event(self, event, now):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            commands = {pygame.K_SPACE:"play", pygame.K_n:"step", pygame.K_u:"undo", pygame.K_r:"new", pygame.K_e:"export"}
            actions = {pygame.K_UP:0, pygame.K_DOWN:1, pygame.K_LEFT:2, pygame.K_RIGHT:3}
            if event.key in commands:
                self.command(commands[event.key], now)
            elif event.key in actions:
                self.take_step(actions[event.key], "manual", now)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if SLIDER.inflate(24, 24).collidepoint(event.pos):
                self.dragging = True
                self.set_speed(event.pos[0], now)
            else:
                for action, rect in self.action_rows():
                    if rect.collidepoint(event.pos):
                        self.take_step(action, "manual", now)
                        break
                for rect, command in ((PLAY,"play"),(STEP,"step"),(UNDO,"undo"),(NEW,"new"),(EXPORT,"export")):
                    if rect.collidepoint(event.pos):
                        self.command(command, now)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_speed(event.pos[0], now)
        return True

    def set_speed(self, mouse_x, now):
        self.speed = speed_from_fraction((mouse_x - SLIDER.left) / SLIDER.width)
        self.next_move = now + 1000 / self.speed

    def tick(self, now):
        if self.autoplay and self.session.decision and now >= self.next_move:
            self.take_step(self.session.decision.recommendation, "autoplay", now)


def run_window(session, speed=3.0, autoplay=False, max_frames=None, screenshot=None):
    pygame.font.init()
    pygame.display.init()
    try:
        screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Threes - policy inspector")
        window = InspectorWindow(session, speed, autoplay)
        window.next_move = pygame.time.get_ticks() + 1000 / window.speed
        clock = pygame.time.Clock()
        frames = 0
        running = True
        while running:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if not window.event(event, now):
                    running = False
            if not running:
                break
            window.tick(now)
            window.draw(screen)
            pygame.display.flip()
            clock.tick(60)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(screenshot))
    finally:
        pygame.quit()
