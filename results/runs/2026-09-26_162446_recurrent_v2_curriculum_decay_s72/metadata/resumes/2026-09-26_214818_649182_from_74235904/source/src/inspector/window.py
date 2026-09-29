"""Pygame policy viewer: legal preferences, human overrides, undo and speed control."""

import math
from pathlib import Path

import pygame

from .session import ACTIONS, tile_value
from .feature_panel import FeaturePanel


SIZE = (1080, 900)
BG, PANEL, INK, MUTED = (16, 24, 38), (28, 41, 60), (239, 244, 251), (159, 176, 199)
GREEN = (92, 222, 175)
TILE_WIDTH, TILE_HEIGHT, TILE_GAP = 90, 132, 12
BOARD = pygame.Rect(80, 202, 396, 564)
PLAY = pygame.Rect(28, 808, 160, 48)
STEP = pygame.Rect(198, 808, 160, 48)
UNDO = pygame.Rect(368, 808, 160, 48)
NEW = pygame.Rect(568, 808, 226, 48)
EXPORT = pygame.Rect(806, 808, 246, 48)
SLIDER = pygame.Rect(588, 722, 444, 22)
FLIP = pygame.Rect(568, 156, 232, 28)
FEATURES = pygame.Rect(568, 185, 232, 20)
ARROWS = {
    0: pygame.Rect(736, 266, 144, 94),
    1: pygame.Rect(736, 476, 144, 94),
    2: pygame.Rect(578, 371, 144, 94),
    3: pygame.Rect(894, 371, 144, 94),
}


def preference_color(probability):
    """Absolute probability scale: almost neutral at zero, dark green at one."""
    p = max(0.0, min(1.0, float(probability)))
    low, middle, high = (225, 233, 227), (95, 187, 126), (5, 65, 30)
    start, end, blend = (low, middle, 2 * p) if p < .5 else (middle, high, 2 * p - 1)
    return tuple(round(a + (b - a) * blend) for a, b in zip(start, end))


def speed_from_fraction(fraction):
    return .25 * 240 ** max(0, min(1, fraction))


def fraction_from_speed(speed):
    return math.log(max(.25, min(60, speed)) / .25, 240)


class InspectorWindow:
    def __init__(self, session, speed=3.0, autoplay=False, features=False):
        self.session = session
        self.speed = max(.25, min(60, speed))
        self.autoplay = autoplay
        self.dragging = False
        self.flip_vertical = False
        self.show_features = features
        self.feature_panel = FeaturePanel()
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
        return [(action, ARROWS[self.display_action(action)]) for action in range(4) if decision.legal[action]]

    def display_action(self, action):
        # Reflection is its own inverse: also maps a keyboard direction back to
        # the canonical simulator action. The policy always sees the same game.
        return 1 - action if self.flip_vertical and action < 2 else action

    def cell_rect(self, index):
        row, col = divmod(index, 4)
        if self.flip_vertical:
            row = 3 - row
        return pygame.Rect(BOARD.x + col * (TILE_WIDTH + TILE_GAP),
                           BOARD.y + row * (TILE_HEIGHT + TILE_GAP), TILE_WIDTH, TILE_HEIGHT)

    def draw_tile(self, screen, rect, value, small=False):
        color = (102, 189, 227) if value == 1 else (242, 136, 162) if value == 2 else (233, 237, 228) if value else PANEL
        pygame.draw.rect(screen, color, rect, border_radius=8 if small else 13)
        if value:
            size = (28 if value < 100 else 20) if small else (54 if value < 1000 else 38 if value < 10000 else 32)
            image = self.fonts[size].render(str(value), True, BG)
            screen.blit(image, image.get_rect(center=rect.center))

    def draw_arrow(self, screen, action, rect, probability, best):
        pygame.draw.rect(screen, preference_color(probability), rect, border_radius=12)
        foreground = (240, 250, 243) if probability >= .65 else (24, 69, 41)
        secondary = (191, 222, 201) if probability >= .65 else (56, 91, 69)
        if best:
            pygame.draw.rect(screen, GREEN, rect, width=3, border_radius=12)
        # Draw geometry instead of relying on arrow glyphs in a platform font.
        points = [(-22, 0), (-10, 0), (-10, 22), (10, 22), (10, 0), (22, 0), (0, -23)]
        direction = self.display_action(action)
        if direction == 1:
            points = [(-x, -y) for x, y in points]
        elif direction == 2:
            points = [(y, -x) for x, y in points]
        elif direction == 3:
            points = [(-y, x) for x, y in points]
        points = [(rect.centerx + x, rect.y + 32 + y) for x, y in points]
        pygame.draw.polygon(screen, foreground, points)
        label = "<0.01" if 0 < probability < .005 else f"{probability:.2f}"
        image = self.fonts[24].render(label, True, foreground)
        screen.blit(image, image.get_rect(center=(rect.centerx, rect.y + 70)))
        logit = self.session.decision.logits[action]
        image = self.fonts[18].render(f"logit {logit:+.2f}", True, secondary)
        screen.blit(image, image.get_rect(center=(rect.centerx, rect.y + 86)))

    def draw_quality(self, screen):
        """Show the active game's last 120 moves; undo removes its last point."""
        samples = self.session.move_quality[-120:]
        total = len(self.session.move_quality)
        blue = (111, 184, 255)
        self.text(screen, "Move quality vs AI", 568, 582, 22)
        if samples:
            self.text(screen, f"Last: {samples[-1].relative_preference:.0%}", 1052, 582, 20, right=True)
        plot = pygame.Rect(602, 609, 438, 60)
        pygame.draw.rect(screen, PANEL, plot)
        for value in (0, .5, 1):
            y = round(plot.bottom - value * plot.height)
            pygame.draw.line(screen, (58, 77, 104), (plot.left, y), (plot.right, y))
            self.text(screen, f"{value:g}", 592, y - 6, 18, MUTED, right=True)
        points = [(round(plot.left + index * plot.width / 119),
                   round(plot.bottom - sample.relative_preference * plot.height))
                  for index, sample in enumerate(samples)]
        if len(points) > 1:
            pygame.draw.lines(screen, MUTED, False, points, 1)
        for point, sample in zip(points, samples):
            pygame.draw.circle(screen, blue if sample.source == "manual" else GREEN, point, 3)
        if not samples:
            self.text(screen, "Play a move to start the chart.", plot.left + 12, plot.y + 17, 20, MUTED)
        self.text(screen, "You", 568, 676, 18, blue)
        self.text(screen, "AI", 606, 676, 18, GREEN)
        span = f"Moves {max(1, total - 119)}-{total}" if samples else "Last 120 moves"
        self.text(screen, span, 638, 676, 18, MUTED)
        self.text(screen, "1 = top AI choice; not proven optimal", 1052, 676, 18, MUTED, right=True)

    def draw(self, screen):
        session = self.session
        screen.fill(BG)
        self.text(screen, "THREES / POLICY INSPECTOR", 28, 22, 32)
        label = session.policy.label
        if len(label) > 85:
            label = label[:82] + "..."
        self.text(screen, label, 28, 58, 20, MUTED)
        self.text(screen, "AUTO" if self.autoplay else "YOUR MOVE", 1052, 24, 24, GREEN, right=True)
        for x, title, value in ((568, "SCORE", f"{session.score:,}"),
                                (732, "LARGEST TILE", f"{session.maximum:,}"),
                                (896, "MOVES", int(session.env.meta[0, 2]))):
            pygame.draw.rect(screen, PANEL, (x, 96, 156, 52), border_radius=10)
            self.text(screen, title, x + 12, 103, 18, MUTED)
            self.text(screen, value, x + 12, 122, 24)
        next_label = self.fonts[18].render("NEXT TILE", True, MUTED)
        screen.blit(next_label, next_label.get_rect(center=(BOARD.centerx, 96)))
        if not session.done:
            preview = session.preview.split(" / ")
            left = BOARD.centerx - (len(preview) * 48 + (len(preview) - 1) * 10) // 2
            for index, value in enumerate(preview):
                self.draw_tile(screen, pygame.Rect(left + index * 58, 110, 48, 70), int(value), small=True)
        self.text(screen, f"Seed {session.seed} | CPU", 1052, 164, 18, MUTED, right=True)
        self.button(screen, FLIP, "Restore view [F]" if self.flip_vertical else "Flip top / bottom [F]")
        self.text(screen, "Move advice [V]" if self.show_features else "Inside the network [V]", FEATURES.x, FEATURES.y, 20, GREEN)
        for index, rank in enumerate(session.env.board[0]):
            self.draw_tile(screen, self.cell_rect(index), tile_value(rank))
        if session.decision:
            self.text(screen, f"Critic V(s): {session.decision.value:+.3f}", 1052, 186, 18, MUTED, right=True)
        if self.show_features:
            self.feature_panel.draw(self, screen)
            slot = self.feature_panel.board_slot()
            if slot is not None and session.decision is not None:
                pygame.draw.rect(screen, GREEN, self.cell_rect(slot), 3, border_radius=13)
        else:
            self.draw_advice(screen)
        self.draw_controls(screen)

    def draw_advice(self, screen):
        session = self.session
        self.text(screen, "Your move + AI advice", 568, 207, 28)
        self.text(screen, "Use your arrow keys or click an arrow below.", 568, 237, 20, MUTED)
        decision = session.decision
        for action, rect in self.action_rows():
            self.draw_arrow(screen, action, rect, decision.probabilities[action], action == decision.recommendation)
        if decision:
            best = ACTIONS[self.display_action(decision.recommendation)]
            for offset, label in ((0, "AI suggests"), (26, best)):
                image = self.fonts[24].render(label, True, MUTED if offset == 0 else GREEN)
                screen.blit(image, image.get_rect(center=(808, 400 + offset)))
        if decision is None:
            self.text(screen, session.end_reason or "No legal moves", 590, 300, 28, GREEN)
        self.draw_quality(screen)

    def draw_controls(self, screen):
        session = self.session
        self.text(screen, f"Autoplay speed: {self.speed:.2g} moves / second", 568, 697, 24)
        pygame.draw.line(screen, (58, 77, 104), (SLIDER.left, SLIDER.centery), (SLIDER.right, SLIDER.centery), 6)
        position = SLIDER.left + fraction_from_speed(self.speed) * SLIDER.width
        pygame.draw.circle(screen, GREEN, (round(position), SLIDER.centery), 10)
        self.text(screen, "0.25 / slow", 568, 752, 18, MUTED)
        self.text(screen, "60 / fast", 1052, 752, 18, MUTED, right=True)
        message = session.end_reason if session.done else self.message
        self.text(screen, message, 28, 782, 20, MUTED)
        self.button(screen, PLAY, "Pause [Space]" if self.autoplay else "Play [Space]", not session.done, self.autoplay)
        self.button(screen, STEP, "AI step [N]", not session.done)
        self.button(screen, UNDO, "Undo [U]", bool(session.history))
        self.button(screen, NEW, "New game [R]")
        self.button(screen, EXPORT, "Export state [E]", session.decision is not None)
        self.text(screen, "Arrows: your move   |   Space: play / pause   |   N: one AI move   |   U: undo   |   Esc: close", 28, 874, 20, MUTED)

    def take_step(self, action, source, now):
        if source != "autoplay":
            self.autoplay = False
        if self.session.move(action, source):
            self.message = f"{source.title()}: {ACTIONS[self.display_action(action)]}. Advice updated."
        else:
            self.message = "That direction is not legal; choose one of the listed moves."
        self.next_move = now + 1000 / self.speed
        if self.session.done:
            self.autoplay = False

    def command(self, name, now):
        if name == "features":
            self.show_features = not self.show_features
            self.session.record("view", features=self.show_features)
        elif name == "flip":
            self.flip_vertical = not self.flip_vertical
            self.session.record("view", flipped_vertical=self.flip_vertical)
            self.message = "View flipped; controls follow the screen." if self.flip_vertical else "Original view restored."
        elif name == "play":
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
            commands = {pygame.K_SPACE:"play", pygame.K_n:"step", pygame.K_u:"undo", pygame.K_r:"new", pygame.K_e:"export", pygame.K_f:"flip", pygame.K_v:"features"}
            actions = {pygame.K_UP:0, pygame.K_DOWN:1, pygame.K_LEFT:2, pygame.K_RIGHT:3}
            if event.key in commands:
                self.command(commands[event.key], now)
            elif event.key in actions:
                self.take_step(self.display_action(actions[event.key]), "manual", now)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if SLIDER.inflate(24, 24).collidepoint(event.pos):
                self.dragging = True
                self.set_speed(event.pos[0], now)
            else:
                if self.show_features:
                    self.feature_panel.click(self.session.decision, event.pos)
                else:
                    for action, rect in self.action_rows():
                        if rect.collidepoint(event.pos):
                            self.take_step(action, "manual", now)
                            break
                for rect, command in ((PLAY,"play"),(STEP,"step"),(UNDO,"undo"),(NEW,"new"),(EXPORT,"export"),(FLIP,"flip"),(FEATURES,"features")):
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


def run_window(session, speed=3.0, autoplay=False, max_frames=None, screenshot=None, features=False):
    pygame.font.init()
    pygame.display.init()
    try:
        screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Threes - policy inspector")
        window = InspectorWindow(session, speed, autoplay, features)
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
