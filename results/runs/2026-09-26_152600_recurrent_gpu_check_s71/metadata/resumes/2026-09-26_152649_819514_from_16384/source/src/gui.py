import pygame
from pathlib import Path

from threes import Direction, Game


WINDOW_SIZE = (600, 880)
BOARD_LEFT = 77
BOARD_TOP = 200
TILE_WIDTH = 106
TILE_HEIGHT = 132
TILE_GAP = 10
NEW_GAME_BUTTON = pygame.Rect(52, 782, 496, 52)

BACKGROUND = (245, 247, 250)
PANEL = (220, 225, 232)
EMPTY_TILE = (229, 233, 238)
WHITE_TILE = (255, 255, 255)
TEXT = (17, 20, 25)
MUTED_TEXT = (83, 91, 101)
TILE_COLORS = {
    1: (139, 203, 235),
    2: (248, 150, 181),
}


def _draw_centered(
    surface: pygame.Surface,
    value: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    center: tuple[int, int],
) -> None:
    text = font.render(value, True, color)
    surface.blit(text, text.get_rect(center=center))


def draw_game(
    surface: pygame.Surface,
    game: Game,
    fonts: dict[int, pygame.font.Font],
    status: str,
    seed: int,
    ai_available: bool = False,
) -> None:
    surface.fill(BACKGROUND)
    _draw_centered(surface, "THREES ML", fonts[46], TEXT, (300, 42))
    _draw_centered(surface, f"Graine {seed}", fonts[22], MUTED_TEXT, (300, 71))

    pygame.draw.rect(surface, PANEL, (52, 92, 238, 72), border_radius=12)
    pygame.draw.rect(surface, PANEL, (310, 92, 238, 72), border_radius=12)
    _draw_centered(surface, "SCORE", fonts[22], MUTED_TEXT, (171, 111))
    _draw_centered(surface, "PROCHAINE", fonts[22], MUTED_TEXT, (429, 111))
    _draw_centered(surface, str(game.current_score), fonts[38], TEXT, (171, 139))
    next_hint = "-" if game.is_over else game.next_hint
    if isinstance(next_hint, tuple):
        _draw_centered(surface, " / ".join(map(str, next_hint)), fonts[26], TEXT, (429, 139))
    else:
        _draw_centered(surface, str(next_hint), fonts[38], TEXT, (429, 139))

    message_color = (171, 58, 58) if status == "Coup impossible" else MUTED_TEXT
    _draw_centered(surface, status, fonts[24], message_color, (300, 177))
    pygame.draw.rect(surface, PANEL, (67, 190, 466, 578), border_radius=18)

    for row_index, row in enumerate(game.board):
        for column_index, value in enumerate(row):
            x_position = BOARD_LEFT + column_index * (TILE_WIDTH + TILE_GAP)
            y_position = BOARD_TOP + row_index * (TILE_HEIGHT + TILE_GAP)
            tile = pygame.Rect(x_position, y_position, TILE_WIDTH, TILE_HEIGHT)
            color = TILE_COLORS.get(value, EMPTY_TILE if value == 0 else WHITE_TILE)
            pygame.draw.rect(surface, color, tile, border_radius=13)
            if value:
                font_size = 62 if value < 100 else 50 if value < 1000 else 39 if value < 10000 else 31
                _draw_centered(surface, str(value), fonts[font_size], TEXT, tile.center)

    pygame.draw.rect(surface, TEXT, NEW_GAME_BUTTON, border_radius=12)
    _draw_centered(surface, "NOUVELLE PARTIE  (R)", fonts[26], (255, 255, 255), NEW_GAME_BUTTON.center)
    _draw_centered(
        surface,
        ("SPACE pause/resume AI - R new game - ESC quit" if ai_available else
         "Fleches, ZQSD ou WASD pour bouger - Echap pour fermer"),
        fonts[20],
        MUTED_TEXT,
        (300, 858),
    )


def run_gui(seed: int, *, ai_model: Path | None = None, ai_speed: int = 10,
            max_frames: int | None = None) -> None:
    if ai_speed < 1:
        raise ValueError("ai_speed must be positive")
    if ai_model is not None:
        from external_agent import KiokOnnxAgent
        agent = KiokOnnxAgent(ai_model)
    else:
        agent = None
    pygame.init()
    try:
        surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Threes ML")
        fonts = {
            size: pygame.font.Font(None, size)
            for size in (20, 22, 24, 26, 31, 38, 39, 46, 50, 62)
        }
        directions: dict[int, Direction] = {
            pygame.K_UP: "up",
            pygame.K_z: "up",
            pygame.K_w: "up",
            pygame.K_DOWN: "down",
            pygame.K_s: "down",
            pygame.K_LEFT: "left",
            pygame.K_q: "left",
            pygame.K_a: "left",
            pygame.K_RIGHT: "right",
            pygame.K_d: "right",
        }
        game = Game(seed=seed)
        autoplay = agent is not None
        last_ai_move = pygame.time.get_ticks()
        status = "Kiok AI running (Space pauses)" if autoplay else "Utilise les fleches pour jouer"
        clock = pygame.time.Clock()
        running = True
        frames = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        seed += 1
                        game = Game(seed=seed)
                        if agent is not None:
                            agent.reset()
                            autoplay = True
                            last_ai_move = pygame.time.get_ticks()
                        status = "Kiok AI running (Space pauses)" if autoplay else "Nouvelle partie"
                    elif event.key == pygame.K_SPACE and agent is not None:
                        autoplay = not autoplay and not game.is_over
                        status = "Kiok AI running (Space pauses)" if autoplay else "Kiok AI paused"
                    elif event.key in directions:
                        if agent is not None:
                            status = "Kiok AI controls this game; press R to restart"
                        elif game.is_over:
                            status = "Partie terminee"
                        elif game.step(directions[event.key]):
                            status = "Partie terminee" if game.is_over else ""
                        else:
                            status = "Coup impossible"
                elif event.type == pygame.MOUSEBUTTONDOWN and NEW_GAME_BUTTON.collidepoint(event.pos):
                    seed += 1
                    game = Game(seed=seed)
                    if agent is not None:
                        agent.reset()
                        autoplay = True
                        last_ai_move = pygame.time.get_ticks()
                    status = "Kiok AI running (Space pauses)" if autoplay else "Nouvelle partie"

            now = pygame.time.get_ticks()
            if autoplay and not game.is_over and now - last_ai_move >= 1000 / ai_speed:
                direction = agent.choose_action(game)
                game.step(direction)
                last_ai_move = now
                if game.is_over:
                    autoplay = False
                    status = f"Kiok AI finished: score {game.current_score}"

            draw_game(surface, game, fonts, status, seed, ai_available=agent is not None)
            pygame.display.flip()
            clock.tick(30)
            frames += 1
            if max_frames is not None and frames >= max_frames:
                running = False
    finally:
        pygame.quit()
