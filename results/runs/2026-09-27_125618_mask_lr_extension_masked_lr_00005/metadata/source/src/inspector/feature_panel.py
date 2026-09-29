"""Interactive layer heatmaps and a selected-unit trace for the policy viewer."""

import math

import numpy as np
import pygame

from .activations import LAYER_LABELS, activation_series


NEGATIVE, NEUTRAL, POSITIVE = (94, 167, 247), (28, 41, 60), (245, 167, 93)


def activation_color(value, scale):
    fraction = min(1., abs(float(value)) / max(float(scale), 1e-12))
    endpoint = POSITIVE if value >= 0 else NEGATIVE
    return tuple(round(a + fraction * (b - a)) for a, b in zip(NEUTRAL, endpoint))


class FeaturePanel:
    def __init__(self):
        self.layer = "hidden"
        self.units = {}

    def buttons(self, decision):
        layers = [name for name in LAYER_LABELS if name in decision.activations]
        return [(name, pygame.Rect(568 + (i % 4) * 123, 260 + (i // 4) * 32, 115, 27))
                for i, name in enumerate(layers)]

    def geometry(self, decision):
        values = decision.activations[self.layer]
        if self.layer == "slots":
            columns, width, height = 8, 32, 13
            left = 718
        else:
            columns = 64 if values.size > 1024 else 32
            width, height, left = 448 // columns, 7, 600
        rows = math.ceil(values.size / columns)
        return pygame.Rect(left, 354, columns * width, rows * height), columns, width, height

    def unit_at(self, decision, position):
        rect, columns, width, height = self.geometry(decision)
        if not rect.collidepoint(position):
            return None
        index = ((position[1] - rect.y) // height) * columns + (position[0] - rect.x) // width
        return index if index < decision.activations[self.layer].size else None

    def click(self, decision, position):
        if decision is None:
            return
        for name, rect in self.buttons(decision):
            if rect.collidepoint(position):
                self.layer = name
                return
        unit = self.unit_at(decision, position)
        if unit is not None:
            self.units[self.layer] = unit

    def board_slot(self):
        if self.layer == "slots":
            slot = self.units.get(self.layer, 0) // 8
            return slot if slot < 16 else None
        return None

    def draw(self, window, screen):
        session = window.session
        decision = session.decision
        text = window.text
        text(screen, "Inside the network", 568, 207, 28)
        text(screen, "Activations, not attention or feature importance.", 568, 236, 20)
        if decision is None:
            text(screen, "Game ended. Undo to inspect the last decision.", 568, 290, 20)
            return
        if self.layer not in decision.activations:
            self.layer = next(iter(decision.activations))
        for name, rect in self.buttons(decision):
            pygame.draw.rect(screen, (92, 222, 175) if name == self.layer else NEUTRAL, rect, border_radius=6)
            color = (16, 24, 38) if name == self.layer else (239, 244, 251)
            label = window.fonts[18].render(LAYER_LABELS[name], True, color)
            screen.blit(label, label.get_rect(center=rect.center))
        vector = decision.activations[self.layer]
        selected = min(self.units.get(self.layer, 0), vector.size - 1)
        self.units[self.layer] = selected
        rect, columns, width, height = self.geometry(decision)
        scale = 1. if self.layer == "hidden" else max(float(np.abs(vector).max()), 1e-12)
        label = "17 slots x 8 channels; original board coordinates" if self.layer == "slots" else f"{vector.size:,} units in index order; grid is not the board"
        text(screen, label, 568, 331, 18)
        for index, value in enumerate(vector.flat):
            row, col = divmod(index, columns)
            cell = pygame.Rect(rect.x + col * width, rect.y + row * height, width, height)
            pygame.draw.rect(screen, activation_color(value, scale), cell)
            if index == selected:
                pygame.draw.rect(screen, (255, 255, 255), cell, 1)
        if self.layer == "slots":
            for slot in range(17):
                label = f"r{slot // 4 + 1} c{slot % 4 + 1}" if slot < 16 else "Next tile"
                text(screen, label, 626, rect.y + slot * height, 18)
        selected_value = float(vector.flat[selected])
        text(screen, f"Unit {selected}: {selected_value:+.4f}  |  Click a cell to track it", 568, 590, 20)
        mode = "fixed" if self.layer == "hidden" else "current-state"
        text(screen, f"Blue -{scale:.3g}    dark 0    orange +{scale:.3g}  ({mode} scale)", 568, 613, 18)
        values = activation_series(session, self.layer, selected)
        plot = pygame.Rect(600, 641, 448, 34)
        pygame.draw.rect(screen, NEUTRAL, plot)
        if len(values):
            trace_scale = max(float(np.abs(values).max()), 1e-12)
            pygame.draw.line(screen, (85, 102, 126), (plot.left, plot.centery), (plot.right, plot.centery))
            points = [(round(plot.left + i * plot.width / 79),
                       round(plot.centery - float(v) / trace_scale * (plot.height / 2 - 2)))
                      for i, v in enumerate(values)]
            if len(points) > 1:
                pygame.draw.lines(screen, (92, 222, 175), False, points, 2)
            pygame.draw.circle(screen, (239, 244, 251), points[-1], 2)
            text(screen, f"Last {len(values)} positions | trace scale +/-{trace_scale:.3g}", 600, 679, 18)
