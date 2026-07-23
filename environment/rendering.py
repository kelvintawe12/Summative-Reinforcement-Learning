"""
rendering.py

Pygame-based visualization for WasteSegregationEnv. Draws a top-down facility
schematic: an incoming item on a conveyor, four bins with live fill and
contamination indicators (jammed bins flash red), an energy gauge, and a
running reward readout.

This module is intentionally decoupled from the environment's core logic --
`custom_env.py` never imports pygame at module scope, only inside `render()`,
so the environment remains fully usable (training, testing) in headless
environments with no display and no pygame video driver.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

WIDTH, HEIGHT = 900, 560
BIN_COLORS = {
    "organic": (110, 168, 90),
    "plastic": (86, 140, 210),
    "metal": (170, 170, 180),
    "glass": (90, 200, 190),
}
BACKGROUND = (24, 26, 31)
PANEL = (36, 39, 46)
TEXT_COLOR = (230, 230, 235)
MUTED_TEXT = (150, 152, 160)
JAM_FLASH = (200, 60, 60)
CONVEYOR_COLOR = (50, 53, 61)


class PygameRenderer:
    """Encapsulates all pygame state so custom_env.py stays pygame-free
    outside of the render() call path."""

    def __init__(self, render_mode: str = "human") -> None:
        import pygame

        self._pygame = pygame
        self.render_mode = render_mode
        pygame.init()
        pygame.font.init()

        if render_mode == "human":
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Waste Segregation Facility")
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("consolas,menlo,monospace", 14)
        self.font_medium = pygame.font.SysFont("consolas,menlo,monospace", 18)
        self.font_large = pygame.font.SysFont("consolas,menlo,monospace", 24)

    def render(self, env) -> Optional[np.ndarray]:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass  # caller decides whether to close; ignore here

        self.screen.fill(BACKGROUND)
        self._draw_header(env)
        self._draw_conveyor_and_item(env)
        self._draw_bins(env)
        self._draw_status_panel(env)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(env.metadata.get("render_fps", 8))
            return None
        else:
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1, 0, 2))

    def _draw_header(self, env) -> None:
        pygame = self._pygame
        title = self.font_large.render(
            "Waste Segregation Facility", True, TEXT_COLOR
        )
        self.screen.blit(title, (24, 16))
        step_txt = self.font_medium.render(
            f"Step {env.step_count}/{env.max_steps}   "
            f"Reward: {env.total_reward:+.1f}   "
            f"Last action: {env.last_action_name or '-'}",
            True, MUTED_TEXT,
        )
        self.screen.blit(step_txt, (24, 50))

    def _draw_conveyor_and_item(self, env) -> None:
        pygame = self._pygame
        belt_rect = pygame.Rect(24, 100, 340, 60)
        pygame.draw.rect(self.screen, CONVEYOR_COLOR, belt_rect, border_radius=6)
        label = self.font_small.render("Incoming item (noisy sensor reading):", True, MUTED_TEXT)
        self.screen.blit(label, (24, 84))

        comp = env.current_item_obs
        names = ["Org", "Pla", "Met", "Gla", "Con"]
        colors = [
            BIN_COLORS["organic"], BIN_COLORS["plastic"],
            BIN_COLORS["metal"], BIN_COLORS["glass"], (150, 90, 90),
        ]
        x = belt_rect.x + 10
        bar_w = (belt_rect.width - 20) / 5
        for i, (name, color) in enumerate(zip(names, colors)):
            h = int(40 * comp[i])
            bar_rect = pygame.Rect(
                int(x + i * bar_w), belt_rect.y + belt_rect.height - h - 10,
                int(bar_w - 4), h,
            )
            pygame.draw.rect(self.screen, color, bar_rect)
            lbl = self.font_small.render(name, True, MUTED_TEXT)
            self.screen.blit(lbl, (bar_rect.x, belt_rect.y + belt_rect.height + 2))

        mass_txt = self.font_small.render(
            f"mass: {env.current_item_mass:.2f}", True, MUTED_TEXT
        )
        self.screen.blit(mass_txt, (belt_rect.x + belt_rect.width + 12, belt_rect.y + 20))

    def _draw_bins(self, env) -> None:
        pygame = self._pygame
        from environment.custom_env import BINS

        bin_w, bin_h = 160, 220
        gap = 24
        start_x = 24
        start_y = 200

        for i, name in enumerate(BINS):
            x = start_x + i * (bin_w + gap)
            rect = pygame.Rect(x, start_y, bin_w, bin_h)
            jammed = env.bin_jam_cooldown[i] > 0
            border_color = JAM_FLASH if jammed else BIN_COLORS[name]
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_color, rect, width=3, border_radius=8)

            fill_frac = min(env.bin_fill[i] / env.bin_capacity, 1.0)
            fill_h = int((bin_h - 20) * fill_frac)
            fill_rect = pygame.Rect(
                x + 10, start_y + bin_h - 10 - fill_h, bin_w - 20, fill_h
            )
            pygame.draw.rect(self.screen, BIN_COLORS[name], fill_rect, border_radius=4)

            contam_frac = 0.0
            if env.bin_total_mass[i] > 1e-9:
                contam_frac = env.bin_contam_mass[i] / env.bin_total_mass[i]

            label = self.font_medium.render(name.capitalize(), True, TEXT_COLOR)
            self.screen.blit(label, (x + 8, start_y - 26))

            info_lines = [
                f"fill: {fill_frac*100:4.0f}%",
                f"contam: {contam_frac*100:4.0f}%",
                "JAMMED" if jammed else "online",
            ]
            for j, line in enumerate(info_lines):
                color = JAM_FLASH if (jammed and j == 2) else MUTED_TEXT
                txt = self.font_small.render(line, True, color)
                self.screen.blit(txt, (x + 8, start_y + bin_h + 6 + j * 16))

    def _draw_status_panel(self, env) -> None:
        pygame = self._pygame
        panel_x = 24
        panel_y = 470
        energy_frac = env.energy / env.energy_budget_max
        pygame.draw.rect(self.screen, PANEL, (panel_x, panel_y, 300, 24), border_radius=4)
        pygame.draw.rect(
            self.screen, (210, 190, 90),
            (panel_x, panel_y, int(300 * max(energy_frac, 0.0)), 24),
            border_radius=4,
        )
        energy_txt = self.font_small.render(
            f"Energy: {env.energy:.1f}/{env.energy_budget_max:.0f}", True, TEXT_COLOR
        )
        self.screen.blit(energy_txt, (panel_x + 8, panel_y + 3))

        stats = env.episode_stats
        stats_txt = self.font_small.render(
            f"Correct sorts: {stats['correct_sorts']}   "
            f"Incorrect: {stats['incorrect_sorts']}   "
            f"Jams: {stats['jams_triggered']}   "
            f"Ship-outs: {stats['shipouts']}",
            True, MUTED_TEXT,
        )
        self.screen.blit(stats_txt, (panel_x, panel_y + 34))

    def close(self) -> None:
        self._pygame.quit()
