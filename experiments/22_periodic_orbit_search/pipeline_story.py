"""Manim animation: The full pipeline story.

Scene 1: Three bodies and the problem
Scene 2: Shape sphere appears, points get colored by basin
Scene 3: Zoom into fractal boundary
Scene 4: Boundary points light up as search candidates
Scene 5: Trajectories launch and loop back as periodic orbits

Run: manim -pql pipeline_story.py PipelineStory
High quality: manim -pqh pipeline_story.py PipelineStory
"""
from manim import *
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
CYAN = "#56B4E9"
DARK_BG = "#0a0a1a"


class PipelineStory(ThreeDScene):
    def construct(self):
        self.camera.background_color = DARK_BG

        # === Scene 1: The Three-Body Problem ===
        title = Text("The Three-Body Problem", font_size=42, color=WHITE)
        subtitle = Text("Finding hidden periodic orbits using machine learning",
                        font_size=22, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP*0.2), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Three bodies falling toward each other
        body1 = Dot(point=LEFT*2 + UP*1.2, radius=0.15, color=BLUE)
        body2 = Dot(point=RIGHT*2 + UP*1.2, radius=0.15, color=ORANGE)
        body3 = Dot(point=DOWN*1.5, radius=0.15, color=GREEN)

        label1 = Text("m₁", font_size=18, color=BLUE).next_to(body1, UP, buff=0.15)
        label2 = Text("m₂", font_size=18, color=ORANGE).next_to(body2, UP, buff=0.15)
        label3 = Text("m₃", font_size=18, color=GREEN).next_to(body3, DOWN, buff=0.15)

        gravity_text = Text("Gravity pulls them together...", font_size=24, color=GRAY_B)
        gravity_text.to_edge(DOWN, buff=0.5)

        self.play(
            FadeIn(body1), FadeIn(body2), FadeIn(body3),
            FadeIn(label1), FadeIn(label2), FadeIn(label3),
            run_time=0.8
        )
        self.play(FadeIn(gravity_text, shift=UP*0.2))

        # Animate bodies moving inward chaotically
        self.play(
            body1.animate.move_to(LEFT*0.5 + UP*0.3),
            body2.animate.move_to(RIGHT*0.8 + DOWN*0.2),
            body3.animate.move_to(LEFT*0.2 + DOWN*0.5),
            label1.animate.move_to(LEFT*0.5 + UP*0.6),
            label2.animate.move_to(RIGHT*0.8 + UP*0.1),
            label3.animate.move_to(LEFT*0.2 + DOWN*0.8),
            run_time=2, rate_func=smooth
        )

        chaos_text = Text("...but the outcome is chaotic", font_size=24, color=RED)
        chaos_text.to_edge(DOWN, buff=0.5)
        self.play(FadeOut(gravity_text), FadeIn(chaos_text))

        # One body escapes
        self.play(
            body2.animate.move_to(RIGHT*5 + UP*2),
            label2.animate.move_to(RIGHT*5 + UP*2.3),
            run_time=1.5, rate_func=rush_from
        )
        escape_text = Text("Which body escapes? It depends on the starting positions.",
                           font_size=20, color=GRAY_B)
        escape_text.to_edge(DOWN, buff=0.5)
        self.play(FadeOut(chaos_text), FadeIn(escape_text))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in [body1, body2, body3, label1, label2, label3, escape_text]])

        # === Scene 2: The Shape Sphere ===
        scene2_title = Text("Step 1: Map all possible starting configurations",
                            font_size=28, color=WHITE)
        scene2_title.to_edge(UP, buff=0.5)
        self.play(FadeIn(scene2_title))

        sphere = Sphere(radius=2, resolution=(30, 30))
        sphere.set_color(BLUE_E)
        sphere.set_opacity(0.15)

        sphere_label = Text("Montgomery Shape Sphere", font_size=20, color=CYAN)
        sphere_label.to_edge(DOWN, buff=0.5)

        self.play(Create(sphere), FadeIn(sphere_label), run_time=2)
        self.wait(0.5)

        # Add colored dots representing classified points
        np.random.seed(42)
        n_points = 200
        colors_list = [BLUE, ORANGE, GREEN, RED]
        dots = VGroup()
        for _ in range(n_points):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            r = 2.02
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            c = np.random.choice(colors_list, p=[0.7, 0.1, 0.1, 0.1])
            dot = Dot3D(point=[x, y, z], radius=0.03, color=c)
            dots.add(dot)

        classify_text = Text("Step 2: Classify each point by outcome", font_size=22, color=WHITE)
        classify_text.to_edge(DOWN, buff=0.5)

        self.play(FadeOut(sphere_label))
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.01),
            FadeIn(classify_text),
            run_time=3
        )
        self.wait(1)

        # === Scene 3: Zoom into boundary ===
        boundary_text = Text("Step 3: Find the fractal boundaries between outcomes",
                             font_size=22, color=WHITE)
        boundary_text.to_edge(DOWN, buff=0.5)
        self.play(FadeOut(classify_text), FadeIn(boundary_text))

        # Highlight some boundary dots
        boundary_dots = VGroup()
        for _ in range(50):
            theta = np.random.uniform(0.3, 2.8)
            phi = np.random.uniform(0, 2 * np.pi)
            r = 2.03
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            dot = Dot3D(point=[x, y, z], radius=0.04, color=YELLOW)
            dot.set_opacity(0.8)
            boundary_dots.add(dot)

        self.play(
            LaggedStart(*[FadeIn(d, scale=2) for d in boundary_dots], lag_ratio=0.02),
            run_time=2
        )
        self.wait(1)

        # === Scene 4: Launch trajectories ===
        search_text = Text("Step 4: Launch trajectories from boundary points",
                           font_size=22, color=WHITE)
        search_text.to_edge(DOWN, buff=0.5)
        self.play(FadeOut(boundary_text), FadeIn(search_text))

        # Create a few trajectory curves spiraling out and back
        trajectories = VGroup()
        for i in range(5):
            theta0 = np.random.uniform(0.5, 2.5)
            phi0 = np.random.uniform(0, 2 * np.pi)
            t = np.linspace(0, 4 * np.pi, 100)
            r = 2.05 + 0.3 * np.sin(t * 2) * np.exp(-t / 8)
            theta_t = theta0 + 0.3 * np.sin(t * 1.5 + i)
            phi_t = phi0 + t * 0.2
            x = r * np.sin(theta_t) * np.cos(phi_t)
            y = r * np.sin(theta_t) * np.sin(phi_t)
            z = r * np.cos(theta_t)
            points = [np.array([x[j], y[j], z[j]]) for j in range(len(t))]
            curve = VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color=[CYAN, GREEN, YELLOW][i % 3], width=1.5, opacity=0.7)
            trajectories.add(curve)

        self.play(
            LaggedStart(*[Create(tr) for tr in trajectories], lag_ratio=0.3),
            run_time=3
        )
        self.wait(1)

        # === Scene 5: Periodic orbits found ===
        found_text = Text("Periodic orbits found!", font_size=32, color=GREEN)
        found_text.to_edge(DOWN, buff=0.5)
        self.play(FadeOut(search_text), FadeIn(found_text))

        # Flash the trajectories that loop back
        self.play(
            *[tr.animate.set_stroke(color=GREEN, width=3, opacity=1) for tr in trajectories[:2]],
            run_time=1
        )
        self.wait(1)

        # === Final stats ===
        self.play(
            *[FadeOut(m) for m in [sphere, dots, boundary_dots, trajectories, found_text, scene2_title]]
        )

        stats = VGroup(
            Text("Results", font_size=36, color=WHITE),
            Text("", font_size=10),
            Text("5,000 boundary candidates searched in 88 seconds", font_size=22, color=GRAY_B),
            Text("543 near-returns found (11% hit rate)", font_size=22, color=GRAY_B),
            Text("4 near-periodic orbits discovered", font_size=22, color=GREEN),
            Text("", font_size=10),
            Text("Basin boundaries → Periodic orbit candidates", font_size=20, color=CYAN),
            Text("ML classification guides the search", font_size=20, color=CYAN),
        ).arrange(DOWN, buff=0.25)

        self.play(
            LaggedStart(*[FadeIn(s, shift=UP*0.2) for s in stats], lag_ratio=0.15),
            run_time=3
        )
        self.wait(2)

        # Credits
        self.play(FadeOut(stats))
        credits = VGroup(
            Text("Fractal Basins of the Three-Body Problem", font_size=28, color=WHITE),
            Text("Aishwarya Das  •  Dirac Labs Inc.", font_size=20, color=GRAY_B),
            Text("github.com/anshu4321/fractal-basins-3bp", font_size=18, color=CYAN),
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(credits, shift=UP*0.3), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(credits))
