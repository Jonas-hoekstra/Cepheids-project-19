from manim import *
import numpy as np

class OrbitDistanceGraph(Scene):
    def construct(self):
        # Basis parameters
        d = 5.0
        r = 1.5

        # Tracker voor de fase (van 0 tot 2*PI)
        phi = ValueTracker(0)

        # ==========================================
        # BOVENSTE DEEL: GEOMETRIE
        # ==========================================
        center_pt = np.array([-2.5, 2.0, 0])
        observer_pt = np.array([-2.5 + d, 2.0, 0])

        center_dot = Dot(center_pt, color=WHITE)
        observer_dot = Dot(observer_pt, color=BLUE)

        orbit = Circle(radius=r, color=DARK_GRAY).move_to(center_pt)

        # Statische lijn voor afstand 'd'
        line_d = Line(center_pt, observer_pt, color=GRAY, stroke_width=2)
        label_d = MathTex("d").next_to(line_d, DOWN, buff=0.1)

        # Dynamische stip op de baan
        # Door -sin en +cos te gebruiken, start hij "boven" (fase 0) en draait tegen de klok in.
        # Hierdoor neemt de afstand in het begin toe, exact zoals in jouw schets.
        def get_orbit_pos():
            val = phi.get_value()
            return center_pt + np.array([-r * np.sin(val), r * np.cos(val), 0])

        orbit_dot = always_redraw(lambda: Dot(get_orbit_pos(), color=RED))

        # Radius lijn 'r'
        line_r = always_redraw(lambda: Line(center_pt, orbit_dot.get_center(), color=RED, stroke_width=2))
        label_r = always_redraw(lambda: MathTex("r").move_to(line_r.get_center() + UP*0.3 + LEFT*0.2).scale(0.8))

        # Dynamische afstandslijn '\Delta s'
        line_s = always_redraw(lambda: Line(orbit_dot.get_center(), observer_pt, color=YELLOW, stroke_width=3))
        label_s = always_redraw(lambda: MathTex(r"\Delta s").move_to(line_s.get_center() + UP*0.3).scale(0.8))

        top_group = VGroup(orbit, line_d, label_d, center_dot, observer_dot, orbit_dot, line_r, label_r, line_s, label_s)

        # Formule rechtsboven (uitgedrukt in r en d zoals gevraagd)
        formula = MathTex(r"\Delta s = \sqrt{r^2 + d^2 + 2rd\sin(\phi)}").to_corner(UR).scale(0.8)
        val_text = always_redraw(
            lambda: MathTex(rf"\Delta s \approx {np.linalg.norm(get_orbit_pos() - observer_pt):.2f}")
            .next_to(formula, DOWN).scale(0.8)
        )

        # ==========================================
        # ONDERSTE DEEL: GRAFIEK
        # ==========================================
        # Assen aanmaken
        ax = Axes(
            x_range=[0, 2 * PI, PI / 2],
            y_range=[d - r - 0.5, d + r + 0.5, r], # Y-as past zich aan aan min/max afstand
            x_length=8,
            y_length=3.5,
            axis_config={"include_tip": False}
        ).to_edge(DOWN, buff=0.8).shift(RIGHT * 0.5)

        # Labels voor de assen (zoals in je schets)
        x_label = MathTex(r"\text{Fase } \phi \rightarrow").next_to(ax.x_axis, RIGHT)
        y_label = MathTex(r"\text{Afstand}(s)").next_to(ax.y_axis, UP)

        # Labels voor \pi en 2\pi op de X-as
        pi_label = MathTex(r"\pi").next_to(ax.c2p(PI, d - r - 0.5), DOWN)
        two_pi_label = MathTex(r"2\pi").next_to(ax.c2p(2*PI, d - r - 0.5), DOWN)

        # Evenwichtsstand (baseline 'd') tekenen
        eq_line = DashedLine(ax.c2p(0, d), ax.c2p(2 * PI, d), color=GRAY)
        eq_label = MathTex("d").next_to(eq_line, LEFT)

        # Het dynamische punt op de grafiek (gekoppeld aan de exacte lengte van line_s)
        def get_graph_y():
            return np.linalg.norm(get_orbit_pos() - observer_pt)

        graph_dot = always_redraw(lambda: Dot(ax.c2p(phi.get_value(), get_graph_y()), color=YELLOW))
        
        # Laat het punt een spoor achterlaten (de sinusgolf)
        trace = TracedPath(graph_dot.get_center, stroke_color=YELLOW, stroke_width=2)

        # Alles toevoegen aan de scene
        self.add(top_group, formula, val_text)
        self.add(ax, x_label, y_label, pi_label, two_pi_label, eq_line, eq_label, graph_dot, trace)

        # ==========================================
        # ANIMATIE STARTEN
        # ==========================================
        self.wait(0.5)
        
        # Verander de fase van 0 naar 2*PI in 6 seconden
        self.play(phi.animate.set_value(2 * PI), run_time=6, rate_func=linear)
        
        self.wait(2)