from manim import *
import numpy as np

class LTTEDecomposition(Scene):
    def construct(self):
        # Basis parameters
        d = 5.0
        a = 1.2
        
        # --- SYSTEEM 1 (Top) ---
        sys1_y = 2.5
        P1 = Dot(radius=0.1, color=WHITE).move_to([-d/2, sys1_y, 0])
        P2 = Dot(radius=0.1, color=BLUE).move_to([d/2, sys1_y, 0])
        v1_1 = MathTex(r"\vec{v}=0", color=WHITE).next_to(P1, UP)
        v1_2 = MathTex(r"\vec{v}=0", color=BLUE).next_to(P2, UP)
        
        brace1 = BraceBetweenPoints(P1.get_center(), P2.get_center(), DOWN, color=RED)
        text1 = MathTex("d", color=RED).next_to(brace1, DOWN)
        sys1 = VGroup(P1, P2, v1_1, v1_2, brace1, text1)
        
        # --- SYSTEEM 2 (Midden) ---
        sys2_y = -0.5
        P3 = Dot(radius=0.1, color=WHITE).move_to([-d/2 + a, sys2_y, 0])
        P4 = Dot(radius=0.1, color=BLUE).move_to([d/2, sys2_y, 0])
        v2_1 = MathTex(r"\vec{v}>0", color=WHITE).next_to(P3, UP)
        v2_2 = MathTex(r"\vec{v}=0", color=BLUE).next_to(P4, UP)
        
        ref_left_2 = np.array([-d/2, sys2_y, 0])
        brace2_a = BraceBetweenPoints(ref_left_2, P3.get_center(), DOWN, color=RED)
        text2_a = MathTex("a", color=RED).next_to(brace2_a, DOWN)
        
        brace2_da = BraceBetweenPoints(P3.get_center(), P4.get_center(), DOWN, color=RED)
        text2_da = MathTex("d-a", color=RED).next_to(brace2_da, DOWN)
        sys2 = VGroup(P3, P4, v2_1, v2_2, brace2_a, text2_a, brace2_da, text2_da)

        # --- SYSTEEM 3 (Onder) ---
        sys3_y = -3.5
        P5 = Dot(radius=0.1, color=WHITE).move_to([-d/2 - a, sys3_y, 0])
        P6 = Dot(radius=0.1, color=BLUE).move_to([d/2, sys3_y, 0])
        v3_1 = MathTex(r"\vec{v}<0", color=WHITE).next_to(P5, UP)
        v3_2 = MathTex(r"\vec{v}=0", color=BLUE).next_to(P6, UP)

        ref_left_3 = np.array([-d/2, sys3_y, 0])
        brace3_a = BraceBetweenPoints(P5.get_center(), ref_left_3, DOWN, color=RED)
        text3_a = MathTex("a", color=RED).next_to(brace3_a, DOWN)
        
        brace3_da = BraceBetweenPoints(P5.get_center(), P6.get_center(), DOWN, color=RED)
        text3_da = MathTex("d+a", color=RED).next_to(brace3_da, DOWN)
        group_da = VGroup(brace3_da, text3_da).shift(DOWN * 0.8)
        
        sys3 = VGroup(P5, P6, v3_1, v3_2, brace3_a, text3_a, group_da)

        # --- Samenvoegen & Framen ---
        all_sys = VGroup(sys1, sys2, sys3)
        all_sys.move_to(ORIGIN) 
        all_sys.scale(0.85)
        self.add(all_sys)

        # ==========================================
        # GOLVEN & FYSICA
        # ==========================================
        
        speed = 2.0          # Constante snelheid van de puls
        packet_len = 1.5     # Fysieke lengte van het golfpakketje
        amp = 0.3            # Hoogte van de golf
        
        # Golflengtes om visueel het Doppler-effect aan te geven
        wl_1 = 0.5   # Groen (Stationair)
        wl_2 = 0.35  # Blauw (Samengedrukt / v>0)
        wl_3 = 0.75  # Rood  (Uitgerekt / v<0)

        # Haal de exacte coördinaten op na het schalen
        x1_s, x1_e = sys1[0].get_x(), sys1[1].get_x()
        x2_s, x2_e = sys2[0].get_x(), sys2[1].get_x()
        x3_s, x3_e = sys3[0].get_x(), sys3[1].get_x()

        y1 = sys1[0].get_y()
        y2 = sys2[0].get_y()
        y3 = sys3[0].get_y()

        time_tracker = ValueTracker(0)

        def create_wave(x_start, x_end, y_pos, wavelength, color):
            # Deze functie wordt elke frame aangeroepen
            def updater():
                t = time_tracker.get_value()
                
                # Bereken waar de voorkant en achterkant van de golf NU zijn
                front_x = x_start + speed * t
                back_x = front_x - packet_len
                
                # Beperk het tekengebied tot TUSSEN de twee stippen
                draw_start = max(x_start, back_x)
                draw_end = min(x_end, front_x)
                
                # Als de golf de stip nog niet uit is, of al volledig binnen is -> niks tekenen
                if draw_end - draw_start < 0.01:
                    return VectorizedPoint(location=[x_start, y_pos, 0])
                
                # Bouw de precieze golf in het toegestane domein
                wave = FunctionGraph(
                    lambda x: amp * np.sin(2 * PI * (x - front_x) / wavelength),
                    x_range=[draw_start, draw_end],
                    color=color
                ).shift(UP * y_pos)
                return wave
            return updater

        # always_redraw zorgt dat de updaters continu de ValueTracker volgen
        wave1 = always_redraw(create_wave(x1_s, x1_e, y1, wl_1, GREEN))
        wave2 = always_redraw(create_wave(x2_s, x2_e, y2, wl_2, BLUE))
        wave3 = always_redraw(create_wave(x3_s, x3_e, y3, wl_3, RED))

        self.add(wave1, wave2, wave3)

        # Bereken hoelang de animatie moet duren (Afstand d+a is de langste)
        max_dist = (x3_e - x3_s) + packet_len
        t_max = max_dist / speed

        # Animatie Loop (4 pulsen)
        for _ in range(4):
            time_tracker.set_value(0) # Reset de tijd
            
            # Laat de tijd oplopen. Hierdoor gaan alle golven perfect tegelijk reizen.
            self.play(time_tracker.animate.set_value(t_max), run_time=t_max, rate_func=linear)
            
            # Pauze voordat de volgende puls start
            self.wait(1.0)