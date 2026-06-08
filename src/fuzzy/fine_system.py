import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FineSystem:
    """
    Sistema de Inferencia Difusa (Mamdani) enfocado a un parqueadero universitario.
    Entrada: Velocidad capturada.
    Salida: Horas de sanción de ingreso.
    """
    def __init__(self):
        # ─────────────────────────────────────────────────────────────
        # 1. UNIVERSOS DE DISCURSO
        # ─────────────────────────────────────────────────────────────
        self.velocidad_universe = np.arange(0, 50.1, 0.1) # [0 – 50 km/h]
        self.sancion_universe   = np.arange(0, 48.1, 0.1) # [0 – 48 horas]

        # ─────────────────────────────────────────────────────────────
        # 2. VARIABLES LINGÜÍSTICAS
        # ─────────────────────────────────────────────────────────────
        self.velocidad = ctrl.Antecedent(self.velocidad_universe, 'velocidad')
        self.sancion   = ctrl.Consequent(self.sancion_universe, 'sancion')

        # ─────────────────────────────────────────────────────────────
        # 3. FUNCIONES DE MEMBRESÍA
        # ─────────────────────────────────────────────────────────────
        # -- Velocidad (entrada) --
        self.velocidad['permitida'] = fuzz.trapmf(self.velocidad_universe, [0, 0, 10, 12])
        self.velocidad['leve']      = fuzz.trimf(self.velocidad_universe,  [10, 13, 16])
        self.velocidad['media']     = fuzz.trimf(self.velocidad_universe,  [14, 18, 22])
        self.velocidad['grave']     = fuzz.trapmf(self.velocidad_universe, [20, 25, 30, 38])
        self.velocidad['critica']   = fuzz.sigmf(self.velocidad_universe,  36, 0.5) # sigmoide

        # -- Sanción (salida) --
        self.sancion['ninguna']  = fuzz.trimf(self.sancion_universe,  [0, 0, 1])
        self.sancion['corta']    = fuzz.trimf(self.sancion_universe,  [0, 1, 2])
        self.sancion['moderada'] = fuzz.trimf(self.sancion_universe,  [1, 2, 4])
        self.sancion['larga']    = fuzz.trapmf(self.sancion_universe, [3, 12, 24, 30])
        self.sancion['maxima']   = fuzz.sigmf(self.sancion_universe,  32, 0.4) # sigmoide para 48h

        # ─────────────────────────────────────────────────────────────
        # 4. BASE DE REGLAS
        # ─────────────────────────────────────────────────────────────
        r01 = ctrl.Rule(self.velocidad['permitida'], self.sancion['ninguna'])
        r02 = ctrl.Rule(self.velocidad['leve'],      self.sancion['corta'])
        r03 = ctrl.Rule(self.velocidad['media'],     self.sancion['moderada'])
        r04 = ctrl.Rule(self.velocidad['grave'],     self.sancion['larga'])
        r05 = ctrl.Rule(self.velocidad['critica'],   self.sancion['maxima'])

        # ─────────────────────────────────────────────────────────────
        # 5. SISTEMA DE CONTROL
        # ─────────────────────────────────────────────────────────────
        self.sistema_ctrl = ctrl.ControlSystem(rules=[r01, r02, r03, r04, r05])
        self.simulador = ctrl.ControlSystemSimulation(self.sistema_ctrl)

    def calculate_fine(self, velocidad_capturada, limite_via_val=None):
        """
        Mantenemos la misma firma de la función para no quebrar el resto del proyecto, 
        pero el límite de vía es ignorado porque es constante (10km/h) en el parqueadero.
        """
        # Limitar al universo
        vel_clipped = max(0, min(50, velocidad_capturada))
        
        # 6. INFERENCIA
        self.simulador.input['velocidad'] = vel_clipped
        self.simulador.compute()
        
        res_sancion = self.simulador.output['sancion']
        
        # Generar grafico de debug en la carpeta 'imagenes'
        self._visualize(vel_clipped, res_sancion)
        
        return float(res_sancion)

    def _visualize(self, val_velocidad, res_sancion):
        """
        7. VISUALIZACIÓN
        Crea el mismo estilo de gráfica solicitado y lo guarda.
        """
        ROJO    = '#e74c3c'
        NARANJA = '#f39c12'
        VERDE   = '#27ae60'
        NAVY    = 'navy'
        PURPURA = '#8e44ad'
        AZUL    = '#2980b9'

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        fig.suptitle('Sistema Difuso Mamdani | Parqueadero Universitario | Salida: Suspensión de Ingreso',
                     fontsize=12, fontweight='bold', y=1.05)

        # ── Velocidad (entrada) ────────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(self.velocidad_universe, fuzz.trapmf(self.velocidad_universe, [0, 0, 10, 12]), color=VERDE,   lw=2, label='Permitida')
        ax.plot(self.velocidad_universe, fuzz.trimf(self.velocidad_universe,  [10, 13, 16]),    color=AZUL,    lw=2, label='Leve')
        ax.plot(self.velocidad_universe, fuzz.trimf(self.velocidad_universe,  [14, 18, 22]),    color=NARANJA, lw=2, label='Media')
        ax.plot(self.velocidad_universe, fuzz.trapmf(self.velocidad_universe, [20, 25, 30, 38]), color=PURPURA, lw=2, label='Grave')
        ax.plot(self.velocidad_universe, fuzz.sigmf(self.velocidad_universe,  36, 0.5),         color=ROJO,    lw=2, label='Crítica (sigmoide)')
        
        ax.axvline(val_velocidad, color=NAVY, linestyle='--', lw=1.8, label=f'Entrada = {val_velocidad:.1f} km/h')
        ax.set_title('Entrada: Velocidad Capturada', fontweight='bold')
        ax.set_xlabel('Velocidad (km/h)'); ax.set_ylabel('Membresia  u(x)')
        ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── Sancion (salida) ──────────────────────────────────────────────────
        ax = axes[1]
        ax.plot(self.sancion_universe, fuzz.trimf(self.sancion_universe,  [0, 0, 1]),      color=VERDE,   lw=2, label='Ninguna')
        ax.plot(self.sancion_universe, fuzz.trimf(self.sancion_universe,  [0, 1, 2]),      color=AZUL,    lw=2, label='Corta')
        ax.plot(self.sancion_universe, fuzz.trimf(self.sancion_universe,  [1, 2, 4]),      color=NARANJA, lw=2, label='Moderada')
        ax.plot(self.sancion_universe, fuzz.trapmf(self.sancion_universe, [3, 12, 24, 30]),color=PURPURA, lw=2, label='Larga')
        ax.plot(self.sancion_universe, fuzz.sigmf(self.sancion_universe,  32, 0.4),        color=ROJO,    lw=2, label='Máxima (sigmoide)')
        
        ax.axvline(res_sancion, color=NAVY, linestyle='--', lw=2, label=f'Sanción = {res_sancion:.2f} h')
        ax.fill_betweenx([0, 1], res_sancion - 0.5, res_sancion + 0.5, alpha=0.25, color=NAVY)
        ax.set_title('Salida: Tiempo de Sanción', fontweight='bold')
        ax.set_xlabel('Horas de Suspensión'); ax.set_ylabel('Membresia  u(x)')
        ax.set_ylim(-0.05, 1.15); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.tight_layout()
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'imagenes'))
        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, 'sistema_difuso_resultado.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
