import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FineSystem:
    """
    Sistema de Inferencia Difusa (Mamdani) para calcular la severidad
    de la multa basada en el exceso de velocidad y el tipo de vía (límite).
    """
    def __init__(self):
        """
        Inicializa las variables lingüísticas y las reglas difusas.
        """
        # 1. Definir los universos de discurso (Antecedentes y Consecuente)
        # Exceso de velocidad (km/h por encima del límite): 0 a 100
        self.exceso_velocidad = ctrl.Antecedent(np.arange(0, 101, 1), 'exceso_velocidad')
        # Límite de la vía (km/h): 30 a 120
        self.limite_via = ctrl.Antecedent(np.arange(30, 121, 1), 'limite_via')
        # Severidad de la multa (Dólares): 0 a 500
        self.severidad_multa = ctrl.Consequent(np.arange(0, 501, 1), 'severidad_multa')

        # 2. Definir las funciones de pertenencia (membresía)
        
        # Funciones para exceso de velocidad
        self.exceso_velocidad['bajo'] = fuzz.trimf(self.exceso_velocidad.universe, [0, 0, 20])
        self.exceso_velocidad['medio'] = fuzz.trapmf(self.exceso_velocidad.universe, [15, 25, 40, 50])
        self.exceso_velocidad['alto'] = fuzz.trapmf(self.exceso_velocidad.universe, [40, 60, 100, 100])
        
        # Funciones para límite de la vía (Urbano vs Carretera)
        self.limite_via['urbano'] = fuzz.trapmf(self.limite_via.universe, [30, 30, 50, 70])
        self.limite_via['carretera'] = fuzz.trapmf(self.limite_via.universe, [60, 80, 120, 120])
        
        # Funciones para la severidad de la multa
        self.severidad_multa['leve'] = fuzz.trimf(self.severidad_multa.universe, [0, 50, 100])
        self.severidad_multa['moderada'] = fuzz.trimf(self.severidad_multa.universe, [80, 150, 250])
        self.severidad_multa['grave'] = fuzz.trapmf(self.severidad_multa.universe, [200, 300, 500, 500])

        # 3. Definir las Reglas Difusas
        # Ir rápido en una zona urbana (límite bajo) es más grave que en carretera.
        rule1 = ctrl.Rule(self.exceso_velocidad['bajo'] & self.limite_via['carretera'], self.severidad_multa['leve'])
        rule2 = ctrl.Rule(self.exceso_velocidad['bajo'] & self.limite_via['urbano'], self.severidad_multa['moderada'])
        rule3 = ctrl.Rule(self.exceso_velocidad['medio'] & self.limite_via['carretera'], self.severidad_multa['moderada'])
        rule4 = ctrl.Rule(self.exceso_velocidad['medio'] & self.limite_via['urbano'], self.severidad_multa['grave'])
        rule5 = ctrl.Rule(self.exceso_velocidad['alto'], self.severidad_multa['grave'])

        # 4. Crear el Sistema de Control
        self.fine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.fine_sim = ctrl.ControlSystemSimulation(self.fine_ctrl)

    def calculate_fine(self, velocidad_capturada, limite_via_val):
        """
        Calcula el monto de la multa usando inferencia difusa.
        
        Args:
            velocidad_capturada (float): Velocidad a la que iba el vehículo.
            limite_via_val (float): Límite de velocidad de la vía.
            
        Returns:
            float: Monto sugerido para la multa (en dólares). 
                   Retorna 0 si no hay exceso de velocidad.
        """
        exceso = velocidad_capturada - limite_via_val
        
        if exceso <= 0:
            return 0.0  # No hay infracción
            
        # Limitar valores a los universos definidos
        exceso_clipped = max(0, min(100, exceso))
        limite_clipped = max(30, min(120, limite_via_val))
        
        self.fine_sim.input['exceso_velocidad'] = exceso_clipped
        self.fine_sim.input['limite_via'] = limite_clipped
        
        # Computar el sistema
        self.fine_sim.compute()
        
        # Retornar el valor defuzificado
        return float(self.fine_sim.output['severidad_multa'])
