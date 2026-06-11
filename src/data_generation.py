import numpy as np
import pandas as pd
from faker import Faker

def generar_dataset(n=350, seed=42):
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)

    edad = np.array([fake.random_int(min=22, max=60) for _ in range(n)])
    anios_empresa = np.clip(np.array([np.random.randint(0, min(21, e-21)) for e in edad]), 0, 20)

    salario_base = 800 + 150 * anios_empresa + np.random.normal(0, 200, n)
    salario_mensual = np.round(np.clip(salario_base, 500, 5000), -1)

    horas_extra = np.clip(np.random.poisson(5, n), 0, 20)
    num_proyectos = np.random.randint(1, 11, n)
    distancia = np.clip(np.random.exponential(10, n), 1, 80).astype(int)
    capacitaciones = np.clip(np.random.poisson(2, n), 0, 5)
    evaluacion = np.round(np.random.beta(2, 2, n) * 0.9 + 0.1, 2)

    prob_asc = np.clip(0.2 + 0.04 * anios_empresa, 0.1, 0.6)
    ascenso = np.random.binomial(1, prob_asc, n)

    satis_base = (3 + (evaluacion - 0.5)*2 - horas_extra/10 + ascenso*1.5 - distancia/40 + np.random.normal(0,0.5,n))
    satisfaccion = np.clip(np.round(satis_base), 1, 5).astype(int)

    # Nuevo logit que incluye salario y años en la empresa de forma coherente
    logit = (1.0 
             - 0.01 * edad 
             - 0.05 * anios_empresa 
             - 0.0008 * salario_mensual 
             + 0.25 * horas_extra 
             - 0.8 * satisfaccion 
             + 0.2 * num_proyectos 
             + 0.02 * distancia 
             - 0.8 * evaluacion 
             - 0.15 * capacitaciones 
             - 1.2 * ascenso
             + np.random.normal(0, 0.5, n))
    prob_ren = 1/(1+np.exp(-logit))
    renuncia = np.random.binomial(1, prob_ren, n)

    df = pd.DataFrame({
        'edad': edad,
        'años_en_empresa': anios_empresa,
        'salario_mensual': salario_mensual,
        'horas_extra_semana': horas_extra,
        'satisfaccion_laboral': satisfaccion,
        'num_proyectos_año': num_proyectos,
        'distancia_casa_trabajo_km': distancia,
        'ultima_evaluacion_desempeño': evaluacion,
        'capacitaciones_recibidas': capacitaciones,
        'tiene_ascenso_ultimos_2_años': ascenso,
        'renuncia': renuncia
    })
    return df
