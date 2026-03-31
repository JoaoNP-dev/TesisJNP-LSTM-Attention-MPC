import numpy as np
import pandas as pd

def calcular_potencia_fv(
    df: pd.DataFrame,
    col_irradiancia: str = "ghi_wm2",   
    area_m2: float = 24.0,
    eficiencia_inversor: float = 0.96
    eficiencia: float = 0.2231,
    correccion_inclinacion: float = 1.10,
    umbral_irrad_wm2: float = 15.0,
    crear_potencia_kw: bool = True
) -> pd.DataFrame:
    """
    Calcula generación FV horaria a partir de irradiancia (W/m²).

    Física:
      Potencia (W) = Irradiancia (W/m²) * Área (m²) * eficiencia * corrección
      Potencia (kW) = W / 1000
      Energía por hora (kWh) = Potencia(kW) * 1h

    Salidas:
      - energia_fv_kwh: energía generada en la hora (kWh)
      - (opcional) potencia_fv_kw: potencia promedio equivalente en la hora (kW)
    """

    df = df.copy()

    if col_irradiancia not in df.columns:
        raise KeyError(f"No se encontró la columna '{col_irradiancia}' en el DataFrame.")

    irr = pd.to_numeric(df[col_irradiancia], errors="coerce").fillna(0.0)

    # Umbral (noche/ruido)
    irr = np.where(irr >= umbral_irrad_wm2, irr, 0.0)

    # Potencia FV promedio equivalente en W
    potencia_w = irr * area_m2 * eficiencia * correccion_inclinacion

    # Convertir a kW y luego a kWh por hora (paso horario)
    potencia_kw = potencia_w / 1000.0
    energia_kwh = potencia_kw * 1.0  # 1h

    df["energia_fv_kwh"] = energia_kwh
    df["unidad_energia_fv"] = "kWh"

    if crear_potencia_kw:
        df["potencia_fv_kw"] = potencia_kw
        df["unidad_potencia_fv"] = "kW"

    return df
