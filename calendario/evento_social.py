import pandas as pd


def obtener_eventos_sociales_2025() -> pd.DatetimeIndex:
    """
   Fechas especiales para Ayacucho durante el año 2025.

    """

    # Lista de feriados nacionales 
    feriados_nacionales = [
        "2025-01-01", "2025-03-20", "2025-03-21", "2025-05-01",
        "2025-06-07", "2025-06-29", "2025-07-23", "2025-07-28",
        "2025-07-29", "2025-08-06", "2025-08-30", "2025-10-08",
        "2025-11-01", "2025-12-08", "2025-12-09", "2025-12-25",
    ]

    feriados_nacionales = pd.to_datetime(feriados_nacionales)

    # Semana Santa en Ayacucho (del 14 al 23 de marzo)
    semana_santa = pd.date_range(start="2025-03-14", end="2025-03-23", freq="D")

    # Fechas familiares
    fechas_familiares = pd.to_datetime([
        "2025-05-11",  # Día de la Madre (2° domingo de mayo)
        "2025-06-15",  # Día del Padre (3° domingo de junio)
        "2025-08-17",  # Día del Niño (3° domingo de agosto)
    ])

    # Unir todas las fechas
    fechas_evento = feriados_nacionales.union(semana_santa).union(fechas_familiares)

    # Normalizar y borrar los duplicados
    fechas_evento = fechas_evento.normalize().unique()

    # Validación final
    if not all(fechas_evento.year == 2025):
        raise ValueError("Se detectaron fechas fuera del año 2025 en eventos sociales.")

    return pd.DatetimeIndex(fechas_evento)
