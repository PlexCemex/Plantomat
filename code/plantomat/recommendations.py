from __future__ import annotations

from typing import Any, Dict, List, Tuple


RANGES = {
    'generic': {
        'air_temp_c': (20.0, 27.0),
        'air_humidity_pct': (55.0, 75.0),
        'soil_moisture_pct': (50.0, 70.0),
        'solution_ph': (5.5, 6.5),
        'ec_ms_cm': (1.5, 3.0),
        'tds_ppm': (900.0, 1600.0),
        'light_lux': (12000.0, 25000.0),
        'co2_ppm': (400.0, 1000.0),
        'leaf_wetness': (0.0, 0.35),
    },
    'flowering': {
        'air_temp_c': (20.0, 26.0),
        'air_humidity_pct': (55.0, 70.0),
        'soil_moisture_pct': (50.0, 68.0),
        'solution_ph': (5.5, 6.4),
        'ec_ms_cm': (2.0, 3.5),
        'tds_ppm': (1000.0, 1700.0),
        'light_lux': (15000.0, 26000.0),
        'co2_ppm': (450.0, 1000.0),
        'leaf_wetness': (0.0, 0.30),
    },
    'fruiting': {
        'air_temp_c': (20.0, 27.0),
        'air_humidity_pct': (50.0, 68.0),
        'soil_moisture_pct': (50.0, 65.0),
        'solution_ph': (5.5, 6.4),
        'ec_ms_cm': (2.2, 4.0),
        'tds_ppm': (1100.0, 1900.0),
        'light_lux': (15000.0, 28000.0),
        'co2_ppm': (450.0, 1100.0),
        'leaf_wetness': (0.0, 0.28),
    },
}

FIELD_NAMES_RU = {
    'air_temp_c': 'температура воздуха',
    'air_humidity_pct': 'влажность воздуха',
    'soil_moisture_pct': 'влажность субстрата',
    'solution_ph': 'pH раствора',
    'ec_ms_cm': 'EC раствора',
    'tds_ppm': 'TDS раствора',
    'light_lux': 'освещённость',
    'co2_ppm': 'CO2',
    'leaf_wetness': 'влажность листа',
}


def analyze_sensor_snapshot(snapshot: Dict[str, Any], stage: str | None = None) -> Tuple[str, List[str], List[str]]:
    stage_key = (stage or snapshot.get('growth_stage') or 'generic').lower()
    ranges = RANGES.get(stage_key, RANGES['generic'])
    issues: List[str] = []
    recommendations: List[str] = []
    for key, (low, high) in ranges.items():
        if key not in snapshot:
            continue
        try:
            value = float(snapshot[key])
        except Exception:
            continue
        ru_name = FIELD_NAMES_RU[key]
        if value < low:
            issues.append(f'{ru_name} ниже нормы ({value:.2f} < {low:.2f})')
            if key == 'air_temp_c':
                recommendations.append('Поднять температуру воздуха или уменьшить охлаждение/проветривание.')
            elif key == 'air_humidity_pct':
                recommendations.append('Повысить влажность воздуха: уменьшить вентиляцию и добавить мягкое увлажнение.')
            elif key == 'soil_moisture_pct':
                recommendations.append('Повысить влажность субстрата: скорректировать полив.')
            elif key == 'solution_ph':
                recommendations.append('Повысить pH раствора до рабочего диапазона.')
            elif key == 'ec_ms_cm':
                recommendations.append('Увеличить концентрацию питательного раствора.')
            elif key == 'tds_ppm':
                recommendations.append('Повысить минерализацию раствора (TDS).')
            elif key == 'light_lux':
                recommendations.append('Увеличить освещённость или длительность досветки.')
            elif key == 'co2_ppm':
                recommendations.append('Улучшить подачу CO2 или уменьшить избыточный воздухообмен.')
        elif value > high:
            issues.append(f'{ru_name} выше нормы ({value:.2f} > {high:.2f})')
            if key == 'air_temp_c':
                recommendations.append('Снизить температуру воздуха: усилить вентиляцию или охлаждение.')
            elif key == 'air_humidity_pct':
                recommendations.append('Снизить влажность воздуха: усилить вентиляцию и исключить переувлажнение.')
            elif key == 'soil_moisture_pct':
                recommendations.append('Снизить влажность субстрата: уменьшить полив.')
            elif key == 'solution_ph':
                recommendations.append('Снизить pH раствора до рабочего диапазона.')
            elif key == 'ec_ms_cm':
                recommendations.append('Снизить концентрацию питательного раствора.')
            elif key == 'tds_ppm':
                recommendations.append('Снизить минерализацию раствора (TDS).')
            elif key == 'light_lux':
                recommendations.append('Уменьшить освещённость или сократить время досветки.')
            elif key == 'co2_ppm':
                recommendations.append('Снизить концентрацию CO2 и проверить вентиляцию.')
            elif key == 'leaf_wetness':
                recommendations.append('Снизить влажность на листьях: улучшить вентиляцию и уменьшить конденсат.')
    if not issues:
        status = 'норма'
        recommendations.append('Параметры среды находятся в рабочем диапазоне.')
    else:
        status = 'риск'
    recommendations = list(dict.fromkeys(recommendations))
    return status, issues, recommendations
