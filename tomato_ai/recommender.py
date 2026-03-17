from __future__ import annotations

from typing import Any, Dict, List

from .utils import load_yaml, normalize_stage, safe_float, slugify_label


def load_rules(path: str) -> Dict[str, Any]:
    return load_yaml(path)


def _coerce_snapshot(snapshot: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in snapshot.items():
        numeric = safe_float(value)
        if numeric == numeric:
            out[key] = numeric

    global_cfg = rules.get('global', {})
    divisor = float(global_cfg.get('tds_to_ec_divisor', 500.0))
    if 'ec_ms_cm' not in out and 'tds_ppm' in out and divisor > 0:
        out['ec_ms_cm'] = out['tds_ppm'] / divisor
    return out


def generate_recommendations(
    sensor_snapshot: Dict[str, Any] | None,
    growth_stage: str | None,
    predicted_label: str | None,
    rules: Dict[str, Any],
    max_items: int = 6,
) -> List[str]:
    sensor_snapshot = sensor_snapshot or {}
    snapshot = _coerce_snapshot(sensor_snapshot, rules)

    stage = normalize_stage(growth_stage, default=rules.get('global', {}).get('default_stage', 'vegetative'))
    stage_rules = rules.get('stages', {}).get(stage, {})
    recommendations: List[str] = []

    for metric, rule in stage_rules.items():
        if metric not in snapshot:
            continue
        value = snapshot[metric]
        low = rule.get('min')
        high = rule.get('max')

        if low is not None and value < float(low):
            msg = rule.get('low_advice', f'{metric} ниже рекомендуемого диапазона.')
            recommendations.append(f'{msg} (текущее значение: {value:.2f})')
        elif high is not None and value > float(high):
            msg = rule.get('high_advice', f'{metric} выше рекомендуемого диапазона.')
            recommendations.append(f'{msg} (текущее значение: {value:.2f})')

    disease_key = slugify_label(predicted_label or '')
    disease_templates = rules.get('disease_templates', {})
    if disease_key in disease_templates:
        recommendations.extend(disease_templates[disease_key])
    elif predicted_label and disease_key != 'healthy' and 'default_disease' in disease_templates:
        recommendations.extend(disease_templates['default_disease'])

    if not recommendations:
        recommendations.append(
            rules.get(
                'healthy_default_message',
                'Сенсорные параметры близки к допустимому диапазону. Продолжай мониторинг и сохрани текущий режим.',
            )
        )

    unique_recommendations: List[str] = []
    for item in recommendations:
        if item not in unique_recommendations:
            unique_recommendations.append(item)
    return unique_recommendations[:max_items]
