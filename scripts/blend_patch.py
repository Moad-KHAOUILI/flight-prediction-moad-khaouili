import pathlib
p = pathlib.Path('webapp/app.py')
src = p.read_text(encoding='utf-8')
old = '    prob = _predict(model_delay, scaler_delay, FEATURE_COLS_DELAY, raw)\n    return jsonify({"probability": prob, "delayed": prob >= 0.68, "asrs_risk_score": asrs_risk_score, "asrs_scored": asrs_scored})'
new_lines = [
    '    prob = _predict(model_delay, scaler_delay, FEATURE_COLS_DELAY, raw)',
    '',
    '    if asrs_scored and asrs_risk_score > 0.55:',
    '        blend = (asrs_risk_score - 0.55) / 0.45',
    '        prob = round(prob + blend * (1.0 - prob) * 0.75, 4)',
    '        prob = min(prob, 0.99)',
    '',
    '    return jsonify({"probability": prob, "delayed": prob >= 0.68, "asrs_risk_score": asrs_risk_score, "asrs_scored": asrs_scored})',
]
new = '\n'.join(new_lines)
assert old in src, f'Pattern not found! Searched: {repr(old[:80])}'
src = src.replace(old, new, 1)
p.write_text(src, encoding='utf-8')
print('Patched OK')
