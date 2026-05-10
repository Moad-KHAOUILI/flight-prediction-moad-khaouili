import pathlib
src = pathlib.Path('webapp/app.py').read_text(encoding='utf-8')

scorer = '''
_nlp_scorer = None
CANDIDATE_LABELS = [
    'likely to cause a flight delay',
    'unlikely to cause a flight delay',
]

def get_scorer():
    global _nlp_scorer
    if _nlp_scorer is None:
        from transformers import pipeline
        _nlp_scorer = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    return _nlp_scorer

def score_narrative(text):
    text = str(text).strip()
    if not text:
        return 0.0
    result = get_scorer()(text[:512], candidate_labels=CANDIDATE_LABELS)
    return round(float(result['scores'][0]), 4)

'''

marker = 'app = Flask(__name__)'
assert marker in src, 'marker not found'
assert 'score_narrative' not in src, 'already patched'
src = src.replace(marker, scorer + marker)

# Patch predict_delay
old1 = '    # Build the one-hot vector from form inputs\n    raw = {}\n    # Numeric fields\n    for col in ['
new1 = '    asrs_text = str(data.get(' + chr(39) + 'asrs_text' + chr(39) + ', ' + chr(39) + chr(39) + ')).strip()\n    asrs_risk_score = score_narrative(asrs_text)\n    asrs_scored = bool(asrs_text)\n\n    raw = {}\n    for col in ['
src = src.replace(old1, new1)

# Remove asrs_risk_score from numeric loop
src = src.replace('        ' + chr(34) + 'asrs_risk_score' + chr(34) + ',\n    ]:\n', '    ]:\n')

# Inject asrs_risk_score into raw before one-hot section
ohe_marker = '    # One-hot: CARRIER_NAME'
src = src.replace(ohe_marker, '    raw[' + chr(34) + 'asrs_risk_score' + chr(34) + '] = asrs_risk_score\n\n' + ohe_marker, 1)

# Fix return
old_r = 'return jsonify({' + chr(34) + 'probability' + chr(34) + ': prob, ' + chr(34) + 'delayed' + chr(34) + ': prob >= 0.68})'
new_r = 'return jsonify({' + chr(34) + 'probability' + chr(34) + ': prob, ' + chr(34) + 'delayed' + chr(34) + ': prob >= 0.68, ' + chr(34) + 'asrs_risk_score' + chr(34) + ': asrs_risk_score, ' + chr(34) + 'asrs_scored' + chr(34) + ': asrs_scored})'
src = src.replace(old_r, new_r)

pathlib.Path('webapp/app.py').write_text(src, encoding='utf-8')
print('Done. score_narrative:', 'score_narrative' in src)
