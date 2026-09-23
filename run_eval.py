import json
from tests.eval.response_quality import evaluate

with open('artifacts/traces/traces_20260923_113222.json') as f:
    traces = json.load(f)

case = traces['eval_cases'][0]
instance = {
    'prompt': case.get('prompt', {}).get('parts', [{}])[0].get('text', ''),
    'response': case.get('responses', [{}])[0].get('response', {}).get('parts', [{}])[0].get('text', ''),
    'agent_data': json.dumps(case.get('agent_data', {})),
}

print('Calling evaluate()...')
result = evaluate(instance)
print('Evaluation result:', result)
