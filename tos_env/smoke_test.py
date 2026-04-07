import sys
sys.path.insert(0, '.')

from graders import grade_task1_binary_risk, grade_task2_category, grade_task3_full_audit
from data.corpus import CLAUSE_BY_ID, DOCUMENTS
from server.tos_environment import TosEnvironment
from models import TosAction

# Test graders
s1, f1 = grade_task1_binary_risk('risky', True)
print(f"Task1 correct risky: score={s1} feedback={f1}")

s1b, _ = grade_task1_binary_risk('safe', False)
print(f"Task1 correct safe:  score={s1b}")

s1c, _ = grade_task1_binary_risk('risky', False)
print(f"Task1 wrong:         score={s1c}")

s2, bd2, fb2 = grade_task2_category('Privacy', 'data personal share collect', 'Privacy')
print(f"Task2 exact+reason:  score={s2:.2f} breakdown={bd2}")

s2b, _, _ = grade_task2_category('Liability', None, 'Privacy')
print(f"Task2 adjacent:      score={s2b:.2f}")

gt = DOCUMENTS[0]['ground_truth_risky_clauses']
perfect_findings = [
    {'clause_text': CLAUSE_BY_ID[c['clause_id']]['text'],
     'category': c['category'],
     'risk_score': c['risk_score']}
    for c in gt
]
s3, bd3, fb3 = grade_task3_full_audit(perfect_findings, gt, CLAUSE_BY_ID)
print(f"Task3 perfect:       score={s3:.2f} F1={bd3['f1']:.2f}")

s3b, _, _ = grade_task3_full_audit([], gt, CLAUSE_BY_ID)
print(f"Task3 empty:         score={s3b:.2f}")

# Test environment
env = TosEnvironment(task_name='binary_risk', seed=42)
obs = env.reset()
print(f"\nEnv task1 reset: task={obs.task_name} text_len={len(obs.document_text)}")

result = env.step(TosAction(verdict='risky'))
print(f"Env task1 step:  reward={result.reward} done={result.done}")
state = env.state()
print(f"Env state:       cumulative_reward={state.cumulative_reward} done={state.done}")

env2 = TosEnvironment(task_name='full_audit', seed=7)
obs2 = env2.reset()
print(f"\nEnv task3 reset: task={obs2.task_name} max_steps={obs2.max_steps}")
r2 = env2.step(TosAction(findings=[
    {'clause_text': 'We may share your personal information with advertisers.', 'category': 'Privacy', 'risk_score': 9}
]))
print(f"Env task3 step1: reward={r2.reward:.2f} done={r2.done}")

print("\n✅ All smoke tests passed!")
