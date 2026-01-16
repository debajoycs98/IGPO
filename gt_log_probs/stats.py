import json
import numpy as np
import matplotlib.pyplot as plt

entropy_down_per_step = []
for i in range(1, 11):
	with open(f"/ossfs/workspace/linyang/FactAgent/DeepResearcher/gt_log_probs/training/gt_log_probs_{i}.json") as f:
		data = json.load(f)
	entropys = data["gt_entropys_per_turn"]
	tot = 0.0
	cnt = 0.0
	for entropy in entropys:
		tot += (np.mean(entropy[0]) - np.mean(entropy[-1]))
		cnt += 1
	entropy_down_per_step.append(tot / cnt)

steps = np.arange(1, len(entropy_down_per_step) + 1)  # [1, 2, ..., 48]

plt.figure(figsize=(12, 6))
plt.plot(steps, entropy_down_per_step,
         marker='o',
         linestyle='-',
         color='#d62728',    # 醒目的红色
         linewidth=2,
         markersize=5,
         label='Avg Entropy Drop per Step')

plt.xlabel('Training Step', fontsize=13)
plt.ylabel('Average Entropy Drop (Initial - Final)', fontsize=13)
plt.title('Average Entropy Drop per Training Step (Step 1 to 48)', fontsize=15, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# ========== 保存图片 ==========
output_file = 'entropy_drop_per_step.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ 图片已保存为: {output_file}")