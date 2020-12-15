import matplotlib.pyplot as plt

scores = []
with open('out.score', 'r') as f:
	for line in f:
		scores.append(float(line))

xs = list(range(len(scores)))
plt.plot(xs, scores, label='Scores')
plt.axvline(xs[int(len(xs)/2)], color='orange', label='Training Ended')
plt.xlabel('Episode #')
plt.ylabel('Agent Score')
plt.legend()
plt.show()
