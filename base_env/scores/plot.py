import matplotlib.pyplot as plt

trainingScores = []
testingScores = []
reading = trainingScores
with open("ghost_out.score", "r") as f:
    for line in f:
        if line == "\n":
            reading = testingScores
            continue
        reading.append(float(line))

allScores = trainingScores + testingScores
xs = list(range(len(allScores)))
plt.plot(xs[: len(trainingScores)], trainingScores, label="Training Score")
plt.plot(
    xs[len(trainingScores) :], testingScores, label="Testing Score", color="green"
)
plt.axvline(xs[len(trainingScores)], color="orange", label="Training Ended")
plt.xlabel("Episode #")
plt.ylabel("Agent Score")
plt.title('Ghost Score by Episode')
plt.legend()
plt.show()
