import matplotlib.pyplot as plt

trainingScores = []
testingScores = []
reading = trainingScores
with open("ghost_out.steps1", "r") as f:
    for line in f:
        if line == "\n":
            reading = testingScores
            continue
        reading.append(float(line))

allScores = trainingScores + testingScores
xs = list(range(len(allScores)))
plt.plot(xs[: len(trainingScores)], trainingScores, label="Training Steps")
plt.plot(
    xs[len(trainingScores) :], testingScores, label="Testing Steps", color="green"
)
plt.axvline(xs[len(trainingScores)], color="orange", label="Training Ended")
plt.xlabel("Episode #")
plt.ylabel("# Steps")
plt.title('Ghost Steps by Episode')
plt.legend()
plt.show()
