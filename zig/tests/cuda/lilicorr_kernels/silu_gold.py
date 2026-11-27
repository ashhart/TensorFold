import math
for r in range(3):
    print(r, [round((t + 6*r - 8.5)/(1 + math.exp(-(t + 6*r - 8.5))), 6) for t in range(6)])
