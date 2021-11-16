import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

with open("losses.txt") as f:
    lines = f.readlines()
    x = []
    y = []

    for line in lines:
        if (int(line.split()[0]) % 10000 == 0):
            x.append(line.split()[0])
            y.append(line.split()[1])

df = pd.DataFrame(
        {"Frames": x,
         "Losses": y
        })

fig = px.line(df, x="Frames", y="Losses", title="Loss over Frames")
fig.show()

with open("all_rewards.txt") as f:
    lines = f.readlines()
    x = []
    y = []

    for line in lines:
        if (int(line.split()[0]) % 10000 == 0):
            x.append(line.split()[0])
            y.append(line.split()[1])

df = pd.DataFrame(
        {"Frames": x,
         "Rewards": y
        })

fig = px.line(df, x="Frames", y="Rewards", title="Reward over Frames")
fig.show()
