from sklearn.manifold import TSNE
import sqlalchemy as sql
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from dotenv import load_dotenv
load_dotenv()

engine = sql.create_engine(
    os.getenv("PG_CONN_STR")
)

query = "SELECT cmetadata, document, embedding FROM langchain_pg_embedding  WHERE collection_id = '3555638c-bca3-4d77-86a9-9e3ce7d4f2f0'"

df = pd.read_sql(query, engine)

df["embeddings"] = df["embedding"].str.replace("[", "").str.replace("]", "").str.split(",")

raw_embeddings = []
for emb in df["embeddings"].values:
    raw_embeddings.append(emb)

np_empbeddings = np.array(raw_embeddings)

tsne = TSNE(n_components=3, random_state=42, perplexity=5)
reduced_vectors = tsne.fit_transform(np_empbeddings)

scatter_plot = go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color='grey', opacity=0.5, line=dict(color='lightgray', width=1)),
    text=[f"{i}" for i in df["document"].values]
)

# Highlight the first point with a different color
highlighted_point = go.Scatter3d(
    x=[reduced_vectors[0, 0]],
    y=[reduced_vectors[0, 1]],
    z=[reduced_vectors[0, 2]],
    mode='markers',
    marker=dict(size=8, color='red', opacity=0.8, line=dict(color='lightgray', width=1)),
    text=["Question"]
    
)

blue_points = go.Scatter3d(
    x=reduced_vectors[1:4, 0],
    y=reduced_vectors[1:4, 1],
    z=reduced_vectors[1:4, 2],
    mode='markers',
    marker=dict(size=8, color='blue', opacity=0.8,  line=dict(color='black', width=1)),
    text=["Top 1 Document","Top 2 Document","Top 3 Document"]
)

# Create the layout for the plot
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    ),
    title='3D Representation after t-SNE (Perplexity=5)'
)


fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Add the scatter plots to the Figure
fig.add_trace(scatter_plot)
fig.add_trace(highlighted_point)
fig.add_trace(blue_points)

fig.update_layout(layout)

pio.write_html(fig, 'interactive_plot.html')
fig.show()

