import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D

# 读取文件路径
file_paths = {
    "chengzhongcun": "C:/lunwen/chengzhongcun_analysis_results.csv",
    "laojiu": "C:/lunwen/laojiu_analysis_results.csv",
    "weilai": "C:/lunwen/weilai_analysis_results.csv"
}
output_path = "C:/lunwen/plots_final/all_scene_object_network.png"

# 合并数据
dfs = []
for name, path in file_paths.items():
    df = pd.read_csv(path)
    df["source"] = name
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# 数据预处理
df_all = df_all.rename(columns={'对象标签': 'object_labels', '场景标签': 'scene_labels'})
df_all.dropna(subset=['object_labels', 'scene_labels'], inplace=True)
df_all['object_labels'] = df_all['object_labels'].str.split(', ')
df_all['scene_labels'] = df_all['scene_labels'].str.split(', ')

# 共现统计
co_occurrence = {}
for _, row in df_all.iterrows():
    for s in set(row['scene_labels']):
        for o in set(row['object_labels']):
            key = (s.strip(), o.strip())
            if key[0] and key[1]:
                co_occurrence[key] = co_occurrence.get(key, 0) + 1

# 构建原始图
G = nx.Graph()
for (scene, obj), weight in co_occurrence.items():
    if weight >= 5:
        G.add_node(scene, type='scene')
        G.add_node(obj, type='object')
        G.add_edge(scene, obj, weight=weight)

# ▶️ 筛选高频节点
degree_threshold = 13  # 你可以改成更大或更小，比如 8 或 5
high_degree_nodes = [n for n in G.nodes() if G.degree(n) >= degree_threshold]
H = G.subgraph(high_degree_nodes).copy()

# 位置
pos = nx.kamada_kawai_layout(H)

# 节点样式
node_types = nx.get_node_attributes(H, 'type')
node_colors = ['#ffbe7a' if node_types[n] == 'scene' else '#fa7f6f' for n in H.nodes()]
node_sizes = [H.degree(n) * 70 for n in H.nodes()]

# 边样式
edge_weights = [H[u][v]['weight'] for u, v in H.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [1 + (w / max_weight) * 3.5 for w in edge_weights]
edge_alphas = [0.3 + (w / max_weight) * 0.6 for w in edge_weights]

# 绘图
plt.figure(figsize=(20, 15))

# 画边
for i, (u, v) in enumerate(H.edges()):
    nx.draw_networkx_edges(
        H, pos,
        edgelist=[(u, v)],
        width=edge_widths[i],
        alpha=edge_alphas[i],
        edge_color='black'
    )

# 画节点
nx.draw_networkx_nodes(
    H, pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="#ffffff",
    linewidths=0.7
)

for node in H.nodes():
    x, y = pos[node]
    txt = plt.text(
        x, y,
        node,
        fontsize=10,
        ha='center',
        va='center',
        zorder=10,
        color='black',
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")]
    )

# 图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Scene Label',
           markerfacecolor='#ffbe7a', markersize=10, markeredgecolor='gray'),
    Line2D([0], [0], marker='o', color='w', label='Object Label',
           markerfacecolor='#fa7f6f', markersize=10, markeredgecolor='gray')
]
plt.legend(handles=legend_elements, title="Node Type", loc='lower left')

# 画布设置
plt.title("High-Degree Bipartite Network: Scene × Object", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()
