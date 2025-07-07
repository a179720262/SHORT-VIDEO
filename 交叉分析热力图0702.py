import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 设置图片保存路径
output_dir = r"C:\lunwen\plots"
os.makedirs(output_dir, exist_ok=True)

# 读取CSV
df = pd.read_csv(r"C:\lunwen\laojiu_analysis_results.csv")

# 字段重命名
df = df.rename(columns={
    '文件名': 'filename',
    '中文转录': 'transcript',
    '情感得分': 'sentiment_score',
    '情感倾向': 'sentiment_label',
    '对象标签': 'object_labels',
    '场景标签': 'scene_labels'
})

# 情感映射
sentiment_map = {'正面': 'Positive', '负面': 'Negative'}
df['sentiment_label'] = df['sentiment_label'].map(sentiment_map)

# 过滤数据
df = df[df['sentiment_label'].isin(['Positive', 'Negative'])]
df = df.dropna(subset=['object_labels', 'scene_labels', 'sentiment_label'])


# 函数：交叉表统计（限制前20高频标签）
def explode_and_group(df, label_column):
    df_exploded = df.copy()
    df_exploded[label_column] = df_exploded[label_column].str.split(', ')
    df_exploded = df_exploded.explode(label_column)
    df_exploded = df_exploded[[label_column, 'sentiment_label']]

    count_table = pd.crosstab(df_exploded[label_column], df_exploded['sentiment_label'])
    count_table = count_table.sort_values(by='Positive', ascending=False).head(20)
    return count_table


# ---------- Object Labels Heatmap ----------
object_table = explode_and_group(df, 'object_labels')
plt.figure(figsize=(12, 8))
sns.heatmap(object_table.fillna(0), annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Top 20 Object Labels vs Sentiment", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Object Label", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "object_sentiment_laojiu.png"), dpi=300)
plt.close()

# ---------- Scene Labels Heatmap ----------
scene_table = explode_and_group(df, 'scene_labels')
plt.figure(figsize=(12, 8))
sns.heatmap(scene_table.fillna(0), annot=True, fmt=".0f", cmap="YlOrBr")
plt.title("Top 20 Scene Labels vs Sentiment", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Scene Label", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scene_sentiment_laojiu.png"), dpi=300)
plt.close()
