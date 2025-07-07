import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置保存路径
output_dir = r"C:\lunwen\plots_final"
os.makedirs(output_dir, exist_ok=True)

# 设置文件路径（请替换为你本地的实际路径）
file_paths = {
    "chengzhongcun": r"C:\lunwen\chengzhongcun_analysis_results.csv",
    "laojiu": r"C:\lunwen\laojiu_analysis_results.csv",
    "weilai": r"C:\lunwen\weilai_analysis_results.csv"
}

# 合并读取所有数据，并添加“source”来源字段
dfs = []
for name, path in file_paths.items():
    df = pd.read_csv(path)
    df["source"] = name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# 字段标准化
combined_df = combined_df.rename(columns={
    '情感倾向': 'sentiment',
    '对象标签': 'object_labels',
    '场景标签': 'scene_labels'
})
combined_df['sentiment'] = combined_df['sentiment'].map({'正面': 'Positive', '负面': 'Negative'})
df = combined_df[combined_df['sentiment'].isin(['Positive', 'Negative'])]

# 你提取出的 Top 20 标签（对象 + 场景）
top_object_labels = [
    'person', 'packaged goods', 'building', 'window', 'top', 'car', 'hat', 'shoe',
    'pants', 'luggage & bags', 'lighting', 'outerwear', 'clothing', 'door', 'tire',
    'mirror', 'wheel', 'table', 'furniture', 'sneakers'
]

top_scene_labels = [
    'building', 'property', 'room', 'vehicle', 'road', 'metropolitan area', 'town',
    'neighbourhood', 'urban area', 'wall', 'transport', 'car', 'residential area',
    'facial expression', 'street', 'glasses', 'eyewear', 'smile', 'architecture',
    'motor vehicle'
]


# 通用数据处理函数
def prepare_data(df, label_col, top_labels):
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].str.split(', ')
    df = df.explode(label_col)
    df = df[df[label_col].isin(top_labels)]

    grouped = df.groupby([label_col, 'source', 'sentiment']).size().reset_index(name='count')
    total = grouped.groupby([label_col, 'source'])['count'].transform('sum')
    grouped['percentage'] = grouped['count'] / total
    return grouped


# 绘图函数（来源 × 情绪分组柱状图 + 柔和配色）
def plot_grouped_sentiment_bar(data, label_col, title, filename):
    data['group'] = data['source'] + " - " + data['sentiment']

    plt.figure(figsize=(20, 8))
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set2")  # 柔和协调的配色

    sns.barplot(
        data=data,
        x=label_col,
        y='count',
        hue='group',
        ci=None,
        dodge=True
    )

    plt.title(title, fontsize=16)
    plt.xlabel(label_col.replace('_', ' ').title())
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Source - Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


# 处理数据并绘图
object_data = prepare_data(df, 'object_labels', top_object_labels)
scene_data = prepare_data(df, 'scene_labels', top_scene_labels)

plot_grouped_sentiment_bar(object_data, 'object_labels',
                           "Top 20 Object Labels - Sentiment by Source", "final_object_sentiment.png")

plot_grouped_sentiment_bar(scene_data, 'scene_labels',
                           "Top 20 Scene Labels - Sentiment by Source", "final_scene_sentiment.png")

# 保存数据为 Excel 文件
excel_path = os.path.join(output_dir, "final_label_sentiment_stats.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    object_data.to_excel(writer, sheet_name="Object Labels", index=False)
    scene_data.to_excel(writer, sheet_name="Scene Labels", index=False)

print(f"Excel 导出成功，路径为：{excel_path}")
