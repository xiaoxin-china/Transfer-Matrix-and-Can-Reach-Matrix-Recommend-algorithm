from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from datetime import datetime
import re
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import random

warnings.filterwarnings('ignore')

# 1. 数据读取
inter_df = pd.read_csv('/Volumes/Elements/深度学习/aicomp/复赛/复赛数据集/inter_reevaluation.csv')
user_df = pd.read_csv('user.csv')
book_df = pd.read_csv('item.csv')

print(f"交互数据形状: {inter_df.shape}")
print(f"用户数据形状: {user_df.shape}")
print(f"书籍数据形状: {book_df.shape}")


# 2. 数据处理
def parse_date(date_str):
    """统一处理日期格式"""
    if pd.isna(date_str) or date_str == '' or date_str == 'NaN' or str(date_str).lower() == 'nan':
        return np.nan

    date_str = str(date_str).strip(" ")

    formats = [
        '%Y/%m/%d%H:%M:%S',
        '%Y-%m-%d%H:%M:%S',
        '%y/%m/%d%H:%M:%S',
        '%y-%m-%d%H:%M:%S',
        '%Y/%m/%d',
        '%Y-%m-%d',
        '%y/%m/%d',
        '%y-%m-%d',
        '%Y%m%d',
        '%y%m%d',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return np.nan


# 处理日期格式
for col in tqdm(['借阅时间', '还书时间', '续借时间'], desc="处理日期格式"):
    if col in inter_df.columns:
        inter_df[col] = inter_df[col].apply(parse_date)

# 删除借阅时间或还书时间缺失的记录
inter_df = inter_df.dropna(subset=['借阅时间', '还书时间'])

# 处理续借次数缺失
if '续借次数' in inter_df.columns:
    inter_df['续借次数'] = inter_df['续借次数'].fillna(0)
    inter_df.loc[inter_df['续借时间'].notna() & (inter_df['续借次数'] == 0), '续借次数'] = 1


# 构建用户下标和书籍下标索引
def make_idxs(inter_df):
    user_idxs = {}
    user_vidx = {}
    for idx, user_id in tqdm(enumerate(inter_df['user_id'].unique()),
                             total=inter_df['user_id'].nunique(),
                             desc="构建用户下标索引"):
        user_idxs[user_id] = idx
        user_vidx[idx] = user_id

    book_idxs = {}
    book_vidx = {}
    for idx, book_id in tqdm(enumerate(inter_df['book_id'].unique()),
                             total=inter_df['book_id'].nunique(),
                             desc="构建书籍下标索引"):
        book_idxs[book_id] = idx
        book_vidx[idx] = book_id

    return user_idxs, book_idxs, user_vidx, book_vidx


user_idxs, book_idxs, user_vidx, book_vidx = make_idxs(inter_df)


# 3. 构建转移矩阵
def make_transition_matrix(inter_df, user_idxs, book_idxs, user_vidx, book_vidx):
    # 按用户和借阅时间排序
    inter_df_sorted = inter_df.sort_values(by=['user_id', '借阅时间'])

    # 记录每个用户的借书序列
    user_sequences = defaultdict(list)

    for _, row in tqdm(inter_df_sorted.iterrows(), total=len(inter_df_sorted), desc="记录用户借书序列"):
        user_id = row['user_id']
        book_id = row['book_id']
        user_sequences[user_id].append(book_id)

    # 构建转移矩阵
    n_books = len(book_idxs)
    transition_matrix = np.zeros((n_books, n_books))

    for user_id, book_ids in tqdm(user_sequences.items(), total=len(user_sequences), desc="构建转移矩阵"):
        for i in range(1, len(book_ids)):
            prev_book = book_ids[i - 1]
            current_book = book_ids[i]

            if prev_book in book_idxs and current_book in book_idxs:
                prev_idx = book_idxs[prev_book]
                current_idx = book_idxs[current_book]
                transition_matrix[prev_idx][current_idx] += 1

    # 归一化转移矩阵
    for i in tqdm(range(n_books), desc="归一化转移矩阵"):
        row_sum = transition_matrix[i].sum()
        if row_sum > 0:
            transition_matrix[i] /= row_sum

    return transition_matrix, user_sequences


transition_matrix, user_sequences = make_transition_matrix(inter_df, user_idxs, book_idxs, user_vidx, book_vidx)


# 4. 推荐方法 - 允许推荐已借阅的书籍
class BookRecommender:
    def __init__(self, transition_matrix, book_idxs, book_vidx, user_sequences):
        self.transition_matrix = transition_matrix
        self.book_idxs = book_idxs
        self.book_vidx = book_vidx
        self.user_sequences = user_sequences

        # 计算书籍流行度作为备选推荐
        self.book_popularity = self._calculate_popularity()

    def _calculate_popularity(self):
        """计算书籍流行度"""
        popularity = defaultdict(int)
        for book_ids in self.user_sequences.values():
            for book_id in book_ids:
                popularity[book_id] += 1
        return popularity

    def recommend_for_user(self, user_id, top_k=1):
        """为单个用户推荐书籍 - 允许推荐已借阅的书籍"""
        if user_id not in self.user_sequences:
            return self._get_popular_books(top_k)

        user_books = self.user_sequences[user_id]
        if not user_books:
            return self._get_popular_books(top_k)

        # 获取用户最近借阅的书籍
        last_book = user_books[-1]

        if last_book not in self.book_idxs:
            return self._get_popular_books(top_k)

        last_book_idx = self.book_idxs[last_book]

        # 获取转移概率
        transition_probs = self.transition_matrix[last_book_idx]

        # 创建(书籍索引, 概率)的列表 - 不移除已借阅的书籍
        book_probs = []
        for book_idx, prob in enumerate(transition_probs):
            if prob > 0:
                book_id = self.book_vidx[book_idx]
                book_probs.append((book_id, prob))

        # 按概率排序
        book_probs.sort(key=lambda x: x[1], reverse=True)

        if book_probs:
            # 返回top_k推荐
            return [book_id for book_id, _ in book_probs[:top_k]]
        else:
            # 如果没有转移推荐，返回热门书籍
            return self._get_popular_books(top_k)

    def recommend_multiple_strategies(self, user_id, top_k=1):
        """使用多种策略为用户推荐书籍"""
        if user_id not in self.user_sequences:
            return self._get_popular_books(top_k)

        user_books = self.user_sequences[user_id]
        if not user_books:
            return self._get_popular_books(top_k)

        # 策略1: 基于最近阅读的转移推荐
        last_book = user_books[-1]
        if last_book in self.book_idxs:
            last_book_idx = self.book_idxs[last_book]
            transition_probs = self.transition_matrix[last_book_idx]

            # 创建转移概率列表
            book_probs = []
            for book_idx, prob in enumerate(transition_probs):
                if prob > 0:
                    book_id = self.book_vidx[book_idx]
                    book_probs.append((book_id, prob))

            # 按概率排序
            book_probs.sort(key=lambda x: x[1], reverse=True)

            if book_probs:
                return [book_id for book_id, _ in book_probs[:top_k]]

        # 策略2: 基于用户历史的所有转移
        all_transitions = defaultdict(float)
        for i in range(len(user_books)):
            book_id = user_books[i]
            if book_id in self.book_idxs:
                book_idx = self.book_idxs[book_id]
                for target_idx, prob in enumerate(self.transition_matrix[book_idx]):
                    if prob > 0:
                        target_book = self.book_vidx[target_idx]
                        all_transitions[target_book] += prob

        if all_transitions:
            sorted_books = sorted(all_transitions.items(), key=lambda x: x[1], reverse=True)
            return [book_id for book_id, _ in sorted_books[:top_k]]

        # 策略3: 返回热门书籍
        return self._get_popular_books(top_k)

    def _get_popular_books(self, top_k=1):
        """获取热门书籍"""
        popular_books = sorted(self.book_popularity.items(), key=lambda x: x[1], reverse=True)

        if popular_books:
            return [book_id for book_id, _ in popular_books[:top_k]]
        else:
            return []


# 5. 创建推荐器
recommender = BookRecommender(transition_matrix, book_idxs, book_vidx, user_sequences)


# 6. 为所有有借阅记录的用户生成推荐
def generate_recommendations(recommender, user_sequences, output_file='submission.csv',
                             use_multiple_strategies=True):
    """为所有用户生成推荐"""
    recommendations = []

    for user_id in tqdm(user_sequences.keys(), desc="生成用户推荐"):
        if use_multiple_strategies:
            recommended_books = recommender.recommend_multiple_strategies(user_id, top_k=1)
        else:
            recommended_books = recommender.recommend_for_user(user_id, top_k=1)

        if recommended_books:
            recommendations.append({
                'user_id': user_id,
                'book_id': recommended_books[0]
            })

    # 创建推荐结果DataFrame
    rec_df = pd.DataFrame(recommendations)

    # 保存结果
    rec_df.to_csv(output_file, index=False)
    print(f"推荐结果已保存到: {output_file}")
    print(f"共为 {len(rec_df)} 个用户生成推荐")

    return rec_df


# 7. 生成推荐 - 使用多种策略
recommendations_df = generate_recommendations(recommender, user_sequences, use_multiple_strategies=True)


# 8. 分析推荐结果
def analyze_recommendations(recommendations_df, user_sequences, book_df):
    """分析推荐结果"""
    print("\n=== 推荐结果分析 ===")

    # 统计推荐书籍的重复情况
    book_recommend_count = recommendations_df['book_id'].value_counts()
    print(f"最常被推荐的10本书:")
    for book_id, count in book_recommend_count.head(10).items():
        book_info = book_df[book_df['book_id'] == book_id]
        if not book_info.empty and '书名' in book_info.columns:
            book_name = book_info.iloc[0]['书名']
            print(f"  {book_id}: {book_name} - 被推荐 {count} 次")
        else:
            print(f"  {book_id}: 被推荐 {count} 次")

    # 统计推荐给用户的书籍是否是他们之前借过的
    same_book_count = 0
    for _, row in recommendations_df.iterrows():
        user_id = row['user_id']
        recommended_book = row['book_id']

        if user_id in user_sequences and recommended_book in user_sequences[user_id]:
            same_book_count += 1

    print(
        f"\n推荐用户已借阅过的书籍比例: {same_book_count / len(recommendations_df):.2%} ({same_book_count}/{len(recommendations_df)})")

    # 计算推荐书籍的覆盖率
    unique_recommended_books = recommendations_df['book_id'].nunique()
    total_books = len(book_idxs)
    print(f"推荐书籍覆盖率: {unique_recommended_books}/{total_books} ({unique_recommended_books / total_books:.2%})")


# 执行分析
analyze_recommendations(recommendations_df, user_sequences, book_df)


# 9. 显示推荐结果示例
def show_recommendation_examples(recommender, user_sequences, book_df, n_examples=10):
    """显示推荐结果示例"""
    print(f"\n=== 推荐结果示例 (前{n_examples}个用户) ===")
    print("=" * 100)

    user_ids = list(user_sequences.keys())[:n_examples]

    for user_id in user_ids:
        # 使用多种策略推荐
        recommended_books = recommender.recommend_multiple_strategies(user_id, top_k=1)
        user_history = user_sequences[user_id][-5:]  # 显示最近5本借阅记录

        print(f"用户 {user_id}:")
        print(f"  借阅历史: {user_history}")

        if recommended_books:
            recommended_book = recommended_books[0]
            # 标记是否用户之前借过这本书
            borrowed_before = "✓" if recommended_book in user_history else "✗"

            # 尝试获取书籍信息
            book_info = book_df[book_df['book_id'] == recommended_book]
            if not book_info.empty and '书名' in book_info.columns:
                book_title = book_info.iloc[0]['书名']
                print(f"  推荐书籍: {recommended_book} - {book_title} [{borrowed_before}已借阅]")
            else:
                print(f"  推荐书籍: {recommended_book} [{borrowed_before}已借阅]")
        else:
            print("  无法生成推荐")
        print("-" * 80)


# 显示示例
if 'book_id' in book_df.columns:
    show_recommendation_examples(recommender, user_sequences, book_df)

print("\n=== 推荐系统完成! ===")
