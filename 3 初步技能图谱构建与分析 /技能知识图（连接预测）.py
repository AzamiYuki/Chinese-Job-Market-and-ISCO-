import pandas as pd
import numpy as np
import json
import networkx as nx
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

class SkillJobKnowledgeGraph:
    def __init__(self, csv_file: str, random_seed: int = 42):
        """
        初始化技能-职业知识图谱构建器
        
        Args:
            csv_file: 包含技能提取结果的CSV文件路径
            random_seed: 随机种子，确保结果可复现
        """
        self.csv_file = csv_file
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 图结构
        self.full_graph = nx.Graph()  # 完整图
        self.training_graph = nx.Graph()  # 训练图
        
        # 节点信息
        self.skill_nodes = set()
        self.job_nodes = set()
        
        # 边信息
        self.all_positive_edges = []
        self.negative_candidate_pool = []
        
        # 数据集分割
        self.train_positive = []
        self.val_positive = []
        self.test_positive = []
        self.train_negative = []
        self.val_negative = []
        self.test_negative = []
        
        # 评分结果
        self.scores = {}
        
        print(f"知识图谱构建器初始化完成（随机种子: {random_seed}）")
    
    def load_data_and_build_graph(self) -> Dict:
        """
        从CSV文件加载数据并构建知识图谱
        
        Returns:
            Dict: 图的基本统计信息
        """
        print("正在加载数据并构建知识图谱...")
        
        # 读取CSV文件
        df = pd.read_csv(self.csv_file)
        print(f"读取数据: {len(df)} 条记录")
        
        # 统计边的权重（技能-职业共现次数）
        edge_weights = defaultdict(int)
        
        # 处理每一行数据
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"处理进度: {idx}/{len(df)}")
            
            # 获取ISCO职业代码
            isco_code = row.get('ISCO_4_Digit_Code_Gemini')
            if pd.isna(isco_code):
                continue
            
            job_node = f"JOB_{int(isco_code)}"
            self.job_nodes.add(job_node)
            
            # 获取标准化技能
            normalized_skills_str = row.get('标准化技能', '[]')
            try:
                normalized_skills = json.loads(normalized_skills_str)
            except:
                continue
            
            # 为每个技能创建边
            for skill in normalized_skills:
                if skill and len(skill.strip()) > 1:  # 过滤空技能
                    skill_node = f"SKILL_{skill.strip()}"
                    self.skill_nodes.add(skill_node)
                    
                    # 记录边的权重
                    edge = (skill_node, job_node)
                    edge_weights[edge] += 1
        
        # 构建完整图
        for (skill_node, job_node), weight in edge_weights.items():
            self.full_graph.add_edge(skill_node, job_node, weight=weight)
            self.all_positive_edges.append((skill_node, job_node))
        
        # 为节点添加类型属性
        for skill_node in self.skill_nodes:
            self.full_graph.add_node(skill_node, node_type='skill')
        for job_node in self.job_nodes:
            self.full_graph.add_node(job_node, node_type='job')
        
        # 计算统计信息
        stats = {
            'total_nodes': len(self.skill_nodes) + len(self.job_nodes),
            'skill_nodes': len(self.skill_nodes),
            'job_nodes': len(self.job_nodes),
            'total_edges': len(self.all_positive_edges),
            'avg_degree': np.mean([self.full_graph.degree(node) for node in self.full_graph.nodes()]),
            'max_degree': max([self.full_graph.degree(node) for node in self.full_graph.nodes()]),
            'density': nx.density(self.full_graph)
        }
        

        print(f"总节点数: {stats['total_nodes']} ({stats['skill_nodes']}个技能 + {stats['job_nodes']}个职业)")
        print(f"总边数: {stats['total_edges']}")
        print(f"平均度数: {stats['avg_degree']:.2f}")
        print(f"最大度数: {stats['max_degree']}")
        print(f"图密度: {stats['density']:.6f}")
        
        return stats
    
    def split_positive_edges(self, train_ratio: float = 0.55, val_ratio: float = 0.15, test_ratio: float = 0.30):
        """
        按照论文比例划分正样本边
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
        """
        print(f"\n正在划分正样本边...")
        
        total_edges = len(self.all_positive_edges)
        
        # 计算各集合大小
        train_size = int(total_edges * train_ratio)
        val_size = int(total_edges * val_ratio)
        test_size = total_edges - train_size - val_size  # 确保总数正确
        
        print(f"总正样本边: {total_edges}")
        print(f"训练集: {train_size} ({train_size/total_edges*100:.1f}%)")
        print(f"验证集: {val_size} ({val_size/total_edges*100:.1f}%)")
        print(f"测试集: {test_size} ({test_size/total_edges*100:.1f}%)")
        
        # 随机打乱边列表
        shuffled_edges = self.all_positive_edges.copy()
        random.shuffle(shuffled_edges)
        
        # 划分数据集
        self.train_positive = shuffled_edges[:train_size]
        self.val_positive = shuffled_edges[train_size:train_size + val_size]
        self.test_positive = shuffled_edges[train_size + val_size:]
        
        # 构建训练图（仅使用训练集的边）
        self.training_graph.clear()
        for skill_node in self.skill_nodes:
            self.training_graph.add_node(skill_node, node_type='skill')
        for job_node in self.job_nodes:
            self.training_graph.add_node(job_node, node_type='job')
        
        for edge in self.train_positive:
            weight = self.full_graph[edge[0]][edge[1]]['weight']
            self.training_graph.add_edge(edge[0], edge[1], weight=weight)
        
        print(f"训练图构建完成: {self.training_graph.number_of_nodes()} 节点, {self.training_graph.number_of_edges()} 边")
        
        return {
            'train_positive': len(self.train_positive),
            'val_positive': len(self.val_positive),
            'test_positive': len(self.test_positive),
            'training_graph_edges': self.training_graph.number_of_edges()
        }
    
    def generate_negative_samples(self):
        """
        生成负样本边
        """

        
        # 计算所有可能的技能-职业配对
        total_possible_pairs = len(self.skill_nodes) * len(self.job_nodes)
        existing_pairs = set(self.all_positive_edges)
        
        print(f"总可能配对数: {total_possible_pairs:,}")
        print(f"已存在边数: {len(existing_pairs):,}")
        print(f"负样本候选池大小: {total_possible_pairs - len(existing_pairs):,}")
        
        # 生成负样本候选池
        negative_candidates = []
        for skill_node in self.skill_nodes:
            for job_node in self.job_nodes:
                if (skill_node, job_node) not in existing_pairs:
                    negative_candidates.append((skill_node, job_node))
        
        # 打乱候选池
        random.shuffle(negative_candidates)
        
        # 为各数据集生成负样本
        train_neg_size = len(self.train_positive)
        val_neg_size = len(self.val_positive)
        test_neg_size = len(self.test_positive)
        
        self.train_negative = negative_candidates[:train_neg_size]
        self.val_negative = negative_candidates[train_neg_size:train_neg_size + val_neg_size]
        self.test_negative = negative_candidates[train_neg_size + val_neg_size:train_neg_size + val_neg_size + test_neg_size]
        
        print(f"✓ 负样本生成完成:")
        print(f"  训练集负样本: {len(self.train_negative)}")
        print(f"  验证集负样本: {len(self.val_negative)}")
        print(f"  测试集负样本: {len(self.test_negative)}")
        
        # 验证负样本确实不存在
        for neg_edge in self.train_negative + self.val_negative + self.test_negative:
            assert neg_edge not in existing_pairs, f"负样本 {neg_edge} 在正样本中存在！"
        
        print("✓ 负样本验证通过")
        
        return {
            'total_possible_pairs': total_possible_pairs,
            'negative_candidates': len(negative_candidates),
            'train_negative': len(self.train_negative),
            'val_negative': len(self.val_negative),
            'test_negative': len(self.test_negative)
        }
    
    def calculate_preferential_attachment_scores(self):
        """
        计算优先连接（Preferential Attachment）得分
        """
        print(f"\n正在计算优先连接得分...")
        
        # 计算所有节点在训练图中的度数
        node_degrees = {}
        for node in self.training_graph.nodes():
            node_degrees[node] = self.training_graph.degree(node)
        
        print(f"节点度数统计:")
        skill_degrees = [degree for node, degree in node_degrees.items() if node.startswith('SKILL_')]
        job_degrees = [degree for node, degree in node_degrees.items() if node.startswith('JOB_')]
        
        print(f"  技能节点度数: 平均{np.mean(skill_degrees):.2f}, 最大{max(skill_degrees) if skill_degrees else 0}")
        print(f"  职业节点度数: 平均{np.mean(job_degrees):.2f}, 最大{max(job_degrees) if job_degrees else 0}")
        
        # 计算所有样本的PA得分
        all_samples = []
        all_samples.extend([(edge, 1) for edge in self.train_positive])  # 正样本标记为1
        all_samples.extend([(edge, 0) for edge in self.train_negative])  # 负样本标记为0
        all_samples.extend([(edge, 1) for edge in self.val_positive])
        all_samples.extend([(edge, 0) for edge in self.val_negative])
        all_samples.extend([(edge, 1) for edge in self.test_positive])
        all_samples.extend([(edge, 0) for edge in self.test_negative])
        
        # 计算PA得分
        pa_scores = []
        for (u, v), label in all_samples:
            degree_u = node_degrees.get(u, 0)
            degree_v = node_degrees.get(v, 0)
            pa_score = degree_u * degree_v
            pa_scores.append(pa_score)
        
        # 归一化得分
        max_pa_score = max(pa_scores) if pa_scores else 1
        normalized_scores = [score / max_pa_score for score in pa_scores]
        
        # 保存得分
        self.scores = {
            'samples': all_samples,
            'pa_scores': pa_scores,
            'normalized_scores': normalized_scores,
            'max_pa_score': max_pa_score
        }
        
        print(f"✓ PA得分计算完成:")
        print(f"  总样本数: {len(all_samples):,}")
        print(f"  PA得分范围: 0 - {max_pa_score}")
        print(f"  归一化得分范围: 0.0 - 1.0")
        print(f"  平均PA得分: {np.mean(pa_scores):.2f}")
        print(f"  平均归一化得分: {np.mean(normalized_scores):.4f}")
        
        return {
            'total_samples': len(all_samples),
            'max_pa_score': max_pa_score,
            'mean_pa_score': np.mean(pa_scores),
            'mean_normalized_score': np.mean(normalized_scores)
        }
    
    def evaluate_link_prediction(self):
        """
        评估链接预测性能
        """
        print(f"\n正在评估链接预测性能...")
        
        # 准备数据
        samples = self.scores['samples']
        predictions = self.scores['normalized_scores']
        
        # 分离各数据集
        train_size = len(self.train_positive) + len(self.train_negative)
        val_size = len(self.val_positive) + len(self.val_negative)
        test_size = len(self.test_positive) + len(self.test_negative)
        
        # 训练集
        train_labels = [label for (edge, label) in samples[:train_size]]
        train_predictions = predictions[:train_size]
        
        # 验证集
        val_labels = [label for (edge, label) in samples[train_size:train_size + val_size]]
        val_predictions = predictions[train_size:train_size + val_size]
        
        # 测试集
        test_labels = [label for (edge, label) in samples[train_size + val_size:]]
        test_predictions = predictions[train_size + val_size:]
        
        # 计算各种评估指标
        def calculate_metrics(y_true, y_scores, threshold=0.5):
            y_pred = (np.array(y_scores) >= threshold).astype(int)
            
            return {
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'auc_roc': roc_auc_score(y_true, y_scores),
                'auc_pr': average_precision_score(y_true, y_scores)
            }
        
        # 计算各数据集的性能
        train_metrics = calculate_metrics(train_labels, train_predictions)
        val_metrics = calculate_metrics(val_labels, val_predictions)
        test_metrics = calculate_metrics(test_labels, test_predictions)
        
        # 打印结果
        print(f"\n链接预测性能评估结果:")
        print(f"{'指标':<15} {'训练集':<10} {'验证集':<10} {'测试集':<10}")
        print("-" * 50)
        for metric in ['precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']:
            print(f"{metric:<15} {train_metrics[metric]:<10.4f} {val_metrics[metric]:<10.4f} {test_metrics[metric]:<10.4f}")
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    
    def analyze_graph_properties(self):
        """
        分析图的拓扑性质
        """
        print(f"\n正在分析图的拓扑性质...")
        
        # 连通性分析
        connected_components = list(nx.connected_components(self.full_graph))
        largest_cc = max(connected_components, key=len)
        
        # 度分布分析
        degrees = [self.full_graph.degree(node) for node in self.full_graph.nodes()]
        skill_degrees = [self.full_graph.degree(node) for node in self.skill_nodes]
        job_degrees = [self.full_graph.degree(node) for node in self.job_nodes]
        
        # 中心性分析（在最大连通分量上计算）
        largest_cc_graph = self.full_graph.subgraph(largest_cc)
        betweenness = nx.betweenness_centrality(largest_cc_graph)
        closeness = nx.closeness_centrality(largest_cc_graph)
        
        properties = {
            'connected_components': len(connected_components),
            'largest_cc_size': len(largest_cc),
            'largest_cc_ratio': len(largest_cc) / len(self.full_graph.nodes()),
            'average_degree': np.mean(degrees),
            'degree_std': np.std(degrees),
            'skill_avg_degree': np.mean(skill_degrees),
            'job_avg_degree': np.mean(job_degrees),
            'max_betweenness': max(betweenness.values()),
            'max_closeness': max(closeness.values())
        }
        
        print(f"  连通分量数: {properties['connected_components']}")
        print(f"  最大连通分量大小: {properties['largest_cc_size']} ({properties['largest_cc_ratio']*100:.1f}%)")
        print(f"  平均度数: {properties['average_degree']:.2f} ± {properties['degree_std']:.2f}")
        print(f"  技能节点平均度数: {properties['skill_avg_degree']:.2f}")
        print(f"  职业节点平均度数: {properties['job_avg_degree']:.2f}")
        
        return properties
    
    def find_top_nodes(self, top_k: int = 10):
        """
        找出度数最高的节点
        """
        print(f"\n正在分析重要节点...")
        
        # 按度数排序
        node_degrees = [(node, self.full_graph.degree(node)) for node in self.full_graph.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        top_skills = []
        top_jobs = []
        
        for node, degree in node_degrees:
            if node.startswith('SKILL_') and len(top_skills) < top_k:
                skill_name = node.replace('SKILL_', '')
                top_skills.append((skill_name, degree))
            elif node.startswith('JOB_') and len(top_jobs) < top_k:
                job_code = node.replace('JOB_', '')
                top_jobs.append((job_code, degree))
        
        print(f"前{top_k}个最重要的技能（按连接的职业数）:")
        for i, (skill, degree) in enumerate(top_skills, 1):
            print(f"  {i:2d}. {skill}: 连接{degree}个职业")
        
        print(f"\n 前{top_k}个最重要的职业（按需要的技能数）:")
        for i, (job, degree) in enumerate(top_jobs, 1):
            print(f"  {i:2d}. ISCO {job}: 需要{degree}个技能")
        
        return {
            'top_skills': top_skills,
            'top_jobs': top_jobs
        }
    
    def save_results(self, output_prefix: str = 'kg_analysis'):
        """
        保存分析结果
        """
        print(f"\n正在保存结果...")
        
        # 保存图结构
        nx.write_gexf(self.full_graph, f"{output_prefix}_full_graph.gexf")
        nx.write_gexf(self.training_graph, f"{output_prefix}_training_graph.gexf")
        
        # 保存数据集分割
        datasets = {
            'train_positive': self.train_positive,
            'val_positive': self.val_positive,
            'test_positive': self.test_positive,
            'train_negative': self.train_negative,
            'val_negative': self.val_negative,
            'test_negative': self.test_negative
        }
        
        with open(f"{output_prefix}_datasets.json", 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            serializable_datasets = {k: [list(edge) for edge in v] for k, v in datasets.items()}
            json.dump(serializable_datasets, f, ensure_ascii=False, indent=2)
        
        # 保存评分结果
        scores_data = {
            'pa_scores': self.scores['pa_scores'],
            'normalized_scores': self.scores['normalized_scores'],
            'max_pa_score': self.scores['max_pa_score']
        }
        
        with open(f"{output_prefix}_scores.json", 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 结果已保存:")
        print(f"  - {output_prefix}_full_graph.gexf: 完整知识图谱")
        print(f"  - {output_prefix}_training_graph.gexf: 训练图")
        print(f"  - {output_prefix}_datasets.json: 数据集分割")
        print(f"  - {output_prefix}_scores.json: PA评分结果")

def main():
    """
    主函数：完整的知识图谱构建和链接预测流程
    """
    print("=" * 80)
    print("技能-职业知识图谱构建与链接预测系统")
    print("=" * 80)
    
    # 初始化系统
    kg_builder = SkillJobKnowledgeGraph(
        csv_file='simplified_jobs_skills.csv',  # 请确保文件存在
        random_seed=42
    )
    
    # 步骤1: 构建知识图谱
    print("\n" + "="*60)
    print("步骤1: 构建知识图谱")
    print("="*60)
    graph_stats = kg_builder.load_data_and_build_graph()
    
    # 步骤2: 划分正样本边
    print("\n" + "="*60)
    print("步骤2: 划分正样本边")
    print("="*60)
    split_stats = kg_builder.split_positive_edges()
    
    # 步骤3: 生成负样本边
    print("\n" + "="*60)
    print("步骤3: 生成负样本边")
    print("="*60)
    negative_stats = kg_builder.generate_negative_samples()
    
    # 步骤4: 计算优先连接得分
    print("\n" + "="*60)
    print("步骤4: 计算优先连接得分")
    print("="*60)
    scoring_stats = kg_builder.calculate_preferential_attachment_scores()
    
    # 步骤5: 评估链接预测性能
    print("\n" + "="*60)
    print("步骤5: 评估链接预测性能")
    print("="*60)
    eval_results = kg_builder.evaluate_link_prediction()
    
    # 步骤6: 分析图性质
    print("\n" + "="*60)
    print("步骤6: 分析图拓扑性质")
    print("="*60)
    graph_properties = kg_builder.analyze_graph_properties()
    
    # 步骤7: 找出重要节点
    print("\n" + "="*60)
    print("步骤7: 分析重要节点")
    print("="*60)
    top_nodes = kg_builder.find_top_nodes(top_k=15)
    
    # 步骤8: 保存结果
    print("\n" + "="*60)
    print("步骤8: 保存分析结果")
    print("="*60)
    kg_builder.save_results('skill_job_kg')
    
    # 生成最终总结报告
    print("\n" + "="*80)

    print("="*80)
    
    print(f"  知识图谱规模: {graph_stats['total_nodes']}个节点, {graph_stats['total_edges']}条边")
    print(f"  技能节点: {graph_stats['skill_nodes']}个")
    print(f"  职业节点: {graph_stats['job_nodes']}个")
    print(f"  图密度: {graph_stats['density']:.6f}")

    print(f"  训练集: {split_stats['train_positive']}条正样本 + {negative_stats['train_negative']}条负样本")
    print(f"  验证集: {split_stats['val_positive']}条正样本 + {negative_stats['val_negative']}条负样本")
    print(f"  测试集: {split_stats['test_positive']}条正样本 + {negative_stats['test_negative']}条负样本")

    test_metrics = eval_results['test_metrics']
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {test_metrics['auc_pr']:.4f}")
    

if __name__ == "__main__":
    main()
