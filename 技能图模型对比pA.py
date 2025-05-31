import pandas as pd
import numpy as np
import json
import networkx as nx
from collections import defaultdict
import random
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


class PALinkPrediction:
    def __init__(self, csv_file: str, random_seed: int = 42):
        """
        PA (Preferential Attachment) 链接预测系统
        
        Args:
            csv_file: 化简后的技能数据CSV文件
            random_seed: 随机种子
        """
        self.csv_file = csv_file
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 图结构
        self.full_graph = nx.Graph()
        self.training_graph = nx.Graph()
        
        # 节点信息
        self.skill_nodes = set()
        self.job_nodes = set()
        
        # 数据集
        self.train_positive = []
        self.val_positive = []
        self.test_positive = []
        self.train_negative = []
        self.val_negative = []
        self.test_negative = []
        
        # PA结果
        self.pa_results = None
        
        print(f"PA链接预测系统初始化完成")
    

    def load_simplified_data_and_build_graph(self) -> Dict:
        """从化简后的CSV加载数据并构建图"""
        print("正在从化简数据构建知识图谱...")
        
        df = pd.read_csv(self.csv_file)
        print(f"读取数据: {len(df)} 条记录")
        
        # 统计边的权重
        edge_weights = defaultdict(int)
        
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"处理进度: {idx}/{len(df)}")
            
            # 获取ISCO职业代码
            isco_code = row.get('ISCO_4_Digit_Code_Gemini')
            if pd.isna(isco_code):
                continue
            
            job_node = f"JOB_{int(isco_code)}"
            self.job_nodes.add(job_node)
            
            # 获取化简后的技能
            skills_str = row.get('标准化技能', '[]')
            try:
                skills = json.loads(skills_str)
            except:
                continue
            
            # 为每个技能创建边
            for skill in skills:
                if skill and len(skill.strip()) > 1:
                    skill_node = f"SKILL_{skill.strip()}"
                    self.skill_nodes.add(skill_node)
                    
                    edge = (skill_node, job_node)
                    edge_weights[edge] += 1
        
        # 构建图
        for (skill_node, job_node), weight in edge_weights.items():
            self.full_graph.add_edge(skill_node, job_node, weight=weight)
        
        # 添加节点类型
        for skill_node in self.skill_nodes:
            self.full_graph.add_node(skill_node, node_type='skill')
        for job_node in self.job_nodes:
            self.full_graph.add_node(job_node, node_type='job')
        
        stats = {
            'total_nodes': len(self.skill_nodes) + len(self.job_nodes),
            'skill_nodes': len(self.skill_nodes),
            'job_nodes': len(self.job_nodes),
            'total_edges': len(edge_weights),
            'density': nx.density(self.full_graph)
        }
        
        print(f"✓ 知识图谱构建完成:")
        print(f"  总节点数: {stats['total_nodes']} ({stats['skill_nodes']}技能 + {stats['job_nodes']}职业)")
        print(f"  总边数: {stats['total_edges']}")
        print(f"  图密度: {stats['density']:.6f}")
        
        return stats
    
    def split_and_prepare_datasets(self):
        """分割数据集并准备训练数据"""
        print("正在分割数据集...")
        
        all_edges = list(self.full_graph.edges())
        total_edges = len(all_edges)
        
        # 按论文比例分割
        train_size = int(total_edges * 0.55)
        val_size = int(total_edges * 0.15)
        
        # 随机打乱
        random.shuffle(all_edges)
        
        self.train_positive = all_edges[:train_size]
        self.val_positive = all_edges[train_size:train_size + val_size]
        self.test_positive = all_edges[train_size + val_size:]
        
        # 构建训练图
        self.training_graph.clear()
        for node in self.full_graph.nodes(data=True):
            self.training_graph.add_node(node[0], **node[1])
        
        for edge in self.train_positive:
            weight = self.full_graph[edge[0]][edge[1]]['weight']
            self.training_graph.add_edge(edge[0], edge[1], weight=weight)
        
        # 生成负样本
        self._generate_negative_samples()
        
        print(f"✓ 数据集分割完成:")
        print(f"  训练集: {len(self.train_positive)} 正样本 + {len(self.train_negative)} 负样本")
        print(f"  验证集: {len(self.val_positive)} 正样本 + {len(self.val_negative)} 负样本")
        print(f"  测试集: {len(self.test_positive)} 正样本 + {len(self.test_negative)} 负样本")
    
    def _generate_negative_samples(self):
        """生成负样本"""
        existing_edges = set(self.full_graph.edges())
        
        # 生成所有可能的技能-职业配对作为负样本候选
        negative_candidates = []
        for skill_node in self.skill_nodes:
            for job_node in self.job_nodes:
                if (skill_node, job_node) not in existing_edges and (job_node, skill_node) not in existing_edges:
                    negative_candidates.append((skill_node, job_node))
        
        random.shuffle(negative_candidates)
        
        # 为各数据集分配负样本
        train_neg_size = len(self.train_positive)
        val_neg_size = len(self.val_positive)
        test_neg_size = len(self.test_positive)
        
        self.train_negative = negative_candidates[:train_neg_size]
        self.val_negative = negative_candidates[train_neg_size:train_neg_size + val_neg_size]
        self.test_negative = negative_candidates[train_neg_size + val_neg_size:train_neg_size + val_neg_size + test_neg_size]
    def find_optimal_threshold(self, scores: List[float], labels: List[int], metric: str = 'f1') -> Tuple[float, float]:
        """在 scores/labels 上扫描阈值，返回 (best_threshold, best_metric_value)。"""
        best_thr, best_score = 0.0, -1.0
        for thr in np.linspace(0.0, 1.0, 200):
            preds = (np.array(scores) >= thr).astype(int)
            # 跳过全 0/全 1 的情况
            if len(set(preds)) < 2:
                continue
            if metric == 'f1':
                score = f1_score(labels, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(labels, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(labels, preds, zero_division=0)
            else:
                score = f1_score(labels, preds, zero_division=0)
            if score > best_score:
                best_score, best_thr = score, thr
        return best_thr, best_score
    def calculate_pa_scores(self) -> Dict:
        """计算并评估 PA 算法，带阈值优化"""
        # —— 1. 计算 raw PA 分数 —— 
        node_degrees = dict(self.training_graph.degree())
        def pa_scores(edges):
            return [node_degrees.get(u,0)*node_degrees.get(v,0) for u,v in edges]

        pos_all = self.train_positive + self.val_positive + self.test_positive
        neg_all = self.train_negative + self.val_negative + self.test_negative
        pos_scores = pa_scores(pos_all)
        neg_scores = pa_scores(neg_all)

        # —— 2. 归一化 —— 
        all_scores = pos_scores + neg_scores
        max_s = max(all_scores) or 1
        pos_norm = [s/max_s for s in pos_scores]
        neg_norm = [s/max_s for s in neg_scores]

        # —— 3. 切分回各集 —— 
        n_train, n_val, n_test = len(self.train_positive), len(self.val_positive), len(self.test_positive)
        train_scores = pos_norm[:n_train]     + neg_norm[:n_train]
        train_labels = [1]*n_train + [0]*n_train
        val_scores   = pos_norm[n_train:n_train+n_val] + neg_norm[n_train:n_train+n_val]
        val_labels   = [1]*n_val   + [0]*n_val
        test_scores  = pos_norm[n_train+n_val:]     + neg_norm[n_train+n_val:]
        test_labels  = [1]*n_test  + [0]*n_test

        # —— 4. 验证集上找最优阈值 —— 
        best_thr, best_f1 = self.find_optimal_threshold(val_scores, val_labels, metric='f1')
        print(f"PA 最优阈值 (验证集): {best_thr:.3f}, F1 = {best_f1:.4f}")

        # —— 5. 用这个阈值评估所有数据集 —— 
        def eval_at_thr(scores, labels):
            preds = (np.array(scores) >= best_thr).astype(int)
            return {
                'precision': precision_score(labels, preds, zero_division=0),
                'recall':    recall_score(labels, preds, zero_division=0),
                'f1_score':  f1_score(labels, preds, zero_division=0),
                'auc_roc':   roc_auc_score(labels, scores),
                'auc_pr':    average_precision_score(labels, scores)
            }

        self.pa_results = {
            'train_metrics': eval_at_thr(train_scores, train_labels),
            'val_metrics':   eval_at_thr(val_scores,   val_labels),
            'test_metrics':  eval_at_thr(test_scores,  test_labels),
            'best_threshold': best_thr
        }

        # —— 6. 打印结果 —— 
        for split in ['train', 'val', 'test']:
            m = self.pa_results[f'{split}_metrics']
            print(f"\n{split.upper()} @ thr={best_thr:.3f}:"
                  f"  Precision={m['precision']:.4f}"
                  f"  Recall={m['recall']:.4f}"
                  f"  F1={m['f1_score']:.4f}"
                  f"  AUC-ROC={m['auc_roc']:.4f}"
                  f"  AUC-PR={m['auc_pr']:.4f}")

        return self.pa_results
    
    def save_results(self, output_file: str = 'pa_link_prediction_results.json'):
        """保存PA算法结果"""
        print(f"\n正在保存结果到 {output_file}...")
        
        if not self.pa_results:
            print("❌ 没有结果可保存")
            return
        
        # 准备保存的数据
        results_data = {
            'algorithm': 'Preferential Attachment (PA)',
            'graph_stats': {
                'total_nodes': len(self.skill_nodes) + len(self.job_nodes),
                'skill_nodes': len(self.skill_nodes),
                'job_nodes': len(self.job_nodes),
                'total_edges': self.full_graph.number_of_edges(),
                'density': nx.density(self.full_graph)
            },
            'dataset_splits': {
                'train_positive': len(self.train_positive),
                'train_negative': len(self.train_negative),
                'val_positive': len(self.val_positive),
                'val_negative': len(self.val_negative),
                'test_positive': len(self.test_positive),
                'test_negative': len(self.test_negative)
            },
            'performance_metrics': self.pa_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 结果已保存到 {output_file}")


def main():
    """主函数：执行PA算法链接预测"""
    print("=" * 80)
    print("PA (Preferential Attachment) 链接预测系统")
    print("=" * 80)
    
    # 初始化系统
    pa_predictor = PALinkPrediction(
        csv_file='simplified_jobs_skills.csv',  # 使用化简后的数据
        random_seed=42
    )
    
    # 步骤1: 构建图
    print("\n" + "="*60)
    print("步骤1: 构建知识图谱")
    print("="*60)
    graph_stats = pa_predictor.load_simplified_data_and_build_graph()
    
    # 步骤2: 分割数据集
    print("\n" + "="*60)
    print("步骤2: 分割数据集")
    print("="*60)
    pa_predictor.split_and_prepare_datasets()
    
    # 步骤3: 计算PA算法性能
    print("\n" + "="*60)
    print("步骤3: 计算PA算法链接预测性能")
    print("="*60)
    pa_results = pa_predictor.calculate_pa_scores()
    
    # 步骤4: 保存结果
    print("\n" + "="*60)
    print("步骤4: 保存结果")
    print("="*60)
    pa_predictor.save_results()
    
    print("\n" + "="*80)
    print("🎉 PA链接预测完成！")
    print("="*80)


if __name__ == "__main__":
    main()