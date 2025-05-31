import pandas as pd
import numpy as np
import json
import networkx as nx
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# 评估指标
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

# Node2Vec相关库
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
    print("✓ Node2Vec库可用")
except ImportError:
    NODE2VEC_AVAILABLE = False
    print("⚠️ Node2Vec库未安装，请运行: pip install node2vec")

# 数值计算
from sklearn.metrics.pairwise import cosine_similarity

class PAvsNode2VecComparison:
    def __init__(self, csv_file: str, random_seed: int = 42):
        """
        PA vs Node2Vec算法比较器
        
        Args:
            csv_file: 化简后的CSV文件路径
            random_seed: 随机种子
        """
        self.csv_file = csv_file
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 图结构
        self.full_graph = nx.Graph()
        self.training_graph = nx.Graph()
        
        # 节点集合
        self.skill_nodes = set()
        self.job_nodes = set()
        
        # 数据集
        self.all_positive_edges = []
        self.train_positive = []
        self.val_positive = []
        self.test_positive = []
        self.train_negative = []
        self.val_negative = []
        self.test_negative = []
        
        # 算法模型
        self.node2vec_model = None
        self.node_embeddings = {}
        
        print(f"PA vs Node2Vec比较器初始化完成 (随机种子: {random_seed})")
    
    def load_simplified_data(self) -> Dict:
        """加载化简后的数据并构建图"""
        print("正在加载化简后的数据...")
        
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
            
            # 创建技能-职业边
            for skill in skills:
                if skill and len(skill.strip()) > 1:
                    skill_node = f"SKILL_{skill.strip()}"
                    self.skill_nodes.add(skill_node)
                    
                    edge = (skill_node, job_node)
                    edge_weights[edge] += 1
        
        # 构建完整图
        for (skill_node, job_node), weight in edge_weights.items():
            self.full_graph.add_edge(skill_node, job_node, weight=weight)
            self.all_positive_edges.append((skill_node, job_node))
        
        # 添加节点类型属性
        for skill_node in self.skill_nodes:
            self.full_graph.add_node(skill_node, node_type='skill')
        for job_node in self.job_nodes:
            self.full_graph.add_node(job_node, node_type='job')
        
        stats = {
            'total_nodes': len(self.skill_nodes) + len(self.job_nodes),
            'skill_nodes': len(self.skill_nodes),
            'job_nodes': len(self.job_nodes),
            'total_edges': len(self.all_positive_edges)
        }
        
        print(f"✓ 图构建完成:")
        print(f"  节点数: {stats['total_nodes']} ({stats['skill_nodes']}技能 + {stats['job_nodes']}职业)")
        print(f"  边数: {stats['total_edges']}")
        
        return stats
    
    def split_dataset(self, train_ratio: float = 0.55, val_ratio: float = 0.15, test_ratio: float = 0.30):
        """划分数据集"""
        print("正在划分数据集...")
        
        total_edges = len(self.all_positive_edges)
        train_size = int(total_edges * train_ratio)
        val_size = int(total_edges * val_ratio)
        test_size = total_edges - train_size - val_size
        
        # 随机打乱并划分正样本
        shuffled_edges = self.all_positive_edges.copy()
        random.shuffle(shuffled_edges)
        
        self.train_positive = shuffled_edges[:train_size]
        self.val_positive = shuffled_edges[train_size:train_size + val_size]
        self.test_positive = shuffled_edges[train_size + val_size:]
        
        # 构建训练图
        self.training_graph.clear()
        for skill_node in self.skill_nodes:
            self.training_graph.add_node(skill_node, node_type='skill')
        for job_node in self.job_nodes:
            self.training_graph.add_node(job_node, node_type='job')
        
        for edge in self.train_positive:
            weight = self.full_graph[edge[0]][edge[1]]['weight']
            self.training_graph.add_edge(edge[0], edge[1], weight=weight)
        
        # 生成负样本
        self._generate_negative_samples()
        
        print(f"✓ 数据集划分完成:")
        print(f"  训练集: {len(self.train_positive)}正样本 + {len(self.train_negative)}负样本")
        print(f"  验证集: {len(self.val_positive)}正样本 + {len(self.val_negative)}负样本") 
        print(f"  测试集: {len(self.test_positive)}正样本 + {len(self.test_negative)}负样本")
        
        return {
            'train_size': len(self.train_positive) + len(self.train_negative),
            'val_size': len(self.val_positive) + len(self.val_negative),
            'test_size': len(self.test_positive) + len(self.test_negative)
        }
    
    def _generate_negative_samples(self):
        """生成负样本"""
        existing_edges = set(self.all_positive_edges)
        
        # 生成负样本候选池
        negative_candidates = []
        for skill_node in self.skill_nodes:
            for job_node in self.job_nodes:
                if (skill_node, job_node) not in existing_edges:
                    negative_candidates.append((skill_node, job_node))
        
        random.shuffle(negative_candidates)
        
        # 为各数据集分配负样本
        train_neg_size = len(self.train_positive)
        val_neg_size = len(self.val_positive)
        test_neg_size = len(self.test_positive)
        
        self.train_negative = negative_candidates[:train_neg_size]
        self.val_negative = negative_candidates[train_neg_size:train_neg_size + val_neg_size]
        self.test_negative = negative_candidates[train_neg_size + val_neg_size:train_neg_size + val_neg_size + test_neg_size]
    
    def train_node2vec(self, dimensions: int = 512, walk_length: int = 16, 
                       num_walks: int = 2500, p: float = 4.0, q: float = 0.5):
        """
        训练Node2Vec模型
        
        Args:
            dimensions: 嵌入维度
            walk_length: 随机游走长度
            num_walks: 每个节点的游走次数
            p: 返回参数
            q: 进出参数
        """
        if not NODE2VEC_AVAILABLE:
            print("❌ Node2Vec库不可用，跳过Node2Vec训练")
            return False
        
        print(f"正在训练Node2Vec模型...")
        print(f"  参数: dim={dimensions}, walk_len={walk_length}, num_walks={num_walks}, p={p}, q={q}")
        
        try:
            # 使用训练图进行Node2Vec训练
            node2vec = Node2Vec(
                self.training_graph,
                dimensions=dimensions,
                walk_length=walk_length,
                num_walks=num_walks,
                p=p,
                q=q,
                workers=4,
                seed=self.random_seed
            )
            
            # 训练模型
            self.node2vec_model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=4,
                sg=1,  # Skip-gram
                epochs=10,
                seed=self.random_seed
            )
            
            # 获取所有节点的嵌入
            for node in self.training_graph.nodes():
                if node in self.node2vec_model.wv:
                    self.node_embeddings[node] = self.node2vec_model.wv[node]
                else:
                    # 对于训练图中没有的节点，使用零向量
                    self.node_embeddings[node] = np.zeros(dimensions)
            
            print(f"✓ Node2Vec训练完成，获得{len(self.node_embeddings)}个节点嵌入")
            return True
            
        except Exception as e:
            print(f"❌ Node2Vec训练失败: {e}")
            return False
    
    def calculate_pa_scores(self, edges: List[Tuple]) -> np.ndarray:
        """计算优先连接(PA)得分"""
        scores = []
        
        # 计算训练图中每个节点的度数
        node_degrees = {}
        for node in self.training_graph.nodes():
            node_degrees[node] = self.training_graph.degree(node)
        
        for u, v in edges:
            degree_u = node_degrees.get(u, 0)
            degree_v = node_degrees.get(v, 0)
            pa_score = degree_u * degree_v
            scores.append(pa_score)
        
        # 归一化到[0,1]
        scores = np.array(scores)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def calculate_node2vec_scores(self, edges: List[Tuple]) -> np.ndarray:
        """计算Node2Vec得分（基于嵌入向量相似度）"""
        if not self.node_embeddings:
            print("❌ Node2Vec嵌入不可用，返回零分数")
            return np.zeros(len(edges))
        
        scores = []
        
        for u, v in edges:
            # 获取节点嵌入
            embed_u = self.node_embeddings.get(u, np.zeros(len(list(self.node_embeddings.values())[0])))
            embed_v = self.node_embeddings.get(v, np.zeros(len(list(self.node_embeddings.values())[0])))
            
            # 计算余弦相似度
            if np.linalg.norm(embed_u) > 0 and np.linalg.norm(embed_v) > 0:
                similarity = cosine_similarity([embed_u], [embed_v])[0][0]
                # 将相似度从[-1,1]映射到[0,1]
                similarity = (similarity + 1) / 2
            else:
                similarity = 0.0
            
            scores.append(similarity)
        
        return np.array(scores)
    
    def evaluate_algorithms(self, threshold: float = 0.5) -> Dict:
        """评估PA和Node2Vec算法性能"""
        print("正在评估算法性能...")
        
        # 准备测试数据
        test_edges = self.test_positive + self.test_negative
        test_labels = [1] * len(self.test_positive) + [0] * len(self.test_negative)
        
        # 计算各算法得分
        pa_scores = self.calculate_pa_scores(test_edges)
        node2vec_scores = self.calculate_node2vec_scores(test_edges)
        
        # 计算预测结果
        pa_predictions = (pa_scores >= threshold).astype(int)
        node2vec_predictions = (node2vec_scores >= threshold).astype(int)
        
        # 计算性能指标
        def calculate_metrics(y_true, y_pred):
            # 分别计算正负样本的指标
            pos_indices = [i for i, label in enumerate(y_true) if label == 1]
            neg_indices = [i for i, label in enumerate(y_true) if label == 0]
            
            # 正样本指标
            pos_true = [y_true[i] for i in pos_indices]
            pos_pred = [y_pred[i] for i in pos_indices]
            
            # 负样本指标  
            neg_true = [y_true[i] for i in neg_indices]
            neg_pred = [y_pred[i] for i in neg_indices]
            
            # 整体指标
            overall_precision = precision_score(y_true, y_pred, average='binary')
            overall_recall = recall_score(y_true, y_pred, average='binary')
            overall_f1 = f1_score(y_true, y_pred, average='binary')
            
            # 分类别指标
            pos_precision = precision_score(pos_true, pos_pred, average='binary') if pos_true else 0
            pos_recall = recall_score(pos_true, pos_pred, average='binary') if pos_true else 0
            pos_f1 = f1_score(pos_true, pos_pred, average='binary') if pos_true else 0
            
            neg_precision = precision_score(neg_true, neg_pred, average='binary', pos_label=0) if neg_true else 0
            neg_recall = recall_score(neg_true, neg_pred, average='binary', pos_label=0) if neg_true else 0
            neg_f1 = f1_score(neg_true, neg_pred, average='binary', pos_label=0) if neg_true else 0
            
            return {
                'overall': {
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1_score': overall_f1
                },
                'positive': {
                    'precision': pos_precision,
                    'recall': pos_recall,
                    'f1_score': pos_f1
                },
                'negative': {
                    'precision': neg_precision,
                    'recall': neg_recall,
                    'f1_score': neg_f1
                }
            }
        
        pa_metrics = calculate_metrics(test_labels, pa_predictions)
        node2vec_metrics = calculate_metrics(test_labels, node2vec_predictions)
        
        return {
            'PA': pa_metrics,
            'Node2Vec': node2vec_metrics,
            'test_samples': len(test_labels),
            'positive_samples': len(self.test_positive),
            'negative_samples': len(self.test_negative)
        }
    
    def print_comparison_table(self, results: Dict):
        """打印类似论文表3的比较表格"""
        print("\n" + "="*80)
        print("算法性能比较表 (类似论文表3)")
        print("="*80)
        
        print(f"\n📊 测试集统计:")
        print(f"  总样本数: {results['test_samples']:,}")
        print(f"  正样本数: {results['positive_samples']:,}")
        print(f"  负样本数: {results['negative_samples']:,}")
        
        # 打印详细比较表
        print(f"\n📈 详细性能比较:")
        print("-" * 80)
        print(f"{'算法':<12} {'类别':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
        print("-" * 80)
        
        # PA算法结果
        pa_pos = results['PA']['positive']
        pa_neg = results['PA']['negative']
        print(f"{'PA':<12} {'1.0':<8} {pa_pos['precision']:<10.4f} {pa_pos['recall']:<10.4f} {pa_pos['f1_score']:<10.4f}")
        print(f"{'PA':<12} {'0.0':<8} {pa_neg['precision']:<10.4f} {pa_neg['recall']:<10.4f} {pa_neg['f1_score']:<10.4f}")
        
        # Node2Vec算法结果
        n2v_pos = results['Node2Vec']['positive'] 
        n2v_neg = results['Node2Vec']['negative']
        print(f"{'Node2Vec':<12} {'1.0':<8} {n2v_pos['precision']:<10.4f} {n2v_pos['recall']:<10.4f} {n2v_pos['f1_score']:<10.4f}")
        print(f"{'Node2Vec':<12} {'0.0':<8} {n2v_neg['precision']:<10.4f} {n2v_neg['recall']:<10.4f} {n2v_neg['f1_score']:<10.4f}")
        
        print("-" * 80)
        
        # 整体性能比较
        print(f"\n🏆 整体性能比较:")
        pa_overall = results['PA']['overall']
        n2v_overall = results['Node2Vec']['overall']
        
        print(f"{'算法':<12} {'整体精确率':<12} {'整体召回率':<12} {'整体F1分数':<12}")
        print("-" * 50)
        print(f"{'PA':<12} {pa_overall['precision']:<12.4f} {pa_overall['recall']:<12.4f} {pa_overall['f1_score']:<12.4f}")
        print(f"{'Node2Vec':<12} {n2v_overall['precision']:<12.4f} {n2v_overall['recall']:<12.4f} {n2v_overall['f1_score']:<12.4f}")
        
        # 判断哪个算法更好
        if n2v_overall['f1_score'] > pa_overall['f1_score']:
            winner = "Node2Vec"
            improvement = ((n2v_overall['f1_score'] - pa_overall['f1_score']) / pa_overall['f1_score']) * 100
        else:
            winner = "PA"
            improvement = ((pa_overall['f1_score'] - n2v_overall['f1_score']) / n2v_overall['f1_score']) * 100
        
        print(f"\n🎯 结论: {winner} 算法在整体F1分数上表现更好 (提升 {improvement:.1f}%)")
    
    def save_results(self, results: Dict, output_file: str = 'pa_vs_node2vec_results.json'):
        """保存比较结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 比较结果已保存到: {output_file}")

def main():
    """主函数：PA vs Node2Vec算法比较"""
    print("=" * 80)
    print("PA vs Node2Vec 算法性能比较")
    print("=" * 80)
    
    # 检查文件是否存在
    csv_file = 'simplified_jobs_skills.csv'
    try:
        pd.read_csv(csv_file, nrows=1)
    except FileNotFoundError:
        print(f"❌ 文件 {csv_file} 不存在，请先运行技能化简处理")
        return
    
    # 初始化比较器
    comparator = PAvsNode2VecComparison(csv_file, random_seed=42)
    
    # 步骤1: 加载数据并构建图
    print("\n" + "="*60)
    print("步骤1: 加载化简数据并构建知识图谱")
    print("="*60)
    graph_stats = comparator.load_simplified_data()
    
    # 步骤2: 划分数据集
    print("\n" + "="*60) 
    print("步骤2: 划分训练/验证/测试数据集")
    print("="*60)
    split_stats = comparator.split_dataset()
    
    # 步骤3: 训练Node2Vec模型
    print("\n" + "="*60)
    print("步骤3: 训练Node2Vec模型")
    print("="*60)
    n2v_success = comparator.train_node2vec(
        dimensions=512,
        walk_length=16, 
        num_walks=2500,
        p=4.0,
        q=0.5
    )
    
    # 步骤4: 评估算法性能
    print("\n" + "="*60)
    print("步骤4: 评估PA和Node2Vec算法性能")
    print("="*60)
    results = comparator.evaluate_algorithms(threshold=0.5)
    
    # 步骤5: 显示比较结果
    print("\n" + "="*60)
    print("步骤5: 算法性能比较结果")
    print("="*60)
    comparator.print_comparison_table(results)
    
    # 步骤6: 保存结果
    print("\n" + "="*60)
    print("步骤6: 保存比较结果")
    print("="*60)
    comparator.save_results(results)
    
    print("\n" + "="*80)
    print("🎉 PA vs Node2Vec 算法比较完成！")
    print("="*80)
    
    if not n2v_success:
        print("\n⚠️  注意: Node2Vec训练失败，结果可能不准确")
        print("   请确保安装: pip install node2vec gensim")

if __name__ == "__main__":
    main()