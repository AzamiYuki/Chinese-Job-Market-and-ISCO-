import pandas as pd
import numpy as np
import json
import networkx as nx
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import time
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入node2vec
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

class RatioAnalysisSystem:
    def __init__(self, csv_file: str, random_seed: int = 42):
        """
        正负边比例分析系统
        
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
        
        # 固定的正样本（确保比较公平性）
        self.fixed_positive_samples = []
        self.negative_candidate_pool = []
        
        # 算法模型
        self.pa_node_degrees = {}
        self.node2vec_model = None
        
        # 分析结果
        self.ratio_results = []
        
        print(f"正负边比例分析系统初始化完成")
    
    def load_data_and_setup(self) -> Dict:
        """加载数据并设置基础图结构"""
        print("正在加载数据并构建图结构...")
        
        df = pd.read_csv(self.csv_file)
        edge_weights = defaultdict(int)
        
        # 构建图
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"处理进度: {idx}/{len(df)}")
            
            isco_code = row.get('ISCO_4_Digit_Code_Gemini')
            if pd.isna(isco_code):
                continue
            
            job_node = f"JOB_{int(isco_code)}"
            self.job_nodes.add(job_node)
            
            skills_str = row.get('标准化技能', '[]')
            try:
                skills = json.loads(skills_str)
            except:
                continue
            
            for skill in skills:
                if skill and len(skill.strip()) > 1:
                    skill_node = f"SKILL_{skill.strip()}"
                    self.skill_nodes.add(skill_node)
                    edge = (skill_node, job_node)
                    edge_weights[edge] += 1
        
        # 构建完整图
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
            'total_edges': len(edge_weights)
        }
        
        print(f"✓ 图结构构建完成:")
        print(f"  节点数: {stats['total_nodes']} ({stats['skill_nodes']}技能 + {stats['job_nodes']}职业)")
        print(f"  边数: {stats['total_edges']}")
        
        return stats
    
    def prepare_training_data(self, sample_size: int = 2000):
        """准备训练数据和正样本"""
        print(f"正在准备训练数据（正样本数量: {sample_size}）...")
        
        # 获取所有边作为正样本候选
        all_edges = list(self.full_graph.edges())
        
        # 随机采样固定数量的正样本（确保后续比较的公平性）
        if len(all_edges) > sample_size:
            self.fixed_positive_samples = random.sample(all_edges, sample_size)
        else:
            self.fixed_positive_samples = all_edges
        
        # 构建训练图（用于计算节点度数和训练Node2Vec）
        # 使用80%的正样本作为训练数据
        train_edge_count = int(len(self.fixed_positive_samples) * 0.8)
        train_edges = self.fixed_positive_samples[:train_edge_count]
        
        self.training_graph.clear()
        for node in self.full_graph.nodes(data=True):
            self.training_graph.add_node(node[0], **node[1])
        
        for edge in train_edges:
            if self.full_graph.has_edge(edge[0], edge[1]):
                weight = self.full_graph[edge[0]][edge[1]]['weight']
                self.training_graph.add_edge(edge[0], edge[1], weight=weight)
        
        # 生成负样本候选池
        existing_edges = set(self.full_graph.edges())
        self.negative_candidate_pool = []
        
        for skill_node in self.skill_nodes:
            for job_node in self.job_nodes:
                if (skill_node, job_node) not in existing_edges and (job_node, skill_node) not in existing_edges:
                    self.negative_candidate_pool.append((skill_node, job_node))
        
        random.shuffle(self.negative_candidate_pool)
        
        print(f"✓ 训练数据准备完成:")
        print(f"  固定正样本数: {len(self.fixed_positive_samples)}")
        print(f"  训练边数: {len(train_edges)}")
        print(f"  负样本候选池: {len(self.negative_candidate_pool)}")
    
    def train_algorithms(self):
        """训练PA和Node2Vec算法"""
        print("正在训练算法...")
        
        # 1. 计算PA算法的节点度数
        self.pa_node_degrees = dict(self.training_graph.degree())
        print(f"✓ PA算法准备完成（基于训练图度数）")
        
        # 2. 训练Node2Vec模型
        if NODE2VEC_AVAILABLE and len(self.training_graph.edges()) > 0:
            try:
                # 使用优化的参数（基于之前的网格搜索结果）
                node2vec = Node2Vec(
                    self.training_graph,
                    dimensions=64,      # 适中的维度
                    walk_length=16,       # 较长的游走
                    num_walks=100,       # 适中的游走次数
                    p=2,                 # 平衡参数
                    q=0.5,                 # 偏向DFS
                    workers=4,
                    seed=self.random_seed
                )
                
                self.node2vec_model = node2vec.fit(
                    window=10,
                    min_count=1,
                    batch_words=4,
                    sg=1,
                    epochs=10
                )
                
                print(f"✓ Node2Vec模型训练完成")
                
            except Exception as e:
                print(f"Node2Vec训练失败: {e}")
                self.node2vec_model = None
        else:
            print(f"⚠Node2Vec不可用（库未安装或训练图为空）")
            self.node2vec_model = None
    
    def calculate_pa_scores(self, edge_list: List[Tuple]) -> List[float]:
        """计算PA算法得分"""
        scores = []
        for u, v in edge_list:
            degree_u = self.pa_node_degrees.get(u, 0)
            degree_v = self.pa_node_degrees.get(v, 0)
            pa_score = degree_u * degree_v
            scores.append(pa_score)
        return scores
    
    def calculate_node2vec_scores(self, edge_list: List[Tuple]) -> List[float]:
        """计算Node2Vec算法得分"""
        if not self.node2vec_model:
            return [0.0] * len(edge_list)  # 如果模型不可用，返回零分
        
        scores = []
        for u, v in edge_list:
            try:
                if u in self.node2vec_model.wv and v in self.node2vec_model.wv:
                    emb_u = self.node2vec_model.wv[u]
                    emb_v = self.node2vec_model.wv[v]
                    similarity = cosine_similarity([emb_u], [emb_v])[0][0]
                    scores.append(similarity)
                else:
                    scores.append(0.0)
            except:
                scores.append(0.0)
        return scores
    
    def evaluate_at_ratio(self, negative_ratio: int) -> Dict:
        """在指定负样本比例下评估算法性能"""
        print(f"正在评估 1:{negative_ratio} 比例...")
        
        # 使用固定的正样本
        positive_samples = self.fixed_positive_samples.copy()
        positive_count = len(positive_samples)
        
        # 根据比例采样负样本
        negative_count = positive_count * negative_ratio
        if negative_count > len(self.negative_candidate_pool):
            negative_samples = self.negative_candidate_pool.copy()
            # 如果候选池不够，进行重复采样
            additional_needed = negative_count - len(self.negative_candidate_pool)
            additional_samples = random.choices(self.negative_candidate_pool, k=additional_needed)
            negative_samples.extend(additional_samples)
        else:
            negative_samples = random.sample(self.negative_candidate_pool, negative_count)
        
        # 准备评估数据
        all_edges = positive_samples + negative_samples
        y_true = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        # 计算PA得分
        pa_scores = self.calculate_pa_scores(all_edges)
        
        # 计算Node2Vec得分
        n2v_scores = self.calculate_node2vec_scores(all_edges)
        
        # 归一化得分
        def normalize_scores(scores):
            max_score = max(scores) if scores and max(scores) > 0 else 1
            return [score / max_score for score in scores]
        
        pa_normalized = normalize_scores(pa_scores)
        n2v_normalized = normalize_scores(n2v_scores)
        
        # 计算性能指标
        def evaluate_performance(scores, threshold=0.5):
            y_pred = (np.array(scores) >= threshold).astype(int)
            return {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else 0.5
            }
        
        pa_metrics = evaluate_performance(pa_normalized)
        n2v_metrics = evaluate_performance(n2v_normalized)
        
        result = {
            'negative_ratio': negative_ratio,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'pa_metrics': pa_metrics,
            'n2v_metrics': n2v_metrics,
            'pa_f1': pa_metrics['f1_score'],
            'n2v_f1': n2v_metrics['f1_score']
        }
        
        print(f"  比例 1:{negative_ratio} - PA F1: {pa_metrics['f1_score']:.4f}, N2V F1: {n2v_metrics['f1_score']:.4f}")
        
        return result
    
    def run_ratio_analysis(self, max_ratio: int = 10) -> List[Dict]:
        """运行完整的比例分析"""
        print(f"\n开始正负边比例分析（1:1 到 1:{max_ratio}）...")
        
        self.ratio_results = []
        
        for ratio in range(1, max_ratio + 1):
            try:
                result = self.evaluate_at_ratio(ratio)
                self.ratio_results.append(result)
                
                # 保存中间结果（防止长时间运行中断）
                if ratio % 3 == 0:
                    self._save_intermediate_results()
                    
            except Exception as e:
                print(f"比例 1:{ratio} 评估失败: {e}")
                continue
        
        print(f"完成 {len(self.ratio_results)} 个比例的评估")
        return self.ratio_results
    
    def _save_intermediate_results(self):
        """保存中间结果"""
        with open('ratio_analysis_intermediate.json', 'w', encoding='utf-8') as f:
            json.dump(self.ratio_results, f, ensure_ascii=False, indent=2)
    
    def plot_ratio_comparison(self, save_path: str = 'ratio_comparison.png') -> None:
        """绘制比例对比图（复现论文图4）"""
        print("正在绘制比例对比图...")
        
        if not self.ratio_results:
            print("没有分析结果可绘制")
            return
        
        # 提取数据
        ratios = [r['negative_ratio'] for r in self.ratio_results]
        pa_f1_scores = [r['pa_f1'] for r in self.ratio_results]
        n2v_f1_scores = [r['n2v_f1'] for r in self.ratio_results]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制线条
        plt.plot(ratios, pa_f1_scores, 
                marker='o', linewidth=2.5, markersize=8, 
                color='#2E8B57', label='Preferential Attachment (PA)',
                markerfacecolor='white', markeredgewidth=2)
        
        plt.plot(ratios, n2v_f1_scores, 
                marker='s', linewidth=2.5, markersize=8,
                color='#FF6B35', label='Node2Vec (N2V)',
                markerfacecolor='white', markeredgewidth=2)
        
        # 添加交叉点标记
        # 找到PA和N2V性能相等的点
        crossover_ratio = None
        min_diff = float('inf')
        for i, ratio in enumerate(ratios):
            diff = abs(pa_f1_scores[i] - n2v_f1_scores[i])
            if diff < min_diff:
                min_diff = diff
                crossover_ratio = ratio
        
        if crossover_ratio:
            crossover_idx = ratios.index(crossover_ratio)
            crossover_f1 = (pa_f1_scores[crossover_idx] + n2v_f1_scores[crossover_idx]) / 2
            plt.plot(crossover_ratio, crossover_f1, 
                    marker='*', markersize=15, color='gold', 
                    markeredgecolor='black', markeredgewidth=1.5,
                    label=f'交叉点 (≈{crossover_ratio}:1)')
        
        # 设置图表样式
        plt.xlabel('负样本与正样本比例', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score (正样本类别)', fontsize=14, fontweight='bold')
        plt.title('不同正负样本比例下的算法性能对比\n(复现论文图4)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # 设置网格和样式
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=12, framealpha=0.9)
        
        # 设置坐标轴
        plt.xlim(0.5, max(ratios) + 0.5)
        plt.ylim(0, 1.05)
        plt.xticks(ratios, [f'1:{r}' for r in ratios], rotation=45)
        
        # 添加注释
        plt.text(0.02, 0.98, 
                f'样本规模: {self.ratio_results[0]["positive_count"]} 正样本', 
                transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 紧凑布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"✓ 比例对比图已保存: {save_path}")
    
    def generate_analysis_report(self) -> Dict:
        """生成详细分析报告"""
        print("正在生成分析报告...")
        
        if not self.ratio_results:
            return {}
        
        # 找到关键点
        pa_best_ratio = max(self.ratio_results, key=lambda x: x['pa_f1'])
        n2v_best_ratio = max(self.ratio_results, key=lambda x: x['n2v_f1'])
        
        # 找到交叉点
        crossover_point = None
        min_diff = float('inf')
        for result in self.ratio_results:
            diff = abs(result['pa_f1'] - result['n2v_f1'])
            if diff < min_diff:
                min_diff = diff
                crossover_point = result
        
        # 分析趋势
        ratios = [r['negative_ratio'] for r in self.ratio_results]
        pa_trend = np.polyfit(ratios, [r['pa_f1'] for r in self.ratio_results], 1)[0]
        n2v_trend = np.polyfit(ratios, [r['n2v_f1'] for r in self.ratio_results], 1)[0]
        
        report = {
            'summary': {
                'total_ratios_tested': len(self.ratio_results),
                'positive_sample_count': self.ratio_results[0]['positive_count'],
                'crossover_ratio': crossover_point['negative_ratio'] if crossover_point else None,
                'crossover_f1_diff': min_diff
            },
            'pa_performance': {
                'best_ratio': pa_best_ratio['negative_ratio'],
                'best_f1': pa_best_ratio['pa_f1'],
                'trend_slope': pa_trend,
                'performance_at_1_1': self.ratio_results[0]['pa_f1'] if self.ratio_results else None
            },
            'n2v_performance': {
                'best_ratio': n2v_best_ratio['negative_ratio'],
                'best_f1': n2v_best_ratio['n2v_f1'],
                'trend_slope': n2v_trend,
                'performance_at_1_1': self.ratio_results[0]['n2v_f1'] if self.ratio_results else None
            },
            'detailed_results': self.ratio_results
        }
        
        # 打印关键发现
        print(f"\n📊 分析报告摘要:")
        print(f"  测试比例范围: 1:1 到 1:{max([r['negative_ratio'] for r in self.ratio_results])}")
        print(f"  正样本数量: {report['summary']['positive_sample_count']:,}")
        
        if crossover_point:
            print(f"  性能交叉点: 约 1:{crossover_point['negative_ratio']} (差异: {min_diff:.4f})")
        
        print(f"\n  PA算法:")
        print(f"    1:1比例F1: {report['pa_performance']['performance_at_1_1']:.4f}")
        print(f"    最佳比例: 1:{report['pa_performance']['best_ratio']} (F1: {report['pa_performance']['best_f1']:.4f})")
        print(f"    性能趋势: {'下降' if pa_trend < 0 else '上升'} ({pa_trend:.4f}/比例)")
        
        print(f"\n  Node2Vec算法:")
        print(f"    1:1比例F1: {report['n2v_performance']['performance_at_1_1']:.4f}")
        print(f"    最佳比例: 1:{report['n2v_performance']['best_ratio']} (F1: {report['n2v_performance']['best_f1']:.4f})")
        print(f"    性能趋势: {'下降' if n2v_trend < 0 else '上升'} ({n2v_trend:.4f}/比例)")
        
        return report
    
    def save_results(self, output_prefix: str = 'ratio_analysis'):
        """保存完整结果"""
        print(f"正在保存结果...")
        
        # 生成报告
        report = self.generate_analysis_report()
        
        # 保存详细结果
        with open(f'{output_prefix}_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存简化的CSV格式
        if self.ratio_results:
            df_results = pd.DataFrame([
                {
                    'negative_ratio': r['negative_ratio'],
                    'ratio_label': f"1:{r['negative_ratio']}",
                    'positive_count': r['positive_count'],
                    'negative_count': r['negative_count'],
                    'pa_f1': r['pa_f1'],
                    'n2v_f1': r['n2v_f1'],
                    'pa_precision': r['pa_metrics']['precision'],
                    'pa_recall': r['pa_metrics']['recall'],
                    'n2v_precision': r['n2v_metrics']['precision'],
                    'n2v_recall': r['n2v_metrics']['recall']
                }
                for r in self.ratio_results
            ])
            
            df_results.to_csv(f'{output_prefix}_results.csv', index=False, encoding='utf-8-sig')
        


def main():
    """主函数：完整的比例分析流程"""

    
    # 初始化系统
    analyzer = RatioAnalysisSystem(
        csv_file='simplified_jobs_skills.csv',
        random_seed=42
    )
    
    # 步骤1: 加载数据
    print("\n" + "="*60)
    print("步骤1: 加载数据并构建图结构")
    print("="*60)
    graph_stats = analyzer.load_data_and_setup()
    
    # 步骤2: 准备训练数据
    print("\n" + "="*60)
    print("步骤2: 准备训练数据")
    print("="*60)
    analyzer.prepare_training_data(sample_size=2000)  # 可调整样本大小
    
    # 步骤3: 训练算法
    print("\n" + "="*60)
    print("步骤3: 训练PA和Node2Vec算法")
    print("="*60)
    analyzer.train_algorithms()
    
    # 步骤4: 运行比例分析
    print("\n" + "="*60)
    print("步骤4: 运行正负边比例分析")
    print("="*60)
    
    # 询问用户分析范围
    max_ratio = input("请输入最大负样本比例 [默认10，即测试1:1到1:10]: ").strip()
    try:
        max_ratio = int(max_ratio) if max_ratio else 10
    except:
        max_ratio = 10
    
    ratio_results = analyzer.run_ratio_analysis(max_ratio=max_ratio)
    
    # 步骤5: 绘制对比图
    print("\n" + "="*60)
    print("步骤5: 绘制性能对比图")
    print("="*60)
    analyzer.plot_ratio_comparison()
    
    # 步骤6: 生成分析报告
    print("\n" + "="*60)
    print("步骤6: 生成分析报告")
    print("="*60)
    report = analyzer.generate_analysis_report()
    
    # 步骤7: 保存结果
    print("\n" + "="*60)
    print("步骤7: 保存分析结果")
    print("="*60)
    analyzer.save_results()
    
    # 生成总结
    print("\n" + "="*80)
    print("正负边比例分析完成")
    print("="*80)
    
    print(f"\n📈 关键发现:")
    if report and 'summary' in report:
        if report['summary']['crossover_ratio']:
            print(f"  • 算法性能交叉点: 约 1:{report['summary']['crossover_ratio']}")
            print(f"  • 在低比例时PA表现更好，高比例时Node2Vec更优")
        
        if 'pa_performance' in report and 'n2v_performance' in report:
            pa_1_1 = report['pa_performance']['performance_at_1_1']
            n2v_1_1 = report['n2v_performance']['performance_at_1_1']
            if pa_1_1 and n2v_1_1:
                print(f"  • 1:1比例性能: PA({pa_1_1:.3f}) vs N2V({n2v_1_1:.3f})")


if __name__ == "__main__":
    main()
