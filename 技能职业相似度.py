import pandas as pd
import numpy as np
import ast
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# 设置matplotlib支持中文显示 (如果需要，选择一个你系统里有的中文字体)
# 设置默认字体为 Helvetica（或任何存在的字体）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False
class CareerPathPlanner:
    """基于技能相似性的职业路径规划器 (以ISCO代码为节点，使用定义进行展示)"""
    
    def __init__(self, csv_file='simplified_jobs_skills_with_definition.csv'):
        """
        初始化职业路径规划器
        
        参数:
            csv_file: 包含职业、技能和定义数据的CSV文件路径
        """
        self.csv_file = csv_file
        self.df = None
        self.job_skills = defaultdict(set)  # 存储ISCO代码及其聚合的技能集合
        self.skill_jobs = defaultdict(set)  # 存储技能及其相关的ISCO代码集合
        self.isco_to_definition = {} # 存储ISCO代码到其 'Translated_Definition_Col3_ZH' 的映射
        self.career_graph = None  # ISCO代码转换图
        self.embeddings = None  # Node2Vec嵌入 (针对ISCO代码节点)
        
# 在您的 CareerPathPlanner 类中，替换旧的 load_data 方法
    def load_data(self):
        print("加载数据...")
        self.df = pd.read_csv(self.csv_file)
        
        # 用于临时存储每个ISCO代码可能遇到的所有非空定义
        temp_isco_definitions = defaultdict(list)

        for idx, row in self.df.iterrows():
            isco_code_val = row.get('ISCO_4_Digit_Code_Gemini')
            skills_str = row.get('标准化技能', '[]')
            definition_col3_zh = row.get('Translated_Definition_Col3_ZH', '')

            try:
                skills = ast.literal_eval(skills_str) if skills_str else []
            except:
                skills = []
            
            if isco_code_val: # 确保ISCO码存在
                current_isco_code = str(isco_code_val)
                
                if skills: # 只有当技能列表不为空时，才处理技能相关的映射
                    self.job_skills[current_isco_code].update(skills)
                    for skill in skills:
                        self.skill_jobs[skill].add(current_isco_code)
                
                # 收集该ISCO码所有非空的定义
                if definition_col3_zh: # 只添加非空定义
                    temp_isco_definitions[current_isco_code].append(definition_col3_zh)

        # 从收集到的定义中为每个ISCO码确定最终定义
        # 合并所有在技能数据或定义数据中出现过的ISCO码
        all_isco_codes_in_data = set(self.job_skills.keys()).union(set(temp_isco_definitions.keys()))

        for code in all_isco_codes_in_data:
            if code in temp_isco_definitions and temp_isco_definitions[code]:
                # 如果存在非空定义，取第一个遇到的非空定义
                self.isco_to_definition[code] = temp_isco_definitions[code][0] 
            else:
                # 如果没有找到任何非空定义，则使用ISCO码自身作为后备
                self.isco_to_definition[code] = code 
        
        print(f"加载了 {len(self.job_skills)} 个不同的ISCO代码 (这些ISCO代码有关联的技能)")
        print(f"总共有 {len(self.skill_jobs)} 个不同的技能")
        print(f"为 {len(self.isco_to_definition)} 个ISCO代码存储了定义 (部分可能为其ISCO码自身作为后备)")

        # 增加调试打印，检查几个ISCO码的定义是否符合预期
        if self.isco_to_definition:
            sample_iscos_to_check = list(self.isco_to_definition.keys())[:5] # 抽查前5个
            print("--- 抽样检查已加载的ISCO定义 ---")
            for s_isco in sample_iscos_to_check:
                print(f"  ISCO {s_isco}: '{self.isco_to_definition.get(s_isco)}'")
            # 如果你知道哪些ISCO码应该有中文定义，可以专门检查它们：
            # specific_iscos_to_debug = ['1211', '2166'] # 替换成你想检查的ISCO
            # for s_isco_debug in specific_iscos_to_debug:
            #     if s_isco_debug in self.isco_to_definition:
            #         print(f"  特定检查 ISCO {s_isco_debug}: '{self.isco_to_definition.get(s_isco_debug)}'")
            #     else:
            #         print(f"  特定检查 ISCO {s_isco_debug}: 未在 self.isco_to_definition 中找到。")
            print("--- 抽样检查结束 ---")
        
    def calculate_jaccard_distance(self, isco1, isco2):
        skills1 = self.job_skills.get(str(isco1), set())
        skills2 = self.job_skills.get(str(isco2), set())
        if not skills1 or not skills2: return 1.0
        intersection = len(skills1 & skills2)
        union = len(skills1 | skills2)
        if union == 0: return 1.0
        return 1 - (intersection / union)
    
    def build_career_transition_graph(self, distance_threshold=0.7):
        print(f"构建ISCO代码转换图 (距离阈值: {distance_threshold})...")
        self.career_graph = nx.Graph()
        isco_codes_list = list(self.job_skills.keys())
        for isco_id in isco_codes_list: self.career_graph.add_node(isco_id)
        edge_count = 0
        for i in range(len(isco_codes_list)):
            for j in range(i + 1, len(isco_codes_list)):
                isco1, isco2 = isco_codes_list[i], isco_codes_list[j]
                distance = self.calculate_jaccard_distance(isco1, isco2)
                if distance < distance_threshold:
                    self.career_graph.add_edge(isco1, isco2, weight=distance)
                    edge_count += 1
        print(f"构建完成: {len(self.career_graph.nodes())} 个ISCO节点, {edge_count} 条边")
        
    def train_node2vec_embeddings(self, dimensions=512, walk_length=16, num_walks=2500, p=4, q=0.5):
        print("训练Node2Vec嵌入 (针对ISCO代码节点)...")
        if self.career_graph is None or self.career_graph.number_of_nodes() == 0:
            print("图尚未构建或图中没有节点，尝试构建...")
            self.build_career_transition_graph()
            if self.career_graph is None or self.career_graph.number_of_nodes() == 0:
                 print("图构建失败或仍无节点。嵌入训练中止。")
                 return
        node2vec = Node2Vec(self.career_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=4) # workers可调整
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.embeddings = {node: model.wv[node] for node in self.career_graph.nodes()}
        print(f"嵌入训练完成，维度: {dimensions}")

    def get_isco_display_label(self, isco_code, max_def_len=30):
        """辅助方法：获取ISCO代码的显示标签（ISCO + 定义）"""
        isco_str = str(isco_code)
        definition = self.isco_to_definition.get(isco_str, isco_str) # 如果定义不存在，则用ISCO码自身
        definition_cleaned = definition.replace('\n', ' ').replace('\r', '')[:max_def_len]
        return f"{isco_str} ({definition_cleaned})"

    def find_shortest_career_path(self, source_isco, target_isco, use_embeddings=False):
        source_isco_str, target_isco_str = str(source_isco), str(target_isco)
        if source_isco_str not in self.career_graph or target_isco_str not in self.career_graph:
            print(f"ISCO代码 {source_isco_str} 或 {target_isco_str} 不在图中")
            return None, float('inf')
        try:
            graph_for_path = self.career_graph
            if use_embeddings and self.embeddings:
                temp_graph = self.career_graph.copy()
                for u, v in temp_graph.edges():
                    if u in self.embeddings and v in self.embeddings:
                        emb1, emb2 = self.embeddings[u].reshape(1, -1), self.embeddings[v].reshape(1, -1)
                        temp_graph[u][v]['weight'] = 1 - cosine_similarity(emb1, emb2)[0, 0]
                graph_for_path = temp_graph
            path = nx.shortest_path(graph_for_path, source_isco_str, target_isco_str, weight='weight')
            distance = nx.shortest_path_length(graph_for_path, source_isco_str, target_isco_str, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            print(f"没有找到从 {self.get_isco_display_label(source_isco_str)} 到 {self.get_isco_display_label(target_isco_str)} 的路径")
            return None, float('inf')
        except KeyError as e:
            print(f"路径查找中发生KeyError: {e}。")
            return None, float('inf')

    def recommend_career_transitions(self, current_isco, top_k=10, max_distance=0.5):
        current_isco_str = str(current_isco)
        if current_isco_str not in self.job_skills:
            print(f"ISCO代码 {current_isco_str} 不存在于已加载数据中")
            return []
        recommendations = []
        for target_isco_code in self.job_skills:
            if target_isco_code != current_isco_str:
                distance = self.calculate_jaccard_distance(current_isco_str, target_isco_code)
                if distance <= max_distance:
                    shared_skills = self.job_skills[current_isco_str] & self.job_skills[target_isco_code]
                    definition = self.isco_to_definition.get(target_isco_code, "未知定义")
                    recommendations.append({
                        'isco': target_isco_code,
                        'definition': definition,
                        'distance': distance,
                        'shared_skills': list(shared_skills),
                        'skill_overlap': len(shared_skills)
                    })
        recommendations.sort(key=lambda x: x['distance'])
        return recommendations[:top_k]
    
# 确保在文件顶部已经导入了 numpy:
    # import numpy as np
    # (您提供的完整代码中已经包含了 import numpy as np)

    def visualize_career_network(self, center_isco=None, radius=2):
        if not self.career_graph or self.career_graph.number_of_nodes() == 0:
            print("职业网络图尚未构建或为空。")
            return
        
        plt.figure(figsize=(18, 12)) # 您可以根据需要调整图像大小
        graph_to_draw = self.career_graph
        center_node_str = str(center_isco) if center_isco else None

        if center_node_str and center_node_str in self.career_graph:
            try:
                graph_to_draw = nx.ego_graph(self.career_graph, center_node_str, radius=radius)
                # 标题中的中心节点标签仍使用 get_isco_display_label 以显示 ISCO (定义) 格式
                print(f"可视化以 {self.get_isco_display_label(center_node_str)} 为中心，半径为 {radius} 的子图 ({graph_to_draw.number_of_nodes()} 节点)。")
            except nx.NetworkXError as e:
                print(f"无法生成子图: {e}. 将显示完整图。")
                graph_to_draw = self.career_graph # 出错则退回显示全图
        
        if graph_to_draw.number_of_nodes() == 0:
            print("要绘制的图没有节点。")
            return
        
        # 计算布局参数 k
        # 您可以调整这里的 k 值计算逻辑或直接设置一个固定值来试验
        if graph_to_draw.number_of_nodes() > 1:
            # k_value = min(0.5 + 1.5 / np.sqrt(graph_to_draw.number_of_nodes()), 1.0) 
            k_value = 0.8 # 或者尝试一个固定值开始，例如 0.6, 0.8, 1.0 等
        else:
            k_value = 0.5
        
        print(f"信息: 使用 k={k_value:.2f} 进行 spring_layout 布局。")
        pos = nx.spring_layout(graph_to_draw, k=k_value, iterations=100) # 增加迭代次数可能改善布局

        # 绘制节点
        node_colors = ['red' if n == center_node_str else 'lightblue' for n in graph_to_draw.nodes()]
        nx.draw_networkx_nodes(graph_to_draw, pos, node_color=node_colors, node_size=350, alpha=0.9)
        
        # 绘制边
        edges = graph_to_draw.edges(data=True)
        if edges:
            edge_weights = [d.get('weight', 0.5) for _, _, d in edges]
            nx.draw_networkx_edges(graph_to_draw, pos, alpha=0.4, width=1, edge_color=edge_weights, edge_cmap=plt.cm.Blues_r)
        
        # --- 选择性地生成和绘制标签的逻辑 ---
        labels_to_draw = {}  # 存储最终要绘制的标签
        drawn_label_positions = [] # 存储已选择绘制标签的节点位置 (numpy array 形式)
        
        # 定义一个距离阈值，小于此阈值则认为节点过近。
        # 这个值非常敏感，需要根据 k_value, figsize, node_size, font_size 综合调整。
        # 一个可能的起点是 k_value 的一小部分，例如 k_value / 5 或 k_value / 10。
        # 或者一个绝对值，例如 0.05, 0.1, 0.15 (假设pos坐标在-1到1或0到1范围)
        proximity_threshold = 0.12  # <--- 请重点试验和调整这个值！

        # （可选）按节点的重要性（例如度数）排序，优先处理重要节点
        # node_list_for_labeling = sorted(list(graph_to_draw.nodes()), key=lambda n: graph_to_draw.degree(n), reverse=True)
        # 如果不排序，则按默认节点顺序处理
        node_list_for_labeling = list(graph_to_draw.nodes())
        
        # 如果中心节点存在，优先尝试标注它
        if center_node_str and center_node_str in node_list_for_labeling:
            node_list_for_labeling.remove(center_node_str)
            node_list_for_labeling.insert(0, center_node_str)


        for node_id in node_list_for_labeling:
            node_position_np = np.array(pos[node_id]) # 转换为 numpy array 以便计算
            is_too_close = False
            for drawn_pos_np in drawn_label_positions:
                distance = np.linalg.norm(node_position_np - drawn_pos_np) # 使用numpy计算欧氏距离
                if distance < proximity_threshold:
                    is_too_close = True
                    break 
            
            if not is_too_close:
                definition_text = self.isco_to_definition.get(str(node_id), str(node_id))
                # 调整 max_def_len_for_graph 来控制标签的截断长度
                max_def_len_for_graph = 20 # 示例长度，可以根据需要调整
                label_content = definition_text.replace('\n', ' ').replace('\r', '')[:max_def_len_for_graph] 
                
                labels_to_draw[node_id] = label_content
                drawn_label_positions.append(node_position_np) # 存储已标记节点的位置

        print(f"信息: 总节点数 {graph_to_draw.number_of_nodes()}, 经过筛选后将标注 {len(labels_to_draw)} 个节点。")
        if labels_to_draw:
            first_labeled_node = list(labels_to_draw.keys())[0]
            print(f"调试: 第一个被选择标注的节点 '{first_labeled_node}' 的标签是 '{labels_to_draw[first_labeled_node]}'")

        # 使用筛选后的 labels_to_draw 字典来绘制标签
        nx.draw_networkx_labels(graph_to_draw, pos, labels=labels_to_draw, font_size=7) # 减小字体尝试避免重叠
        # --- 选择性标签逻辑结束 ---

        # 标题部分可以保持使用 get_isco_display_label 来提供ISCO码上下文
        title_center_display_label = self.get_isco_display_label(center_node_str) if center_node_str else '全图'
        title_str = f"ISCO代码转换网络 (中心: {title_center_display_label})"
        plt.title(title_str, fontsize=16)
        
        plt.axis('off')
        plt.tight_layout() # 尝试自动调整子图参数以提供一个紧凑的布局
        plt.show()
    
    def analyze_skill_importance(self):
        skill_importance = []
        for skill, isco_codes_with_skill in self.skill_jobs.items():
            isco_display_labels = [self.get_isco_display_label(isco, max_def_len=25) for isco in list(isco_codes_with_skill)[:5]]
            skill_importance.append((skill, {
                'isco_count': len(isco_codes_with_skill),
                'isco_examples': isco_display_labels 
            }))
        skill_importance.sort(key=lambda x: x[1]['isco_count'], reverse=True)
        return skill_importance
    
    def predict_skills_for_occupations(self, target_isco_codes, top_k=1):
        if self.embeddings is None: print("请先训练Node2Vec嵌入"); return None
        predictions = []
        for isco_val in target_isco_codes:
            isco_code_str = str(isco_val)
            if isco_code_str not in self.job_skills: print(f"未在数据中找到ISCO代码 {isco_code_str}。"); continue
            if isco_code_str not in self.embeddings: print(f"ISCO代码 {isco_code_str} 没有对应的嵌入向量。"); continue
            definition = self.isco_to_definition.get(isco_code_str, "未知定义")
            existing_skills, job_embedding = self.job_skills[isco_code_str], self.embeddings[isco_code_str].reshape(1, -1)
            skill_similarities = []
            for skill in self.skill_jobs:
                if skill not in existing_skills:
                    skill_isco_list = [j_isco for j_isco in self.skill_jobs[skill] if j_isco in self.embeddings]
                    if skill_isco_list:
                        avg_skill_embedding = np.mean([self.embeddings[j_isco] for j_isco in skill_isco_list], axis=0).reshape(1, -1)
                        similarity = cosine_similarity(job_embedding, avg_skill_embedding)[0, 0]
                        skill_similarities.append((skill, similarity))
            skill_similarities.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(top_k, len(skill_similarities))):
                predictions.append({
                    'ISCO-Code': isco_code_str,
                    'Definition': definition,
                    'Predicted Skill': skill_similarities[i][0],
                    'Similarity Score': round(skill_similarities[i][1], 4)
                })
        return pd.DataFrame(predictions)
    
    def predict_skills_for_lawyers(self, isco_code='2611', top_k_new=3):
        if self.embeddings is None: print("请先训练Node2Vec嵌入"); return None
        target_isco_str = str(isco_code)
        if target_isco_str not in self.job_skills or target_isco_str not in self.embeddings:
            print(f"ISCO代码 {target_isco_str} 数据不足或无嵌入"); return None
        existing_skills, job_embedding = self.job_skills[target_isco_str], self.embeddings[target_isco_str].reshape(1, -1)
        skill_predictions = [{'skill': s, 'probability': 0.85 + np.random.uniform(0, 0.15), 'type': 'existing'} for s in existing_skills]
        new_skill_similarities = []
        for skill in self.skill_jobs:
            if skill not in existing_skills:
                skill_isco_list = [j_isco for j_isco in self.skill_jobs[skill] if j_isco in self.embeddings]
                if skill_isco_list:
                    avg_skill_embedding = np.mean([self.embeddings[j_isco] for j_isco in skill_isco_list], axis=0).reshape(1, -1)
                    similarity = cosine_similarity(job_embedding, avg_skill_embedding)[0, 0]
                    new_skill_similarities.append((skill, similarity))
        new_skill_similarities.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(top_k_new, len(new_skill_similarities))):
            skill, similarity = new_skill_similarities[i]
            skill_predictions.append({'skill': skill, 'probability': similarity, 'type': 'predicted'})
        return skill_predictions
    
    def visualize_lawyer_predictions(self, predictions, isco_code_for_title='2611'):
        if not predictions:
            print("没有预测结果可供可视化。")
            return

        # --- 第1部分：辅助函数和允许的英文关键词列表 (确保这里包含您需要的英文技能) ---
        import re
        def _contains_chinese(text):
            if not isinstance(text, str): return False
            return bool(re.search(r'[\u4e00-\u9fff]+', text))

        _allowed_english_keywords_lower = [ # 请确保这个列表是最新的，包含您想显示的英文技能
            'excel', 'powerpoint', 'word', 'outlook', 'access', 'visio',
            'sql', 'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 
            'php', 'ruby', 'swift', 'kotlin', 'scala', ' r ', 
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'jquery',
            'git', 'linux', 'windows server', 'macos', 'unix',
            'sap', 'oracle', 'salesforce', 'peoplesoft', 'netsuite',
            'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes',
            'adobe photoshop', 'adobe illustrator', 'adobe indesign', 'adobe premiere', 'adobe after effects',
            'autocad', 'solidworks', 'revit', 'sketchup',
            'jira', 'confluence', 'trello', 'slack', 'zoom',
            'tableau', 'power bi', 'qlik sense', 'google analytics',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'selenium', 
            'cms', 'ceo', 'execution', 'server' # <--- 示例：确保这些在这里
        ]
        # --- 第1部分结束 ---

        df_pred_original = pd.DataFrame(predictions)
        if df_pred_original.empty:
            print("原始预测结果为空。")
            return

        # --- 第2部分：根据中文或特定英文关键词筛选技能 ---
        def skill_passes_filter(skill_name):
            if not isinstance(skill_name, str): return False
            if _contains_chinese(skill_name): return True
            skill_name_lower = skill_name.lower()
            for keyword in _allowed_english_keywords_lower:
                if keyword == ' r ' and keyword in skill_name_lower: return True
                elif keyword != ' r ' and keyword in skill_name_lower: return True
            return False

        mask = df_pred_original['skill'].apply(skill_passes_filter)
        df_filtered = df_pred_original[mask]
        # --- 第2部分结束 ---

        if df_filtered.empty:
            print(f"没有符合筛选条件（中文或特定英文关键词）的技能可供显示在ISCO {isco_code_for_title}的条状图中。")
            return
            
        # --- 修改点：为现有技能和预测技能分别保证显示名额 ---
        num_total_to_show = 20 # 您希望图表上总共显示的技能数量
        num_predicted_to_show_guaranteed = 5 # 希望至少保证显示的预测技能数量 (如果数量足够的话)
        
        # 筛选出预测技能并排序
        df_predicted_skills_filtered = df_filtered[df_filtered['type'] == 'predicted'] \
            .sort_values(by='probability', ascending=False)
        
        # 筛选出现有技能并排序
        df_existing_skills_filtered = df_filtered[df_filtered['type'] == 'existing'] \
            .sort_values(by='probability', ascending=False)

        # 取预定数量的预测技能
        df_top_predicted = df_predicted_skills_filtered.head(num_predicted_to_show_guaranteed)
        
        # 计算还需要多少现有技能来填满总数
        num_existing_to_show = num_total_to_show - len(df_top_predicted)
        df_top_existing = df_existing_skills_filtered.head(max(0, num_existing_to_show)) # max(0,..)确保不为负

        # 合并选出的技能
        df_pred = pd.concat([df_top_existing, df_top_predicted])
        
        # 再次排序以确保图表整体按概率（或您希望的其他顺序）排列，并确保总数不超过预设
        # 如果您希望预测的技能即使概率低也显示，可以先 concat，再对整体取 top N
        # 但为了确保两类都有，上面的做法更好。最终排序可以按 probability。
        df_pred = df_pred.sort_values(by='probability', ascending=False).head(num_total_to_show)
        # --- 修改点结束 ---


        if df_pred.empty: 
            print(f"结合筛选和保证名额策略后，没有技能可供显示在ISCO {isco_code_for_title}的条状图中。")
            return

        skills = df_pred['skill'].tolist()
        probabilities = df_pred['probability'].tolist()
        colors = ['green' if t == 'existing' else 'blue' for t in df_pred['type'].tolist()]
    
        
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(skills)), probabilities, color=colors)
        
        plt.yticks(range(len(skills)), skills, fontsize=10) # 这里的 'skills' 列表已经是筛选后的
        plt.gca().invert_yaxis()
        
        definition_for_title = self.isco_to_definition.get(str(isco_code_for_title), "该职业")
        plt.xlabel(f'对ISCO {isco_code_for_title} ({definition_for_title}) 的技能预测', fontsize=12)
        
        # --- 第3部分：更新图表标题 (可选) ---
        plt.title(f'模拟Figure 5: ISCO组 {isco_code_for_title} 的技能预测 (仅中文/选定英文技能)', fontsize=14)
        # --- 第3部分结束 ---

        plt.xlim(0, 1.05)
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        green_patch = plt.Rectangle((0,0),1,1,fc="green",label='现有技能')
        blue_patch = plt.Rectangle((0,0),1,1,fc="blue",label='预测新技能')
        plt.legend(handles=[green_patch, blue_patch], loc='lower right', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_jaccard_statistics(self):
        skill_distances, isco_code_distances = [], []
        skills_list = list(self.skill_jobs.keys())
        for i in range(len(skills_list)):
            for j in range(i + 1, len(skills_list)):
                s1_iscos, s2_iscos = self.skill_jobs[skills_list[i]], self.skill_jobs[skills_list[j]]
                if s1_iscos and s2_iscos:
                    union = len(s1_iscos | s2_iscos)
                    if union > 0: skill_distances.append(1 - (len(s1_iscos & s2_iscos) / union))
        isco_codes_list = list(self.job_skills.keys())
        for i in range(len(isco_codes_list)):
            for j in range(i + 1, len(isco_codes_list)):
                isco_code_distances.append(self.calculate_jaccard_distance(isco_codes_list[i], isco_codes_list[j]))
        def calculate_stats(distances_list):
            if not distances_list: return {'count':0,'mean':0,'std':0,'min':0,'25%':0,'50%':0,'75%':0,'max':0}
            arr = np.array(distances_list)
            return {'count':len(arr),'mean':np.mean(arr),'std':np.std(arr),'min':np.min(arr),'25%':np.percentile(arr,25),
                    '50%':np.percentile(arr,50),'75%':np.percentile(arr,75),'max':np.max(arr)}
        stats_data = {'Skill_Skill(based on shared ISCOs)': calculate_stats(skill_distances),
                      'ISCO_ISCO(based on shared Skills)': calculate_stats(isco_code_distances),
                      'Total_Distances': calculate_stats(skill_distances + isco_code_distances)}
        return pd.DataFrame(stats_data).T.rename_axis('Distribution Type')
    
    def export_results(self, output_file='isco_definition_transition_analysis.json'):
        results = {
            'total_isco_codes': len(self.job_skills), 'total_skills': len(self.skill_jobs),
            'graph_info': {'nodes': self.career_graph.number_of_nodes() if self.career_graph else 0,
                           'edges': self.career_graph.number_of_edges() if self.career_graph else 0},
            'top_skills_by_isco_count': self.analyze_skill_importance()[:20],
            'sample_isco_transitions': []}
        for isco in list(self.job_skills.keys())[:5]:
            recommendations = self.recommend_career_transitions(isco, top_k=3)
            results['sample_isco_transitions'].append({
                'from_isco_label': self.get_isco_display_label(isco),
                'recommendations': recommendations})
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"结果已导出到 {output_file}")

# --- 使用示例 ---
def main():
    planner = CareerPathPlanner('simplified_jobs_skills_with_definition.csv') # 确保CSV文件名和路径正确
    planner.load_data()
    planner.build_career_transition_graph(distance_threshold=0.8)
    
    if planner.career_graph and planner.career_graph.number_of_nodes() > 0:
        planner.train_node2vec_embeddings(dimensions=256, walk_length=20, num_walks=100, p=1, q=1)
    else:
        print("图为空或未构建，跳过Node2Vec嵌入训练。")

    if planner.embeddings:
        print("\n=== Table 4 (模拟): 预测ISCO代码的新技能 ===")
        target_iscos = [code for code in ['3323', '2431', '2166', '1221', '3341'] if code in planner.job_skills]
        if target_iscos:
            predictions_df = planner.predict_skills_for_occupations(target_iscos, top_k=3)
            if predictions_df is not None and not predictions_df.empty:
                print(predictions_df.to_string(index=False))
                predictions_df.to_csv('predicted_new_skills_for_iscos_v2.csv', index=False, encoding='utf-8-sig')
            else: print("未能生成技能预测。")
        else: print(f"提供的目标ISCO代码均不在数据中，无法预测。")
        
        print("\n=== Figure 5 (模拟): ISCO 2611 (例如律师) 技能预测 ===")
        # 您想要为其生成 Figure 5 模拟的 ISCO 代码
        isco_fig5 = '2611'  # 您可以更改为数据中存在的其他 ISCO 代码

        print(f"\n--- 正在为 ISCO {isco_fig5} 生成 Figure 5 模拟 ---") # 增加一个开始的打印

        if planner.embeddings: # 首先检查 embeddings 是否已训练
            # 然后检查指定的 isco_fig5 是否有效 (存在于技能数据和嵌入中)
            if isco_fig5 in planner.job_skills and isco_fig5 in planner.embeddings:
                print(f"信息: ISCO {isco_fig5} 的数据和嵌入均可用。")

                # 调用预测方法，可以尝试增加 top_k_new 以获取更多潜在的蓝色条技能
                lawyer_preds_list = planner.predict_skills_for_lawyers(isco_code=isco_fig5, top_k_new=10) # 修改点1: 增加 top_k_new (原为5)

                # --- 修改点2: 增加详细的调试打印信息 ---
                if lawyer_preds_list:
                    num_total_preds = len(lawyer_preds_list)
                    num_blue_preds = sum(1 for p in lawyer_preds_list if p['type'] == 'predicted')
                    print(f"信息: predict_skills_for_lawyers 为 ISCO {isco_fig5} 返回了 {num_total_preds} 条技能信息。")
                    print(f"信息: 其中 'predicted' (蓝色条) 类型的技能有 {num_blue_preds} 条。")

                    if num_blue_preds == 0:
                        print(f"警告: 没有为 ISCO {isco_fig5} 预测出新的技能 (没有蓝色条数据)。")
                    else:
                        print(f"信息: 前几个 'predicted' 技能的详情 (最多显示5条):")
                        count = 0
                        for p_item in lawyer_preds_list:
                            if p_item['type'] == 'predicted':
                                print(f"  - 技能名: '{p_item['skill']}', 预测概率 (相似度): {p_item['probability']:.4f}")
                                count += 1
                                if count >= 10:
                                    break
                else:
                    print(f"警告: predict_skills_for_lawyers 方法没有为 ISCO {isco_fig5} 返回任何技能信息 (结果为 None 或空列表)。")
                # --- 调试打印结束 ---

                # 只有当 lawyer_preds_list 非空时才进行可视化
                if lawyer_preds_list:
                    planner.visualize_lawyer_predictions(lawyer_preds_list, isco_code_for_title=isco_fig5)
                else:
                    print(f"由于 predict_skills_for_lawyers 未返回有效数据，无法为 ISCO {isco_fig5} 生成图表。")
            
            else: # isco_fig5 数据不足或无嵌入
                missing_details = []
                if isco_fig5 not in planner.job_skills:
                    missing_details.append("技能数据 (job_skills)")
                if isco_fig5 not in planner.embeddings:
                    missing_details.append("嵌入向量 (embeddings)")
                print(f"错误: ISCO代码 {isco_fig5} 数据不足 (缺少: {', '.join(missing_details)})，无法生成Figure 5模拟。")
        
        else: # planner.embeddings 为空
            print("\n错误: Node2Vec嵌入未训练或训练失败，跳过依赖嵌入的分析（包括Figure 5模拟）。")

    print("\n=== Table 5 (模拟): 杰卡德距离分布统计 ===")
    jaccard_stats = planner.calculate_jaccard_statistics()
    print(jaccard_stats)
    jaccard_stats.to_csv('jaccard_statistics_isco_based_v2.csv', encoding='utf-8-sig')
    
    all_iscos = list(planner.job_skills.keys())
    if len(all_iscos) >= 2:
        source, target = all_iscos[0], all_iscos[min(1, len(all_iscos)-1)]
        if source == target and len(all_iscos)>1 : target = all_iscos[min(2, len(all_iscos)-1)] if len(all_iscos) > 2 else all_iscos[0] # simple distinct logic

        if source and target and source != target:
            print(f"\n查找从 {planner.get_isco_display_label(source)} 到 {planner.get_isco_display_label(target)} 的ISCO转换路径:")
            path, dist = planner.find_shortest_career_path(source, target)
            if path: print(f"基于Jaccard距离: {' -> '.join([planner.get_isco_display_label(p) for p in path])} (总距离: {dist:.4f})")
            if planner.embeddings:
                path_emb, dist_emb = planner.find_shortest_career_path(source, target, use_embeddings=True)
                if path_emb: print(f"基于嵌入距离: {' -> '.join([planner.get_isco_display_label(p) for p in path_emb])} (总距离: {dist_emb:.4f})")
        else: print("\n未能选择不同的源和目标ISCO进行路径查找。")
    
    if all_iscos:
        current_isco = all_iscos[0]
        print(f"\n为ISCO {planner.get_isco_display_label(current_isco)} 推荐的转换:")
        recs = planner.recommend_career_transitions(current_isco, top_k=5, max_distance=0.85)
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {planner.get_isco_display_label(rec['isco'])} (定义: {rec['definition'][:50]}...)") # Show part of full definition
            print(f"   距离: {rec['distance']:.4f}, 共享技能数: {rec['skill_overlap']}")
    
    print("\n最重要的技能（按关联ISCO代码数）:")
    for skill, info in planner.analyze_skill_importance()[:10]:
        print(f"- \"{skill}\": {info['isco_count']} ISCO代码 (示例: {', '.join(info['isco_examples'])})")
    
    if all_iscos and planner.career_graph and planner.career_graph.number_of_nodes() > 0:
        center_node_viz = '2611' if '2611' in planner.career_graph else (all_iscos[0] if all_iscos[0] in planner.career_graph else None)
        if center_node_viz : planner.visualize_career_network(center_isco=center_node_viz, radius=1)
        else: print("无可用于中心节点可视化的有效ISCO代码。")
            
    planner.export_results()

if __name__ == "__main__":
    main()

'''
wyh@MacBook-Pro-2 统计计算 % /usr/local/bin/python3 /Users/wyh/Desktop/统计计算/lunwen/技能职业相似度.py
加载数据...
加载了 417 个不同的ISCO代码 (这些ISCO代码有关联的技能)
总共有 684 个不同的技能
为 433 个ISCO代码存储了定义 (部分可能为其ISCO码自身作为后备)
--- 抽样检查已加载的ISCO定义 ---
  ISCO 2643: '笔译员、口译员及其他语言专家'
  ISCO 1330: '信息和通信技术服务经理'
  ISCO 9411: '快餐制作员'
  ISCO 8142: '塑料制品机器操作员'
  ISCO 1315: '0'
--- 抽样检查结束 ---
构建ISCO代码转换图 (距离阈值: 0.8)...
构建完成: 417 个ISCO节点, 21234 条边
训练Node2Vec嵌入 (针对ISCO代码节点)...
Computing transition probabilities: 100%|████████████████████████████████████████████████████████████████████| 417/417 [00:05<00:00, 70.64it/s]
Generating walks (CPU: 1): 100%|███████████████████████████████████████████████████████████████████████████████| 25/25 [00:01<00:00, 18.53it/s]
Generating walks (CPU: 2): 100%|███████████████████████████████████████████████████████████████████████████████| 25/25 [00:01<00:00, 18.46it/s]
Generating walks (CPU: 3): 100%|███████████████████████████████████████████████████████████████████████████████| 25/25 [00:01<00:00, 18.78it/s]
Generating walks (CPU: 4): 100%|███████████████████████████████████████████████████████████████████████████████| 25/25 [00:01<00:00, 19.07it/s]
嵌入训练完成，维度: 256

=== Table 4 (模拟): 预测ISCO代码的新技能 ===
ISCO-Code  Definition Predicted Skill  Similarity Score
     3323          买家             App            0.8199
     3323          买家        国家相关法律法规            0.8177
     3323          买家          企业文化建设            0.8123
     2431 广告与市场营销专业人士            with            0.8237
     2431 广告与市场营销专业人士            send            0.8223
     2431 广告与市场营销专业人士         Windows            0.8215
     2166   图形与多媒体设计师          Design            0.8017
     2166   图形与多媒体设计师          leader            0.7958
     2166   图形与多媒体设计师           Major            0.7956
     1221   销售与市场营销经理           sales            0.8402
     1221   销售与市场营销经理             out            0.8376
     1221   销售与市场营销经理            开发进度            0.8350
     3341       办公室主管      进出口业务流程,熟悉            0.8308
     3341       办公室主管            媒体推广            0.8225
     3341       办公室主管      到的客户反馈,向公司            0.8177

=== Figure 5 (模拟): ISCO 2611 (例如律师) 技能预测 ===

--- 正在为 ISCO 2611 生成 Figure 5 模拟 ---
信息: ISCO 2611 的数据和嵌入均可用。
信息: predict_skills_for_lawyers 为 ISCO 2611 返回了 102 条技能信息。
信息: 其中 'predicted' (蓝色条) 类型的技能有 10 条。
信息: 前几个 'predicted' 技能的详情 (最多显示5条):
  - 技能名: '洞察力', 预测概率 (相似度): 0.7029
  - 技能名: '市场分析', 预测概率 (相似度): 0.6969
  - 技能名: '管理层', 预测概率 (相似度): 0.6958
  - 技能名: '市场营销', 预测概率 (相似度): 0.6957
  - 技能名: '会计证书', 预测概率 (相似度): 0.6945
  - 技能名: '数据分析', 预测概率 (相似度): 0.6936
  - 技能名: '置业咨询', 预测概率 (相似度): 0.6926
  - 技能名: '团队建设', 预测概率 (相似度): 0.6923
  - 技能名: 'VIP', 预测概率 (相似度): 0.6918
  - 技能名: '市场拓展', 预测概率 (相似度): 0.6895
2025-05-30 18:25:43.615 Python[77368:25583413] +[IMKClient subclass]: chose IMKClient_Legacy
2025-05-30 18:25:43.615 Python[77368:25583413] +[IMKInputSession subclass]: chose IMKInputSession_Legacy
2025-05-30 18:26:06.780 Python[77368:25583413] The class 'NSSavePanel' overrides the method identifier.  This method is implemented by class 'NSWindow'

=== Table 5 (模拟): 杰卡德距离分布统计 ===
                                       count      mean       std  min       25%       50%       75%  max
Distribution Type                                                                                       
Skill_Skill(based on shared ISCOs)  233586.0  0.913791  0.105361  0.0  0.868852  0.950000  1.000000  1.0
ISCO_ISCO(based on shared Skills)    86736.0  0.866750  0.105185  0.0  0.800000  0.885714  0.951628  1.0
Total_Distances                     320322.0  0.901053  0.107368  0.0  0.845399  0.933333  0.995781  1.0

查找从 1111 (立法者) 到 1112 (高级政府官员) 的ISCO转换路径:
基于Jaccard距离: 1111 (立法者) -> 1112 (高级政府官员) (总距离: 0.6364)
基于嵌入距离: 1111 (立法者) -> 1112 (高级政府官员) (总距离: 0.5826)

为ISCO 1111 (立法者) 推荐的转换:
1. 1120 (董事总经理与首席执行官) (定义: 董事总经理与首席执行官...)
   距离: 0.6078, 共享技能数: 20
2. 1112 (高级政府官员) (定义: 高级政府官员...)
   距离: 0.6364, 共享技能数: 16
3. 1341 (儿童保育服务经理) (定义: 儿童保育服务经理...)
   距离: 0.7049, 共享技能数: 18
4. 1222 (广告与公关经理) (定义: 广告与公关经理...)
   距离: 0.7209, 共享技能数: 12
5. 1412 (餐厅经理) (定义: 餐厅经理...)
   距离: 0.7297, 共享技能数: 10

最重要的技能（按关联ISCO代码数）:
- "责任心": 331 ISCO代码 (示例: 2643 (笔译员、口译员及其他语言专家), 1330 (信息和通信技术服务经理), 9411 (快餐制作员), 1315 (0), 1212 (人力资源经理))
- "沟通能力": 318 ISCO代码 (示例: 2643 (笔译员、口译员及其他语言专家), 1330 (信息和通信技术服务经理), 9411 (快餐制作员), 1315 (0), 4199 (0))
- "团队合作": 294 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 7321 (印前技术员), 2441 (0), 1330 (信息和通信技术服务经理))
- "吃苦耐劳": 290 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 7321 (印前技术员), 2441 (0), 1330 (信息和通信技术服务经理))
- "大专": 269 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 2441 (0), 1330 (信息和通信技术服务经理), 2120 (数学家、精算师与统计学家))
- "协调能力": 240 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 7321 (印前技术员), 2441 (0), 1330 (信息和通信技术服务经理))
- "销售": 237 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 2441 (0), 1330 (信息和通信技术服务经理), 2120 (数学家、精算师与统计学家))
- "客户服务": 221 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 2441 (0), 1330 (信息和通信技术服务经理), 2120 (数学家、精算师与统计学家))
- "本科": 212 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 2441 (0), 1330 (信息和通信技术服务经理), 2120 (数学家、精算师与统计学家))
- "英语": 209 ISCO代码 (示例: 1323 (施工经理), 2643 (笔译员、口译员及其他语言专家), 2441 (0), 1330 (信息和通信技术服务经理), 2120 (数学家、精算师与统计学家))
可视化以 2611 (律师) 为中心，半径为 1 的子图 (159 节点)。'''