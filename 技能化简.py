import pandas as pd
import json
import numpy as np
from collections import Counter, defaultdict
import re
from typing import Dict, List, Set, Tuple
from fuzzywuzzy import fuzz
import jieba

class SkillSimplifier:
    def __init__(self, csv_file: str, target_skill_count: int = 900):
        """
        技能化简处理器
        
        Args:
            csv_file: 包含技能提取结果的CSV文件
            target_skill_count: 目标技能数量
        """
        self.csv_file = csv_file
        self.target_skill_count = target_skill_count
        
        # 技能统计
        self.skill_frequency = Counter()
        self.skill_job_mapping = defaultdict(set)  # 技能->使用该技能的职位ISCO集合
        
        # 化简规则
        self.merge_rules = {}  # 合并规则：原技能 -> 标准技能
        self.filtered_skills = set()  # 被过滤掉的技能
        self.final_skills = {}  # 最终技能集合：技能名 -> 频次
        
        print(f"技能化简器初始化完成，目标：{target_skill_count}种技能")
    
    def load_and_analyze_skills(self) -> Dict:
        """加载并分析技能分布"""
        print("正在加载和分析技能...")
        
        df = pd.read_csv(self.csv_file)
        total_jobs = len(df)
        
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"分析进度: {idx}/{total_jobs}")
            
            # 获取ISCO代码
            isco_code = row.get('ISCO_4_Digit_Code_Gemini')
            if pd.isna(isco_code):
                continue
            
            # 获取标准化技能
            normalized_skills_str = row.get('标准化技能', '[]')
            try:
                skills = json.loads(normalized_skills_str)
            except:
                continue
            
            # 统计技能频次和技能-职业映射
            for skill in skills:
                if skill and len(skill.strip()) > 1:
                    skill = skill.strip()
                    self.skill_frequency[skill] += 1
                    self.skill_job_mapping[skill].add(int(isco_code))
        
        print(f"✓ 技能分析完成:")
        print(f"  总职位数: {total_jobs:,}")
        print(f"  原始技能数: {len(self.skill_frequency):,}")
        print(f"  技能总频次: {sum(self.skill_frequency.values()):,}")
        
        return {
            'total_jobs': total_jobs,
            'original_skill_count': len(self.skill_frequency),
            'total_skill_frequency': sum(self.skill_frequency.values())
        }
    
    def define_core_skill_categories(self) -> Dict[str, List[str]]:
        """定义核心技能类别和标准词汇"""
        return {
            # 编程技术类
            'programming': {
                'Python': ['Python', 'python', 'PYTHON'],
                'Java': ['Java', 'java', 'JAVA', 'J2EE', 'Spring', 'SpringBoot'],
                'JavaScript': ['JavaScript', 'JS', 'js', 'jQuery', 'React', 'Vue', 'Angular'],
                'C++': ['C++', 'C/C++', 'CPP'],
                'C#': ['C#', '.NET', 'ASP.NET'],
                'PHP': ['PHP', 'php'],
                'Go': ['Go', 'Golang'],
                'SQL': ['SQL', 'MySQL', 'PostgreSQL', 'Oracle', 'SQL Server'],
                'HTML/CSS': ['HTML', 'CSS', 'HTML5', 'CSS3'],
                '移动开发': ['Android', 'iOS', 'APP开发', '移动开发', 'Swift', 'Kotlin'],
                '数据库': ['数据库', '数据库设计', '数据库管理', 'MongoDB', 'Redis'],
                '前端开发': ['前端', '前端开发', 'Web开发', '网页设计'],
                '后端开发': ['后端', '后端开发', '服务器开发'],
                '算法': ['算法', '数据结构', '算法设计']
            },
            
            # 办公软件类
            'office_software': {
                'Excel': ['Excel', 'excel', 'EXCEL', 'Excel表格'],
                'Word': ['Word', 'word', 'Word文档'],
                'PowerPoint': ['PowerPoint', 'PPT', 'ppt', '幻灯片'],
                'Office套件': ['Office', 'office', 'Microsoft Office', 'WPS'],
                'Photoshop': ['Photoshop', 'PS', 'ps', 'Adobe Photoshop'],
                'AutoCAD': ['AutoCAD', 'CAD', 'cad', '制图软件'],
                'SAP': ['SAP', 'sap'],
                '用友': ['用友', '用友U8', 'U8'],
                '金蝶': ['金蝶', 'K3', 'k3']
            },
            
            # 沟通表达类
            'communication': {
                '沟通能力': ['沟通能力', '交流能力', '表达能力', '沟通技巧', '交流技巧', '沟通'],
                '演讲能力': ['演讲能力', '汇报能力', '公众演讲', '口齿清晰'],
                '写作能力': ['写作能力', '文字功底', '公文写作', '文案'],
                '倾听能力': ['倾听能力', '善于倾听'],
                '谈判能力': ['谈判能力', '商务谈判', '谈判技巧']
            },
            
            # 管理领导类
            'leadership': {
                '领导力': ['领导力', '领导能力', '领导技能'],
                '团队管理': ['团队管理', '人员管理', '管理能力', '管理技能'],
                '项目管理': ['项目管理', 'PMP', '项目管理能力'],
                '决策能力': ['决策能力', '决策'],
                '组织能力': ['组织能力', '组织协调', '组织'],
                '计划能力': ['计划能力', '规划能力', '计划'],
                '执行力': ['执行力', '执行能力', '执行']
            },
            
            # 团队协作类
            'teamwork': {
                '团队合作': ['团队合作', '协作能力', '配合能力', '团队精神', '合作能力'],
                '协调能力': ['协调能力', '协调', '沟通协调'],
                '服务意识': ['服务意识', '客户服务意识', '服务精神']
            },
            
            # 分析思维类
            'analytical': {
                '分析能力': ['分析能力', '分析技能', '数据分析能力'],
                '逻辑思维': ['逻辑思维', '逻辑思维能力', '逻辑分析'],
                '问题解决': ['问题解决', '解决问题', '问题解决能力'],
                '数据分析': ['数据分析', '统计分析', '数据处理'],
                '财务分析': ['财务分析', '财务数据分析'],
                '市场分析': ['市场分析', '市场调研', '市场研究']
            },
            
            # 个人素质类
            'personal_qualities': {
                '学习能力': ['学习能力', '快速学习', '持续学习', '自学能力'],
                '适应能力': ['适应能力', '应变能力', '灵活性'],
                '抗压能力': ['抗压能力', '抗压', '承受压力'],
                '创新能力': ['创新能力', '创新思维', '创新'],
                '责任心': ['责任心', '责任感', '敬业精神', '职业道德'],
                '主动性': ['主动性', '积极主动', '自我激励'],
                '时间管理': ['时间管理', '时间管理能力'],
                '细心': ['细心', '认真细致', '注意细节'],
                '耐心': ['耐心', '耐心细致']
            },
            
            # 语言能力类
            'languages': {
                '英语': ['英语', '英文', '英语能力', '英语听说读写'],
                '日语': ['日语', '日文', '日语能力'],
                '韩语': ['韩语', '韩文', '韩语能力'],
                '普通话': ['普通话', '标准普通话'],
                '英语证书': ['CET-4', 'CET-6', '四级', '六级', '雅思', '托福']
            },
            
            # 专业证书类
            'certificates': {
                '会计证书': ['CPA', '注册会计师', '中级会计师', '初级会计师', '会计从业资格证'],
                '工程证书': ['建造师', '造价师', '工程师', '注册工程师'],
                '驾驶证': ['驾驶证', '驾照', 'C照', 'B照', 'A照'],
                '其他证书': ['PMP', '教师资格证', '律师资格证', '执业药师']
            },
            
            # 业务技能类
            'business_skills': {
                '销售': ['销售', '销售技巧', '销售能力'],
                '市场营销': ['市场营销', '营销', '营销策划'],
                '客户服务': ['客户服务', '客户关系', '客户管理'],
                '人力资源': ['人力资源', '人力资源管理', '招聘', 'HR'],
                '财务管理': ['财务管理', '财务', '会计', '成本控制'],
                '采购': ['采购', '采购管理', '供应商管理'],
                '物流': ['物流', '仓储', '供应链', '配送'],
                '质量管理': ['质量管理', 'ISO', '质量控制', '品质管理'],
                '生产管理': ['生产管理', '生产计划', '现场管理'],
                '运营管理': ['运营管理', '运营', '业务运营']
            }
        }
    
    def apply_merge_rules(self, core_categories: Dict) -> Dict:
        """应用合并规则"""
        print("正在应用技能合并规则...")
        
        merge_count = 0
        
        # 为每个核心类别建立合并规则
        for category, skills_dict in core_categories.items():
            for standard_skill, variants in skills_dict.items():
                for variant in variants:
                    if variant in self.skill_frequency:
                        if variant != standard_skill:
                            self.merge_rules[variant] = standard_skill
                            merge_count += 1
                        else:
                            # 标准技能直接保留
                            self.merge_rules[variant] = standard_skill
        
        # 使用模糊匹配找到更多相似技能
        fuzzy_merge_count = self._apply_fuzzy_merge()
        
        print(f"✓ 合并规则应用完成:")
        print(f"  精确匹配合并: {merge_count}个技能")
        print(f"  模糊匹配合并: {fuzzy_merge_count}个技能")
        print(f"  总合并规则: {len(self.merge_rules)}个")
        
        return {
            'exact_merges': merge_count,
            'fuzzy_merges': fuzzy_merge_count,
            'total_merge_rules': len(self.merge_rules)
        }
    
    def _apply_fuzzy_merge(self, similarity_threshold: int = 85) -> int:
        """使用模糊匹配合并相似技能"""
        
        # 获取已有的标准技能
        standard_skills = set(self.merge_rules.values())
        unmatched_skills = [skill for skill in self.skill_frequency.keys() 
                          if skill not in self.merge_rules]
        
        fuzzy_merge_count = 0
        
        for skill in unmatched_skills:
            best_match = None
            best_score = 0
            
            # 与所有标准技能比较
            for standard_skill in standard_skills:
                score = fuzz.ratio(skill, standard_skill)
                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = standard_skill
            
            if best_match:
                self.merge_rules[skill] = best_match
                fuzzy_merge_count += 1
        
        return fuzzy_merge_count
    
    def filter_low_frequency_skills(self, min_job_count: int = 5, min_frequency: int = 10) -> Dict:
        """过滤低频技能"""
        print(f"正在过滤低频技能（最少{min_job_count}个职位，最少出现{min_frequency}次）...")
        
        filtered_count = 0
        
        for skill, frequency in self.skill_frequency.items():
            job_count = len(self.skill_job_mapping[skill])
            
            # 过滤条件：职位数太少或频次太低
            if job_count < min_job_count or frequency < min_frequency:
                self.filtered_skills.add(skill)
                filtered_count += 1
        
        print(f"✓ 低频技能过滤完成:")
        print(f"  过滤掉的技能数: {filtered_count}")
        print(f"  剩余技能数: {len(self.skill_frequency) - filtered_count}")
        
        return {
            'filtered_count': filtered_count,
            'remaining_count': len(self.skill_frequency) - filtered_count
        }
    
    def build_final_skill_set(self) -> Dict:
        """构建最终技能集合"""
        print("正在构建最终技能集合...")
        
        # 应用合并规则并过滤
        merged_frequency = Counter()
        
        for skill, frequency in self.skill_frequency.items():
            # 跳过被过滤的技能
            if skill in self.filtered_skills:
                continue
            
            # 应用合并规则
            final_skill = self.merge_rules.get(skill, skill)
            merged_frequency[final_skill] += frequency
        
        # 如果技能数量仍然超过目标，进一步过滤
        if len(merged_frequency) > self.target_skill_count:
            # 按频次排序，保留前N个
            top_skills = merged_frequency.most_common(self.target_skill_count)
            self.final_skills = dict(top_skills)
            
            additional_filtered = len(merged_frequency) - self.target_skill_count
            print(f"  额外过滤低频技能: {additional_filtered}个")
        else:
            self.final_skills = dict(merged_frequency)
        
        print(f"✓ 最终技能集合构建完成:")
        print(f"  最终技能数: {len(self.final_skills)}")
        print(f"  技能总频次: {sum(self.final_skills.values()):,}")
        
        return {
            'final_skill_count': len(self.final_skills),
            'final_total_frequency': sum(self.final_skills.values())
        }
    
    def analyze_simplification_results(self) -> Dict:
        """分析化简结果"""
        print("正在分析化简结果...")
        
        original_count = len(self.skill_frequency)
        final_count = len(self.final_skills)
        reduction_rate = (original_count - final_count) / original_count * 100
        
        # 分析技能分类分布
        category_distribution = defaultdict(int)
        core_categories = self.define_core_skill_categories()
        
        for skill in self.final_skills.keys():
            found_category = False
            for category, skills_dict in core_categories.items():
                if skill in skills_dict.keys():
                    category_distribution[category] += 1
                    found_category = True
                    break
            if not found_category:
                category_distribution['其他'] += 1
        
        # 找出最重要的技能
        top_skills = sorted(self.final_skills.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print(f"📊 化简结果分析:")
        print(f"  原始技能数: {original_count:,}")
        print(f"  最终技能数: {final_count:,}")
        print(f"  压缩率: {reduction_rate:.1f}%")
        
        print(f"\n📈 技能分类分布:")
        for category, count in sorted(category_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}个技能")
        
        print(f"\n🏆 前20个最重要技能:")
        for i, (skill, freq) in enumerate(top_skills, 1):
            print(f"  {i:2d}. {skill}: {freq:,}次")
        
        return {
            'original_count': original_count,
            'final_count': final_count,
            'reduction_rate': reduction_rate,
            'category_distribution': dict(category_distribution),
            'top_skills': top_skills
        }
    
    def generate_simplified_csv(self, output_file: str) -> str:
        """生成化简后的CSV文件"""
        print(f"正在生成化简后的CSV文件: {output_file}")
        
        df = pd.read_csv(self.csv_file)
        
        # 处理每一行的技能
        simplified_count = 0
        
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"处理进度: {idx}/{len(df)}")
            
            # 获取原始技能
            normalized_skills_str = row.get('标准化技能', '[]')
            try:
                original_skills = json.loads(normalized_skills_str)
            except:
                continue
            
            # 应用化简规则
            simplified_skills = []
            for skill in original_skills:
                if skill in self.filtered_skills:
                    continue  # 跳过被过滤的技能
                
                # 应用合并规则
                final_skill = self.merge_rules.get(skill, skill)
                
                # 检查是否在最终技能集合中
                if final_skill in self.final_skills:
                    simplified_skills.append(final_skill)
            
            # 去重并更新
            simplified_skills = list(set(simplified_skills))
            df.at[idx, '标准化技能'] = json.dumps(simplified_skills, ensure_ascii=False)
            df.at[idx, '技能数量'] = len(simplified_skills)
            
            if len(simplified_skills) != len(original_skills):
                simplified_count += 1
        
        # 保存结果
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 化简后CSV文件已生成:")
        print(f"  输出文件: {output_file}")
        print(f"  处理记录数: {len(df):,}")
        print(f"  技能变化记录数: {simplified_count:,}")
        
        return output_file
    
    def save_simplification_report(self, output_file: str = 'skill_simplification_report.json'):
        """保存化简报告"""
        report = {
            'original_skill_count': len(self.skill_frequency),
            'final_skill_count': len(self.final_skills),
            'merge_rules': self.merge_rules,
            'filtered_skills': list(self.filtered_skills),
            'final_skills': self.final_skills,
            'top_skills': sorted(self.final_skills.items(), key=lambda x: x[1], reverse=True)[:50]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 化简报告已保存: {output_file}")

def main():
    """主函数：完整的技能化简流程"""
    print("=" * 80)
    print("技能化简处理系统")
    print("=" * 80)
    
    # 初始化化简器
    simplifier = SkillSimplifier(
        csv_file='all_jobs_skills_extracted.csv',
        target_skill_count=900
    )
    
    # 步骤1: 分析原始技能分布
    print("\n" + "="*60)
    print("步骤1: 分析原始技能分布")
    print("="*60)
    analysis_stats = simplifier.load_and_analyze_skills()
    
    # 步骤2: 定义并应用合并规则
    print("\n" + "="*60)
    print("步骤2: 应用技能合并规则")
    print("="*60)
    core_categories = simplifier.define_core_skill_categories()
    merge_stats = simplifier.apply_merge_rules(core_categories)
    
    # 步骤3: 过滤低频技能
    print("\n" + "="*60)
    print("步骤3: 过滤低频技能")
    print("="*60)
    filter_stats = simplifier.filter_low_frequency_skills(
        min_job_count=5,    # 至少5个职位需要
        min_frequency=10    # 至少出现10次
    )
    
    # 步骤4: 构建最终技能集合
    print("\n" + "="*60)
    print("步骤4: 构建最终技能集合")
    print("="*60)
    final_stats = simplifier.build_final_skill_set()
    
    # 步骤5: 分析化简结果
    print("\n" + "="*60)
    print("步骤5: 分析化简结果")
    print("="*60)
    result_analysis = simplifier.analyze_simplification_results()
    
    # 步骤6: 生成化简后的CSV
    print("\n" + "="*60)
    print("步骤6: 生成化简后的CSV文件")
    print("="*60)
    simplified_csv = simplifier.generate_simplified_csv('simplified_jobs_skills.csv')
    
    # 步骤7: 保存化简报告
    print("\n" + "="*60)
    print("步骤7: 保存化简报告")
    print("="*60)
    simplifier.save_simplification_report()
    
    # 生成总结
    print("\n" + "="*80)
    print("🎉 技能化简处理完成！")
    print("="*80)
    
    print(f"\n📊 化简效果总结:")
    print(f"  原始技能数: {analysis_stats['original_skill_count']:,}")
    print(f"  最终技能数: {final_stats['final_skill_count']:,}")
    print(f"  压缩率: {result_analysis['reduction_rate']:.1f}%")
    print(f"  合并规则数: {merge_stats['total_merge_rules']:,}")
    print(f"  过滤技能数: {filter_stats['filtered_count']:,}")
    
    print(f"\n📁 输出文件:")
    print(f"  • simplified_jobs_skills.csv - 化简后的技能数据")
    print(f"  • skill_simplification_report.json - 详细化简报告")
    
    print(f"\n🚀 下一步:")
    print(f"  使用 simplified_jobs_skills.csv 进行知识图谱构建")
    print(f"  预期图规模: ~{final_stats['final_skill_count']}个技能节点")

if __name__ == "__main__":
    main()