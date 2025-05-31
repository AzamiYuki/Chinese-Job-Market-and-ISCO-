import pandas as pd
import jieba
import jieba.posseg as pseg
import re
from collections import Counter, defaultdict
import numpy as np
import json
from typing import List, Dict, Set, Tuple
import warnings
from fuzzywuzzy import fuzz, process

warnings.filterwarnings('ignore')

class ImprovedSkillExtractor:
    def __init__(self):
        """改进版技能提取器"""
        # 核心技能词典
        self.skill_keywords = self._load_core_skills()
        
        # 同义词映射
        self.synonyms_dict = self._load_synonyms()
        
        # 停用词和噪声词
        self.stop_words = self._load_stop_words()
        
        # 编译正则表达式
        self.skill_patterns = self._compile_skill_patterns()
        
        # 添加自定义词典
        self._add_custom_dict()
        
        print("改进版技能提取器初始化完成")
    
    def _load_core_skills(self) -> Dict[str, List[str]]:
        """加载核心技能词典（整合人工总结的技能）"""
        return {
            # 编程与开发
            'programming': [
                'Python', 'Java', 'JavaScript', 'C++', 'PHP', 'Go', 'C#', 'Swift', 'Objective-C',
                'HTML', 'CSS', 'React', 'Vue', 'Angular', 'Node.js', 'Django', 'Flask',
                'J2EE', 'Hibernate', 'iBatis', 'Spring', 'Struts2', 'Eclipse',
                '.NET', 'ASP.NET', 'jQuery', 'Ajax', 'Web Service',
                'C/C++', 'Cocos2d-x', 'Android开发', 'iOS开发', 'APP开发',
                'MFC/WTL框架', 'ActiveX组件', 'DirectX', 'Windows GDI'
            ],
            
            # 数据库技术
            'database': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQL Server', 'DB2',
                '数据库应用开发', '数据库设计', 'SQL', 'NoSQL'
            ],
            
            # 系统与运维
            'system_ops': [
                'Linux', 'Git', 'Docker', 'SVN', 'JIRA', 'Confluence',
                'Tomcat', 'Jboss', 'WebLogic', 'WebSphere',
                '网络综合布线', '服务器配置', '交换机调试', '防火墙调试',
                'IT运维', '机房管理', '网络监控', '数据备份', '故障排除'
            ],
            
            # 办公软件
            'office_tools': [
                'Excel', 'Word', 'PowerPoint', 'PPT', 'Office', 'Outlook',
                'Photoshop', 'PS', 'Illustrator', 'AI', 'CorelDRAW', 'Dreamweaver', 'Flash',
                'InDesign', '3DMAX', 'Maya', 'AutoCAD', 'CAD', 'Pro/E', 'SolidWorks',
                'Catia', 'Inventor', 'VRay', 'SketchUp', 'SAI',
                'WPS', 'Visio', 'Project'
            ],
            
            # 企业软件
            'enterprise_software': [
                '用友U8', 'SAP', 'K3', '金蝶', 'ERP系统', 'MES系统', 'CRM',
                'OA办公自动化系统', 'SPSS', 'SAS', '广联达软件'
            ],
            
            # 语言能力
            'languages': [
                '英语', '英语听说读写', '德语', '日语', '普通话', '粤语', '英文', '日文', '韩文',
                'CET-4', 'CET-6', '四级', '六级', '雅思', '托福', 'TOEFL', 'IELTS',
                'TESOL'
            ],
            
            # 专业证书
            'certificates': [
                'PMP', 'CPA', 'CFA', '注册会计师', '中级会计师', '初级会计师',
                '执业药师资格证', '律师执业资格证', '会计从业资格证',
                '报关员资格证', '报检员资格证', '驾驶证', 'A照', 'B照', 'C照',
                '电工证', '焊工证', '教师资格证', '人力资源管理师',
                '造价员资格证', '工程师职称', '建造师证书', '安全员证',
                'AFP', 'CFP', '证券从业资格证', '保险代理人资格证'
            ],
            
            # 沟通与表达技能
            'communication_skills': [
                '沟通能力', '交流能力', '表达能力', '演讲能力', '汇报能力',
                '倾听能力', '谈判能力', '商务沟通', '文字功底', '写作能力',
                '口齿清晰', '善于倾听', '公文写作'
            ],
            
            # 领导与管理技能
            'leadership_skills': [
                '领导力', '领导能力', '管理能力', '团队管理', '人员管理',
                '项目管理', '团队建设', '决策能力', '组织能力', '计划能力',
                '控制能力', '执行能力', '软件项目管理'
            ],
            
            # 团队协作技能
            'teamwork_skills': [
                '团队合作', '协作能力', '协调能力', '配合能力', '团队精神',
                '团队合作精神', '独立工作能力'
            ],
            
            # 分析与思维技能
            'analytical_skills': [
                '分析能力', '逻辑思维', '数据分析', '财务分析', '市场分析',
                '问题分析', '统计分析', '判断能力', '问题解决能力',
                '逻辑思维能力', '洞察力', '归纳总结能力'
            ],
            
            # 个人素质技能
            'personal_skills': [
                '学习能力', '适应能力', '抗压能力', '创新能力', '执行力',
                '责任心', '主动性', '时间管理', '自我管理', '应变能力',
                '积极主动', '吃苦耐劳', '自我激励', '亲和力', '耐心', '细心',
                '时间管理能力', '敬业精神', '职业道德', '有激情', '同理心',
                '沉稳', '隐忍', '乐观', '自律性'
            ],
            
            # 客户服务技能
            'service_skills': [
                '客户服务', '客户服务意识', '服务意识', '客户关系管理',
                '客户开发', '客户需求挖掘', '客户信息收集', '售后服务'
            ],
            
            # 市场营销技能
            'marketing_skills': [
                '市场营销', '市场营销学', '品牌运作', '渠道维护管理', '网络推广',
                '搜索引擎竞价', '百度推广', 'Google推广', 'SEO', 'SEM',
                'B2B平台操作', 'B2C平台操作', '外贸平台操作', '店铺建设',
                '产品推广', '电话营销', '网络营销', '市场调研', '招商管理',
                '促销策划', '广告策划', '媒体推广', '公关活动策划',
                '整合营销', '整合推广', '电子商务', '网络销售'
            ],
            
            # 销售技能
            'sales_skills': [
                '销售', '销售技巧', '客户开发', '潜在客户开发', '销售数据分析',
                '市场拓展', 'KA系统操作', '经销商管理', '终端管理',
                '投标商务文件编制', '合同起草', '房产经纪', '置业咨询',
                '房屋过户手续办理', '案场接待', '销讲', '答客问', '余款催缴'
            ],
            
            # 人力资源技能
            'hr_skills': [
                '人力资源管理', '招聘流程管理', '入职手续办理', '离职手续办理',
                '劳动合同管理', '薪酬管理', '绩效考核', '培训与发展',
                '培训需求分析', '培训计划制定', '培训教材编写', '培训效果评估',
                '员工关系管理', '企业文化建设'
            ],
            
            # 行政管理技能
            'admin_skills': [
                '前台接待技巧', '电话接转技巧', '文件管理', '档案管理',
                '会议管理', '办公用品管理', '资产管理', '考勤管理',
                '员工通讯信息管理', '后勤保障', '行政事务处理', '商务礼仪'
            ],
            
            # 财务会计技能
            'finance_skills': [
                '财务预算', '会计核算', '成本核算', '成本控制', '财务报表编制',
                '财务分析', '税务申报', '出口退税', '软件产品退税', '总账处理',
                '应收应付账款管理', '银行结算业务', '现金管理', '票据管理',
                '内部审计', '资金管理', '融资管理', '投资管理', '税务筹划',
                '风险控制能力', '成本意识'
            ],
            
            # 生产制造技能
            'production_skills': [
                '生产计划', '生产调度', '生产管理', '质量管理', '品质检验',
                'IQC', 'PQC', '工艺开发', '工艺流程规划', '现场管理',
                '设备管理', '设备维护保养', '模具设计', '模具维修',
                '作业指导书编写', 'BOM单管理', '产能评估', '焊接技术',
                'CNC编程与操作', '钳工', '机修'
            ],
            
            # 质量管理技能
            'quality_skills': [
                'ISO9000', 'ISO9001', 'TS16949', 'ISO14001', 'OHSAS18001',
                'GMP', 'HACCP', '6sigma', '六西格玛', '5S管理', '8D报告'
            ],
            
            # 物流仓储技能
            'logistics_skills': [
                '仓库管理', '库存管理', '进出库管理', '盘点', '物流配送',
                '货物跟踪', '制单', '标签制作', '托运书管理',
                '国际货运代理', '海运操作', '空运操作', '报关', '清关',
                '单证制作', 'L/C审单制单', 'SHIPPING流程操作', '供应链管理'
            ],
            
            # 设计创意技能
            'design_skills': [
                '平面设计', '网页设计', '美工', '图片处理', '视觉设计',
                '广告设计', '包装设计', '海报设计', '宣传册设计',
                '店铺装修', '详情页设计', '创意构思', '色彩搭配',
                '手绘能力', 'UI设计', 'GUI设计', '交互设计'
            ],
            
            # 多媒体技能
            'multimedia_skills': [
                '动画制作', '视频拍摄', '视频剪辑', '视频后期处理',
                'EDIUS', 'Final Cut Pro', 'Premiere Pro', '3D角色动作设计',
                '场景原画设计', '游戏原画', '摄影', '图片修调', '播音主持'
            ],
            
            # 工程建筑技能
            'engineering_skills': [
                '工程造价', '工程预结算', '土木工程', '建筑施工',
                '施工组织设计', '图纸会审', '工程验收', '工程变更管理',
                '签证工作', '建筑安全管理', '结构工程检测', '岩土工程',
                '测量放样', '给排水工程', '暖通工程', '空气净化工程'
            ],
            
            # 教育培训技能
            'education_skills': [
                '教学能力', '课程开发', '教案制作', '教具制作', '课堂管理',
                '公开课演讲', '家校互动', '学员管理', '营养师培训',
                '育婴师培训', '小儿推拿师培训', '语文教学', '教材编辑',
                '校对', '钢琴教学', '视奏能力', '因材施教', '英语教学'
            ],
            
            # 法律专业技能
            'legal_skills': [
                '法律知识', '合同法', '公司法', '诉讼法', '知识产权法',
                '经济法', '民商法', '劳动法', '司法诉讼程序', '法律风险防范',
                '律师执业', '商标处理', '专利处理'
            ],
            
            # 医疗健康技能
            'medical_skills': [
                '中西药学知识', '执业药师', '临床医学', '男科手术',
                '口腔修复技术', '根管治疗技术', '皮肤护理', '美容美体',
                '美甲', '化妆', '中医养生保健', '中药材性状用法',
                '体外诊断试剂研发', '体外诊断试剂生产'
            ],
            
            # 技术专业技能
            'technical_skills': [
                '嵌入式系统设计', '单片机', 'ARM处理器', 'DSP数字信号处理器',
                'FPGA', 'PLC编程', '电机伺服控制', 'LabVIEW编程',
                '系统集成', '用户培训', '技术方案编写', '实施方案编写',
                '验收文档编写', '微波天线通讯技术', '光纤光栅制作',
                '光纤激光器设计', '光学器件测试', '光纤传感系统'
            ],
            
            # 其他专业技能
            'other_professional': [
                '土地房产管理', '精算评估', 'Prophet模型', '资产负债管理',
                '驾驶技术', '车辆保养维修', '电工技能', '高压入网',
                '电梯操作', '水电维修安装', '消防安全管理', '物业管理',
                '餐饮管理', '后厨管理', '菜品开发', '宴会服务', '餐厅服务',
                '酒水调制', '茶艺', '西餐服务', '食品检验', '微生物检验',
                '酿造工艺', '保密意识', '全局观', '大局观', '使命感'
            ]
        }
    
    def _load_synonyms(self) -> Dict[str, str]:
        """加载同义词映射（整合人工总结的同义词）"""
        return {
            # 沟通相关同义词
            '交流能力': '沟通能力',
            '表达能力': '沟通能力',
            '沟通技巧': '沟通能力',
            '交流技巧': '沟通能力',
            '谈判能力': '沟通能力',
            '商务沟通': '沟通能力',
            '口齿清晰': '沟通能力',
            '善于倾听': '沟通能力',
            
            # 团队相关同义词
            '协作能力': '团队合作',
            '配合能力': '团队合作',
            '团队精神': '团队合作',
            '合作能力': '团队合作',
            '团队合作精神': '团队合作',
            
            # 领导管理相关同义词
            '领导能力': '领导力',
            '管理能力': '领导力',
            '管理技能': '领导力',
            '团队管理': '领导力',
            '人员管理': '领导力',
            '组织能力': '领导力',
            '计划能力': '领导力',
            '控制能力': '领导力',
            
            # 分析思维相关同义词
            '分析技能': '分析能力',
            '逻辑分析': '分析能力',
            '数据分析能力': '分析能力',
            '逻辑思维能力': '分析能力',
            '判断能力': '分析能力',
            '问题分析': '分析能力',
            '问题解决能力': '分析能力',
            
            # 个人素质相关同义词
            '执行能力': '执行力',
            '应变能力': '适应能力',
            '抗压能力': '适应能力',
            '时间管理能力': '时间管理',
            '自我激励': '主动性',
            '积极主动': '主动性',
            '职业道德': '责任心',
            '敬业精神': '责任心',
            
            # 技术相关同义词
            'JS': 'JavaScript',
            'js': 'JavaScript',
            '面向对象设计': 'OOD',
            '面向对象编程': 'OOP',
            
            # 办公软件同义词
            'PPT': 'PowerPoint',
            'ppt': 'PowerPoint',
            'PS': 'Photoshop',
            'ps': 'Photoshop',
            'AI': 'Illustrator',
            'office': 'Office',
            'OFFICE': 'Office',
            'Word文档': 'Word',
            'Excel表格': 'Excel',
            
            # 数据库同义词
            'mysql': 'MySQL',
            'MYSQL': 'MySQL',
            'oracle': 'Oracle',
            'ORACLE': 'Oracle',
            
            # 学历相关同义词
            '本科以上': '本科',
            '大专以上': '大专',
            '专科以上': '专科',
            '硕士以上': '硕士',
            '研究生以上': '研究生',
            
            # 语言能力同义词
            '英语听说读写': '英语',
            '英文': '英语',
            '日文': '日语',
            '韩文': '韩语',
            '四级': 'CET-4',
            '六级': 'CET-6',
            
            # 证书同义词
            '注册会计师': 'CPA',
            '项目管理': 'PMP',
            '驾照': '驾驶证',
            'C照': '驾驶证',
            'B照': '驾驶证',
            'A照': '驾驶证',
            
            # 业务技能同义词
            '市场营销学': '市场营销',
            '客户关系管理': 'CRM',
            '客户服务意识': '客户服务',
            '服务意识': '客户服务',
            '网络推广': '网络营销',
            '电话营销': '销售',
            '网络销售': '销售',
            
            # 质量管理同义词
            '质量管理': 'ISO9001',
            '品质检验': '质量管理',
            'IQC': '质量管理',
            'PQC': '质量管理',
            '六西格玛': '6sigma',
            
            # 物流相关同义词
            '仓储管理': '仓库管理',
            '库存管理': '仓库管理',
            '物流配送': '物流管理',
            '供应链管理': '物流管理',
            
            # 设计相关同义词
            '美工': '平面设计',
            '图片处理': '平面设计',
            'UI设计': '界面设计',
            'GUI设计': '界面设计',
            '视觉设计': '平面设计',
            
            # 财务相关同义词
            '会计核算': '财务管理',
            '成本核算': '成本控制',
            '财务报表编制': '财务分析',
            '税务申报': '税务管理',
            '资金管理': '财务管理',
            
            # 生产相关同义词
            '生产计划': '生产管理',
            '生产调度': '生产管理',
            '现场管理': '生产管理',
            '设备管理': '设备维护',
            '工艺开发': '工艺管理',
            
            # 人力资源同义词
            '招聘管理': '人力资源管理',
            '薪酬管理': '人力资源管理',
            '绩效考核': '人力资源管理',
            '培训管理': '培训与发展',
            '员工关系管理': '人力资源管理'
        }
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词和噪声词"""
        return {
            # 数字和符号
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '年', '月', '日', '次', '个', '位', '名', '人',
            'com', 'cn', 'www', 'http', 'https',
            
            # 无意义的形容词
            '良好', '较强', '优秀', '出色', '具备', '具有', '拥有',
            '能够', '可以', '应该', '需要', '要求', '希望',
            
            # 常见的连接词
            '以上', '以下', '或者', '以及', '并且', '同时',
            '相关', '相应', '对应', '合适', '适合',
            
            # 职位相关但非技能的词
            '工作', '岗位', '职位', '负责', '承担', '完成',
            '实施', '执行', '推进', '落实', '开展'
        }
    
    def _compile_skill_patterns(self) -> Dict[str, re.Pattern]:
        """编译技能相关正则表达式"""
        return {
            'education': re.compile(r'(本科|硕士|研究生|博士|大专|专科|MBA|EMBA)(?:以上|及以上)?'),
            'experience_years': re.compile(r'(\d+)年以上.*?经验'),
            'experience_field': re.compile(r'(\d+)年以上.*?(管理|开发|销售|设计|运营|市场|技术)经验'),
            'skill_proficiency': re.compile(r'(熟练掌握|精通|熟悉|了解)\s*([^\s，。；]{2,10})'),
            'ability_requirement': re.compile(r'(具备|具有|拥有).*?(沟通|协调|管理|分析|学习|创新|领导)能力'),
            'good_ability': re.compile(r'(良好|较强|优秀|出色)的\s*(沟通|团队|分析|学习|管理|协调|组织)能力')
        }
    
    def _add_custom_dict(self):
        """添加自定义词典"""
        all_skills = []
        for skills in self.skill_keywords.values():
            all_skills.extend(skills)
        
        for skill in all_skills:
            jieba.add_word(skill, freq=1000, tag='skill')
        
        # 添加同义词
        for synonym in self.synonyms_dict.keys():
            jieba.add_word(synonym, freq=800, tag='skill')
    
    def clean_text(self, text: str) -> str:
        """深度清理文本"""
        if pd.isna(text) or text == '':
            return ''
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除邮箱和网址
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'http[s]?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # 移除电话号码
        text = re.sub(r'\d{3,4}-?\d{7,8}', '', text)
        text = re.sub(r'1[3-9]\d{9}', '', text)
        
        # 标准化标点符号
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('，', ',').replace('。', '.')
        text = text.replace('；', ';').replace('：', ':')
        
        # 移除特殊字符，保留中英文、数字和常用标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\+\-\.\s,，。；：！？、（）\[\]{}()]', ' ', text)
        
        # 统一空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def filter_skill(self, skill: str) -> bool:
        """过滤技能，去除噪声"""
        if not skill or len(skill.strip()) < 2:
            return False
        
        skill = skill.strip()
        
        # 过滤停用词
        if skill in self.stop_words:
            return False
        
        # 过滤纯数字
        if skill.isdigit():
            return False
        
        # 过滤包含过多数字的词（如"1管理"、"3年经验"）
        if re.match(r'^\d+\s*\w*$', skill):
            return False
        
        # 过滤单个字符或太长的词
        if len(skill) < 2 or len(skill) > 15:
            return False
        
        # 过滤只有标点符号的
        if re.match(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', skill):
            return False
        
        return True
    
    def extract_by_keywords(self, text: str) -> Dict[str, List[str]]:
        """基于关键词提取技能"""
        text_lower = text.lower()
        extracted = defaultdict(list)
        
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                # 精确匹配
                if keyword.lower() in text_lower:
                    extracted[category].append(keyword)
        
        # 清理结果
        cleaned_extracted = {}
        for category, skills in extracted.items():
            cleaned_skills = [skill for skill in set(skills) if self.filter_skill(skill)]
            if cleaned_skills:
                cleaned_extracted[category] = cleaned_skills
        
        return cleaned_extracted
    
    def extract_by_patterns(self, text: str) -> Dict[str, List[str]]:
        """基于正则表达式提取"""
        extracted = defaultdict(list)
        
        for pattern_name, pattern in self.skill_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    # 处理分组匹配
                    if pattern_name == 'skill_proficiency':
                        skill = match[1].strip()
                        if self.filter_skill(skill):
                            extracted['proficiency_skills'].append(skill)
                    elif pattern_name in ['good_ability', 'ability_requirement']:
                        ability = match[-1] + '能力'  # 最后一个匹配组 + '能力'
                        if self.filter_skill(ability):
                            extracted['extracted_abilities'].append(ability)
                    else:
                        clean_match = ' '.join(match).strip()
                        if self.filter_skill(clean_match):
                            extracted[pattern_name].append(clean_match)
                else:
                    if self.filter_skill(match):
                        extracted[pattern_name].append(match)
        
        return dict(extracted)
    
    def extract_by_segmentation(self, text: str) -> List[str]:
        """基于分词提取潜在技能"""
        words = jieba.lcut(text)
        skills = []
        
        # 技能指示词
        skill_indicators = ['能力', '技能', '技巧', '经验', '管理', '开发', '设计', '分析', '运营']
        
        for word in words:
            # 过滤基本条件
            if not self.filter_skill(word):
                continue
            
            # 包含技能指示词的复合词
            if any(indicator in word for indicator in skill_indicators) and len(word) >= 3:
                skills.append(word)
            # 英文技术词汇
            elif re.match(r'^[a-zA-Z][a-zA-Z0-9\+\-\.]*$', word) and len(word) >= 3:
                skills.append(word)
            # 在已知技能列表中
            elif word in [skill for skills in self.skill_keywords.values() for skill in skills]:
                skills.append(word)
        
        return list(set(skills))
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """标准化技能（处理同义词）"""
        normalized = []
        
        for skill in skills:
            # 清理空格和标点
            skill = re.sub(r'\s+', '', skill)
            skill = skill.strip('.,;:!?()[]{}')
            
            if not self.filter_skill(skill):
                continue
            
            # 查找同义词映射
            normalized_skill = self.synonyms_dict.get(skill, skill)
            normalized.append(normalized_skill)
        
        return list(set(normalized))
    
    def extract_skills_from_description(self, description: str) -> Dict:
        """从岗位描述中提取技能"""
        if pd.isna(description) or description.strip() == '':
            return {
                'keywords_skills': {},
                'pattern_skills': {},
                'segmentation_skills': [],
                'normalized_skills': [],
                'total_skills': 0,
                'text_length': 0
            }
        
        # 深度清理文本
        clean_text = self.clean_text(description)
        
        # 多种方法提取
        keywords_skills = self.extract_by_keywords(clean_text)
        pattern_skills = self.extract_by_patterns(clean_text)
        segmentation_skills = self.extract_by_segmentation(clean_text)
        
        # 收集所有技能
        all_skills = []
        for skill_list in keywords_skills.values():
            all_skills.extend(skill_list)
        for skill_list in pattern_skills.values():
            all_skills.extend(skill_list)
        all_skills.extend(segmentation_skills)
        
        # 标准化处理
        normalized_skills = self.normalize_skills(all_skills)
        
        return {
            'keywords_skills': keywords_skills,
            'pattern_skills': pattern_skills,
            'segmentation_skills': segmentation_skills,
            'normalized_skills': normalized_skills,
            'total_skills': len(normalized_skills),
            'text_length': len(clean_text)
        }
    
    def process_job_descriptions(self, df: pd.DataFrame, desc_column: str = '岗位描述') -> pd.DataFrame:
        """处理整个数据集的岗位描述列"""
        print(f"开始处理岗位描述数据，共 {len(df)} 条记录...")
        
        result_df = df.copy()
        
        # 添加新列
        result_df['提取的技能'] = None
        result_df['技能数量'] = 0
        result_df['编程技能'] = None
        result_df['沟通技能'] = None
        result_df['领导技能'] = None
        result_df['分析技能'] = None
        result_df['业务技能'] = None
        result_df['标准化技能'] = None
        
        # 统计信息
        total_skills_found = 0
        jobs_with_skills = 0
        
        for idx, row in result_df.iterrows():
            if idx % 500 == 0:
                print(f"已处理 {idx}/{len(df)} 条记录...")
            
            # 提取技能
            skill_data = self.extract_skills_from_description(row[desc_column])
            
            # 保存详细结果
            result_df.at[idx, '提取的技能'] = json.dumps(skill_data, ensure_ascii=False)
            result_df.at[idx, '技能数量'] = skill_data['total_skills']
            result_df.at[idx, '标准化技能'] = json.dumps(skill_data['normalized_skills'], ensure_ascii=False)
            
            # 分类保存
            keywords_skills = skill_data['keywords_skills']
            result_df.at[idx, '编程技能'] = json.dumps(keywords_skills.get('programming', []), ensure_ascii=False)
            result_df.at[idx, '沟通技能'] = json.dumps(keywords_skills.get('communication_skills', []), ensure_ascii=False)
            result_df.at[idx, '领导技能'] = json.dumps(keywords_skills.get('leadership_skills', []), ensure_ascii=False)
            result_df.at[idx, '分析技能'] = json.dumps(keywords_skills.get('analytical_skills', []), ensure_ascii=False)
            result_df.at[idx, '业务技能'] = json.dumps(keywords_skills.get('business_skills', []), ensure_ascii=False)
            
            # 统计
            if skill_data['total_skills'] > 0:
                jobs_with_skills += 1
                total_skills_found += skill_data['total_skills']
        
        print(f"\n处理完成！")
        print(f"总职位数: {len(df)}")
        print(f"包含技能的职位数: {jobs_with_skills}")
        print(f"技能覆盖率: {jobs_with_skills/len(df)*100:.1f}%")
        print(f"平均每职位技能数: {total_skills_found/len(df):.1f}")
        
        return result_df
    
    def generate_skill_report(self, df: pd.DataFrame) -> Dict:
        """生成技能分析报告"""
        print("生成技能分析报告...")
        
        all_skills = Counter()
        skill_categories = Counter()
        
        for idx, row in df.iterrows():
            # 统计标准化技能
            normalized_skills_str = row.get('标准化技能', '[]')
            try:
                normalized_skills = json.loads(normalized_skills_str)
                for skill in normalized_skills:
                    all_skills[skill] += 1
            except:
                continue
            
            # 统计分类技能
            skill_data_str = row.get('提取的技能', '{}')
            try:
                skill_data = json.loads(skill_data_str)
                keywords_skills = skill_data.get('keywords_skills', {})
                for category, skills in keywords_skills.items():
                    skill_categories[category] += len(skills)
            except:
                continue
        
        # 转换pandas数据类型为Python原生类型
        skill_stats = df['技能数量'].describe()
        skill_distribution = {
            'count': int(skill_stats['count']),
            'mean': float(skill_stats['mean']),
            'std': float(skill_stats['std']),
            'min': int(skill_stats['min']),
            '25%': float(skill_stats['25%']),
            '50%': float(skill_stats['50%']),
            '75%': float(skill_stats['75%']),
            'max': int(skill_stats['max'])
        }
        
        # 生成报告
        report = {
            'summary': {
                'total_jobs': int(len(df)),
                'jobs_with_skills': int(len(df[df['技能数量'] > 0])),
                'total_unique_skills': int(len(all_skills)),
                'avg_skills_per_job': float(df['技能数量'].mean()),
                'max_skills_per_job': int(df['技能数量'].max())
            },
            'top_skills': [(skill, int(count)) for skill, count in all_skills.most_common(50)],
            'skill_categories': {k: int(v) for k, v in skill_categories.items()},
            'skill_distribution': skill_distribution
        }
        
        return report

def main():
    """主函数"""
    # 读取数据
    print("读取数据...")
    try:
        df = pd.read_csv('lunwen/newjob1_sortall.csv')
        print(f"成功读取数据，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return
    
    # 检查岗位描述列
    if '岗位描述' not in df.columns:
        print("错误：未找到'岗位描述'列")
        print(f"可用列: {list(df.columns)}")
        return
    
    # 初始化改进版提取器
    extractor = ImprovedSkillExtractor()
    
    # ===== 修改这里来处理全部数据 =====
    # 选择要处理的数据量
    process_all_data = True  # 设置为True处理全部数据，False处理样本
    
    if process_all_data:
        # 处理全部数据
        sample_df = df.copy()
        print(f"将处理全部 {len(sample_df)} 条记录")
        output_file = 'all_jobs_skills_extracted.csv'  # 全量数据输出文件名
        report_file = 'all_jobs_skill_analysis_report.json'
        isco_file = 'all_jobs_isco_skill_distribution.json'
    else:
        # 处理样本数据（用于测试）
        sample_size = min(5000, len(df))
        sample_df = df.head(sample_size).copy()
        print(f"将处理前 {len(sample_df)} 条记录（样本模式）")
        output_file = 'sample_jobs_skills_extracted.csv'
        report_file = 'sample_skill_analysis_report.json'
        isco_file = 'sample_isco_skill_distribution.json'
    
    # 处理数据
    print("开始技能提取...")
    start_time = pd.Timestamp.now()
    
    result_df = extractor.process_job_descriptions(sample_df)
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"处理完成，总耗时: {processing_time:.2f} 秒")
    print(f"平均每条记录处理时间: {processing_time/len(sample_df):.3f} 秒")
    
    # 保存结果
    print(f"保存结果到: {output_file}")
    try:
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSV文件保存成功: {output_file}")
        
        # 显示文件大小
        import os
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"✓ 文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"✗ CSV文件保存失败: {e}")
        return
    
    # 生成报告
    report = extractor.generate_skill_report(result_df)
    
    print("\n" + "="*60)
    print("改进版技能提取报告")
    print("="*60)
    
    summary = report['summary']
    print(f"总职位数: {summary['total_jobs']:,}")
    print(f"包含技能的职位数: {summary['jobs_with_skills']:,}")
    print(f"技能覆盖率: {summary['jobs_with_skills']/summary['total_jobs']*100:.1f}%")
    print(f"识别的唯一技能数: {summary['total_unique_skills']:,}")
    print(f"平均每职位技能数: {summary['avg_skills_per_job']:.1f}")
    print(f"最多技能数: {summary['max_skills_per_job']:,}")
    
    print(f"\n前30个最常见技能:")
    for i, (skill, count) in enumerate(report['top_skills'][:30], 1):
        print(f"  {i:2d}. {skill}: {count:,}")
    
    print(f"\n技能分类统计:")
    for category, count in sorted(report['skill_categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count:,}")
    
    # 保存详细报告
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 详细报告已保存到: {report_file}")
    except Exception as e:
        print(f"\n✗ 报告保存失败: {e}")
    
    # 显示一些示例结果
    print(f"\n示例提取结果:")
    for i in range(min(5, len(result_df))):
        row = result_df.iloc[i]
        job_title = row.get('岗位', 'N/A')
        skills_count = row.get('技能数量', 0)
        try:
            normalized_skills = json.loads(row.get('标准化技能', '[]'))
            skills_display = normalized_skills[:8]  # 显示前8个技能
        except:
            skills_display = []
        
        print(f"\n{i+1}. 【{job_title}】- 技能数量: {skills_count}")
        print(f"   主要技能: {skills_display}")
        
        # 显示分类技能
        try:
            programming_skills = json.loads(row.get('编程技能', '[]'))
            communication_skills = json.loads(row.get('沟通技能', '[]'))
            business_skills = json.loads(row.get('业务技能', '[]'))
            
            if programming_skills:
                print(f"   编程技能: {programming_skills}")
            if communication_skills:
                print(f"   沟通技能: {communication_skills}")
            if business_skills:
                print(f"   业务技能: {business_skills[:5]}")  # 只显示前5个
        except:
            pass
    
    # 按ISCO代码分析技能分布
    if 'ISCO_4_Digit_Code_Gemini' in result_df.columns:
        print(f"\n" + "="*60)
        print("按ISCO职业代码分析技能分布")
        print("="*60)
        
        isco_skill_analysis = analyze_skills_by_isco(result_df)
        
        # 按职位数量排序，显示更多ISCO类别
        isco_sorted = sorted(isco_skill_analysis.items(), 
                           key=lambda x: x[1]['job_count'], reverse=True)
        
        # 显示前20个ISCO类别（或全部，如果少于20个）
        display_count = min(20, len(isco_sorted))
        print(f"显示前{display_count}个职业类别的技能分布：\n")
        
        for i, (isco_code, analysis) in enumerate(isco_sorted[:display_count]):
            print(f"{i+1:2d}. ISCO {isco_code} (共{analysis['job_count']:,}个职位):")
            print(f"    平均技能数: {analysis['avg_skills']:.1f}")
            print(f"    技能多样性: {analysis['skill_diversity']}种不同技能")
            
            # 显示前8个常见技能，格式化输出
            top_skills = analysis['top_skills'][:8]
            if top_skills:
                skills_str = ", ".join([f"{skill}({count})" for skill, count in top_skills])
                print(f"    主要技能: {skills_str}")
            else:
                print(f"    主要技能: 暂无")
            print()  # 空行分隔
        
        # 添加技能分布统计
        print("="*60)
        print("ISCO职业技能分布统计")
        print("="*60)
        
        # 计算各种统计指标
        job_counts = [analysis['job_count'] for analysis in isco_skill_analysis.values()]
        avg_skills = [analysis['avg_skills'] for analysis in isco_skill_analysis.values()]
        skill_diversity = [analysis['skill_diversity'] for analysis in isco_skill_analysis.values()]
        
        print(f"总职业类别数: {len(isco_skill_analysis)}")
        print(f"职位数量分布: 最多{max(job_counts):,}个, 最少{min(job_counts)}个, 平均{sum(job_counts)/len(job_counts):.1f}个")
        print(f"平均技能数分布: 最高{max(avg_skills):.1f}, 最低{min(avg_skills):.1f}, 总体平均{sum(avg_skills)/len(avg_skills):.1f}")
        print(f"技能多样性分布: 最高{max(skill_diversity)}, 最低{min(skill_diversity)}, 平均{sum(skill_diversity)/len(skill_diversity):.1f}")
        
        # 找出技能要求最高和最低的职业
        max_skill_isco = max(isco_skill_analysis.items(), key=lambda x: x[1]['avg_skills'])
        min_skill_isco = min(isco_skill_analysis.items(), key=lambda x: x[1]['avg_skills'])
        
        print(f"\n技能要求最高的职业: ISCO {max_skill_isco[0]} (平均{max_skill_isco[1]['avg_skills']:.1f}个技能)")
        print(f"技能要求最低的职业: ISCO {min_skill_isco[0]} (平均{min_skill_isco[1]['avg_skills']:.1f}个技能)")
        
        # 找出技能最多样化的职业
        max_diversity_isco = max(isco_skill_analysis.items(), key=lambda x: x[1]['skill_diversity'])
        print(f"技能最多样化的职业: ISCO {max_diversity_isco[0]} ({max_diversity_isco[1]['skill_diversity']}种不同技能)")
        
        # 展示不同类别的代表性职业
        print(f"\n" + "="*60)
        print("不同技能要求水平的代表职业")
        print("="*60)
        
        # 将职业按平均技能数分组
        high_skill_jobs = [item for item in isco_sorted if item[1]['avg_skills'] >= 10]
        medium_skill_jobs = [item for item in isco_sorted if 5 <= item[1]['avg_skills'] < 10]
        low_skill_jobs = [item for item in isco_sorted if item[1]['avg_skills'] < 5]
        
        if high_skill_jobs:
            print(f"高技能要求职业 (≥10个技能, 共{len(high_skill_jobs)}类):")
            for isco_code, analysis in high_skill_jobs[:5]:
                top_3_skills = [skill for skill, count in analysis['top_skills'][:3]]
                print(f"  ISCO {isco_code}: {analysis['avg_skills']:.1f}个技能 - {', '.join(top_3_skills)}")
        
        if medium_skill_jobs:
            print(f"\n中等技能要求职业 (5-9个技能, 共{len(medium_skill_jobs)}类):")
            for isco_code, analysis in medium_skill_jobs[:5]:
                top_3_skills = [skill for skill, count in analysis['top_skills'][:3]]
                print(f"  ISCO {isco_code}: {analysis['avg_skills']:.1f}个技能 - {', '.join(top_3_skills)}")
        
        if low_skill_jobs:
            print(f"\n基础技能要求职业 (<5个技能, 共{len(low_skill_jobs)}类):")
            for isco_code, analysis in low_skill_jobs[:5]:
                top_3_skills = [skill for skill, count in analysis['top_skills'][:3]]
                print(f"  ISCO {isco_code}: {analysis['avg_skills']:.1f}个技能 - {', '.join(top_3_skills)}")
        
        # 保存ISCO分析结果
        try:
            with open(isco_file, 'w', encoding='utf-8') as f:
                json.dump(isco_skill_analysis, f, ensure_ascii=False, indent=2)
            print(f"\n✓ ISCO技能分析已保存到: {isco_file}")
        except Exception as e:
            print(f"\n✗ ISCO分析保存失败: {e}")
    
    # 提供数据使用建议
    print(f"\n" + "="*60)
    print("数据使用建议")
    print("="*60)
    print("📁 主要输出文件:")
    print(f"  • {output_file} - 包含技能的完整数据（CSV格式）")
    print(f"  • {report_file} - 详细统计报告（JSON格式）")
    print(f"  • {isco_file} - ISCO技能分析（JSON格式）")
    
    print("\n📊 后续分析建议:")
    print("  • 可直接用Excel/Python/R分析CSV文件")
    print("  • '标准化技能'列适合做知识图谱节点")
    print("  • 各分类技能列便于行业分析")
    print("  • ISCO分析可用于职业技能画像")
    
    print("\n🔗 知识图谱构建:")
    print("  • 节点: ISCO职业代码 + 标准化技能")
    print("  • 边权重: 技能在职业中的出现频率")
    print("  • 可进一步计算技能相似度和职业转换路径")

# 添加批量处理的辅助函数
def process_in_batches(df: pd.DataFrame, extractor, batch_size: int = 1000) -> pd.DataFrame:
    """分批处理大数据集，避免内存问题"""
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"将分 {total_batches} 批处理，每批 {batch_size} 条记录")
    
    processed_dfs = []
    
    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        batch_df = df.iloc[i:i+batch_size].copy()
        
        print(f"处理第 {batch_num}/{total_batches} 批 ({len(batch_df)} 条记录)...")
        
        batch_result = extractor.process_job_descriptions(batch_df)
        processed_dfs.append(batch_result)
        
        # 可选：每批处理完后释放内存
        import gc
        gc.collect()
    
    # 合并所有批次的结果
    print("合并所有批次结果...")
    final_result = pd.concat(processed_dfs, ignore_index=True)
    
    return final_result
    with open('improved_skill_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存到: improved_skill_analysis_report.json")
    
    # 显示一些示例结果
    print(f"\n示例提取结果:")
    for i in range(min(5, len(result_df))):
        row = result_df.iloc[i]
        job_title = row.get('岗位', 'N/A')
        skills_count = row.get('技能数量', 0)
        try:
            normalized_skills = json.loads(row.get('标准化技能', '[]'))
            skills_display = normalized_skills[:8]  # 显示前8个技能
        except:
            skills_display = []
        
        print(f"\n{i+1}. 【{job_title}】- 技能数量: {skills_count}")
        print(f"   主要技能: {skills_display}")
        
        # 显示分类技能
        try:
            programming_skills = json.loads(row.get('编程技能', '[]'))
            communication_skills = json.loads(row.get('沟通技能', '[]'))
            business_skills = json.loads(row.get('业务技能', '[]'))
            
            if programming_skills:
                print(f"   编程技能: {programming_skills}")
            if communication_skills:
                print(f"   沟通技能: {communication_skills}")
            if business_skills:
                print(f"   业务技能: {business_skills[:5]}")  # 只显示前5个
        except:
            pass
    
    # 按ISCO代码分析技能分布
    if 'ISCO_4_Digit_Code_Gemini' in result_df.columns:
        print(f"\n" + "="*60)
        print("按ISCO职业代码分析技能分布")
        print("="*60)
        
        isco_skill_analysis = analyze_skills_by_isco(result_df)
        
        # 显示前5个ISCO类别的技能分布
        for i, (isco_code, analysis) in enumerate(list(isco_skill_analysis.items())[:5]):
            print(f"\nISCO {isco_code} (共{analysis['job_count']}个职位):")
            print(f"  平均技能数: {analysis['avg_skills']:.1f}")
            print(f"  前5个常见技能: {analysis['top_skills'][:5]}")
        
        # 保存ISCO分析结果
        with open('isco_skill_distribution.json', 'w', encoding='utf-8') as f:
            json.dump(isco_skill_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\nISCO技能分析已保存到: isco_skill_distribution.json")

def analyze_skills_by_isco(df: pd.DataFrame) -> Dict:
    """按ISCO代码分析技能分布"""
    isco_analysis = defaultdict(lambda: {
        'job_count': 0,
        'total_skills': 0,
        'skill_counter': Counter(),
        'avg_skills': 0.0,
        'top_skills': []
    })
    
    for idx, row in df.iterrows():
        isco_code = row.get('ISCO_4_Digit_Code_Gemini')
        if pd.isna(isco_code):
            continue
        
        isco_code = str(int(isco_code)) if isinstance(isco_code, float) else str(isco_code)
        
        # 统计职位数量
        isco_analysis[isco_code]['job_count'] += 1
        
        # 统计技能
        skills_count = row.get('技能数量', 0)
        isco_analysis[isco_code]['total_skills'] += skills_count
        
        # 统计具体技能
        try:
            normalized_skills = json.loads(row.get('标准化技能', '[]'))
            for skill in normalized_skills:
                isco_analysis[isco_code]['skill_counter'][skill] += 1
        except:
            pass
    
    # 计算平均值和排序
    final_analysis = {}
    for isco_code, data in isco_analysis.items():
        if data['job_count'] > 0:
            avg_skills = data['total_skills'] / data['job_count']
            top_skills = [(skill, count) for skill, count in data['skill_counter'].most_common(20)]
            
            final_analysis[isco_code] = {
                'job_count': data['job_count'],
                'total_skills': data['total_skills'],
                'avg_skills': round(avg_skills, 2),
                'top_skills': top_skills,
                'skill_diversity': len(data['skill_counter'])
            }
    
    return final_analysis

if __name__ == "__main__":
    main()