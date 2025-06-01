import csv
from openai import OpenAI # We can still use the openai library due to the compatible endpoint
import time
import os

# --- 配置参数 ---
# GEMINI API (via OpenAI-compatible proxy) Configuration
API_KEY = "sk-eUO******"  # Gemini API卡号
BASE_URL = "https://www.chataiapi.com/v1" 
MODEL_NAME = "gemini-2.0-flash"  # 或 "gemini-2.5-pro-preview-03-25" (确保全称无误)

=
INPUT_CSV_FILE = '/Users/wyh/Desktop/统计计算/newjob1.csv'
# 新的输出文件路径，反映使用的是Gemini API
OUTPUT_CSV_FILE = '/Users/wyh/Desktop/统计计算/newjob1_isco_4digit_gemini.csv'

# CSV中需要读取的列名
COLUMNS_TO_READ = ['岗位', '岗位描述', '岗位职能']
# 新增的列名
NEW_COLUMN_CODE = 'ISCO_4_Digit_Code_Gemini' # 列名也反映API来源

# API调用间的等待时间（秒），避免频率限制
API_CALL_DELAY = 0.6 # 根据API提供商的限制调整, Gemini可能也需要

# 处理的最大行数 (设置为None则处理所有行, 设置为具体数字如1000则只处理前1000行)
MAX_ROWS_TO_PROCESS = None # 设置为 50000 或 None 处理全部

# --- 初始化API客户端 (使用OpenAI库，但配置为Gemini的代理) ---
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def query_gemini_for_isco_code(job_title, job_description, job_function):
    """
    调用Gemini API (via proxy) 获取ISCO四级分类代码。
    """
    prompt_content = f"""
请根据以下提供的中文招聘信息，判断其最匹配的四位数字ISCO编码。

岗位名称：{job_title if job_title else "未提供"}
岗位描述：{job_description if job_description else "未提供"}
岗位职能：{job_function if job_function else "未提供"}

请仅返回四位数字的ISCO编码。例如：2351
如果无法准确判断，请返回 "N/A"。
不要包含任何其他解释、说明文字或标签（如“ISCO编码：”）。
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一位精通国际标准职业分类 (ISCO) 的专家，能够根据职业描述准确识别ISCO四位编码。请直接输出四位数字编码或N/A。"},
                {"role": "user", "content": prompt_content}
            ],
            stream=False,
            max_tokens=10,  # 足够容纳四位数字编码或 "N/A"
            temperature=0.0  # 为了结果的确定性
        )
        # 直接获取返回的文本
        code = response.choices[0].message.content.strip()
        return code
    except Exception as e:
        print(f"  API调用时发生错误: {e}")
        return f"API_ERROR" # 返回错误信息，以便记录

def validate_api_response_code(response_code):
    """
    验证API返回的编码是否符合预期（四位数字或N/A）。
    """
    if response_code == "API_ERROR":
        return response_code # API错误直接透传

    if response_code == "N/A":
        return "N/A"
    
    if len(response_code) == 4 and response_code.isdigit():
        return response_code
    else:
        print(f"  警告: API返回内容 '{response_code}' 不符合预期的四位数字或N/A格式。将记录为 'Format_Error'.")
        return "Format_Error"


def process_csv_classification():
    """
    读取CSV文件，进行ISCO四级分类（仅编码，使用Gemini API），并将结果写入新的CSV文件。
    """
    processed_rows_count = 0
    print(f"准备处理文件: {INPUT_CSV_FILE} (使用Gemini API)")
    print(f"结果将保存到: {OUTPUT_CSV_FILE}")
    if MAX_ROWS_TO_PROCESS is not None:
        print(f"注意：本次运行最多处理 {MAX_ROWS_TO_PROCESS} 行数据。")

    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8-sig') as infile, \
             open(OUTPUT_CSV_FILE, mode='w', encoding='utf-8', newline='') as outfile:

            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"错误：输入文件 '{INPUT_CSV_FILE}' 为空或没有表头。")
                return

            missing_cols = [col for col in COLUMNS_TO_READ if col not in reader.fieldnames]
            if missing_cols:
                print(f"错误：输入文件 '{INPUT_CSV_FILE}' 缺少以下列: {', '.join(missing_cols)}")
                return

            fieldnames_out = list(reader.fieldnames) + [NEW_COLUMN_CODE]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames_out)
            writer.writeheader()

            print("开始处理数据行...")
            for i, row in enumerate(reader):
                if MAX_ROWS_TO_PROCESS is not None and processed_rows_count >= MAX_ROWS_TO_PROCESS:
                    print(f"已达到处理上限 {MAX_ROWS_TO_PROCESS} 行。")
                    break
                
                current_row_number = i + 1
                job_title = row.get(COLUMNS_TO_READ[0], "")
                job_description = row.get(COLUMNS_TO_READ[1], "")
                job_function = row.get(COLUMNS_TO_READ[2], "")

                api_isco_code = "N/A_NoInputData"

                if job_title or job_description or job_function:
                    raw_api_response = query_gemini_for_isco_code(job_title, job_description, job_function)
                    api_isco_code = validate_api_response_code(raw_api_response)
                    if processed_rows_count % 10 == 0 : 
                         print(f"  第 {current_row_number} 行 - API (Gemini) 返回编码: {api_isco_code}")
                else:
                    if processed_rows_count % 10 == 0 :
                        print(f"  第 {current_row_number} 行 - 输入数据为空，跳过API调用。")

                row_to_write = row.copy() 
                row_to_write[NEW_COLUMN_CODE] = api_isco_code
                
                writer.writerow(row_to_write)
                processed_rows_count += 1

                if processed_rows_count % 100 == 0: 
                    print(f"  已处理 {processed_rows_count} 行。正在写入文件...")
                    outfile.flush() 

                time.sleep(API_CALL_DELAY)

            print(f"\n处理完成。总共处理了 {processed_rows_count} 行数据。")
            print(f"结果已保存到: {OUTPUT_CSV_FILE}")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{INPUT_CSV_FILE}' 未找到。请检查路径是否正确。")
    except Exception as e:
        print(f"处理CSV文件时发生未预料的错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not API_KEY or API_KEY == "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx": 
        print("错误：请在脚本顶部将 'API_KEY' 变量替换为您的有效Gemini API卡号。")
    elif not os.path.exists(INPUT_CSV_FILE):
        print(f"错误: 输入文件 '{INPUT_CSV_FILE}' 未找到。请确保文件路径正确。")
        print("程序将不会运行。")
    else:
        process_csv_classification()
