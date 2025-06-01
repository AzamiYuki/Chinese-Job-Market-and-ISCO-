import csv
from openai import OpenAI
import time
import os
import pandas as pd
import re # For parsing REVIEW_NEEDED responses
import traceback # For detailed error logging

# --- Configuration Parameters ---
API_KEY = "sk-eUOQ45HD9kB25oU1dJhmRfmjLGi7tYwDFopXmNoGnGv95vuF"
BASE_URL = "https://www.chataiapi.com/v1"

# Model names for each pass
MODEL_NAME_PASS1 = "gemini-2.0-flash" # Model for initial classification
MODEL_NAME_PASS2 = "gemini-2.0-flash" # Model for validation and re-classification

# --- Pass 1: Initial Classification Configuration ---
INPUT_CSV_FILE_PASS1 = '/Users/wyh/Desktop/统计计算/newjob1.csv'
OUTPUT_CSV_FILE_PASS1 = '/Users/wyh/Desktop/统计计算/newjob1_isco_4digit_gemini_sorted.csv' # Output of Pass 1
COLUMNS_TO_READ_PASS1 = ['岗位', '岗位描述', '岗位职能']
NEW_COLUMN_CODE_PASS1 = 'ISCO_4_Digit_Code_Gemini' # Column added by Pass 1

# --- Pass 2: Validation and Re-classification Configuration ---
# Input for Pass 2 is the output of Pass 1.
# User confirmed this file is already sorted as needed or sorting within Pass 2 is not required.
INPUT_CSV_FILE_PASS2 = OUTPUT_CSV_FILE_PASS1
OUTPUT_CSV_FILE_PASS2 = '/Users/wyh/Desktop/统计计算/newjob1_isco_validated_gemini_pro.csv' # Final output
NEW_COLUMN_VALIDATION_PASS2 = 'ISCO_Validation_Gemini_Pro' # Column added by Pass 2

# Column names expected in the input for Pass 2
JOB_TITLE_COL_PASS2 = '岗位'
JOB_DESC_COL_PASS2 = '岗位描述'
JOB_FUNC_COL_PASS2 = '岗位职能'
ISCO_CODE_COL_PASS2 = NEW_COLUMN_CODE_PASS1 # This is 'ISCO_4_Digit_Code_Gemini'

# General Configuration
API_CALL_DELAY = 0.6
MAX_ROWS_TO_PROCESS = None # Set to a number for testing (e.g., 100), None for all
GROUP_ASSESSMENT_CHUNK_SIZE = 10 # Max entries per chunk for group assessment in Pass 2

# --- Initialize API Client ---
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Utility Function (Used by both passes potentially) ---
def validate_api_response_code(response_code):
    """
    Validates API response for 4-digit ISCO code.
    """
    if response_code == "API_ERROR":
        return response_code
    if response_code == "N/A":
        return "N/A"
    if isinstance(response_code, str) and len(response_code) == 4 and response_code.isdigit():
        return response_code
    else:
        # This print can be verbose if many format errors occur.
        # print(f"  Warning: API response '{response_code}' not a 4-digit code or N/A. Recording as 'Format_Error'.")
        return "Format_Error"

# --- Pass 1 Functions ---
def query_gemini_for_isco_code_pass1(job_title, job_description, job_function):
    """
    Calls API for initial ISCO classification (Pass 1).
    Uses MODEL_NAME_PASS1.
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
            model=MODEL_NAME_PASS1,
            messages=[
                {"role": "system", "content": "你是一位精通国际标准职业分类 (ISCO) 的专家，能够根据职业描述准确识别ISCO四位编码。请直接输出四位数字编码或N/A。"},
                {"role": "user", "content": prompt_content}
            ],
            stream=False, max_tokens=10, temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API call error (Pass 1, {MODEL_NAME_PASS1}): {e}")
        return "API_ERROR"

def process_csv_classification_pass1():
    """
    Reads CSV, performs initial ISCO classification (Pass 1), writes to new CSV.
    This function is based on the code you provided.
    """
    processed_rows_count = 0
    print(f"--- Starting Pass 1: Initial ISCO Classification ---")
    print(f"Processing file: {INPUT_CSV_FILE_PASS1} (using {MODEL_NAME_PASS1})")
    print(f"Output will be saved to: {OUTPUT_CSV_FILE_PASS1}")
    if MAX_ROWS_TO_PROCESS is not None:
        print(f"Note: Max {MAX_ROWS_TO_PROCESS} rows will be processed for Pass 1.")

    try:
        with open(INPUT_CSV_FILE_PASS1, mode='r', encoding='utf-8-sig') as infile, \
             open(OUTPUT_CSV_FILE_PASS1, mode='w', encoding='utf-8', newline='') as outfile:

            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: Input file '{INPUT_CSV_FILE_PASS1}' is empty or has no header.")
                return

            missing_cols = [col for col in COLUMNS_TO_READ_PASS1 if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: Input file '{INPUT_CSV_FILE_PASS1}' is missing required columns for API prompt: {', '.join(missing_cols)}")
                return

            # Preserve all original columns and add the new one
            fieldnames_out = list(reader.fieldnames) + [NEW_COLUMN_CODE_PASS1]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames_out)
            writer.writeheader()

            print("Starting data row processing (Pass 1)...")
            for i, row in enumerate(reader):
                if MAX_ROWS_TO_PROCESS is not None and processed_rows_count >= MAX_ROWS_TO_PROCESS:
                    print(f"Reached processing limit of {MAX_ROWS_TO_PROCESS} rows for Pass 1.")
                    break
                
                current_row_number = i + 1 # For user-friendly row numbering in logs (1-based)
                job_title = row.get(COLUMNS_TO_READ_PASS1[0], "")
                job_description = row.get(COLUMNS_TO_READ_PASS1[1], "")
                job_function = row.get(COLUMNS_TO_READ_PASS1[2], "")

                api_isco_code = "N/A_NoInputData" # Default if all input fields for API are empty

                if job_title or job_description or job_function: # Only call API if there's data
                    raw_api_response = query_gemini_for_isco_code_pass1(job_title, job_description, job_function)
                    api_isco_code = validate_api_response_code(raw_api_response) # Validate the format
                    if processed_rows_count % 10 == 0 : 
                         print(f"  Row {current_row_number} (Pass 1) - API ({MODEL_NAME_PASS1}) Raw: '{raw_api_response}', Validated: {api_isco_code}")
                else:
                    if processed_rows_count % 10 == 0 :
                        print(f"  Row {current_row_number} (Pass 1) - Input data for API empty, assigned '{api_isco_code}'.")
                
                row_to_write = row.copy() # Start with all original data from the row
                row_to_write[NEW_COLUMN_CODE_PASS1] = api_isco_code # Add the new ISCO code
                
                writer.writerow(row_to_write)
                processed_rows_count += 1

                if processed_rows_count > 0 and processed_rows_count % 100 == 0: 
                    print(f"  Processed {processed_rows_count} rows (Pass 1). Flushing to file...")
                    outfile.flush() # Write data to disk periodically

                time.sleep(API_CALL_DELAY)

            print(f"\nPass 1 Finished. Total rows processed: {processed_rows_count}.")
            print(f"Results saved to: {OUTPUT_CSV_FILE_PASS1}")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV_FILE_PASS1}' not found. Please check the path.")
    except Exception as e:
        print(f"Unexpected error during CSV processing (Pass 1): {e}")
        traceback.print_exc()

# --- Pass 2 Functions ---
def query_gemini_for_reclassification_pass2(job_title, job_description, job_function):
    """Handles re-classification for error/N/A codes in Pass 2. Uses MODEL_NAME_PASS2."""
    prompt_content = f"请根据以下提供的中文招聘信息，判断其最匹配的四位数字ISCO编码。\n\n岗位名称：{job_title if job_title else '未提供'}\n岗位描述：{job_description if job_description else '未提供'}\n岗位职能：{job_function if job_function else '未提供'}\n\n请仅返回四位数字的ISCO编码。例如：2351\n如果无法准确判断，请返回 \"N/A\"。\n不要包含任何其他解释、说明文字或标签。"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME_PASS2,
            messages=[
                {"role": "system", "content": "你是一位精通国际标准职业分类 (ISCO) 的专家，能够根据职业描述准确识别ISCO四位编码。请直接输出四位数字编码或N/A。"},
                {"role": "user", "content": prompt_content}
            ],
            stream=False, max_tokens=10, temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API call error (Re-classification Pass 2, {MODEL_NAME_PASS2}): {e}")
        return "API_ERROR"

def query_gemini_for_validation_pass2(job_title, job_description, job_function, original_isco_code):
    """Handles individual row validation in Pass 2. Uses MODEL_NAME_PASS2."""
    prompt_content = f"Given the following job information:\n岗位名称：{job_title if job_title else '未提供'}\n岗位描述：{job_description if job_description else '未提供'}\n岗位职能：{job_function if job_function else '未提供'}\n\nThe job was previously classified with the ISCO-08 four-digit code: {original_isco_code}.\n\nPlease verify if this classification is correct for this specific job entry.\n- If the code {original_isco_code} is appropriate for the job described, please respond with the single word: correct\n- If the code {original_isco_code} is NOT appropriate, please provide the correct four-digit ISCO-08 code followed by a brief reason for the change, formatted EXACTLY as: XXXX - Brief reason.\n- Do not include any other explanations or introductory text. Only \"correct\" or \"XXXX - Brief reason.\""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME_PASS2,
            messages=[
                {"role": "system", "content": "You are an expert in ISCO-08 classification. Verify the provided code for the single job entry or suggest a correction with a reason, following the specified format."},
                {"role": "user", "content": prompt_content}
            ],
            stream=False, max_tokens=100, temperature=0.0
        )
        validation_result = response.choices[0].message.content.strip()
        if validation_result.lower() == "correct": return "correct"
        if len(validation_result) > 7 and validation_result[0:4].isdigit() and " - " in validation_result:
            return validation_result
        print(f"  Warning: Validation API ({MODEL_NAME_PASS2}) response '{validation_result}' for code {original_isco_code} not in expected format.")
        return f"Validation_Format_Unexpected:{validation_result}" # Return full unexpected response
    except Exception as e:
        print(f"  API call error (Validation Pass 2 for code {original_isco_code}, {MODEL_NAME_PASS2}): {e}")
        return f"Validation_API_ERROR"

def query_gemini_for_group_assessment(job_entries_str, original_isco_code_of_group):
    """Asks LLM to assess a group/chunk of job entries. Uses MODEL_NAME_PASS2."""
    prompt_content = f"""The following job entries were all previously classified with ISCO code {original_isco_code_of_group}.
Each entry is identified by its original DataFrame index prefix (e.g., EntryDFIndex_XXX).
Please review them:
{job_entries_str}

Are all of these job entries correctly classified under ISCO code {original_isco_code_of_group}?
- If ALL entries are correctly classified, respond ONLY with the word: ALL_CORRECT
- If one or more entries appear misclassified or require individual review, respond with "REVIEW_NEEDED" followed by a comma-separated list of the full EntryDFIndex_XXX identifiers that need closer inspection. For example: REVIEW_NEEDED: EntryDFIndex_123, EntryDFIndex_456
- If you cannot determine for the group, respond with "UNCLEAR".
Please ensure your response strictly follows one of these three formats.
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME_PASS2,
            messages=[
                {"role": "system", "content": f"You are an ISCO classification review expert. Assess the provided list of job entries against ISCO code {original_isco_code_of_group} and respond in the specified format."},
                {"role": "user", "content": prompt_content}
            ],
            stream=False, max_tokens=250, temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API call error (Group Assessment for code {original_isco_code_of_group}, {MODEL_NAME_PASS2}): {e}")
        return "GroupAssessment_API_ERROR"

def process_assessment_response_for_sub_group(sub_group_df, assessment_response, original_isco_code_of_group, main_df_to_update, current_api_calls):
    """Processes group/chunk assessment response, updates DataFrame, returns updated API call count."""
    if assessment_response == "ALL_CORRECT":
        print(f"    Sub-group/Chunk for {original_isco_code_of_group}: ALL_CORRECT.")
        for index, row in sub_group_df.iterrows():
            main_df_to_update.loc[index, NEW_COLUMN_VALIDATION_PASS2] = "correct (group_assessed)"
    elif assessment_response.startswith("REVIEW_NEEDED"):
        print(f"    Sub-group/Chunk for {original_isco_code_of_group}: REVIEW_NEEDED. Details: {assessment_response}")
        flagged_df_indices = []
        try:
            raw_indices = re.findall(r"EntryDFIndex_(\d+)", assessment_response)
            flagged_df_indices = [int(idx_str) for idx_str in raw_indices]
            valid_indices_in_sub_group = set(sub_group_df.index) # Get actual indices from the sub_group_df
            flagged_df_indices = [idx for idx in flagged_df_indices if idx in valid_indices_in_sub_group] # Filter
            print(f"      Flagged DataFrame indices within this sub-group/chunk for {original_isco_code_of_group}: {flagged_df_indices}")
        except Exception as e_parse:
            print(f"      Error parsing REVIEW_NEEDED indices: {e_parse}. Response: {assessment_response}")
        
        for index, row in sub_group_df.iterrows():
            if index in flagged_df_indices:
                job_title = str(row.get(JOB_TITLE_COL_PASS2, ""))
                job_description = str(row.get(JOB_DESC_COL_PASS2, ""))
                job_function = str(row.get(JOB_FUNC_COL_PASS2, ""))
                if job_title or job_description or job_function:
                    print(f"      Individually validating flagged row (Original DF Index {index}) for {original_isco_code_of_group}...")
                    result = query_gemini_for_validation_pass2(job_title, job_description, job_function, original_isco_code_of_group)
                    current_api_calls += 1
                    main_df_to_update.loc[index, NEW_COLUMN_VALIDATION_PASS2] = result
                    time.sleep(API_CALL_DELAY)
                else:
                    main_df_to_update.loc[index, NEW_COLUMN_VALIDATION_PASS2] = f"NoInput_FlaggedRow_{original_isco_code_of_group}"
            else:
                main_df_to_update.loc[index, NEW_COLUMN_VALIDATION_PASS2] = "correct (not_flagged_in_group_review)"
    elif assessment_response == "UNCLEAR" or assessment_response == "GroupAssessment_API_ERROR" or assessment_response.startswith("Validation_Format_Unexpected"):
        print(f"    Sub-group/Chunk for {original_isco_code_of_group}: Response was '{assessment_response}'.")
        for index, row in sub_group_df.iterrows():
            main_df_to_update.loc[index, NEW_COLUMN_VALIDATION_PASS2] = assessment_response
    else: 
        print(f"    Sub-group/Chunk for {original_isco_code_of_group}: Unexpected group assessment response: '{assessment_response}'.")
        for index, row in sub_group_df.iterrows():
            main_df_to_update.loc[index, NEW_COLUMN_VALIDATION_PASS2] = f"GroupReview_Unexpected:{assessment_response}"
    return current_api_calls

def process_csv_validation_and_reclassification_pass2():
    """Reads Pass 1 output, performs chunked group validation, and writes final output."""
    print(f"\n--- Starting Pass 2: ISCO Code Validation and Re-classification (Chunked Group Approach) ---")
    print(f"Input file for Pass 2 (expects pre-sorted or accepts natural group order): {INPUT_CSV_FILE_PASS2}")
    print(f"Final validated output will be saved to: {OUTPUT_CSV_FILE_PASS2}")
    if MAX_ROWS_TO_PROCESS is not None:
        print(f"Note: Max {MAX_ROWS_TO_PROCESS} rows will be processed in Pass 2.")

    try:
        print(f"\nStep 2.1: Reading file '{INPUT_CSV_FILE_PASS2}' for processing (using {MODEL_NAME_PASS2}).")
        if not os.path.exists(INPUT_CSV_FILE_PASS2):
            print(f"Error: Input file for Pass 2 '{INPUT_CSV_FILE_PASS2}' not found. Run Pass 1 first."); return

        df = pd.read_csv(INPUT_CSV_FILE_PASS2, dtype=str) # Read all as string
        if df.empty: print(f"Error: Input file '{INPUT_CSV_FILE_PASS2}' is empty."); return
        
        if ISCO_CODE_COL_PASS2 not in df.columns:
            print(f"Error: ISCO code column '{ISCO_CODE_COL_PASS2}' not found in '{INPUT_CSV_FILE_PASS2}'.")
            return
            
        df[NEW_COLUMN_VALIDATION_PASS2] = "" # Initialize new column
        api_calls_pass2 = 0
        total_rows_processed_in_pass2 = 0

        # Group by the ISCO code from Pass 1. sort=False preserves existing order if file is pre-sorted.
        grouped = df.groupby(ISCO_CODE_COL_PASS2, sort=False) 

        for original_isco_code, group_df in grouped:
            print(f"\nProcessing group with ISCO code: '{original_isco_code}' (Total entries in group: {len(group_df)})")
            current_group_rows_processed_this_group = 0 # Counter for rows processed within the current original_isco_code group

            # Handle 4-digit numeric ISCO codes with chunked group assessment
            if original_isco_code.isdigit() and len(original_isco_code) == 4:
                num_entries_in_group = len(group_df)
                for i in range(0, num_entries_in_group, GROUP_ASSESSMENT_CHUNK_SIZE):
                    chunk_df = group_df.iloc[i:i + GROUP_ASSESSMENT_CHUNK_SIZE]
                    
                    if num_entries_in_group > GROUP_ASSESSMENT_CHUNK_SIZE:
                        chunk_num = i // GROUP_ASSESSMENT_CHUNK_SIZE + 1
                        total_chunks = (num_entries_in_group + GROUP_ASSESSMENT_CHUNK_SIZE - 1) // GROUP_ASSESSMENT_CHUNK_SIZE
                        print(f"  Processing chunk {chunk_num}/{total_chunks} for {original_isco_code} (Entries in chunk: {len(chunk_df)}, DF Indices: {chunk_df.index.min()}-{chunk_df.index.max()})...")
                    else:
                        print(f"  Processing group {original_isco_code} as a single unit (Entries: {num_entries_in_group}).")

                    chunk_entries_for_prompt_list = []
                    for index_in_chunk, row_in_chunk in chunk_df.iterrows(): # index_in_chunk is original DF index
                        title = str(row_in_chunk.get(JOB_TITLE_COL_PASS2, "N/A")).replace("\n", " ").strip()
                        chunk_entries_for_prompt_list.append(f"EntryDFIndex_{index_in_chunk}: {title[:100]}") 
                    
                    chunk_entries_str = "\n".join(chunk_entries_for_prompt_list)
                    if not chunk_entries_str.strip() : 
                        print(f"    Skipping group assessment for this chunk of {original_isco_code}: all job titles are empty or invalid.")
                        for index_in_chunk, _ in chunk_df.iterrows():
                             df.loc[index_in_chunk, NEW_COLUMN_VALIDATION_PASS2] = "NoInput_For_Chunk_Assessment"
                        current_group_rows_processed_this_group += len(chunk_df)
                        continue # Move to the next chunk or group

                    assessment_response = query_gemini_for_group_assessment(chunk_entries_str, original_isco_code)
                    api_calls_pass2 += 1
                    time.sleep(API_CALL_DELAY)
                    
                    api_calls_pass2 = process_assessment_response_for_sub_group(chunk_df, assessment_response, original_isco_code, df, api_calls_pass2)
                    current_group_rows_processed_this_group += len(chunk_df)

            # Handle non-4-digit codes (API_ERROR, N/A, etc.) - row by row re-classification
            elif original_isco_code in ["API_ERROR", "N/A", "Format_Error"]:
                print(f"  Re-classifying entries for code '{original_isco_code}' individually...")
                for index, row in group_df.iterrows():
                    job_title = str(row.get(JOB_TITLE_COL_PASS2, ""))
                    job_description = str(row.get(JOB_DESC_COL_PASS2, ""))
                    job_function = str(row.get(JOB_FUNC_COL_PASS2, ""))
                    if job_title or job_description or job_function:
                        result = query_gemini_for_reclassification_pass2(job_title, job_description, job_function)
                        api_calls_pass2 += 1
                        df.loc[index, NEW_COLUMN_VALIDATION_PASS2] = f"Reclassified_from_{original_isco_code}_to:_{validate_api_response_code(result)}"
                        time.sleep(API_CALL_DELAY)
                    else:
                        df.loc[index, NEW_COLUMN_VALIDATION_PASS2] = f"Skipped_Reclass_NoInputData_Was_{original_isco_code}"
                    current_group_rows_processed_this_group += 1
            elif original_isco_code == "N/A_NoInputData":
                for index, row in group_df.iterrows():
                    df.loc[index, NEW_COLUMN_VALIDATION_PASS2] = "Original_NoInputData"
                    current_group_rows_processed_this_group += 1
            else: # Other unexpected codes
                for index, row in group_df.iterrows():
                    df.loc[index, NEW_COLUMN_VALIDATION_PASS2] = f"Unhandled_Original_Code:_{original_isco_code}"
                    current_group_rows_processed_this_group += 1
            
            total_rows_processed_in_pass2 += current_group_rows_processed_this_group
            
            # Intermediate save logic
            save_threshold = 200 
            if total_rows_processed_in_pass2 > 0 and \
               (total_rows_processed_in_pass2 // save_threshold) > \
               ((total_rows_processed_in_pass2 - current_group_rows_processed_this_group) // save_threshold):
                print(f"  Processed {total_rows_processed_in_pass2} total rows. API calls this pass: {api_calls_pass2}. Flushing to file...")
                df.to_csv(OUTPUT_CSV_FILE_PASS2, index=False, encoding='utf-8')
            
            if MAX_ROWS_TO_PROCESS is not None and total_rows_processed_in_pass2 >= MAX_ROWS_TO_PROCESS:
                print(f"Reached processing limit of {MAX_ROWS_TO_PROCESS} total rows for Pass 2.")
                break # Break from the loop over groups
        
        print(f"\nPass 2 Finished. Total rows processed: {total_rows_processed_in_pass2}. Total API calls in Pass 2: {api_calls_pass2}.")
        df.to_csv(OUTPUT_CSV_FILE_PASS2, index=False, encoding='utf-8') # Final save
        print(f"Final validated results saved to: {OUTPUT_CSV_FILE_PASS2}")

    except FileNotFoundError:
        print(f"Error: Input file for Pass 2 '{INPUT_CSV_FILE_PASS2}' not found.")
    except Exception as e:
        print(f"Unexpected error during CSV processing (Pass 2 main logic): {e}")
        traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    if not API_KEY or API_KEY == "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" or API_KEY == "YOUR_PLACEHOLDER_API_KEY_HERE": 
        print("Error: Please replace 'API_KEY' at the top of the script with your valid Gemini API key.")
    else:
        # --- Option to run Pass 1: Initial Classification ---
        run_pass_1 = False # SET TO True TO RUN PASS 1
        if run_pass_1:
            if not os.path.exists(INPUT_CSV_FILE_PASS1):
                print(f"Error: Input file for Pass 1 '{INPUT_CSV_FILE_PASS1}' not found.")
            else:
                process_csv_classification_pass1()
        else:
            print(f"Pass 1 (Initial Classification) is SKIPPED as run_pass_1 is False.")
            
        # --- Option to run Pass 2: Validation and Re-classification ---
        run_pass_2 = True # SET TO True TO RUN PASS 2
        if run_pass_2:
            # Pass 2 expects the output of Pass 1 as its input
            if not os.path.exists(INPUT_CSV_FILE_PASS2): 
                 print(f"\nError: Input file for Pass 2 '{INPUT_CSV_FILE_PASS2}' not found. Ensure Pass 1 has run or the file exists.")
            else:
                process_csv_validation_and_reclassification_pass2()
        else:
            print(f"\nPass 2 (Validation and Re-classification) is SKIPPED as run_pass_2 is False.")
        
        print("\n--- Script Execution Finished ---")
