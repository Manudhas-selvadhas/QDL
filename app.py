import os
import requests
import pandas as pd
import shutil
import tiktoken # For token counting, if used by existing logic
import time
import logging
import uuid
from flask import Flask, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
import psycopg2 # For PostgreSQL interaction
from psycopg2 import sql as psycopg2_sql # For safe SQL query construction
import pandasql as ps
import markdown
import re # Added re import
import json # Ensure json is imported
import io # Ensure io is imported
import base64 # Ensure base64 is imported
import subprocess
# import threading # Threading might not be needed for AI viz if config is returned directly

app = Flask(__name__)
CORS(app,
     resources={r"/api/*": {"origins": "*"}},
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
TXT_FOLDER = os.path.join(PROCESSED_FOLDER, "txt")
SPLIT_FOLDER = os.path.join(PROCESSED_FOLDER, "split")
SPLIT_OUTPUT_FOLDER = os.path.join(PROCESSED_FOLDER, "split_output")
MERGE_FOLDER = os.path.join(PROCESSED_FOLDER, "merge")
FINAL_RESULT_FOLDER = os.path.join(PROCESSED_FOLDER, "final_result")
AI_VIZ_TEMP_CONTEXT_FOLDER = os.path.join(PROCESSED_FOLDER, "ai_visualizations_temp_data") # For preparing context for AI


# PostgreSQL Credentials
PG_HOST = os.environ.get("PG_HOST", "database-1.cf4mucoqok1c.eu-north-1.rds.amazonaws.com")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "ASC_Postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "aws_postgres")
PG_DATABASE = os.environ.get("PG_DATABASE", "DE")

# API details
API_URL = "https://avaplus-internal.avateam.io/force/platform/pipeline/api/v1/execute"
ACCESS_KEY = os.environ.get("AVAPLUS_ACCESS_KEY", "eyJraWQiOiIwODMwYWI0MS05MzM5LTQwZTgtODhhZC1iMGQ3NDY1ZDAyM2MiLCJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2FzY2VuZGlvbi5jb20iLCJpZGVudGlmaWVyIjoiYXZhIy0qZ3B0IiwibmFtZSI6Imdsb2JhbCBhY2Nlc3MiLCJlbWFpbCI6Imdsb2JhbC5hY2Nlc3NrZXlAYXNjZW5kaW9uLmNvbSIsImlhdCI6MTcyNjc1MDk3MSwiZXhwIjoxNzY2MTAyNDAwfQ.ZvoB40XIXqWPfKtvCvLOCk2mSiOJ6cc11vSkSoTgu2I4q0TZ87ChXFGh9f6iJjyY7C2QaFJ5NzsUeq83g6mc_w")
HEADERS = {"access-key": ACCESS_KEY, "Content-Type": "application/json"}

TOKEN_LIMIT = 130000  # Token limit for main analysis processing
CONTEXT_TOKEN_LIMIT = 1000  # Token limit for ask question and AI visualization context
MERGE_TOKEN_LIMIT = 130000
DEFAULT_API_TIMEOUT = 1200
SUMMARIZATION_API_TIMEOUT = 1200 # Also used for AI viz config generation

processing_results = {}

for folder_path in [UPLOAD_FOLDER, PROCESSED_FOLDER, TXT_FOLDER, SPLIT_FOLDER,
                    SPLIT_OUTPUT_FOLDER, MERGE_FOLDER, FINAL_RESULT_FOLDER,
                    AI_VIZ_TEMP_CONTEXT_FOLDER, # Ensure this base temp folder for AI viz context exists
                    os.path.join(PROCESSED_FOLDER, "ai_visualizations")]: # Old folder, maybe still used for something else or can be removed if only for code execution
    os.makedirs(folder_path, exist_ok=True)

# --- Helper Functions ---
def convert_to_txt(input_file, txt_file):
    try:
        logger.info(f"Converting {input_file} to text format at {txt_file}")
        if input_file.endswith(".csv"):
            try:
                # Try to sniff the separator
                df_check = pd.read_csv(input_file, on_bad_lines='skip', sep=None, engine='python', iterator=True, nrows=5) 
                first_chunk = df_check.get_chunk(1)
                if first_chunk.empty: # Handle completely empty CSV
                     df = pd.DataFrame()
                else:
                    # Basic separator sniffing based on counts in the first non-empty line
                    sep_to_use = '\t' if first_chunk.iloc[0,0].count('\t') > first_chunk.iloc[0,0].count(',') else ','
                    df = pd.read_csv(input_file, on_bad_lines='skip', sep=sep_to_use)
            except Exception as sniff_err:
                logger.warning(f"CSV Sniffing failed for {input_file}: {sniff_err}. Falling back to comma.")
                df = pd.read_csv(input_file, on_bad_lines='skip') # Fallback to comma

        elif input_file.endswith((".xls", ".xlsx")):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported file format! Provide an Excel or CSV file.")

        os.makedirs(os.path.dirname(txt_file), exist_ok=True)
        if df.empty:
            logger.warning(f"DataFrame from {input_file} is empty after reading. Creating empty text file.")
            open(txt_file, 'w', encoding='utf-8').close() # Create an empty file
        else:
            df = df.astype(str) # Ensure all data is string to prevent to_csv errors with mixed types
            df.to_csv(txt_file, sep="\t", index=False, encoding='utf-8') # Use tab as a robust separator for text
        logger.info(f"Successfully converted to {txt_file}")
        return f"Converted {input_file} to text file: {txt_file}"
    except pd.errors.EmptyDataError:
        logger.warning(f"File {input_file} is empty or contains no data (pandas EmptyDataError).")
        os.makedirs(os.path.dirname(txt_file), exist_ok=True)
        open(txt_file, 'w', encoding='utf-8').close() # Create an empty file
        return f"File {input_file} was empty. Created empty text file: {txt_file}"
    except Exception as e:
        logger.exception(f"Error converting file to text: {str(e)}")
        raise

def count_tokens(file_path, model="gpt-4"):
    try:
        logger.info(f"Counting tokens in {file_path}")
        enc = tiktoken.encoding_for_model(model)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        token_count = len(enc.encode(text))
        logger.info(f"Token count for {file_path}: {token_count}")
        return token_count
    except FileNotFoundError:
        logger.error(f"File not found for token counting: {file_path}")
        return 0
    except Exception as e:
        logger.exception(f"Error counting tokens in {file_path}: {str(e)}")
        raise

def split_text_file(txt_file, output_dir, token_limit=TOKEN_LIMIT, create_only_first_split=False):
    try:
        logger.info(f"Splitting text file {txt_file} into {output_dir} with token limit {token_limit}, create_only_first_split: {create_only_first_split}")
        enc = tiktoken.encoding_for_model("gpt-4")
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if not lines:
            logger.warning(f"File {txt_file} is empty, no splitting needed.")
            os.makedirs(output_dir, exist_ok=True)
            # Create a dummy empty split file if the source is empty
            dummy_file_path = os.path.join(output_dir, "split_part_1.txt")
            with open(dummy_file_path, "w", encoding="utf-8") as df_dummy: df_dummy.write("")
            return f"Source file was empty. Created 1 empty split file."

        header = lines[0]
        data_rows = lines[1:]

        current_tokens_in_part = len(enc.encode(header))
        part_num = 1
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"split_part_{part_num}.txt")

        with open(output_file_path, "w", encoding="utf-8") as f_current_part:
            f_current_part.write(header)

            for row_content in data_rows:
                row_tokens = len(enc.encode(row_content))

                # Check if adding this row exceeds the limit AND if the current part already has some data (beyond just header)
                if current_tokens_in_part + row_tokens > token_limit and current_tokens_in_part > len(enc.encode(header)): 
                    if create_only_first_split:
                        logger.info(f"First split part ({output_file_path}) is complete with {current_tokens_in_part} tokens. create_only_first_split is True. Stopping.")
                        return f"First split part created. Created 1 file."
                    # Start a new part
                    part_num += 1
                    output_file_path = os.path.join(output_dir, f"split_part_{part_num}.txt")
                    with open(output_file_path, "w", encoding="utf-8") as f_new_part_header: # Write header to new part
                        f_new_part_header.write(header)
                    current_tokens_in_part = len(enc.encode(header)) # Reset token count for the new part

                # Append the current row to the current part's file
                with open(output_file_path, "a", encoding="utf-8") as f_append_to_current:
                     f_append_to_current.write(row_content)
                current_tokens_in_part += row_tokens

        if create_only_first_split:
            logger.info(f"All content fit into the first split part ({output_file_path}) with {current_tokens_in_part} tokens.")
            return f"First split part created (all content fit). Created 1 file."
        else:
            logger.info(f"Splitting complete. Created {part_num} files in {output_dir}.")
            return f"Splitting complete. Created {part_num} files."
    except Exception as e:
        logger.exception(f"Error splitting text file {txt_file}: {str(e)}")
        raise

def send_request_with_retries(payload, timeout_duration, max_retries=3, initial_delay=1):
    # Utility function to send API requests with retry logic
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending API request (attempt {attempt + 1}/{max_retries}, timeout: {timeout_duration}s) to {API_URL} with pipelineId {payload.get('pipeLineId')}")
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout_duration)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            logger.info(f"API request successful (status {response.status_code}) for pipelineId {payload.get('pipeLineId')}")
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text} for pipelineId {payload.get('pipeLineId')}")
            if e.response.status_code == 429 or e.response.status_code >= 500: # Retry on 429 (Too Many Requests) or 5xx server errors
                pass # Continue to retry logic
            else: # Don't retry for other client errors (e.g., 400, 401, 403, 404)
                return None # Or re-raise the exception if specific handling is needed
        except requests.exceptions.RequestException as e: # Catches other network issues (DNS failure, connection timeout, etc.)
            logger.exception(f"Request failed: {str(e)} for pipelineId {payload.get('pipeLineId')}")
        
        if attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt) # Exponential backoff
            logger.info(f"Retrying API call for pipelineId {payload.get('pipeLineId')} in {delay} seconds...")
            time.sleep(delay)
    logger.error(f"API request failed after {max_retries} attempts for pipelineId {payload.get('pipeLineId')}.")
    return None

def process_files(input_folder, output_folder, pipeline_id, user_question=None):
    # Processes text files in a folder using a specified API pipeline
    try:
        logger.info(f"Processing files from {input_folder} (to: {output_folder if output_folder else 'memory/direct_aggregation'}) with pipeline {pipeline_id}")
        if output_folder: os.makedirs(output_folder, exist_ok=True)
        files_to_process = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
        if not files_to_process:
            logger.warning(f"No .txt files found in {input_folder} to process.")
            return ("No .txt files found to process.", []) if pipeline_id == 1887 or pipeline_id == 2141 else "No .txt files found to process."

        results_log, all_processed_text_outputs = [], []
        current_api_timeout = SUMMARIZATION_API_TIMEOUT if pipeline_id == 901 or pipeline_id == 2141 else DEFAULT_API_TIMEOUT
        if pipeline_id == 901 or pipeline_id == 2141: 
            logger.info(f"Using extended timeout ({current_api_timeout}s) for pipeline {pipeline_id}.")

        for file_name in files_to_process:
            file_path = os.path.join(input_folder, file_name)
            logger.info(f"Processing file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file: file_content = file.read().strip()
            except UnicodeDecodeError: # Fallback for files not in UTF-8
                with open(file_path, "r", encoding="latin-1", errors="ignore") as file: file_content = file.read().strip()

            if not file_content:
                logger.warning(f"Skipping empty file: {file_path}"); results_log.append(f"Skipped empty file: {file_name}")
                if pipeline_id == 1887 or pipeline_id == 2141: all_processed_text_outputs.append(f"[Skipped empty part: {file_name}]")
                continue
            token_count = count_tokens(file_path)
            results_log.append(f"Processing {file_name} - Tokens: {token_count}"); logger.info(f"File {file_name} has {token_count} tokens.")

            pipeline_specific_token_limit = TOKEN_LIMIT
            if pipeline_id == 1887 or pipeline_id == 2141: # Ask Question or AI Viz context
                pipeline_specific_token_limit = CONTEXT_TOKEN_LIMIT

            if token_count > pipeline_specific_token_limit:
                logger.warning(f"Skipping {file_name} (pipeline {pipeline_id}) - Exceeds Token Limit ({token_count} > {pipeline_specific_token_limit})")
                results_log.append(f"Skipping {file_name} (pipeline {pipeline_id}) - Exceeds Token Limit: {token_count}")
                if pipeline_id == 1887 or pipeline_id == 2141: all_processed_text_outputs.append(f"[Skipped part due to token limit: {file_name}]")
                continue

            execution_id = str(uuid.uuid4())
            user_inputs = {"data": file_content} # Default for pipeline 895 (initial processing)
            if pipeline_id == 1887 and user_question: # Ask Question pipeline
                user_inputs = {"{{user_questions}}": user_question, "{{data_file}}": file_content}
            elif pipeline_id == 2141 and user_question: # AI Visualization config pipeline
                user_inputs = {"{{Data}}": file_content, "{{User_Prompt}}": user_question}
            # Add other pipeline-specific user_inputs here if needed for 901 (summarization)


            payload = {"pipeLineId": pipeline_id, "userInputs": user_inputs, "executionId": execution_id, "user": "manudhas.selvadhas@ascendion.com"}
            output_data = send_request_with_retries(payload, timeout_duration=current_api_timeout)

            if output_data and "pipeline" in output_data and "output" in output_data["pipeline"]:
                formatted_output = str(output_data["pipeline"]["output"]).strip()
                if pipeline_id == 1887 or pipeline_id == 2141: # For Ask Question or AI Viz, aggregate outputs in memory
                    all_processed_text_outputs.append(formatted_output)
                elif output_folder: # For other pipelines like 895, 901, save to file
                    output_file_path = os.path.join(output_folder, f"Processed_{file_name}")
                    with open(output_file_path, "w", encoding="utf-8") as out_f: out_f.write(formatted_output)
                    logger.info(f"Successfully processed {file_name} for pipeline {pipeline_id}, saved to {output_file_path}")
                results_log.append(f"Successfully processed {file_name} for pipeline {pipeline_id}")
            else:
                logger.warning(f"Invalid API response for {file_name} (pipeline {pipeline_id}). Response: {output_data}")
                results_log.append(f"API response issue for {file_name} (pipeline {pipeline_id})")
                if pipeline_id == 1887 or pipeline_id == 2141: 
                    all_processed_text_outputs.append(f"[Error processing part: {file_name} with API pipeline {pipeline_id}]")
            time.sleep(1) # Small delay between API calls
        
        if pipeline_id == 1887 or pipeline_id == 2141: # Return aggregated text for these pipelines
            return ("\n".join(results_log), all_processed_text_outputs)
        else:
            return "\n".join(results_log) # Return log string for other pipelines
            
    except Exception as e:
        logger.exception(f"Error in process_files (input: {input_folder}, pipeline: {pipeline_id}): {str(e)}")
        if pipeline_id == 1887 or pipeline_id == 2141: # Ensure tuple is returned for consistency
            return (f"Error: {str(e)}", [])
        else:
            raise


def merge_files_with_token_limit(input_folder, output_folder, token_limit=MERGE_TOKEN_LIMIT):
    # Merges multiple text files into fewer files, respecting a token limit per merged file
    try:
        logger.info(f"Merging files from {input_folder} to {output_folder} with token limit {token_limit}")
        os.makedirs(output_folder, exist_ok=True); enc = tiktoken.encoding_for_model("gpt-4")
        files_to_merge = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])
        if not files_to_merge: logger.warning(f"No .txt files found in {input_folder} to merge."); return 0

        part_num, current_tokens, current_text_content = 1, 0, []
        for file_name in files_to_merge:
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
            if not content.strip(): logger.info(f"Skipping empty file during merge: {file_name}"); continue

            # Add a separator if this is not the first piece of content in the current merged part
            content_to_add = ("\n\n" + "="*20 + f" Content from {file_name} " + "="*20 + "\n\n" + content) if current_text_content else content
            content_tokens = len(enc.encode(content_to_add))

            if current_tokens + content_tokens > token_limit and current_text_content: # If adding exceeds limit and there's content
                # Save current merged part
                out_path = os.path.join(output_folder, f"merged_part_{part_num}.txt")
                with open(out_path, "w", encoding="utf-8") as out_f: out_f.write("".join(current_text_content))
                logger.info(f"Saved merged part: {out_path}"); part_num += 1
                # Start new part with the current content (which was too large for the previous part)
                current_text_content, current_tokens = [content], len(enc.encode(content)) # Start new part with the *original* content, not content_to_add
            else:
                current_text_content.append(content_to_add); current_tokens += content_tokens

        if current_text_content: # Save any remaining content
            out_path = os.path.join(output_folder, f"merged_part_{part_num}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f: out_f.write("".join(current_text_content))
            logger.info(f"Saved final merged part: {out_path}")
        elif part_num == 1 and not files_to_merge: # Edge case: no files to merge resulted in no output
            logger.warning(f"No content to merge from {input_folder}. No merged files created."); return 0

        logger.info(f"Merging complete. Created {part_num if current_text_content else part_num -1} merged file(s) in {output_folder}.")
        return part_num if current_text_content else part_num -1 # Return number of files created
    except Exception as e: logger.exception(f"Error merging files (input: {input_folder}): {str(e)}"); raise


def recursive_merge_and_process(input_folder, pipeline_id, temp_base_dir, final_dir, token_limit, session_id):
    # Recursively merges and processes files until a single summary file is produced
    try:
        logger.info(f"Starting recursive merge for session {session_id}, input: {input_folder}")
        round_num, current_input_folder, operation_logs = 1, input_folder, []

        while True:
            logger.info(f"Recursive Merge - Round {round_num} for session {session_id}")
            round_merged_dir = os.path.join(temp_base_dir, f"round_{round_num}_merged_data")
            round_processed_dir = os.path.join(temp_base_dir, f"round_{round_num}_processed_data")
            os.makedirs(round_merged_dir, exist_ok=True); os.makedirs(round_processed_dir, exist_ok=True)

            num_merged_files = merge_files_with_token_limit(current_input_folder, round_merged_dir, token_limit)
            operation_logs.append(f"Round {round_num}: Merged from '{os.path.basename(current_input_folder)}' into {num_merged_files} file(s) in '{os.path.basename(round_merged_dir)}'.")

            if num_merged_files == 0:
                # If no files were merged, it might mean the input folder was empty or all files were empty.
                # Check if this is the first round and if the original input_folder actually had non-empty .txt files.
                if round_num > 1 or not any(f.endswith(".txt") and os.path.getsize(os.path.join(current_input_folder,f)) > 0 for f in os.listdir(current_input_folder)):
                    msg = f"Round {round_num}: No data to merge or process further from {'initial files' if current_input_folder == input_folder else 'previous round processing'}."
                    logger.warning(msg); operation_logs.append(msg)
                    # Attempt to find if a final result was already created in a previous iteration (e.g. if this is a re-run or error recovery)
                    existing_final_files = [f for f in os.listdir(final_dir) if f.startswith(f"Final_Analysis_Result_{session_id}")]
                    if existing_final_files:
                        return "\n".join(operation_logs), os.path.join(final_dir, existing_final_files[0]) # Return existing if found
                    return "\n".join(operation_logs), None # No result to return
                
            processing_log_summary = process_files(round_merged_dir, round_processed_dir, pipeline_id) # Process the merged files
            operation_logs.append(f"Round {round_num}: Processing summary for files in '{os.path.basename(round_merged_dir)}': {processing_log_summary}")

            processed_files_list = [f for f in os.listdir(round_processed_dir) if f.endswith(".txt") and os.path.getsize(os.path.join(round_processed_dir, f)) > 0]

            if len(processed_files_list) == 1: # Base case: only one processed file remains
                final_tmp_path = os.path.join(round_processed_dir, processed_files_list[0])
                os.makedirs(final_dir, exist_ok=True)
                final_path = os.path.join(final_dir, f"Final_Analysis_Result_{session_id}.txt")
                shutil.copy(final_tmp_path, final_path)
                logger.info(f"Final processing complete for session {session_id}. Output: {final_path}")
                operation_logs.append(f"Final result for session {session_id}: {os.path.basename(final_path)}")
                return "\n".join(operation_logs), final_path
            elif not processed_files_list: # No files produced after processing
                msg = f"Round {round_num}: Processing of merged files in '{os.path.basename(round_merged_dir)}' produced no output. Halting."
                logger.warning(msg); operation_logs.append(msg)
                existing_final_files = [f for f in os.listdir(final_dir) if f.startswith(f"Final_Analysis_Result_{session_id}")]
                if existing_final_files: # Check if a result from a previous attempt exists
                    return "\n".join(operation_logs), os.path.join(final_dir, existing_final_files[0]) 
                return "\n".join(operation_logs), None # No result
            
            current_input_folder = round_processed_dir # Next round's input is this round's output
            round_num += 1
            if round_num > 10: # Safety break for too many rounds
                logger.error(f"Max recursion depth reached for session {session_id}. Halting.");
                operation_logs.append("Error: Max recursion depth reached.")
                # Try to save the first file from the last processed set as a partial result
                last_processed_files = [os.path.join(current_input_folder, f) for f in os.listdir(current_input_folder) if f.endswith(".txt")]
                if last_processed_files: 
                    shutil.copy(last_processed_files[0], os.path.join(final_dir, f"Final_Analysis_Result_{session_id}_MaxDepth.txt"))
                    return "\n".join(operation_logs), os.path.join(final_dir, f"Final_Analysis_Result_{session_id}_MaxDepth.txt")
                return "\n".join(operation_logs), None
        logger.warning(f"Recursive processing for {session_id} exited loop unexpectedly.") # Should not be reached if logic is correct
        return "\n".join(operation_logs), None
    except Exception as e: logger.exception(f"Error in recursive_merge_and_process for {session_id}: {e}"); return f"Error in recursive merge: {e}", None

def process_analysis_workflow(file_path, session_id):
    # Main workflow for data analysis
    workflow_logs = []
    try:
        logger.info(f"Starting ANALYSIS workflow for session {session_id} with file {file_path}")
        if not os.path.exists(file_path): return {"success": False, "message": f"File not found: {file_path}", "logs": [f"File not found: {file_path}"]}

        # Define session-specific folders
        session_txt_folder = os.path.join(TXT_FOLDER, session_id)
        session_split_folder = os.path.join(SPLIT_FOLDER, session_id)
        session_split_output_folder = os.path.join(SPLIT_OUTPUT_FOLDER, session_id)
        session_merge_temp_base_dir = os.path.join(MERGE_FOLDER, session_id) # Temp for recursive merge
        session_final_folder = os.path.join(FINAL_RESULT_FOLDER, session_id) # Final results for this session
        for folder in [session_txt_folder, session_split_folder, session_split_output_folder, session_merge_temp_base_dir, session_final_folder]:
            os.makedirs(folder, exist_ok=True)

        logger.info(f"Analysis Step 1: Converting to TXT for session {session_id}")
        txt_file = os.path.join(session_txt_folder, "converted_data.txt")
        workflow_logs.append(convert_to_txt(file_path, txt_file))
        if not os.path.exists(txt_file) or os.path.getsize(txt_file) == 0: 
             workflow_logs.append("Text conversion failed or produced an empty file.")
             if not os.path.exists(txt_file): # If file doesn't exist at all
                return {"success": False, "message": "Text conversion failed.", "logs": workflow_logs}
             # If file exists but is empty, the workflow might still proceed if that's handled downstream,
             # but it's good to log. The main check is token_count.
             
        token_count = count_tokens(txt_file); workflow_logs.append(f"Total Token Count: {token_count}")

        if token_count <= TOKEN_LIMIT: # If file is small enough, process directly
            logger.info(f"File under token limit ({token_count} <= {TOKEN_LIMIT}), processing directly with pipeline 895 for session {session_id}.")
            workflow_logs.append("File under token limit, processing directly with pipeline 895.")

            # Prepare a "split" folder with just this one file for process_files consistency
            single_file_input_dir = os.path.join(session_split_folder, "direct_processing") # Use session_split_folder as base
            os.makedirs(single_file_input_dir, exist_ok=True)
            shutil.copy(txt_file, os.path.join(single_file_input_dir, "part_1.txt")) # Name it like a split part

            logger.info(f"Analysis Step 2: Direct processing with pipeline 895 from {single_file_input_dir} for session {session_id}.")
            process_result_log = process_files(single_file_input_dir, session_split_output_folder, 895) # Output to session_split_output_folder
            workflow_logs.append(f"Direct Processing Log: {process_result_log}")

            # Check if direct processing yielded any output
            processed_files = [f for f in os.listdir(session_split_output_folder) if f.endswith('.txt') and os.path.getsize(os.path.join(session_split_output_folder, f)) > 0]
            if not processed_files:
                return {"success": False, "message": "Direct processing (pipeline 895) yielded no results.", "logs": workflow_logs + ["Warning: Direct processing generated no output."]}

            # The result of direct processing is the final result
            processed_file_path = os.path.join(session_split_output_folder, processed_files[0])
            final_result_path = os.path.join(session_final_folder, f"Final_Analysis_Result_{session_id}.txt")
            shutil.copy(processed_file_path, final_result_path)

            logger.info(f"Analysis workflow (direct) complete for session {session_id}. Result: {final_result_path}")
            return {"success": True, "message": "File processed directly for analysis.", "logs": workflow_logs, "result_file": final_result_path, "result_filename": os.path.basename(final_result_path)}

        # File is large, needs splitting and recursive summarization
        logger.info(f"Token count {token_count} exceeds limit {TOKEN_LIMIT} for session {session_id}, splitting.")
        workflow_logs.append(split_text_file(txt_file, session_split_folder, TOKEN_LIMIT, create_only_first_split=False)) # Full split

        logger.info(f"Analysis Step 3: Initial processing (pipeline 895) on split files for session {session_id}.")
        initial_processing_log = process_files(session_split_folder, session_split_output_folder, 895) # Process all splits
        workflow_logs.append(f"Initial Processing (Pipeline 895) Log: {initial_processing_log}")

        # Check if initial processing yielded any output
        if not [f for f in os.listdir(session_split_output_folder) if f.endswith('.txt') and os.path.getsize(os.path.join(session_split_output_folder, f)) > 0]:
            return {"success": False, "message": "Initial processing of split files (pipeline 895) yielded no results.", "logs": workflow_logs + ["Warning: Initial processing of split files generated no output."]}

        logger.info(f"Analysis Step 4: Recursive merge and summarization (pipeline 901) for session {session_id}.")
        recursive_summary, final_result_path = recursive_merge_and_process(
            session_split_output_folder, 901, session_merge_temp_base_dir, 
            session_final_folder, MERGE_TOKEN_LIMIT, session_id
        )
        workflow_logs.append(f"Recursive Merge/Process (Pipeline 901) Log: {recursive_summary}")
        if not final_result_path or not os.path.exists(final_result_path):
            return {"success": False, "message": "Analysis summarization (recursive, pipeline 901) did not yield a final result.", "logs": workflow_logs + ["Warning: Recursive summarization failed to produce a final file."]}

        logger.info(f"Analysis workflow (recursive) complete for session {session_id}. Result: {final_result_path}")
        return {"success": True, "message": "File processed through full analysis workflow.", "logs": workflow_logs, "result_file": final_result_path, "result_filename": os.path.basename(final_result_path)}
    except Exception as e:
        logger.exception(f"Error in ANALYSIS workflow for {session_id}: {e}")
        return {"success": False, "message": f"Analysis error: {e}", "logs": workflow_logs + [f"Workflow Error: {e}"]}


# --- PostgreSQL Connection Function ---
def get_pg_connection():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        raise

# --- New PostgreSQL Endpoints ---
@app.route('/api/postgres/schemas', methods=['GET', 'OPTIONS'])
def get_postgres_schemas():
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)
    conn = None
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT schema_name FROM information_schema.schemata
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            AND schema_name NOT LIKE 'pg_temp_%' AND schema_name NOT LIKE 'pg_toast_temp_%'
            ORDER BY schema_name;
        """)
        schemas = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return jsonify(schemas), 200
    except Exception as e:
        logger.exception("Error fetching PostgreSQL schemas")
        return jsonify({"error": f"Failed to fetch schemas: {str(e)}"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/postgres/tables/<path:schema_name>', methods=['GET', 'OPTIONS'])
def get_postgres_tables(schema_name):
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)
    conn = None
    try:
        conn = get_pg_connection()
        cursor = conn.cursor()
        # Use psycopg2.sql for safe schema name insertion
        query = psycopg2_sql.SQL("""
            SELECT tablename FROM pg_catalog.pg_tables
            WHERE schemaname = %s
            ORDER BY tablename;
        """)
        cursor.execute(query, (schema_name,))
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return jsonify(tables), 200
    except Exception as e:
        logger.exception(f"Error fetching tables for schema {schema_name}")
        return jsonify({"error": f"Failed to fetch tables: {str(e)}"}), 500
    finally:
        if conn: conn.close()

@app.route('/api/postgres/upload-table', methods=['POST', 'OPTIONS'])
def upload_table_from_postgres():
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)
    conn = None
    try:
        data = request.get_json()
        schema_name = data.get('schema_name')
        table_name = data.get('table_name')

        if not schema_name or not table_name:
            return jsonify({"error": "Schema name and table name are required."}), 400

        logger.info(f"Attempting to import from PostgreSQL: {schema_name}.{table_name}")
        conn = get_pg_connection()

        # Safely construct the SQL query
        query_str = psycopg2_sql.SQL("SELECT * FROM {}.{}").format(
            psycopg2_sql.Identifier(schema_name),
            psycopg2_sql.Identifier(table_name)
        ).as_string(conn) # Get the string representation for pandas
        
        df = pd.read_sql_query(query_str, conn)

        session_id = str(uuid.uuid4())
        # Use only the table name for the filename without appending session ID
        excel_filename = f"{table_name}.xlsx"

        session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_upload_folder, exist_ok=True)
        excel_file_path = os.path.join(session_upload_folder, excel_filename)

        df.to_excel(excel_file_path, index=False, engine='openpyxl')
        logger.info(f"Table {schema_name}.{table_name} converted to Excel: {excel_file_path}")

        # Prepare data for immediate visualization if needed by frontend
        data_for_viz = df.to_dict(orient='records')
        column_headers = list(df.columns)

        # Store info in processing_results, similar to file upload
        processing_results[session_id] = {
            'status': 'uploaded_from_postgres',
            'original_filename': excel_filename, # Use just the table name as filename
            'file_path': excel_file_path,
            'logs': [f"Table '{schema_name}.{table_name}' imported successfully as '{excel_filename}'."]
        }

        return jsonify({
            'message': f"Table '{schema_name}.{table_name}' imported successfully.",
            'session_id': session_id,
            'file_path': excel_file_path, # Path to the saved Excel file
            'original_filename': excel_filename,
            'status': 'success',
            'data_for_visualization': data_for_viz, # Send data for frontend parsing
            'column_headers': column_headers
        }), 200
    except psycopg2.Error as pg_err:
        logger.exception(f"PostgreSQL error during table import: {pg_err}")
        error_message = str(pg_err).split('\n')[0] # Get a concise error message
        return jsonify({"error": f"Database error: {error_message}"}), 500
    except Exception as e:
        logger.exception("Error importing table from PostgreSQL")
        return jsonify({"error": f"Failed to import table: {str(e)}"}), 500
    finally:
        if conn: conn.close()

# --- Existing Flask Routes ---
@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file_route():
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)
    try:
        logger.info("Upload endpoint called")
        if 'file' not in request.files: return jsonify({'error': 'No file part in request.'}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({'error': 'No file selected.'}), 400
        session_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_upload_folder, exist_ok=True)
        file_path = os.path.join(session_upload_folder, original_filename)
        file.save(file_path)
        logger.info(f"File '{original_filename}' uploaded for session {session_id}, saved to {file_path}")
        processing_results[session_id] = {'status': 'uploaded', 'original_filename': original_filename, 'file_path': file_path, 'logs': [f"File {original_filename} uploaded."]}
        return jsonify({'message': 'File uploaded.', 'session_id': session_id, 'file_path': file_path, 'original_filename': original_filename, 'status': 'success'}), 200
    except Exception as e: logger.exception("Error in /api/upload"); return jsonify({'error': f'Upload error: {e}'}), 500

@app.route('/api/process/<session_id>', methods=['POST', 'OPTIONS'])
def process_file_route(session_id):
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)
    try:
        logger.info(f"PROCESS (Analysis) endpoint for session: {session_id}")
        uploaded_file_path = processing_results.get(session_id, {}).get('file_path')
        if request.is_json: # Allow overriding file_path via JSON body if needed (e.g., for re-processing)
            data = request.get_json()
            if data and 'file_path' in data: 
                uploaded_file_path = data['file_path']
        
        if not uploaded_file_path or not os.path.exists(uploaded_file_path):
            return jsonify({'error': 'Uploaded file not found for session.'}), 404
        
        logger.info(f"Starting analysis for session {session_id}, file: {uploaded_file_path}")
        result_details = process_analysis_workflow(uploaded_file_path, session_id)
        
        if session_id not in processing_results: processing_results[session_id] = {} # Ensure session entry exists
        processing_results[session_id].update({
            'status': 'analysis_complete' if result_details['success'] else 'analysis_failed',
            'success': result_details['success'], 'message': result_details['message'],
            'logs': processing_results[session_id].get('logs', []) + result_details.get('logs', []), # Append new logs
            'result_file': result_details.get('result_file'), 'result_filename': result_details.get('result_filename')
        })
        logger.info(f"Analysis for {session_id} finished. Success: {result_details['success']}")
        return jsonify({'message': result_details['message'], 'success': result_details['success'], 'logs': processing_results[session_id]['logs'], 'result_filename': result_details.get('result_filename')}), 200 if result_details['success'] else 500
    except Exception as e:
        logger.exception(f"Critical error in /api/process/{session_id}: {e}")
        if session_id in processing_results: processing_results[session_id].update({'status': 'analysis_error', 'success': False, 'message': f'Server error: {e}'})
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/api/result/<session_id>', methods=['GET'])
def get_result_route(session_id):
    try:
        logger.info(f"GET RESULT (Analysis or Question) for session: {session_id}")
        session_data = processing_results.get(session_id) 

        # Attempt to load from disk if not in memory (e.g., after server restart or for long-running jobs)
        if not session_data: 
            is_question_result = "_q" in session_id # Heuristic to check if it's a question result session ID
            if is_question_result:
                q_final_dir = os.path.join(FINAL_RESULT_FOLDER, session_id) # Question results are in a folder named with session_id_qXXXX
                if os.path.exists(q_final_dir):
                    possible_files = [f for f in os.listdir(q_final_dir) if f.endswith(".txt")]
                    if possible_files:
                        # Prefer files without "Error" or "Failed" in their name
                        success_files = [f for f in possible_files if "Error" not in f and "Failed" not in f]
                        chosen_file = success_files[0] if success_files else possible_files[0] # Pick first success or first overall
                        disk_path = os.path.join(q_final_dir, chosen_file)
                        session_data = {'success': "Error" not in chosen_file, 'result_file': disk_path, 'result_filename': chosen_file, 'logs': ['Retrieved from disk.'], 'is_markdown': True, 'html_content': None, 'is_markdown_checked': False} # Assume markdown for questions
                        logger.info(f"Found question result on disk: {disk_path}")
            else: # Try to load analysis result
                analysis_disk_path = os.path.join(FINAL_RESULT_FOLDER, session_id, f"Final_Analysis_Result_{session_id}.txt")
                if os.path.exists(analysis_disk_path):
                    session_data = {'success': True, 'result_file': analysis_disk_path, 'result_filename': os.path.basename(analysis_disk_path), 'logs': ['Retrieved analysis from disk.'], 'is_markdown': True, 'html_content': None, 'is_markdown_checked': False} # Assume markdown for analysis
                    logger.info(f"Found analysis result on disk: {analysis_disk_path}")
            
            if session_data: # If loaded from disk, store in memory for subsequent faster access
                processing_results[session_id] = session_data


        if not session_data:
            return jsonify({'error': 'Result not found for session.'}), 404

        # If processing failed and no HTML content (e.g. error message) is already set
        if not session_data.get('success', True) and not session_data.get('html_content'): # Default success to True if not set but content exists
            return jsonify({
                'error': 'Processing failed or no content available for this result.',
                'full_content': session_data.get('message', 'Failed.'), # Use message if available
                'logs': session_data.get('logs', []),
                'success': False,
                'filename': session_data.get('result_filename') or session_data.get('filename') # Try both keys
            }), 200 # Return 200 as the request itself was successful, but processing wasn't

        result_file_path = session_data.get('result_file')

        # If HTML content is already processed and stored (e.g., for question answers)
        if session_data.get('html_content'):
            return jsonify({
                'full_content': session_data.get('markdown_content', ''), # Send raw markdown too if available
                'html_content': session_data.get('html_content'),
                'is_markdown': session_data.get('is_markdown', False), # is_markdown should be true if html_content is set from markdown
                'filename': session_data.get('result_filename') or session_data.get('filename'),
                'logs': session_data.get('logs', []),
                'success': session_data.get('success', False) # Reflect the success status of the operation
            }), 200

        if not result_file_path or not os.path.exists(result_file_path):
            return jsonify({'error': 'Result file missing.', 'full_content': 'File missing.', 'logs': session_data.get('logs', []), 'success': False, 'filename': session_data.get('result_filename')}), 404

        with open(result_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_content = f.read()

        # Basic Markdown detection (can be improved)
        is_likely_markdown = session_data.get('is_markdown', False) # Respect if already set
        if not session_data.get('is_markdown_checked'): # Only detect if not already checked/set
            is_likely_markdown = '##' in full_content or '*' in full_content or '```' in full_content or '| ---' in full_content or full_content.startswith("#")
            session_data['is_markdown_checked'] = True # Mark as checked
            session_data['is_markdown'] = is_likely_markdown


        if is_likely_markdown:
            html_content = markdown.markdown(full_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            # Cache the converted HTML and raw markdown for future requests
            session_data['markdown_content'] = full_content 
            session_data['html_content'] = html_content     
            processing_results[session_id] = session_data # Update cache

            return jsonify({
                'full_content': full_content,
                'html_content': html_content,
                'is_markdown': True,
                'filename': os.path.basename(result_file_path),
                'logs': session_data.get('logs', []),
                'success': True # If we have content, assume success of retrieval
            }), 200
        else: # Plain text result
            return jsonify({
                'full_content': full_content,
                'is_markdown': False,
                'filename': os.path.basename(result_file_path),
                'logs': session_data.get('logs', []),
                'success': True
            }), 200
    except Exception as e:
        logger.exception(f"Error in /api/result/{session_id}: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

# --- New Endpoint for Prompt Enhancement ---
@app.route('/api/enhance-prompt', methods=['POST', 'OPTIONS'])
def enhance_prompt_route():
    if request.method == 'OPTIONS':
        return make_response(jsonify(success=True), 200)

    try:
        data = request.get_json()
        if not data or 'prompt' not in data or not data['prompt'].strip():
            return jsonify({'error': 'No prompt provided.'}), 400

        user_prompt = data['prompt'].strip()
        logger.info(f"Enhance Prompt endpoint called with prompt: \"{user_prompt[:50]}...\"")

        execution_id = str(uuid.uuid4())
        user_inputs = {"{{Prompt}}": user_prompt} # As per Avaplus pipeline expectation
        payload = {
            "pipeLineId": 2245,  # Specific pipeline ID for prompt enhancement
            "userInputs": user_inputs,
            "executionId": execution_id,
            "user": "manudhas.selvadhas@ascendion.com" # As specified
        }
        # Assuming DEFAULT_API_TIMEOUT is suitable, otherwise define a specific one
        api_output = send_request_with_retries(payload, timeout_duration=DEFAULT_API_TIMEOUT) 

        if api_output and "pipeline" in api_output and "output" in api_output["pipeline"]:
            enhanced_prompt = str(api_output["pipeline"]["output"]).strip()
            logger.info(f"Prompt enhanced successfully. Original: \"{user_prompt[:50]}...\", Enhanced: \"{enhanced_prompt[:50]}...\"")
            return jsonify({'success': True, 'original_prompt': user_prompt, 'enhanced_prompt': enhanced_prompt}), 200
        else:
            logger.error(f"Failed to get enhanced prompt from API. Response: {api_output}")
            error_message = "Failed to enhance prompt using the AI service."
            # Try to get a more specific error from the API response if available
            if isinstance(api_output, dict):
                pipeline_info = api_output.get("pipeline", {})
                if isinstance(pipeline_info, dict): # Check if pipeline_info itself is a dict
                    error_details = pipeline_info.get("error") # Now safe to use .get()
                    if error_details:
                        error_message = str(error_details) # Use the specific error from API
                    elif "output" in pipeline_info and "Error" in str(pipeline_info["output"]): # Check if output field contains error string
                         error_message = f"API indicated an error: {str(pipeline_info['output'])[:200]}"

            return jsonify({'error': error_message, 'details': str(api_output)[:500]}), 500 # Send generic and detailed error

    except Exception as e:
        logger.exception(f"Error in /api/enhance-prompt: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/ask-question/<session_id>', methods=['POST', 'OPTIONS'])
def ask_question_route(session_id):
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)

    q_logs = []
    user_question = ""
    question_instance_id = str(uuid.uuid4())[:8] # Unique ID for this question instance
    q_tag = f"{session_id}_q{question_instance_id}" # Unique session tag for this question's artifacts
    q_final_dir = os.path.join(FINAL_RESULT_FOLDER, q_tag) # Dedicated folder for this question's final answer
    q_txt_folder = None # Initialize for finally block
    q_split_folder = None # Initialize for finally block

    try:
        logger.info(f"ASK QUESTION for original session: {session_id}, Q_ID: {question_instance_id}")
        data = request.get_json()
        if not data or 'question' not in data or not data['question'].strip():
            return jsonify({'error': 'No question provided.'}), 400

        user_question = data['question'].strip()
        q_logs.append(f"Q ID {question_instance_id} for session {session_id}. Question: \"{user_question}\"")

        # --- Retrieve original file path ---
        original_session_data = processing_results.get(session_id)
        original_file_path = None
        original_filename = None

        if not original_session_data or 'file_path' not in original_session_data:
            # Attempt to reconstruct from UPLOAD_FOLDER if not in memory (e.g., server restart)
            session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
            if os.path.exists(session_upload_folder):
                uploaded_files = [f for f in os.listdir(session_upload_folder) if os.path.isfile(os.path.join(session_upload_folder, f))]
                if uploaded_files:
                    original_file_path = os.path.join(session_upload_folder, uploaded_files[0]) # Assume first file
                    original_filename = uploaded_files[0]
                    # Update processing_results if found on disk
                    if session_id not in processing_results: processing_results[session_id] = {}
                    processing_results[session_id]['file_path'] = original_file_path
                    processing_results[session_id]['original_filename'] = original_filename
                    logger.info(f"Loaded original file {original_filename} from disk for session {session_id}")

            if not original_file_path: # Still not found
                return jsonify({'error': 'Original data file for session not found in memory or on disk.'}), 404
        else:
            original_file_path = original_session_data['file_path']
            original_filename = original_session_data.get('original_filename', os.path.basename(original_file_path))

        if not os.path.exists(original_file_path):
            return jsonify({'error': f'Original data file missing: {original_file_path}'}), 404
        
        # Derive table name for pandasql from the original filename
        table_name = os.path.splitext(original_filename)[0]
        table_name = ''.join(c if c.isalnum() else '_' for c in table_name) # Sanitize
        if not table_name: table_name = "data_table" # Default if sanitization results in empty
        logger.info(f"Using table name '{table_name}' derived from filename '{original_filename}' for pandasql")


        # --- Prepare data context for API (first chunk of the file) ---
        q_txt_folder = os.path.join(TXT_FOLDER, q_tag) # Temp folder for this question's text conversion
        q_split_folder = os.path.join(SPLIT_FOLDER, q_tag) # Temp folder for this question's split
        for folder in [q_txt_folder, q_split_folder, q_final_dir]:
            os.makedirs(folder, exist_ok=True)

        q_txt_file = os.path.join(q_txt_folder, "converted_for_q.txt")
        q_logs.append(convert_to_txt(original_file_path, q_txt_file))

        if not os.path.exists(q_txt_file): # Check if conversion was successful
            return jsonify({'success': False, 'message': "Text conversion failed for question data."}), 500

        q_token_count = count_tokens(q_txt_file)
        q_logs.append(f"Data token count for Q: {q_token_count}")

        if q_token_count > CONTEXT_TOKEN_LIMIT:
            q_logs.append(f"Data for Q exceeds context token limit ({q_token_count} > {CONTEXT_TOKEN_LIMIT}). Splitting to get first chunk.")
            split_log_msg = split_text_file(q_txt_file, q_split_folder, CONTEXT_TOKEN_LIMIT, create_only_first_split=True)
            q_logs.append(split_log_msg)
        else:
            # If small enough, copy the whole text file to the split folder as "part_1"
            target_split_file_path = os.path.join(q_split_folder, "split_part_1.txt")
            shutil.copy(q_txt_file, target_split_file_path)
            q_logs.append(f"Data for Q under context token limit ({q_token_count}), using directly (copied as {target_split_file_path}).")

        first_split_file_path = os.path.join(q_split_folder, "split_part_1.txt")
        
        if not os.path.exists(first_split_file_path): # Ensure the first part (chunk or whole) exists
             error_msg_no_split = "Data preparation for question failed: first split part not found."
             q_logs.append(error_msg_no_split)
             return jsonify({'success': False, 'message': error_msg_no_split, 'logs': q_logs }), 500


        q_logs.append(f"Using data from: {os.path.basename(first_split_file_path)} for AI agent context")

        with open(first_split_file_path, "r", encoding="utf-8", errors="ignore") as file:
            file_content_for_api = file.read().strip()

        if not file_content_for_api: # Handle case where the first chunk is empty
            error_msg_empty_data = "Data file for AI context (used for SQL generation) is empty after preparation."
            q_logs.append(error_msg_empty_data)
            md_parts = [f"## Question\n\n> {user_question}\n", f"## Data Error\n\n{error_msg_empty_data}\n"]
            final_md_content = "\n\n".join(md_parts)
            html_content = markdown.markdown(final_md_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            answer_fname = f"Answer_To_Q_{question_instance_id}_Sess_{session_id}_DataEmptyError.txt" # Informative filename
            final_answer_path = os.path.join(q_final_dir, answer_fname)
            with open(final_answer_path, "w", encoding="utf-8") as f_ans: f_ans.write(final_md_content)
            processing_results[q_tag] = { 'status': 'question_failed_data_empty', 'success': False, 'markdown_content': final_md_content, 'html_content': html_content, 'is_markdown': True, 'logs': q_logs, 'filename': answer_fname, 'result_file': final_answer_path }
            return jsonify({ 'success': False, 'message': error_msg_empty_data, 'html_content': html_content, 'is_markdown': True, 'filename': answer_fname, 'logs': q_logs }), 400


        # --- Call API (pipeline 1887) for SQL generation ---
        formatted_data_for_api = f"table name : {table_name}\n\n{file_content_for_api}" # Include table name in context
        execution_id_api = str(uuid.uuid4())
        user_inputs_api = {"{{user_questions}}": user_question, "{{data_file}}": formatted_data_for_api}
        payload_api = {"pipeLineId": 1887, "userInputs": user_inputs_api, "executionId": execution_id_api, "user": "manudhas.selvadhas@ascendion.com"}

        q_logs.append(f"Sending request to API (1887) for SQL generation. Execution ID: {execution_id_api}")
        api_output_data = send_request_with_retries(payload_api, timeout_duration=DEFAULT_API_TIMEOUT)
        sql_query = None

        if not api_output_data or "pipeline" not in api_output_data or "output" not in api_output_data["pipeline"]:
            error_msg = "Failed to get SQL query from the AI assistance API (pipeline 1887)."
            q_logs.append(error_msg)
            api_response_snippet = str(api_output_data)[:500] if api_output_data else "No response received."
            markdown_parts = [f"## Question\n\n> {user_question}\n", f"## API Error\n\n{error_msg}\n", f"### API Response (Details)\n\n```json\n{api_response_snippet}...\n```\n"]
            final_markdown_content = "\n\n".join(markdown_parts)
            html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            answer_fname = f"Answer_To_Q_{question_instance_id}_Sess_{session_id}_ApiError.txt"
            final_answer_path = os.path.join(q_final_dir, answer_fname)
            with open(final_answer_path, "w", encoding="utf-8") as f_ans: f_ans.write(final_markdown_content)
            processing_results[q_tag] = { 'status': 'question_failed_api', 'success': False, 'markdown_content': final_markdown_content, 'html_content': html_content, 'is_markdown': True, 'logs': q_logs, 'filename': answer_fname, 'result_file': final_answer_path }
            return jsonify({ 'success': False, 'message': error_msg, 'html_content': html_content, 'is_markdown': True, 'filename': answer_fname, 'logs': q_logs }), 500

        # --- Extract SQL query from API response ---
        api_response_str = str(api_output_data["pipeline"]["output"]).strip()
        # Try to extract SQL from markdown code block first
        if "```sql" in api_response_str:
            sql_query_match = re.search(r"```sql\s*([\s\S]*?)\s*```", api_response_str, re.IGNORECASE)
            if sql_query_match:
                sql_query = sql_query_match.group(1).strip()
            else: # If ```sql is present but regex fails, take the whole string as potential SQL
                sql_query = api_response_str # This might need refinement if API returns more text
        elif "SQL_QUERY:" in api_response_str.upper(): # Fallback to "SQL_QUERY:" prefix
            parts = re.split("SQL_QUERY:", api_response_str, 1, re.IGNORECASE)
            if len(parts) > 1:
                # Try to split by "Answer:" if it exists after SQL_QUERY to isolate the query
                sql_parts = re.split("Answer:", parts[1], 1, re.IGNORECASE) if "Answer:" in parts[1] else [parts[1]]
                sql_query = sql_parts[0].strip()
        else: # If no clear markers, assume the whole output is the query (less robust)
            sql_query = api_response_str

        if not sql_query: # If SQL query is still empty or couldn't be extracted
            error_msg = "AI assistance API returned a response, but no SQL query could be extracted."
            q_logs.append(error_msg)
            markdown_parts = [f"## Question\n\n> {user_question}\n", f"## API Output Issue\n\n{error_msg}\n", f"### Full API Output\n\n```\n{api_response_str}\n```\n"]
            final_markdown_content = "\n\n".join(markdown_parts)
            html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            answer_fname = f"Answer_To_Q_{question_instance_id}_Sess_{session_id}_NoSqlError.txt"
            final_answer_path = os.path.join(q_final_dir, answer_fname)
            with open(final_answer_path, "w", encoding="utf-8") as f_ans: f_ans.write(final_markdown_content)
            processing_results[q_tag] = { 'status': 'question_failed_no_sql', 'success': False, 'markdown_content': final_markdown_content, 'html_content': html_content, 'is_markdown': True, 'logs': q_logs, 'filename': answer_fname, 'result_file': final_answer_path }
            return jsonify({ 'success': False, 'message': error_msg, 'html_content': html_content, 'is_markdown': True, 'filename': answer_fname, 'logs': q_logs }), 500


        q_logs.append(f"Received SQL query from API: {sql_query}")

        # --- Execute SQL query using pandasql ---
        try:
            # Load the original data file into a DataFrame
            if original_file_path.endswith('.csv'):
                df_data = pd.read_csv(original_file_path, on_bad_lines='skip', low_memory=False)
            else: # .xlsx or .xls
                df_data = pd.read_excel(original_file_path)

            # Execute query
            query_globals = {table_name: df_data, 'pd': pd} # Make DataFrame available to pandasql
            result_df = ps.sqldf(sql_query, query_globals)
            q_logs.append(f"SQL query executed successfully against the data.")

            # Format result as Markdown
            md_parts = [f"## Question\n\n> {user_question}\n"]
            if not result_df.empty:
                md_parts.append("## Results\n")
                # Sanitize column names for Markdown table compatibility (e.g., remove pipes)
                result_df.columns = [str(col).replace('|', '-') for col in result_df.columns]
                md_parts.append(result_df.to_markdown(index=False))
            else:
                md_parts.append("## Results\n\nNo records returned by the query.\n")
            final_markdown_content = "\n\n".join(md_parts)
            html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            answer_fname = f"Answer_To_Q_{question_instance_id}_Sess_{session_id}.txt"
            final_answer_path = os.path.join(q_final_dir, answer_fname)
            with open(final_answer_path, "w", encoding="utf-8") as f_ans: f_ans.write(final_markdown_content)
            q_logs.append(f"Results saved to {answer_fname}")
            processing_results[q_tag] = { 'status': 'answered', 'success': True, 'markdown_content': final_markdown_content, 'html_content': html_content, 'is_markdown': True, 'logs': q_logs, 'filename': answer_fname, 'result_file': final_answer_path, 'question': user_question }
            return jsonify({ 'success': True, 'message': 'Question processed.', 'html_content': html_content, 'is_markdown': True, 'filename': answer_fname, 'logs': q_logs }), 200

        except Exception as sql_exec_error:
            error_msg_sql_exec = f"Error executing SQL query: {str(sql_exec_error)}"
            logger.exception(error_msg_sql_exec)
            q_logs.append(error_msg_sql_exec)
            md_parts = [f"## Question\n\n> {user_question}\n", f"## SQL Query (Execution Failed)\n\n```sql\n{sql_query}\n```\n", f"## Execution Error Details\n\n```\n{str(sql_exec_error)}\n```\n"]
            final_markdown_content = "\n\n".join(md_parts)
            html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            answer_fname = f"Answer_To_Q_{question_instance_id}_Sess_{session_id}_SQLExecError.txt"
            final_answer_path = os.path.join(q_final_dir, answer_fname)
            with open(final_answer_path, "w", encoding="utf-8") as f_ans: f_ans.write(final_markdown_content)
            processing_results[q_tag] = { 'status': 'question_failed_sql_exec', 'success': False, 'markdown_content': final_markdown_content, 'html_content': html_content, 'is_markdown': True, 'logs': q_logs, 'filename': answer_fname, 'result_file': final_answer_path, 'question': user_question }
            return jsonify({ 'success': False, 'message': error_msg_sql_exec, 'html_content': html_content, 'is_markdown': True, 'filename': answer_fname, 'logs': q_logs }), 500

    except Exception as e_general:
        logger.exception(f"General error processing question for session {session_id}, Q_ID {question_instance_id}: {e_general}")
        final_user_question = user_question if user_question else "Not available or error before question parsing."
        q_logs.append(f"General Q Processing Error: {str(e_general)}")
        md_parts = [f"## Question\n\n> {final_user_question}\n", f"## General Processing Error\n\nAn unexpected error occurred.\n", f"### Error Details\n\n```\n{str(e_general)}\n```\n"]
        final_markdown_content = "\n\n".join(md_parts)
        html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
        answer_fname = f"Answer_To_Q_{question_instance_id}_Sess_{session_id}_GeneralError.txt"
        final_answer_path = os.path.join(q_final_dir, answer_fname)
        try: os.makedirs(q_final_dir, exist_ok=True); # Ensure dir exists before writing error file
        except: pass # Ignore if it fails (e.g., already exists)
        with open(final_answer_path, "w", encoding="utf-8") as f_ans: f_ans.write(final_markdown_content)
        processing_results[q_tag] = { 'status': 'question_failed_general', 'success': False, 'markdown_content': final_markdown_content, 'html_content': html_content, 'is_markdown': True, 'logs': q_logs, 'filename': answer_fname, 'result_file': final_answer_path, 'question': final_user_question }
        return jsonify({ 'success': False, 'message': f"Error processing question: {str(e_general)}", 'html_content': html_content, 'is_markdown': True, 'filename': answer_fname, 'logs': q_logs }), 500
    finally:
        # Cleanup temporary folders for this question instance
        if q_txt_folder and os.path.exists(q_txt_folder):
            try: shutil.rmtree(q_txt_folder)
            except Exception as e_clean_txt: logger.warning(f"Could not cleanup temp Q TXT folder {q_txt_folder}: {e_clean_txt}")
        if q_split_folder and os.path.exists(q_split_folder):
            try: shutil.rmtree(q_split_folder)
            except Exception as e_clean_split: logger.warning(f"Could not cleanup temp Q SPLIT folder {q_split_folder}: {e_clean_split}")


@app.route('/api/download/<session_id>/<filename>', methods=['GET'])
def download_file_route(session_id, filename):
    try:
        logger.info(f"Download request for session: {session_id}, filename: {filename}")
        path_to_file = None
        # Determine path based on filename prefix (analysis vs question result)
        if filename.startswith("Final_Analysis_Result_"):
            # Analysis results are in FINAL_RESULT_FOLDER/session_id/filename
            path_to_file = os.path.join(FINAL_RESULT_FOLDER, session_id, filename)
        elif filename.startswith("Answer_To_Q_"):
            # Question results are in FINAL_RESULT_FOLDER/session_id_qXXXX/filename
            # The session_id in the URL for download should already be the q_tag (e.g., originalsession_q1234)
            path_to_file = os.path.join(FINAL_RESULT_FOLDER, session_id, filename)
        else:
            logger.warning(f"Unrecognized file type for download: {filename}")
            return jsonify({'error': 'Unrecognized file type for download.'}), 400

        if path_to_file and os.path.exists(path_to_file):
            logger.info(f"Serving file: {path_to_file}")
            return send_file(path_to_file, as_attachment=True, download_name=filename)
        else:
            logger.error(f"File not found for download: {path_to_file if path_to_file else 'path not constructed or file missing'}")
            return jsonify({'error': 'Requested file not found on server.'}), 404
    except Exception as e:
        logger.exception(f"Error in download/{session_id}/{filename}: {e}")
        return jsonify({'error': f'Download error: {e}'}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def check_status_route(session_id):
    try:
        logger.info(f"Status check for session: {session_id}")
        if session_id in processing_results:
            data = processing_results[session_id]
            return jsonify({
                'status': data.get('status', 'unknown'),
                'success': data.get('success'), # Can be True, False, or None
                'message': data.get('message', 'Status available.'),
                'original_filename': data.get('original_filename'),
                'result_filename': data.get('result_filename'),
                'question': data.get('question') # If it was a question session
            }), 200
        
        # Check disk for analysis result if not in memory
        analysis_disk_path = os.path.join(FINAL_RESULT_FOLDER, session_id, f"Final_Analysis_Result_{session_id}.txt")
        if os.path.exists(analysis_disk_path):
             return jsonify({'status': 'analysis_completed_on_disk', 'message': 'Analysis result found on disk.', 'result_filename': os.path.basename(analysis_disk_path)}), 200

        # Check disk for question result if not in memory (session_id would be like original_id_qXXXX)
        if "_q" in session_id: # Heuristic for question session ID
            q_final_dir = os.path.join(FINAL_RESULT_FOLDER, session_id)
            if os.path.exists(q_final_dir):
                possible_files = [f for f in os.listdir(q_final_dir) if f.endswith(".txt")]
                if possible_files:
                    success_files = [f for f in possible_files if "Error" not in f and "Failed" not in f]
                    chosen_file = success_files[0] if success_files else possible_files[0]
                    return jsonify({'status': 'answered_on_disk', 'message': 'Question result found on disk.', 'result_filename': chosen_file, 'question_id': session_id.split('_q')[-1]}), 200


        # Check if original upload folder exists (means file was uploaded at some point)
        if os.path.exists(os.path.join(UPLOAD_FOLDER, session_id)): # Check original session_id for upload
             return jsonify({'status': 'uploaded_on_disk', 'message': 'File originally uploaded, specific job status not in memory.'}), 200

        return jsonify({'status': 'not_found', 'message': 'No job or upload for this session ID in memory or common disk paths.'}), 404
    except Exception as e:
        logger.exception(f"Error in /api/status/{session_id}: {e}")
        return jsonify({'error': str(e)}), 500


# --- AI Visualization Endpoint (MODIFIED to return JSON config) ---
@app.route('/api/ai-visualization', methods=['POST', 'OPTIONS'])
def ai_visualization_route():
    if request.method == 'OPTIONS': return make_response(jsonify(success=True), 200)
    
    viz_job_context_folder = None # To ensure cleanup in finally block
    try:
        logger.info("AI Visualization endpoint called")
        data = request.get_json()
        original_uploaded_file_path = None

        if not data or 'user_prompt' not in data or not data['user_prompt'].strip():
            return jsonify({'error': 'No prompt provided.'}), 400

        session_id = data.get('session_id') # This is the original data upload session ID
        user_prompt = data['user_prompt'].strip()

        if not session_id :
            return jsonify({'error': 'Session ID is required for AI visualization.'}), 400

        # Retrieve the path of the originally uploaded file for this session
        if session_id in processing_results and 'file_path' in processing_results[session_id]:
            original_uploaded_file_path = processing_results[session_id]['file_path']
        else: # Attempt to find it on disk if not in memory
            session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
            if os.path.exists(session_upload_folder):
                uploaded_files = [f for f in os.listdir(session_upload_folder) if os.path.isfile(os.path.join(session_upload_folder, f))]
                if uploaded_files:
                    original_file_path_from_disk = os.path.join(session_upload_folder, uploaded_files[0])
                    if session_id not in processing_results: processing_results[session_id] = {}
                    processing_results[session_id]['file_path'] = original_file_path_from_disk
                    processing_results[session_id]['original_filename'] = uploaded_files[0]
                    original_uploaded_file_path = original_file_path_from_disk
                    logger.info(f"Using file {original_uploaded_file_path} from disk for AI visualization for session {session_id}")
                else:
                    return jsonify({'error': 'Session data folder found on disk, but no specific file. Please ensure data is uploaded first.'}), 404
            else:
                return jsonify({'error': 'Invalid session or original data file not found for session.'}), 404

        if not original_uploaded_file_path or not os.path.exists(original_uploaded_file_path):
            return jsonify({'error': f'Session file {original_uploaded_file_path if original_uploaded_file_path else "path not determined"} not found on disk.'}), 404

        # Create temporary folders for this specific AI visualization job's context data
        viz_ctx_id = str(uuid.uuid4())[:8] # Unique ID for this visualization context prep
        viz_job_context_folder = os.path.join(AI_VIZ_TEMP_CONTEXT_FOLDER, f"viz_ctx_{session_id}_{viz_ctx_id}")
        viz_txt_context_folder = os.path.join(viz_job_context_folder, "txt")
        viz_split_context_folder = os.path.join(viz_job_context_folder, "split")

        for folder in [viz_job_context_folder, viz_txt_context_folder, viz_split_context_folder]:
            os.makedirs(folder, exist_ok=True)

        # Convert the original data file to TXT for context preparation
        txt_file_for_viz_context = os.path.join(viz_txt_context_folder, "data_for_viz_context.txt")
        convert_to_txt(original_uploaded_file_path, txt_file_for_viz_context)

        token_count_viz_context = count_tokens(txt_file_for_viz_context)
        logger.info(f"Token count for AI visualization data context (session {session_id}, ctx_id {viz_ctx_id}): {token_count_viz_context}")

        # Get the first chunk of the data for API context
        path_to_first_chunk_for_context = ""
        if token_count_viz_context > CONTEXT_TOKEN_LIMIT:
            logger.info(f"Splitting data for AI visualization context (session {session_id}, ctx_id {viz_ctx_id})")
            split_text_file(txt_file_for_viz_context, viz_split_context_folder, CONTEXT_TOKEN_LIMIT, create_only_first_split=True)
            path_to_first_chunk_for_context = os.path.join(viz_split_context_folder, "split_part_1.txt")
        else:
            target_single_part_path = os.path.join(viz_split_context_folder, "split_part_1.txt")
            shutil.copy(txt_file_for_viz_context, target_single_part_path)
            path_to_first_chunk_for_context = target_single_part_path

        if not os.path.exists(path_to_first_chunk_for_context):
            logger.error(f"Failed to prepare data context for AI viz. Path not found: {path_to_first_chunk_for_context}")
            return jsonify({'error': 'Failed to prepare data context for AI visualization.'}), 500

        first_chunk_file_content_for_context = ""
        with open(path_to_first_chunk_for_context, "r", encoding="utf-8", errors="ignore") as f_chunk:
            first_chunk_file_content_for_context = f_chunk.read().strip()

        if not first_chunk_file_content_for_context:
            logger.error(f"Data context for AI visualization is empty after preparation for session {session_id}")
            return jsonify({'error': 'Data context for AI visualization is empty after preparation.'}), 400

        # Prepare payload for API (pipeline 2141)
        original_filename_for_api = os.path.basename(original_uploaded_file_path)
        formatted_api_data_input = f"Filename: {original_filename_for_api}\n\nData Sample:\n{first_chunk_file_content_for_context}"

        execution_id_viz_config = str(uuid.uuid4())
        user_inputs_viz_config = {
            "{{Data}}": formatted_api_data_input,
            "{{User_Prompt}}": user_prompt
        }
        payload_viz_config = {
            "pipeLineId": 2141, # Pipeline for AI visualization config generation
            "userInputs": user_inputs_viz_config,
            "executionId": execution_id_viz_config,
            "user": "manudhas.selvadhas@ascendion.com"
        }

        logger.info(f"Sending AI visualization config request to API (pipeline 2141) for session {session_id}, exec_id {execution_id_viz_config}.")
        api_output_viz_config = send_request_with_retries(payload_viz_config, timeout_duration=SUMMARIZATION_API_TIMEOUT) # Use longer timeout


        if not api_output_viz_config or "pipeline" not in api_output_viz_config or "output" not in api_output_viz_config["pipeline"]:
            error_msg_viz = "Failed to get visualization configuration from AI API (empty or malformed response structure)."
            logger.error(f"{error_msg_viz} Response: {str(api_output_viz_config)[:500]}")
            return jsonify({'success': False, 'message': error_msg_viz, 'details': str(api_output_viz_config)[:500]}), 500

        chart_config_json_str = str(api_output_viz_config["pipeline"]["output"]).strip()
        logger.info(f"AI Visualization API response (raw string): {chart_config_json_str[:1000]}...") # Log more for debugging

        # Robust JSON parsing
        try:
            # Pre-cleaning common issues
            cleaned_json_str = chart_config_json_str.replace('\u00a0', ' ') # Replace non-breaking spaces
            # Remove JS-style comments (single and multi-line)
            cleaned_json_str = re.sub(r'//.*?(\r\n?|\n)|/\*.*?\*/', '', cleaned_json_str, flags=re.DOTALL)
            # Remove trailing commas before ] or }
            cleaned_json_str = re.sub(r',\s*([]}])', r'\1', cleaned_json_str)
            cleaned_json_str = cleaned_json_str.strip()
            
            # Attempt to find the actual start of JSON (either { or [)
            # This helps if the API returns some prefix text before the JSON.
            json_start_brace = cleaned_json_str.find('{')
            json_start_bracket = cleaned_json_str.find('[')

            if json_start_brace == -1 and json_start_bracket == -1:
                # No JSON start found
                raise json.JSONDecodeError("Valid JSON start '{' or '[' not found in response.", cleaned_json_str, 0)
            
            # Determine the actual start index of the JSON content
            if json_start_brace != -1 and (json_start_bracket == -1 or json_start_brace < json_start_bracket) :
                # Likely a JSON object starting with {
                actual_json_start_index = json_start_brace
            elif json_start_bracket != -1 :
                # Likely a JSON array starting with [
                 actual_json_start_index = json_start_bracket
            # This case should be covered by the initial check, but as a fallback:
            # else: actual_json_start_index = 0 
            
            json_to_parse = cleaned_json_str[actual_json_start_index:]


            chart_config = json.loads(json_to_parse) # Parse the extracted/cleaned JSON string
            
            logger.info(f"Successfully parsed AI visualization config. Type: {type(chart_config)}")
            if isinstance(chart_config, list):
                logger.info(f"Parsed as a list of {len(chart_config)} configurations.")
            elif isinstance(chart_config, dict):
                logger.info(f"Parsed as a single dictionary configuration.")
            else:
                logger.warning(f"Parsed AI config is neither a list nor a dict: {type(chart_config)}. Treating as error.")
                raise json.JSONDecodeError("Parsed JSON is not an object or array.", cleaned_json_str, 0)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON configuration from AI API for session {session_id}. Error: {e.msg} at char {e.pos}. String part: '{cleaned_json_str[max(0,e.pos-20):e.pos+20]}'")
            return jsonify({'success': False, 'message': f'AI returned invalid configuration format. Parsing error: {e.msg}', 'raw_response': chart_config_json_str[:1000]}), 500
        
        # If parsing successful, respond with the config
        return jsonify({
            'success': True,
            'message': 'AI visualization configuration received.',
            'chart_config': chart_config # This will be a list or a dict
        }), 200

    except Exception as e:
        logger.exception(f"Error in AI visualization endpoint: {e}")
        return jsonify({'success': False, 'message': f'Server error during AI visualization: {str(e)}'}), 500
    finally:
        if viz_job_context_folder and os.path.exists(viz_job_context_folder):
            try:
                shutil.rmtree(viz_job_context_folder)
                logger.info(f"Cleaned up temporary AI viz context folder: {viz_job_context_folder}")
            except Exception as e_clean:
                logger.warning(f"Could not cleanup temp AI viz context folder {viz_job_context_folder}: {e_clean}")


@app.route('/api/execute-sql', methods=['POST', 'OPTIONS'])
def execute_sql_route():
    if request.method == 'OPTIONS': 
        return make_response(jsonify(success=True), 200)
    
    try:
        logger.info("Execute SQL endpoint called")
        data = request.get_json()
        
        if not data or 'sql_query' not in data or not data['sql_query'].strip():
            return jsonify({'error': 'No SQL query provided.'}), 400 # Bad request
            
        session_id = data.get('session_id')
        sql_query = data['sql_query'].strip()
        
        if not session_id:
            return jsonify({'error': 'Session ID is required for SQL execution.'}), 400 # Bad request
            
        # --- Enhanced: Check if it's a SELECT statement ---
        if not sql_query.upper().startswith("SELECT"):
            error_message = "Unsupported SQL operation. Only SELECT statements are allowed for querying the uploaded data."
            logger.warning(f"Unsupported SQL attempt by session {session_id}: {sql_query[:100]}...")
            md_parts = [
                f"## SQL Query (Unsupported)\n\n{sql_query}\n", # <--- NEW LINE
                f"## Error\n\n{error_message}\n"
            ]
            final_markdown_content = "\n\n".join(md_parts)
            html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            return jsonify({
                'success': False,
                'message': error_message,
                'full_content': final_markdown_content,
                'html_content': html_content,
                'is_markdown': True
            }), 400 # Bad Request
            
        # Get the original file path associated with the session
        original_uploaded_file_path = None
        original_filename = "data_file" # Default
        session_info = processing_results.get(session_id)

        if session_info and 'file_path' in session_info:
            original_uploaded_file_path = session_info['file_path']
            original_filename = session_info.get('original_filename', os.path.basename(original_uploaded_file_path))
        else: # Attempt to retrieve from disk if not in memory
            session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
            if os.path.exists(session_upload_folder):
                uploaded_files = [f for f in os.listdir(session_upload_folder) if os.path.isfile(os.path.join(session_upload_folder, f))]
                if uploaded_files:
                    original_uploaded_file_path = os.path.join(session_upload_folder, uploaded_files[0])
                    original_filename = uploaded_files[0]
                    if session_id not in processing_results: processing_results[session_id] = {}
                    processing_results[session_id]['file_path'] = original_uploaded_file_path
                    processing_results[session_id]['original_filename'] = original_filename
                    logger.info(f"Using file {original_uploaded_file_path} from disk for SQL execution for session {session_id}")
                else:
                    return jsonify({'error': 'Session data folder found on disk, but no specific file. Please ensure data is uploaded first.'}), 404
            else:
                return jsonify({'error': 'Invalid session or original data file not found for session.'}), 404
                
        if not original_uploaded_file_path or not os.path.exists(original_uploaded_file_path):
            return jsonify({'error': f'Session file {original_uploaded_file_path if original_uploaded_file_path else "path not determined"} not found on disk.'}), 404
            
        # Derive table name from filename for pandasql
        table_name = os.path.splitext(original_filename)[0]
        table_name = ''.join(c if c.isalnum() else '_' for c in table_name) # Sanitize
        if not table_name: table_name = "data_table" # Default table name
        
        # Load the data into a DataFrame and execute query
        try:
            if original_uploaded_file_path.endswith('.csv'):
                df_data = pd.read_csv(original_uploaded_file_path, on_bad_lines='skip', low_memory=False)
            else: # .xlsx or .xls
                df_data = pd.read_excel(original_uploaded_file_path)
                
            query_globals = {table_name: df_data, 'pd': pd}
            logger.info(f"Executing SQL for session {session_id} on table '{table_name}': {sql_query[:200]}...")
            result_df = ps.sqldf(sql_query, query_globals)
            
            if result_df.empty:
                # Ensure 'msg' is defined here:
                msg = 'SQL query executed successfully, but no records were returned.' 
                logger.info(f"{msg} for session {session_id}")
                
                # This is the line where your error likely occurs (line 1353 in your file)
                # It correctly uses 'msg' as defined above.
                md_content = f"## SQL Query\n\n{sql_query}\n\n## Results\n\n{msg}\n" 
                
                html_c = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'sane_lists'])
                return jsonify({
                    'success': True, 
                    'message': msg, # 'msg' is also used here
                    'rows': 0, 
                    'columns': [], 
                    'data': [],
                    'full_content': md_content, 
                    'html_content': html_c, 
                    'is_markdown': True
                }), 200
            
            # This is the 'else' part for when result_df is NOT empty
            else: 
                result_data = result_df.to_dict(orient='records')
                column_headers = list(result_df.columns)
                
                # This was the other line you changed to remove backticks for the SQL query
                md_parts = [f"## SQL Query\n\n{sql_query}\n\n## Results\n"] 
                result_df.columns = [str(col).replace('|', '-') for col in result_df.columns]
                md_parts.append(result_df.to_markdown(index=False))
                final_markdown_content = "\n\n".join(md_parts)
                html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
                
                logger.info(f"SQL query successful for session {session_id}, returned {len(result_data)} records.")
                return jsonify({
                    'success': True, 
                    'message': f'SQL query executed successfully. Returned {len(result_data)} records.',
                    'rows': len(result_data), 
                    'columns': column_headers, 
                    'data': result_data,
                    'full_content': final_markdown_content, 
                    'html_content': html_content, 
                    'is_markdown': True
                }), 200
                
        except Exception as sql_exec_error: # Catch pandasql execution errors
            logger.exception(f"Error executing pandasql query for session {session_id} on table '{table_name}': {sql_exec_error}")
            
            error_str = str(sql_exec_error)
            user_friendly_message = f"Error executing SQL query: {error_str}" # Default

            # Provide more specific feedback for common errors
            if "no such table" in error_str.lower():
                # Try to extract the incorrect table name mentioned in the error, if possible
                match = re.search(r"no such table:\s*([\w.-]+)", error_str.lower()) # Allow dots and hyphens in captured name
                incorrect_table_in_error = match.group(1) if match else "the one in your query"
                user_friendly_message = (
                    f"Error: The table '{incorrect_table_in_error}' was not found. "
                    f"Please ensure your query uses the correct table name for your uploaded data, which is: '{table_name}'."
                )
            elif "syntax error" in error_str.lower():
                user_friendly_message = f"SQL Syntax Error: {error_str}. Please check your query details."
            
            md_parts = [
                f"## SQL Query (Execution Failed)\n\n{sql_query}\n", # <--- NEW LINE
                f"## Error Details\n\n```\n{user_friendly_message}\n```\n" 
            ]
            final_markdown_content = "\n\n".join(md_parts)
            html_content = markdown.markdown(final_markdown_content, extensions=['tables', 'fenced_code', 'sane_lists'])
            
            return jsonify({
                'success': False,
                'message': user_friendly_message,
                'full_content': final_markdown_content,
                'html_content': html_content,
                'is_markdown': True
            }), 400 # Bad Request, as the query itself is problematic
            
    except Exception as e: # Catch other unexpected server errors
        logger.exception(f"Unexpected error in execute_sql_route for session {session_id}: {e}")
        return jsonify({'error': f'An unexpected server error occurred: {str(e)}'}), 500

# --- NEW: Endpoint for SQL Query Enhancement ---

@app.route('/api/enhance-sql', methods=['POST', 'OPTIONS'])
def enhance_sql_route():
    if request.method == 'OPTIONS':
        return make_response(jsonify(success=True), 200)

    try:
        data = request.get_json()
        if not data or 'sql_query' not in data or not data['sql_query'].strip():
            return jsonify({'error': 'No SQL query provided for enhancement.'}), 400

        user_sql_query = data['sql_query'].strip()
        logger.info(f"Enhance SQL Query endpoint called with query: \"{user_sql_query[:100]}...\"")

        # Get the session ID from the request
        session_id = data.get('session_id')
        logger.info(f"Session ID for SQL enhancement: {session_id}")
        
        formatted_data_for_api = ""
        
        # Only proceed with data preparation if session_id is provided
        if session_id:
            try:
                # Get the original file path associated with the session
                original_uploaded_file_path = None
                session_info = processing_results.get(session_id)

                if session_info and 'file_path' in session_info:
                    original_uploaded_file_path = session_info['file_path']
                    logger.info(f"Found file path in session: {original_uploaded_file_path}")
                else:
                    # Attempt to retrieve from disk if not in memory
                    session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
                    if os.path.exists(session_upload_folder):
                        uploaded_files = [f for f in os.listdir(session_upload_folder) 
                                         if os.path.isfile(os.path.join(session_upload_folder, f))]
                        if uploaded_files:
                            original_uploaded_file_path = os.path.join(session_upload_folder, uploaded_files[0])
                            logger.info(f"Found file on disk: {original_uploaded_file_path}")
                
                # If we found the file, convert it to text and split based on token length
                if original_uploaded_file_path and os.path.exists(original_uploaded_file_path):
                    # Create temporary folder for text conversion
                    temp_txt_folder = os.path.join(TXT_FOLDER, f"sql_enhance_{session_id}")
                    os.makedirs(temp_txt_folder, exist_ok=True)
                    

                    # Convert to text
                    temp_txt_file = os.path.join(temp_txt_folder, "data_for_sql.txt")
                    convert_to_txt(original_uploaded_file_path, temp_txt_file)
                    
                    # Split based on token length (1000 tokens)
                    temp_split_folder = os.path.join(SPLIT_FOLDER, f"sql_enhance_{session_id}")
                    os.makedirs(temp_split_folder, exist_ok=True)
                    split_text_file(temp_txt_file, temp_split_folder, 1000, create_only_first_split=True)
                    
                    # Read the first split file
                    first_split_file = os.path.join(temp_split_folder, "split_part_1.txt")
                    if os.path.exists(first_split_file):
                        with open(first_split_file, 'r', encoding='utf-8', errors='ignore') as f:
                            formatted_data_for_api = f.read().strip()
                            logger.info(f"Successfully prepared data sample for SQL enhancement: {len(formatted_data_for_api)} chars")
                    
             
                    # Clean up temporary folders
                    try:
                        shutil.rmtree(temp_txt_folder)
                        shutil.rmtree(temp_split_folder)
                    except Exception as e_clean:
                        logger.warning(f"Could not clean up temp folders for SQL enhancement: {e_clean}")
            
            except Exception as data_prep_error:
                logger.exception(f"Error preparing data for SQL enhancement: {data_prep_error}")
                # Continue with empty data if there's an error - the enhancement will still work without data context

        execution_id = str(uuid.uuid4())
        # Construct userInputs with both SQL query and data file
        user_inputs = {
            "{{SQL}}": user_sql_query, 
            "{{data_file}}": formatted_data_for_api
        }
        
        payload = {
            "pipeLineId": 2279,  # Pipeline ID for SQL enhancement
            "userInputs": user_inputs,
            "executionId": execution_id,
            "user": "manudhas.selvadhas@ascendion.com"
        }

        # Send request to the API
        api_output = send_request_with_retries(payload, timeout_duration=DEFAULT_API_TIMEOUT) 

        if api_output and "pipeline" in api_output and "output" in api_output["pipeline"]:
            enhanced_sql = str(api_output["pipeline"]["output"]).strip()
            logger.info(f"SQL Query enhanced successfully. Original: \"{user_sql_query[:100]}...\", Enhanced: \"{enhanced_sql[:100]}...\"")
            return jsonify({'success': True, 'original_sql_query': user_sql_query, 'enhanced_sql_query': enhanced_sql}), 200
        else:
            logger.error(f"Failed to get enhanced SQL query from API (pipeline 2279). Response: {api_output}")
            error_message = "Failed to enhance SQL query using the AI service."
            # Attempt to extract more specific error details from the API response
            if isinstance(api_output, dict):
                pipeline_info = api_output.get("pipeline", {})
                if isinstance(pipeline_info, dict): 
                    error_details = pipeline_info.get("error") 
                    if error_details:
                        error_message = str(error_details) 
                    elif "output" in pipeline_info and "Error" in str(pipeline_info["output"]): 
                         error_message = f"API indicated an error: {str(pipeline_info['output'])[:200]}"
            return jsonify({'error': error_message, 'details': str(api_output)[:500]}), 500 

    except Exception as e:
        logger.exception(f"Error in /api/enhance-sql: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
if __name__ == '__main__':
    if not ACCESS_KEY or "your_avaplus_access_key_here" in ACCESS_KEY : # pragma: no cover
        logger.warning("AVAPLUS_ACCESS_KEY is not set or is a placeholder. API calls may fail.")
    if "database-1.cf4mucoqok1c.eu-north-1.rds.amazonaws.com" in PG_HOST: # pragma: no cover
         logger.warning("Using default PostgreSQL host. Ensure it's accessible and credentials are correct.")
    app.run(debug=True, host='0.0.0.0', port=5001)
