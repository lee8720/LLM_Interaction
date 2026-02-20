import os
import time
import json
import re
import numpy as np
import pandas as pd
import openai

# Load the Excel file
file_path = "/path/to/Problem.xlsx"

# Load image URL JSON (sorted by file number per case)
with open("/path/to/case_download_links.json", "r", encoding="utf-8") as f:
    case_image_links = json.load(f)

# Extract numeric index from filename for sorting
def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else float('inf')

# Sort images within each case by filename number
for case_no, images in case_image_links.items():
    images.sort(key=lambda x: extract_number(x["file_name"]))

# Read the Excel file
df = pd.read_excel(file_path)
case_no_list = df['Case_No'].tolist()
sex_list = df['Sex'].tolist()
age_list = df['Age'].tolist()
complaint_list = df['Chief Complaint'].tolist()
findings_list = df['Radiologic Findings'].tolist()
category_list = df['Category'].tolist() if 'Category' in df.columns else [''] * len(df)
choice_lists = [df[f'choice {i}'].tolist() for i in range(5)]

openai.api_key = "YOUR_OPENAI_API_KEY"


def normalize_scores(score_dict):
    diags = list(score_dict.keys())
    scores = np.array([max(0.0, float(score_dict.get(d, 0))) for d in diags], dtype=float)
    if scores.sum() == 0:
        scores = np.ones(len(diags)) / len(diags)
    else:
        scores = scores / scores.sum()
    scores = np.round(scores, 2)
    diff = 1.0 - scores.sum()
    for i in np.argsort(-scores):
        if abs(diff) < 1e-6:
            break
        step = 0.01 if diff > 0 else -0.01
        if 0 <= scores[i] + step <= 1:
            scores[i] += step
            diff -= step
    return dict(zip(diags, scores.tolist()))


def generate_prompt(sex, age, complaint, legend, choices, case_id):
    meta = f"\n- Sex: {sex}\n- Age: {age}\n- Chief complaint: {complaint}"
    choice_lines = "\n".join([f"{i+1}) {c}" for i, c in enumerate(choices)])
    legend_text = str(legend).strip() if legend and not pd.isna(legend) and str(legend).strip() else "Not available"

    prompt = f"""
You are a board-certified thoracic radiologist.
This is a challenging thoracic imaging quiz designed for radiologists in training and practice.
Given patient metadata, provided chest images, and five diagnosis choices, your tasks are:
1) rank the provided diagnoses from most to least likely with calibrated scores that sum to 1,
2) provide a single-sentence rationale for the top three diagnoses,
   **and in each rationale explicitly reference the provided patient information and/or imaging findings that support the choice.**

CASE_ID: {case_id}
PATIENT METADATA: {meta}

IMAGE LEGEND:
{legend_text}

DIAGNOSIS CHOICES:
{choice_lines}

TASK:
- Use the exact diagnosis strings in your ranking.
- Provide rationale only for top 3 diagnoses.
- Each rationale must clearly mention specific patient demographics, clinical complaint, or radiologic findings that justify the diagnosis.

Output your result strictly in the following JSON format:

{{
  "case_id": "001",
  "ranking": [
    {{"diagnosis": "Pulmonary tuberculosis", "score": 0.42, "rationale": "Classic upper lobe cavitary lesion"}},
    {{"diagnosis": "Sarcoidosis", "score": 0.31, "rationale": "Perilymphatic nodules in upper lobe"}},
    {{"diagnosis": "Invasive mucinous adenocarcinoma", "score": 0.15, "rationale": "Confluent consolidation with air bronchogram"}},
    {{"diagnosis": "Organizing pneumonia", "score": 0.07}},
    {{"diagnosis": "Metastatic disease", "score": 0.05}}
  ],
}}
"""
    return prompt


# Output file path
output_file_path = "/path/to/explanation_4.xlsx"

# Load existing output file if it exists; otherwise initialize an empty DataFrame
if os.path.exists(output_file_path):
    output_df = pd.read_excel(output_file_path)
else:
    output_df = pd.DataFrame({
        'Case Number': pd.Series(dtype='str'),
        'Diagnosis': pd.Series(dtype='str'),
        'Score': pd.Series(dtype='float'),
        'Rationale': pd.Series(dtype='str'),
    })

max_retries = 5
correct_count = 0

for idx in range(len(df)):
    case_no = case_no_list[idx]
    sex = sex_list[idx]
    age = age_list[idx]
    complaint = complaint_list[idx]
    legend = findings_list[idx]
    category = category_list[idx]
    choices = [choice_lists[i][idx] for i in range(5)]

    case_no_str = str(case_no)
    image_list = case_image_links.get(case_no_str, [])

    message = generate_prompt(sex, age, complaint, legend, choices, case_no)
    retries = 0
    success = False

    # Build multimodal content: text prompt + image URLs
    content = [{"type": "text", "text": message}]
    for img in image_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": img["url"]}
        })

    while retries < max_retries and not success:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-2025-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced, board-certified thoracic radiologist."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=1.0
            )

            if not response.choices or not response.choices[0].message.content.strip():
                raise ValueError("Received an empty response from the API.")

            response_json = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", response_json, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON object found in the response.")

            response_data = json.loads(json_match.group(0))

            if "ranking" not in response_data:
                raise ValueError("Invalid JSON: 'ranking' missing.")

            # Top-1 prediction and ground truth
            top_pred = response_data["ranking"][0]["diagnosis"]
            ground_truth = choices[0]

            # Check if prediction matches ground truth
            if top_pred.strip().lower() == ground_truth.strip().lower():
                correct_count += 1
                print(f"✅ Case {case_no}: Correct! ({top_pred})")
            else:
                print(f"❌ Case {case_no}: Wrong. Predicted {top_pred}, Answer {ground_truth}")

            # Print running accuracy
            print(f"👉 Running correct: {correct_count}/{idx + 1} (Accuracy: {correct_count / (idx + 1):.2%})\n")

            # Normalize scores
            score_raw = {item["diagnosis"]: item.get("score", 0.0) for item in response_data["ranking"]}
            score_norm = normalize_scores(score_raw)

            for item in response_data["ranking"]:
                diag = item["diagnosis"]
                score = score_norm.get(diag, 0.0)
                rationale = item.get("rationale", "") if score >= sorted(score_norm.values(), reverse=True)[2] else ""
                output_df = pd.concat([output_df, pd.DataFrame([{
                    'Case Number': case_no,
                    'Diagnosis': diag,
                    'Score': score,
                    'Rationale': rationale,
                }])], ignore_index=True)

            print(f"\n✅ Case {case_no} processed.")
            for item in response_data["ranking"]:
                diag = item["diagnosis"]
                score = score_norm.get(diag, 0.0)
                rationale = item.get("rationale", "")
                if rationale:
                    print(f"{diag} ({score:.2f}): {rationale}")
                else:
                    print(f"{diag} ({score:.2f})")

            success = True  # Mark success only after all processing is complete

        except Exception as e:
            retries += 1
            print(f"❌ Error processing {case_no}: {str(e)}")
            print(f"Retrying {retries}/{max_retries}...")
            time.sleep(2)

    if not success:
        output_df = pd.concat([output_df, pd.DataFrame([{
            'Case Number': case_no,
            'Diagnosis': "ERROR",
            'Score': 0.0,
            'Rationale': "",
        }])], ignore_index=True)

    output_df.to_excel(output_file_path, index=False)

# Print final accuracy
print(f"\nTotal correct: {correct_count}/{len(df)}")
print(f"Final Accuracy: {correct_count/len(df):.2%}")