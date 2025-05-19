import copy  # Added import for deepcopy
import json
import time

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from tqdm import tqdm
from logzero import logger
from sklearn.metrics import recall_score, accuracy_score, precision_score, classification_report


def run(**kwargs):
    start_time = time.time()
    script_start_dt = datetime.now()
    logger.info("--------------------------------")
    logger.info(f"Starting Calculation of Adjudication Metrics at {script_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("--------------------------------")

    ocr_output_dir = kwargs.get("ocr_output_dir", None)
    metrics_output_dir = kwargs.get("metrics_output_dir", None)
    adjudication_output_dir = kwargs.get("adjudication_output_dir", None)
    ground_truth_file = kwargs.get("ground_truth_file", None)

    if ocr_output_dir is None:
        raise ValueError("ocr_output_dir is required")
    ocr_output_dir = Path(ocr_output_dir)
    if not ocr_output_dir.exists():
        raise ValueError(f"ocr_output_dir does not exist: {ocr_output_dir}")
    if not ocr_output_dir.is_dir():
        raise ValueError(f"ocr_output_dir is not a directory: {ocr_output_dir}")

    if adjudication_output_dir is None:
        raise ValueError("adjudication_output_dir is required")

    adjudication_output_dir = Path(adjudication_output_dir)
    if not adjudication_output_dir.exists():
        raise ValueError(f"adjudication_output_dir does not exist: {adjudication_output_dir}")

    if not adjudication_output_dir.is_dir():
        raise ValueError(f"adjudication_output_dir is not a directory: {adjudication_output_dir}")

    if ground_truth_file is None:
        raise ValueError("ground_truth_file is required")

    ground_truth_file = Path(ground_truth_file)

    if not ground_truth_file.exists():
        raise ValueError(f"ground_truth_file does not exist: {ground_truth_file}")

    if not ground_truth_file.is_file():
        raise ValueError(f"ground_truth_file is not a file: {ground_truth_file}")

    if metrics_output_dir is None:
        raise ValueError("metrics_output_dir is required")

    metrics_output_dir = Path(metrics_output_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading OCR output data...")

    def did_complete_ocr(claim) -> bool:
        return len(claim["documents"]) > 0

    claims_batch_df = pd.DataFrame()
    ocr_files = list(ocr_output_dir.glob("*.json"))
    if not ocr_files:
        logger.warning(f"No JSON files found in OCR output directory: {ocr_output_dir}")
    else:
        logger.info(f"Found {len(ocr_files)} JSON files in {ocr_output_dir}.")
        for file in tqdm(ocr_files, desc="Processing OCR files"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    number_of_documents_in_claim = len(data.get("documents", []))
                    did_claim_complete_ocr_flag = did_complete_ocr(data)
                    new_row = pd.DataFrame(
                        {
                            "policy_id": [data.get("policy_id")],
                            "claim_id": [data.get("claim_id")],
                            "number_of_documents_in_claim": [number_of_documents_in_claim],
                            "did_claim_complete_ocr": [did_claim_complete_ocr_flag],
                        }
                    )
                    claims_batch_df = pd.concat([claims_batch_df, new_row], ignore_index=True)
            except Exception as e:
                logger.error(f"Error processing OCR file {file}: {e}")

    if claims_batch_df.empty:
        logger.warning("OCR claims_batch_df is empty. Further processing might yield no results.")
    else:
        logger.info(f"Loaded {len(claims_batch_df)} records from OCR outputs.")
        logger.debug(f"claims_batch_df head:\n{claims_batch_df.head()}")

    if "policy_id" in claims_batch_df.columns:
        claims_batch_df["policy_id"] = claims_batch_df["policy_id"].astype(str)
    if "claim_id" in claims_batch_df.columns:
        claims_batch_df["claim_id"] = claims_batch_df["claim_id"].astype(str)

    logger.info("Step 2: Loading and processing ground truth data...")
    gt_df = pd.read_csv(ground_truth_file, dtype={"policy_number": str, "claim_number": str})
    gt_df["decision"] = gt_df["decision"].astype(str).str.upper()
    gt_df["decision"] = gt_df["decision"].replace({"PAID": "PAY", "DENIED": "DENY"})
    gt_df = gt_df.rename(
        columns={
            "policy_number": "policy_id",
            "claim_number": "claim_id",
            "decision": "gt_decision",
            "status": "gt_status",
            "status_reason": "gt_status_reason",
            "in_data_set": "gt_in_data_set",
        }
    )
    logger.debug(f"Ground truth 'gt_decision' unique values after initial load: {gt_df['gt_decision'].unique()}")

    gt_df_deduplicated = gt_df.drop_duplicates(subset=["policy_id", "claim_id"], keep="first")
    logger.info(
        f"Ground truth data loaded. {len(gt_df)} original rows, {len(gt_df_deduplicated)} rows after deduplication on policy_id and claim_id."
    )

    if not claims_batch_df.empty:
        logger.info("Step 3: Merging OCR data with ground truth data...")
        claims_batch_df = pd.merge(
            claims_batch_df,
            gt_df_deduplicated[
                ["policy_id", "claim_id", "gt_decision", "gt_status", "gt_status_reason", "gt_in_data_set"]
            ],
            on=["policy_id", "claim_id"],
            how="left",
        )
        logger.info(f"Merge complete. claims_batch_df now has {len(claims_batch_df)} rows.")
        missing_gt_decision_count = claims_batch_df["gt_decision"].isnull().sum()
        logger.info(f"Number of rows in merged claims_batch_df with missing gt_decision: {missing_gt_decision_count}")
        logger.debug(
            f"Unique values in 'gt_decision' in merged claims_batch_df: {claims_batch_df['gt_decision'].unique()}"
        )
        logger.debug(
            f"Value counts for 'gt_decision' in merged claims_batch_df:\n{claims_batch_df['gt_decision'].value_counts(dropna=False)}"
        )
    else:
        logger.warning("Skipping merge with ground truth as claims_batch_df is empty.")

    if not claims_batch_df.empty:
        logger.info("Step 4: Loading adjudication data...")
        adjudication_data_df = pd.DataFrame()
        adj_files = list(adjudication_output_dir.glob("*.json"))
        if not adj_files:
            logger.warning(f"No JSON files found in adjudication output directory: {adjudication_output_dir}")
        else:
            logger.info(f"Found {len(adj_files)} JSON files in {adjudication_output_dir}.")
            for file in tqdm(adj_files, desc="Processing adjudication files"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        claim_id = data.get("claim_id")
                        policy_id = data.get("policy_id")
                        adjudication_decision = data.get("decision", {}).get("status", "UNKNOWN").upper()

                        new_row = pd.DataFrame(
                            {
                                "policy_id": [policy_id],
                                "claim_id": [claim_id],
                                "adjudication_decision": [adjudication_decision],
                                "adjudication_data": [data],
                            }
                        )
                        adjudication_data_df = pd.concat([adjudication_data_df, new_row], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error processing adjudication file {file}: {e}")

        if adjudication_data_df.empty:
            logger.warning("Adjudication data_df is empty.")
        else:
            logger.info(f"Loaded {len(adjudication_data_df)} records from adjudication outputs.")
            if "policy_id" in adjudication_data_df.columns:
                adjudication_data_df["policy_id"] = adjudication_data_df["policy_id"].astype(str)
            if "claim_id" in adjudication_data_df.columns:
                adjudication_data_df["claim_id"] = adjudication_data_df["claim_id"].astype(str)

        if not claims_batch_df.empty:
            logger.info("Step 5: Merging OCR-completed data with adjudication data...")
            claims_batch_df_with_ocr_results = pd.merge(
                claims_batch_df, adjudication_data_df, on=["policy_id", "claim_id"], how="left"
            )
            logger.info(
                f"Merge complete. claims_batch_df_with_ocr_results now has {len(claims_batch_df_with_ocr_results)} rows."
            )

            # Calculate adjudication decision counts immediately after merge,
            # before any filtering for metrics calculation. These are for ALL adjudicated claims.
            num_adj_pay_unfiltered = 0
            num_adj_deny_unfiltered = 0
            num_adj_refer_unfiltered = 0
            num_adj_missing_unfiltered = 0

            if (
                not claims_batch_df_with_ocr_results.empty
                and "adjudication_decision" in claims_batch_df_with_ocr_results.columns
            ):
                adj_decision_counts_unfiltered = claims_batch_df_with_ocr_results["adjudication_decision"].value_counts(
                    dropna=False
                )
                num_adj_pay_unfiltered = int(adj_decision_counts_unfiltered.get("PAY", 0))
                num_adj_deny_unfiltered = int(adj_decision_counts_unfiltered.get("DENY", 0))
                num_adj_refer_unfiltered = int(adj_decision_counts_unfiltered.get("REFER", 0))
                num_adj_missing_unfiltered = int(
                    claims_batch_df_with_ocr_results["adjudication_decision"].isnull().sum()
                )
                logger.info(
                    f"Unfiltered adjudication counts (from all OCR-completed claims): PAY={num_adj_pay_unfiltered}, DENY={num_adj_deny_unfiltered}, REFER={num_adj_refer_unfiltered}, MISSING/NaN={num_adj_missing_unfiltered}"
                )
            else:
                logger.warning(
                    "Cannot calculate unfiltered adjudication counts as claims_batch_df_with_ocr_results is empty or 'adjudication_decision' column is missing."
                )

            # Calculate Ground Truth counts for ALL adjudicated claims (before REFER filter) for the summary section
            gt_pay_for_all_adjudicated_summary = 0
            gt_deny_for_all_adjudicated_summary = 0
            gt_missing_for_all_adjudicated_summary = 0

            if not claims_batch_df_with_ocr_results.empty and "gt_decision" in claims_batch_df_with_ocr_results.columns:
                # This uses claims_batch_df_with_ocr_results *before* 'REFER' is filtered out
                gt_decision_counts_for_summary = claims_batch_df_with_ocr_results["gt_decision"].value_counts(
                    dropna=False
                )  # Use dropna=False to count NaNs
                gt_pay_for_all_adjudicated_summary = int(gt_decision_counts_for_summary.get("PAY", 0))
                gt_deny_for_all_adjudicated_summary = int(gt_decision_counts_for_summary.get("DENY", 0))
                gt_missing_for_all_adjudicated_summary = int(
                    claims_batch_df_with_ocr_results["gt_decision"].isnull().sum()
                )
                logger.info(
                    f"Ground Truth counts for summary (from all {len(claims_batch_df_with_ocr_results)} OCR-completed & adjudicated claims): "
                    f"PAY={gt_pay_for_all_adjudicated_summary}, DENY={gt_deny_for_all_adjudicated_summary}, MISSING_GT={gt_missing_for_all_adjudicated_summary}"
                )
            else:
                logger.warning(
                    "Cannot calculate ground truth counts for summary as claims_batch_df_with_ocr_results is empty or 'gt_decision' column is missing at this stage."
                )

            original_count_before_refer_filter = len(claims_batch_df_with_ocr_results)
            claims_batch_df_with_ocr_results = claims_batch_df_with_ocr_results[
                claims_batch_df_with_ocr_results["adjudication_decision"] != "REFER"
            ]
            logger.info(
                f"Removed {original_count_before_refer_filter - len(claims_batch_df_with_ocr_results)} 'REFER' claims. {len(claims_batch_df_with_ocr_results)} claims remaining for metrics."
            )
        elif claims_batch_df.empty:
            logger.warning("Skipping merge with adjudication data as claims_batch_df_with_ocr_results is empty.")
            claims_batch_df_with_ocr_results = pd.DataFrame(
                columns=claims_batch_df.columns.tolist() + ["adjudication_decision", "adjudication_data"]
                if not claims_batch_df.empty
                else [
                    "policy_id",
                    "claim_id",
                    "number_of_documents_in_claim",
                    "did_claim_complete_ocr",
                    "gt_decision",
                    "gt_status",
                    "gt_status_reason",
                    "gt_in_data_set",
                    "adjudication_decision",
                    "adjudication_data",
                ]
            )

        logger.info("Step 6: Calculating adjudication metrics...")
        true_column = "gt_decision"
        pred_column = "adjudication_decision"

        logger.info(f"--- Initial Data Overview for '{true_column}' and '{pred_column}' ---")
        if true_column in claims_batch_df_with_ocr_results:
            logger.info(
                f"Value counts for '{true_column}':\n{claims_batch_df_with_ocr_results[true_column].value_counts(dropna=False)}"
            )
        else:
            logger.error(
                f"True column '{true_column}' not found in the DataFrame. Metrics calculation will be impacted."
            )

        if pred_column in claims_batch_df_with_ocr_results:
            logger.info(
                f"Value counts for '{pred_column}':\n{claims_batch_df_with_ocr_results[pred_column].value_counts(dropna=False)}"
            )
        else:
            logger.error(
                f"Predicted column '{pred_column}' not found in the DataFrame. Metrics calculation will be impacted."
            )

        metrics_df = pd.DataFrame()
        if (
            not claims_batch_df_with_ocr_results.empty
            and true_column in claims_batch_df_with_ocr_results
            and pred_column in claims_batch_df_with_ocr_results
        ):
            metrics_df = claims_batch_df_with_ocr_results[[true_column, pred_column]].copy()
            initial_rows = len(metrics_df)
            metrics_df.dropna(subset=[true_column], inplace=True)
            rows_after_gt_dropna = len(metrics_df)
            removed_rows_count = initial_rows - rows_after_gt_dropna
            logger.info(
                f"Removed {removed_rows_count} rows due to missing '{true_column}'. {rows_after_gt_dropna} rows remaining for metrics."
            )
        else:
            logger.warning(
                "Metrics DataFrame cannot be prepared due to empty input or missing columns. Skipping metrics calculation."
            )
            removed_rows_count = 0

        if metrics_df.empty:
            logger.warning("No data remaining for metrics calculation after filtering.")
            y_true_for_report = pd.Series(dtype="object")
            y_pred_original_nans_for_report = 0
            labels_for_report = []
            accuracy = 0.0
            precision_per_class_dict = {}
            recall_per_class_dict = {}
            report_str = "No data available to generate classification report."

        else:
            y_true = metrics_df[true_column]
            y_pred = metrics_df[pred_column]

            y_pred_original_nans = y_pred.isnull().sum()
            if y_pred_original_nans > 0:
                logger.info(
                    f"Found {y_pred_original_nans} NaN values in '{pred_column}' for rows with valid '{true_column}'. Replacing with 'NO_PREDICTION'."
                )
                y_pred = y_pred.fillna("NO_PREDICTION")
            else:
                logger.info(f"No NaN values found in '{pred_column}' for rows with valid '{true_column}'.")

            y_true_for_report = y_true
            y_pred_original_nans_for_report = int(y_pred_original_nans)

            labels = sorted(list(set(y_true) | set(y_pred)))
            labels_for_report = labels
            logger.info(f"Unique labels considered for metrics: {labels}")

            if not labels:
                logger.warning("No labels found to calculate metrics.")
                accuracy = 0.0
                precision_per_class_dict = {}
                recall_per_class_dict = {}
                report_str = "No labels found to generate classification report."
            else:
                accuracy = accuracy_score(y_true, y_pred)
                precision_per_class_values = precision_score(
                    y_true, y_pred, labels=labels, average=None, zero_division=0
                )
                precision_per_class_dict = {label: score for label, score in zip(labels, precision_per_class_values)}
                recall_per_class_values = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                recall_per_class_dict = {label: score for label, score in zip(labels, recall_per_class_values)}
                report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0, target_names=labels)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"adjudication_summary_statistics_{timestamp}"
        json_filename = metrics_output_dir / f"{base_filename}.json"
        md_filename = metrics_output_dir / f"{base_filename}.md"

        metrics_summary_json = {
            "generation_timestamp": datetime.now().isoformat(),
            "script_parameters": {
                "ocr_output_dir": str(ocr_output_dir),
                "metrics_output_dir": str(metrics_output_dir),
                "adjudication_output_dir": str(adjudication_output_dir),
                "ground_truth_file": str(ground_truth_file),
            },
            "data_summary": {
                "initial_ocr_claims_count": len(ocr_files) if "ocr_files" in locals() else 0,
                "loaded_ocr_claims_df_rows": len(claims_batch_df) if not claims_batch_df.empty else 0,
                "ground_truth_rows_before_dedup": len(gt_df) if "gt_df" in locals() else 0,
                "ground_truth_rows_after_dedup": len(gt_df_deduplicated) if "gt_df_deduplicated" in locals() else 0,
                "claims_completed_ocr_count": len(claims_batch_df_with_ocr_results)
                if not claims_batch_df_with_ocr_results.empty
                and "adjudication_decision" not in claims_batch_df_with_ocr_results
                else (
                    len(claims_batch_df[claims_batch_df["did_claim_complete_ocr"] == True])
                    if "claims_batch_df" in locals() and not claims_batch_df.empty
                    else 0
                ),
                "claims_after_refer_filter": len(metrics_df) + removed_rows_count
                if not metrics_df.empty
                else (len(claims_batch_df_with_ocr_results) if not claims_batch_df_with_ocr_results.empty else 0),
            },
            "metrics_calculation_summary": {
                "true_column": true_column,
                "predicted_column": pred_column,
                "data_points_used_for_metrics": len(y_true_for_report),
                "rows_removed_due_to_missing_ground_truth": removed_rows_count,
                "predicted_nans_replaced_with_NO_PREDICTION": y_pred_original_nans_for_report,
                "unique_labels_in_metrics": labels_for_report,
            },
            "overall_accuracy": accuracy,
            "precision_per_class": precision_per_class_dict,
            "recall_per_class": recall_per_class_dict,
            "classification_report_text": report_str,
        }
        with open(json_filename, "w", encoding="utf-8") as f_json:
            json.dump(metrics_summary_json, f_json, indent=4)
        logger.info(f"Metrics summary saved to JSON: {json_filename}")

        md_content = []
        md_content.append("# Adjudication Summary Statistics\n")
        md_content.append(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        md_content.append("## Input Parameters")
        md_content.append(f"*   OCR Output Directory: `{ocr_output_dir}`")
        md_content.append(f"*   Adjudication Output Directory: `{adjudication_output_dir}`")
        md_content.append(f"*   Ground Truth File: `{ground_truth_file}`")
        md_content.append(f"*   Metrics Output Directory: `{metrics_output_dir}`\n")

        md_content.append("## Data Processing Summary")
        md_content.append(
            f"*   Initial OCR JSON files found: {metrics_summary_json['data_summary']['initial_ocr_claims_count']}"
        )
        md_content.append(
            f"*   Rows in DataFrame after loading OCR data: {metrics_summary_json['data_summary']['loaded_ocr_claims_df_rows']}"
        )
        md_content.append(
            f"*   Ground truth rows (before deduplication): {metrics_summary_json['data_summary']['ground_truth_rows_before_dedup']}"
        )
        md_content.append(
            f"*   Ground truth rows (after deduplication): {metrics_summary_json['data_summary']['ground_truth_rows_after_dedup']}"
        )
        md_content.append(
            f"*   Claims that completed OCR (before REFER filter): {metrics_summary_json['data_summary']['claims_completed_ocr_count']}"
        )
        md_content.append(
            f"*   Claims after 'REFER' filter (before GT null removal): {metrics_summary_json['data_summary']['claims_after_refer_filter']}\n"
        )

        # --- New Section: Ground Truth Data for Adjudicated Claims ---
        md_content.append("## Ground Truth Data (for Adjudicated Claims)")
        md_content.append(
            "This section shows the ground truth distribution for all claims that completed OCR and were processed by the adjudication model (i.e., received a PAY, DENY, or REFER decision from the model)."
        )

        # Use the pre-calculated unfiltered GT counts for the summary
        md_content.append(f"*   Number marked PAY (Ground Truth): **{gt_pay_for_all_adjudicated_summary}**")
        md_content.append(f"*   Number marked DENY (Ground Truth): **{gt_deny_for_all_adjudicated_summary}**")
        if gt_missing_for_all_adjudicated_summary > 0:
            md_content.append(f"*   Number with Missing Ground Truth: **{gt_missing_for_all_adjudicated_summary}**")

        total_gt_accounted_for = (
            gt_pay_for_all_adjudicated_summary
            + gt_deny_for_all_adjudicated_summary
            + gt_missing_for_all_adjudicated_summary
        )
        total_adjudicated_claims_for_comparison = (
            num_adj_pay_unfiltered + num_adj_deny_unfiltered + num_adj_refer_unfiltered + num_adj_missing_unfiltered
        )
        md_content.append(
            f"  (Total claims with ground truth status accounted for: {total_gt_accounted_for}. This is based on all {total_adjudicated_claims_for_comparison} adjudicated claims.)\n"
        )

        # --- New Section: Adjudication Decision Data ---
        md_content.append("## Adjudication Decision Data")
        md_content.append(
            "This section shows the distribution of decisions made by the adjudication model for all claims that completed OCR."
        )

        # Use the pre-calculated unfiltered counts
        md_content.append(f"*   Number predicted PAY: **{num_adj_pay_unfiltered}**")
        md_content.append(f"*   Number predicted DENY: **{num_adj_deny_unfiltered}**")
        md_content.append(f"*   Number predicted REFER: **{num_adj_refer_unfiltered}**")
        if num_adj_missing_unfiltered > 0:
            md_content.append(f"*   Number with Missing/Unknown Adjudication (NaN): **{num_adj_missing_unfiltered}**")
        md_content.append("\n")

        md_content.append("## Metrics Calculation Overview")
        md_content.append(
            f"*   Metrics calculated on **{len(y_true_for_report)}** data points after removing {removed_rows_count} rows with missing '{true_column}'."
        )
        if y_pred_original_nans_for_report > 0:
            md_content.append(
                f"*   **{y_pred_original_nans_for_report}** NaN values in '{pred_column}' were treated as 'NO_PREDICTION'."
            )
        else:
            md_content.append(
                f"*   No NaN values in '{pred_column}' needed replacement (for rows with valid ground truth)."
            )
        md_content.append(f"*   Unique labels considered for metrics: `{labels_for_report}`\n")

        md_content.append("## Overall Accuracy")
        md_content.append(
            "Accuracy measures the overall correctness of the adjudication model. It is the proportion of claims (both PAY and DENY) that the model correctly classified compared to the ground truth decisions."
        )
        md_content.append(
            "For example, an accuracy of 0.85 means that 85% of the claims processed were correctly adjudicated by the model as either PAY or DENY, matching the ground truth."
        )
        md_content.append(f"Accuracy: {accuracy:.4f}\n")

        md_content.append("## Precision per Class")
        md_content.append(
            'Precision for a specific class (e.g., PAY or DENY) answers the question: "Of all claims that the model predicted as this class, how many were actually correct?"'
        )
        md_content.append(
            "*   **Precision (PAY):** Of all claims the model predicted as PAY, what proportion were actually PAY according to the ground truth? High precision for PAY indicates that when the model decides to PAY a claim, it is very likely correct, minimizing overpayments."
        )
        md_content.append(
            "*   **Precision (DENY):** Of all claims the model predicted as DENY, what proportion were actually DENY according to the ground truth? High precision for DENY indicates that when the model decides to DENY a claim, it is very likely correct, minimizing incorrect denials."
        )
        if precision_per_class_dict:
            for label, score in precision_per_class_dict.items():
                md_content.append(f"*   {label}: {score:.4f}")
        else:
            md_content.append("*   N/A (No data or labels for precision)")
        md_content.append("\n")

        md_content.append("## Recall per Class")
        md_content.append(
            'Recall for a specific class (e.g., PAY or DENY) answers the question: "Of all claims that truly belong to this class, how many did the model correctly identify?"'
        )
        md_content.append(
            "*   **Recall (PAY):** Of all claims that should have been adjudicated as PAY (according to ground truth), what proportion did the model correctly identify as PAY? High recall for PAY means the model is effective at identifying most of the claims that are genuinely payable, ensuring claimants receive due benefits."
        )
        md_content.append(
            "*   **Recall (DENY):** Of all claims that should have been adjudicated as DENY (according to ground truth), what proportion did the model correctly identify as DENY? High recall for DENY means the model is effective at identifying most of the claims that should be denied, protecting against paying out on non-covered or fraudulent claims."
        )
        if recall_per_class_dict:
            for label, score in recall_per_class_dict.items():
                md_content.append(f"*   {label}: {score:.4f}")
        else:
            md_content.append("*   N/A (No data or labels for recall)")
        md_content.append("\n")

        md_content.append("## Classification Report")
        md_content.append("```")
        md_content.append(report_str)
        md_content.append("```\n")

        # --- New Section for Detailed Claims Data Table (from Notebook) ---
        md_content.append("## Detailed Claims Data (Metrics Set)\n")
        md_content.append(
            "The following table shows the specific claims that were included in the metrics calculation (after filtering for OCR completion, non-'REFER' adjudication, and presence of a ground truth decision).\n"
        )

        if not metrics_df.empty and not y_true_for_report.empty:  # y_true_for_report is y_true used for metrics
            columns_to_dump = [
                "policy_id",
                "claim_id",
                "number_of_documents_in_claim",
                "did_claim_complete_ocr",
                "gt_decision",
                "gt_status_reason",
                "adjudication_decision",
                "adjudication_data",
            ]

            # Create the DataFrame for the table based on the indices of y_true_for_report
            # This ensures it only contains rows used in the metrics.
            # claims_batch_df_with_ocr_results should be the df *before* gt_decision NaNs were dropped for metrics_df
            # but *after* 'REFER' was filtered.
            # y_true_for_report.index will give indices relative to metrics_df, which came from claims_batch_df_with_ocr_results
            # after dropping NaNs in gt_decision. So, we use these indices on claims_batch_df_with_ocr_results.

            df_for_markdown_table_source = claims_batch_df_with_ocr_results.loc[y_true_for_report.index].copy()

            # Override 'adjudication_decision' with y_pred (which has NaNs filled from metrics calculation)
            # y_pred was derived from metrics_df, so its index aligns with y_true_for_report.index
            # We need to ensure y_pred (potentially with 'NO_PREDICTION') is used here.
            # The y_pred used for report was created from metrics_df[pred_column].fillna("NO_PREDICTION")
            # So, we can re-create it or use a version of it.
            # Let's use the y_pred that was used for classification_report

            # Reconstruct y_pred as it was for the report (with NO_PREDICTION)
            # This y_pred aligns with y_true_for_report
            y_pred_for_table = metrics_df[pred_column].copy()
            if y_pred_original_nans_for_report > 0:
                y_pred_for_table = y_pred_for_table.fillna("NO_PREDICTION")

            df_for_markdown_table_source[pred_column] = y_pred_for_table.values  # Assign based on aligned indices

            temp_table_data = {}
            for col_name in columns_to_dump:
                if col_name in df_for_markdown_table_source.columns:
                    temp_table_data[col_name] = df_for_markdown_table_source[col_name]
                else:
                    logger.warning(f"Column '{col_name}' not found for detailed table, will be empty/NA.")
                    temp_table_data[col_name] = pd.Series(
                        [pd.NA] * len(df_for_markdown_table_source), index=df_for_markdown_table_source.index
                    )

            df_for_markdown_table = pd.DataFrame(temp_table_data, columns=columns_to_dump)

            def format_json_cell(data):
                if pd.isna(data):
                    return ""

                # Make a deep copy to avoid modifying the original data in the DataFrame
                data_copy = copy.deepcopy(data)

                try:
                    # Remove 'text' field from 'claim_documents' if it exists
                    if (
                        isinstance(data_copy, dict)
                        and "claim_documents" in data_copy
                        and isinstance(data_copy["claim_documents"], list)
                    ):
                        for doc in data_copy["claim_documents"]:
                            if isinstance(doc, dict) and "text" in doc:
                                del doc["text"]

                    json_str = json.dumps(data_copy, separators=(",", ":"))
                    json_str_escaped = json_str.replace("`", "\\\\`")  # Escape backticks for Markdown code block
                    return f"`{json_str_escaped}`"
                except TypeError:
                    return "`Error: Not JSON serializable`"
                except Exception as e:
                    return f"`Error formatting JSON: {str(e)}`"

            if "adjudication_data" in df_for_markdown_table.columns:
                df_for_markdown_table["adjudication_data"] = df_for_markdown_table["adjudication_data"].apply(
                    format_json_cell
                )

            df_for_markdown_table = df_for_markdown_table.fillna("")

            if not df_for_markdown_table.empty:
                markdown_table_string = df_for_markdown_table.to_markdown(index=False)
                md_content.append(markdown_table_string)
            else:
                md_content.append(
                    "No data available to display in the detailed claims table (metrics dataset was empty or table generation failed)."
                )
        else:
            md_content.append("No data available for metrics, so detailed claims table is omitted.")
        md_content.append("\n")
        # --- End of New Section ---

        md_content.append("## Interpretation Notes")
        md_content.append(
            f"*   Metrics are calculated on {len(y_true_for_report)} data points after removing rows with missing '{true_column}'."
        )
        if y_pred_original_nans_for_report > 0:
            md_content.append(
                f"*   If '{pred_column}' had NaN values for records with ground truth, they were treated as a 'NO_PREDICTION' category."
            )
        md_content.append(
            f"*   'Support' in the classification report refers to the number of actual instances of each class in '{true_column}'."
        )
        md_content.append(
            "*   If a class has 0 support (no true instances), its recall and F1-score will be 0 (sklearn `zero_division=0` behavior)."
        )
        md_content.append(
            "*   If a class was never predicted, its precision might be 0 (sklearn `zero_division=0` behavior)."
        )

        if not metrics_df.empty and "DENY" not in y_true.unique() and "DENY" in labels_for_report:
            md_content.append(
                "\n**WARNING:** The ground truth data used for these metrics (after filtering) does not appear to contain 'DENY' labels, though 'DENY' was a possible label."
            )
            md_content.append(
                "This means recall for 'DENY' will be 0, and precision for 'DENY' will also be 0 if 'DENY' was predicted for any non-DENY true case."
            )

        with open(md_filename, "w", encoding="utf-8") as f_md:
            f_md.write("\n".join(md_content))
        logger.info(f"Metrics summary saved to Markdown: {md_filename}")

        logger.info("\n--- Adjudication Metrics Summary ---")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Precision per class:")
        if precision_per_class_dict:
            for label, score in precision_per_class_dict.items():
                logger.info(f"  {label}: {score:.4f}")
        else:
            logger.info("  N/A")
        logger.info("Recall per class:")
        if recall_per_class_dict:
            for label, score in recall_per_class_dict.items():
                logger.info(f"  {label}: {score:.4f}")
        else:
            logger.info("  N/A")
        logger.info("Classification Report:\n" + report_str)

    script_end_dt = datetime.now()
    total_time = time.time() - start_time
    logger.info("--------------------------------")
    logger.info(f"Finished Calculation of Adjudication Metrics at {script_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time taken: {timedelta(seconds=total_time)}")
    logger.info("--------------------------------")
