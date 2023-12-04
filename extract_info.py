import re
import os
from collections import defaultdict
import numpy as np

def extract_losses_from_txt(file_path):
    # 지역
    region_pattern = re.compile(r"test mode: (\S+)")
    
    # 로스
    loss_patterns = {
        "original_loss": re.compile(r"original_loss:\s*([\d\.]+)"),
        "trend_loss": re.compile(r"trend_loss:\s*([\d\.]+)"),
        "seasonal_loss": re.compile(r"seasonal_loss:\s*([\d\.]+)"),
        "resid_loss": re.compile(r"resid_loss:\s*([\d\.]+)"),
        "percent_alltime_error": re.compile(r"percent_error_orignal:\s*([\d\.]+)")
    }

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 지역별 데이터 디렉토리.
    region_metrics = {}
    for section in content.split('--------------------'):
        region_match = region_pattern.search(section)
        if region_match:
            current_region = region_match.group(1)
            region_metrics[current_region] = {}
            for key, pattern in loss_patterns.items():
                match = pattern.search(section)
                if match:
                    region_metrics[current_region][key] = float(match.group(1))
                else:
                    region_metrics[current_region][key] = None  # If no match, assign None

    return region_metrics

if __name__ == "__main__":
    test_record_paths = ['train_models/2-stageRR_2000_drop0.5/test_record.txt'
                         ,'train_models/2-stageRR_2000_drop0.5_2/test_record.txt'
                         ,'train_models/2-stageRR_2000_drop0.5_3/test_record.txt'
                         ,'train_models/2-stageRL_2000_drop0.5/test_record.txt'
                         ,'train_models/2-stageRL_2000_drop0.5_2/test_record.txt'
                         ,'train_models/2-stageRL_2000_drop0.5_3/test_record.txt'
                         ,'train_models/2-stageLR_2000_drop0.5/test_record.txt'
                         ,'train_models/2-stageLR_2000_drop0.5_2/test_record.txt'
                         ,'train_models/2-stageLR_2000_drop0.5_3/test_record.txt'
                         ,'train_models/2-stageLL_2000_drop0.5/test_record.txt'
                         ,'train_models/2-stageLL_2000_drop0.5_2/test_record.txt'
                         ,'train_models/2-stageLL_2000_drop0.5_3/test_record.txt']
    
    markdown_content = ""
    aggregated_losses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for path in test_record_paths:
        losses = extract_losses_from_txt(path)
        directory_name = os.path.dirname(path).split('/')[-1]
        base_directory_name = re.sub(r'_\d+$', '', directory_name)
    
        markdown_content += f"### {directory_name}\n\n"
        for region, loss_values in losses.items():
            markdown_content += f"- **{region}**\n\n"
            markdown_content += "| Loss Type | Value |\n"
            markdown_content += "| --- | --- |\n"
            for loss_type, value in loss_values.items():
                markdown_content += f"| {loss_type} | {value} |\n"
                # 지역별로 로스 값을 집계
                aggregated_losses[loss_type][region][base_directory_name].append(value)
            markdown_content += "\n"
    
    # 평균 및 표준편차 계산을 위한 새로운 섹션
    stats_content = "\n\n## Aggregated Statistics\n\n"
    for loss_type, regions in aggregated_losses.items():
        stats_content += f"### {loss_type}\n\n"
        
        # 첫 번째 모델의 이름 목록을 얻기 위한 수정
        first_model_name = next(iter(regions.values())).keys()
        stats_content += "| Region | " + " | ".join([f"{model} (Avg, Std)" for model in first_model_name]) + " |\n"
        stats_content += "| --- |" + " --- |" * len(first_model_name) + "\n"
    
        for region, models in regions.items():
            stats_content += f"| {region} |"
            for model, values in models.items():
                avg = np.mean(values)
                std = np.std(values)
                stats_content += f" {avg:.3f} ({std:.3f}) |"
            stats_content += "\n"

    # 이 부분에서 두 개의 markdown 파일에 내용을 작성합니다.
    with open("results.md", 'w') as file:
        file.write(markdown_content)

    with open("results_stat.md", 'w') as file:
        file.write(stats_content)
