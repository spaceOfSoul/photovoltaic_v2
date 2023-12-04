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
    aggregated_losses = defaultdict(lambda: defaultdict(list))

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
                aggregated_losses[base_directory_name][loss_type].append(value)
            markdown_content += "\n"

    stats_content = "\n\n## Aggregated Statistics\n\n"
    for base_directory, losses in aggregated_losses.items():
        stats_content += f"### {base_directory}\n\n"
        for loss_type, values in losses.items():
            avg = np.mean(values)
            std = np.std(values)
            stats_content += f"- **{loss_type}**\n"
            stats_content += f"  - Average: {avg:.3f}\n"
            stats_content += f"  - Standard Deviation: {std:.3f}\n"
        stats_content += "\n"

    # 이 부분에서 두 개의 markdown 파일에 내용을 작성합니다.
    with open("results.md", 'w') as file:
        file.write(markdown_content)

    with open("results_stat.md", 'w') as file:
        file.write(stats_content)
