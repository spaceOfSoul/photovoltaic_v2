import re

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

test_record_path = 'train_models/2-stageRR_2000_drop0.5/test_record.txt'
losses = extract_losses_from_txt(test_record_path)
for region_key in losses.keys():
    print(region_key)
    print(losses[region_key])
