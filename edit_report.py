import re

train_record_path = 'path...'
with open(train_record_path, 'r', encoding='utf-8') as file:
    train_record_content = file.read()

# Define regular expressions for the performance metrics
metrics_patterns = {
    "original_loss": re.compile(r"original_loss:\s*([\d\.]+)"),
    "trend_loss": re.compile(r"trend_loss:\s*([\d\.]+)"),
    "seasonal_loss": re.compile(r"seasonal_loss:\s*([\d\.]+)"),
    "resid_loss": re.compile(r"resid_loss:\s*([\d\.]+)"),
    "percent_alltime_error": re.compile(r"percent_error_orignal:\s*([\d\.]+)")
}

# Extract the performance metrics
performance_metrics = {}
for key, pattern in metrics_patterns.items():
    match = pattern.search(train_record_content)
    if match:
        performance_metrics[key] = float(match.group(1))

md_path = 'RNN,LSTM,RCNN,2stage-LSTM compare.md'
with open(md_path, 'r', encoding='utf-8') as file:
    md_content = file.read()

# Define a function to insert performance metrics into the markdown table
def insert_metrics_to_md_table(md_content, region, model_name, dropout, metrics):
    # Split the content into lines
    lines = md_content.split('\n')
    
    # Find the table that corresponds to the region and model
    table_start = None
    for i, line in enumerate(lines):
        if region in line and model_name in line:
            table_start = i
            break
            
    # If the table is not found, return the original content without changes
    if table_start is None:
        return md_content
    
    # Construct the new row for the table with the metrics
    new_row = f"| {region} | {model_name} | {dropout} | {metrics['original_loss']} | {metrics['trend_loss']} | {metrics['seasonal_loss']} | {metrics['resid_loss']} | {metrics['percent_alltime_error']} |"
    
    # Insert the new row into the content
    lines.insert(table_start + 2, new_row)  # +2 to account for the header row and the delimiter row
    
    # Join the lines back into a single string
    updated_md_content = '\n'.join(lines)
    return updated_md_content

# write report content info
updated_md_content = insert_metrics_to_md_table(md_content, "삼척", "correction_LSTMs", 0.1, performance_metrics)

# save
with open(md_path, 'w', encoding='utf-8') as file:
    file.write(updated_md_content)
