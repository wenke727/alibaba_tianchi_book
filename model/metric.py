from sklearn.metrics import classification_report

def flatten_classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, 
                                  digits=2, keep_avg=False, output_dict=False, desc=None,
                                  metric=set(['accuracy', 'precision', 'recall'])):
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, 
        sample_weight=sample_weight, digits=digits, output_dict=True)
    
    flat_report = {}
    for key, value in report_dict.items():
        # if key == 'accuracy' and not keep_avg:
            # continue
        if key.startswith('weighted avg') and not keep_avg:
            continue
        if key.startswith('macro avg') and not keep_avg:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key in metric:
                    flat_report[f'{key}_{sub_key}'] = sub_value
        else:
            flat_report[key] = value
            
    if desc:
        flat_report['desc'] = desc
    
    return flat_report

if __name__ == "__main__":
    # 示例用法
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 1, 0]
    target_names = ['class 0', 'class 1']

    flat_report = flatten_classification_report(y_true, y_pred, target_names=target_names, keep_avg=False, output_dict=False)
    print(flat_report)
