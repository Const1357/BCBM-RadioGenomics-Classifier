logs = 'threshold_opt_logs.txt'

import re
from typing import Dict, List, Tuple

import pandas as pd

def parse_optuna_log(filepath: str = logs) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Parse an Optuna log file containing study and trial info.
    
    Returns:
        dict: {study_name: [(ER_threshold, PR_threshold, HER2_threshold, value), ...]}
    """
    study_pattern = re.compile(
        r"A new study created in memory with name: (.+)"
    )
    trial_pattern = re.compile(
        r"Trial \d+ finished with value: ([\d\.eE+-]+).*?"
        r"ER_threshold': ([\d\.eE+-]+).*?"
        r"PR_threshold': ([\d\.eE+-]+).*?"
        r"HER2_threshold': ([\d\.eE+-]+)"
    )
    
    data = {}
    current_study = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Skip warnings
            if "WARNING" in line or "WARN" in line:
                continue
            
            # Detect new study
            study_match = study_pattern.search(line)
            if study_match:
                current_study = study_match.group(1).strip()
                data[current_study] = []
                continue
            
            # Detect trial results
            trial_match = trial_pattern.search(line)
            if trial_match and current_study:
                value = float(trial_match.group(1))
                ER = float(trial_match.group(2))
                PR = float(trial_match.group(3))
                HER2 = float(trial_match.group(4))
                data[current_study].append((ER, PR, HER2, value))
    
    return data

data = parse_optuna_log()
# print(data.keys())

rows = []
for study, trials in data.items():
    for ER, PR, HER2, value in trials:
        rows.append({
            "study": study,
            "ER_threshold": ER,
            "PR_threshold": PR,
            "HER2_threshold": HER2,
            "f1": value
        })

df = pd.DataFrame(rows)
# print(df)

top5_by_study = (
    df.groupby("study", group_keys=False)
      .apply(lambda x: x.nlargest(5, "f1"), include_groups=True)
      .reset_index(drop=True)
)

print(top5_by_study)

#       study  ER_threshold  PR_threshold  HER2_threshold        f1
# 0   trial27      0.367025      0.466898        0.301082  0.643406
# 1   trial27      0.120512      0.581769        0.134629  0.643009
# 2   trial27      0.112595      0.445868        0.123663  0.641842
# 3   trial27      0.197189      0.569386        0.125732  0.641822
# 4   trial27      0.216527      0.523055        0.013701  0.641073

#       study  ER_threshold  PR_threshold  HER2_threshold        f1
# 5   trial32      0.204312      0.618743        0.384851  0.652432
# 6   trial32      0.202706      0.600860        0.295874  0.642934
# 7   trial32      0.051066      0.502173        0.350429  0.642593
# 8   trial32      0.288116      0.586419        0.295793  0.641345
# 9   trial32      0.368970      0.173453        0.473336  0.640935

#       study  ER_threshold  PR_threshold  HER2_threshold        f1
# 10  trial42      0.562326      0.186415        0.165311  0.663173
# 11  trial42      0.561281      0.094707        0.168443  0.656151
# 12  trial42      0.545679      0.078935        0.091315  0.648162
# 13  trial42      0.540558      0.375525        0.052166  0.648106
# 14  trial42      0.547619      0.313153        0.285086  0.646503

#       study  ER_threshold  PR_threshold  HER2_threshold        f1
# 15  trial43      0.118519      0.486719        0.438465  0.643503
# 16  trial43      0.114662      0.527354        0.407823  0.641659
# 17  trial43      0.098400      0.430399        0.451291  0.641441
# 18  trial43      0.143447      0.409773        0.454422  0.641332
# 19  trial43      0.197479      0.322465        0.450315  0.640616

#       study  ER_threshold  PR_threshold  HER2_threshold        f1
# 20  trial46      0.032011      0.661297        0.386311  0.665440
# 21  trial46      0.022595      0.663792        0.368584  0.660239
# 22  trial46      0.462181      0.288870        0.416913  0.644499
# 23  trial46      0.437263      0.089147        0.378561  0.643286
# 24  trial46      0.013806      0.070174        0.026881  0.643277