
from azureml.core import Run
import pandas as pd

df = pd.read_csv("diabetes.csv")

# This is the main difference from the interactive run from Jupyter
run = Run.get_context()

cat_cols = ["Pregnancies"]
num_cols = ['PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI']

# Log some basic metrics
row_count = df.shape[0]
run.log("observations", row_count)  # Log a number
run.log_list("data columns", df.columns)  # Log a list of strings
run.log_list("categorical columns", cat_cols)
run.log_list("numerical columns", num_cols)

# Log descriptive statistics
summary_stats = df[num_cols].describe().to_dict()
for col_name, stat_dict in summary_stats.items():
    for stat_name, val in stat_dict.items():
        run.log_row(col_name, stat=stat_name, value=val)  # Logging rows


# Log some image output
import matplotlib.pyplot as plt

diabetic_counts = df['Diabetic'].value_counts()

fig = plt.figure(figsize=(6,6))
ax = fig.gca()    
diabetic_counts.plot.bar(ax = ax) 
ax.set_title('Patients with Diabetes') 
ax.set_xlabel('Diagnosis') 
ax.set_ylabel('Patients')

run.log_image(name='label distribution', plot=fig)  # Log an image

# Log some file
os.makedirs("outputs", exist_ok=True)
df.sample(100).to_csv("outputs/sample.csv", index=False)

run.complete()
