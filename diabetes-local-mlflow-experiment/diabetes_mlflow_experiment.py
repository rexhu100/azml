import pandas as pd
import mlflow
import os

# mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
# experiment = Experiment(workspace=ws, name="diabetes-local-mlflow")
# mlflow.set_experiment(experiment.name)

with mlflow.start_run():
    df = pd.read_csv("diabetes.csv")
    
    # Log some basic metrics
    row_count = df.shape[0]
    
    # Unlike AzureML's native logging functions, which append value to each metric, MLflow doesn't 
    # allow changing the value assigned to each key. Additionally, log_metric only supports float value
    mlflow.log_metric("observations", row_count)  # Log a number
    # Use log_param for other types of data that need to be logged
    mlflow.log_param("columns", df.columns)
    
    cat_cols = ["Pregnancies"]
    num_cols = ['PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI']
    mlflow.log_param("categorical columns", cat_cols)
    mlflow.log_param("numerical columns", num_cols)

    # Log some image output
    import matplotlib.pyplot as plt

    diabetic_counts = df['Diabetic'].value_counts()

    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()    
    diabetic_counts.plot.bar(ax = ax) 
    ax.set_title('Patients with Diabetes') 
    ax.set_xlabel('Diagnosis') 
    ax.set_ylabel('Patients')
    
    mlflow.log_figure(fig, "img_log.png")  # Log an image

    # Log some file
    os.makedirs("outputs", exist_ok=True)
    df.sample(100).to_csv("outputs/sample.csv", index=False)
    
#     mlflow.log_artifact("outputs/sample.csv")