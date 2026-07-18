import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Steel Industry Analytics Infrastructure")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_pipeline = joblib.load('model.joblib')

# MATCHED TO YOUR EXACT NOTEBOOK OUTPUT SIGNATURE
EXPECTED_FEATURES = [
    'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 
    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 
    'NSM', 'hour_of_day', 'extracted_day_of_week', 'month', 'is_weekend', 
    'Power_Factor_Ratio', 'WeekStatus_Weekend', 'Day_of_week_Monday', 
    'Day_of_week_Saturday', 'Day_of_week_Sunday', 'Day_of_week_Thursday', 
    'Day_of_week_Tuesday', 'Day_of_week_Wednesday', 'Load_Type_Maximum_Load', 
    'Load_Type_Medium_Load'
]

@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

@app.get("/predict", response_class=HTMLResponse)
async def predict_get(request: Request):
    return templates.TemplateResponse(request=request, name="predict.html", context={"prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_post(
    request: Request,
    lagging_reactive: float = Form(...),
    leading_reactive: float = Form(...),
    co2: float = Form(...),
    lagging_pf: float = Form(...),
    leading_pf: float = Form(...),
    nsm: float = Form(...),
    hour_of_day: int = Form(...),
    extracted_day_of_week: int = Form(...),
    month: int = Form(...),
    is_weekend: int = Form(...),
    pf_ratio: float = Form(...),
    week_status: str = Form(...),
    load_type: str = Form(...),
    day_of_week: str = Form(...)
):
    # Initialize array structured with 0s
    input_data = {feat: 0.0 for feat in EXPECTED_FEATURES}
    
    # Structural Numerical mappings
    input_data['Lagging_Current_Reactive.Power_kVarh'] = lagging_reactive
    input_data['Leading_Current_Reactive_Power_kVarh'] = leading_reactive
    input_data['CO2(tCO2)'] = co2
    input_data['Lagging_Current_Power_Factor'] = lagging_pf
    input_data['Leading_Current_Power_Factor'] = leading_pf
    input_data['NSM'] = nsm
    input_data['hour_of_day'] = hour_of_day
    input_data['extracted_day_of_week'] = extracted_day_of_week
    input_data['month'] = month
    input_data['is_weekend'] = is_weekend
    input_data['Power_Factor_Ratio'] = pf_ratio
    
    # Conditional One-Hot Target Flag updates matching drop_first rules
    if week_status == "Weekend":
        input_data['WeekStatus_Weekend'] = 1.0
        
    if day_of_week == "Monday": input_data['Day_of_week_Monday'] = 1.0
    elif day_of_week == "Saturday": input_data['Day_of_week_Saturday'] = 1.0
    elif day_of_week == "Sunday": input_data['Day_of_week_Sunday'] = 1.0
    elif day_of_week == "Thursday": input_data['Day_of_week_Thursday'] = 1.0
    elif day_of_week == "Tuesday": input_data['Day_of_week_Tuesday'] = 1.0
    elif day_of_week == "Wednesday": input_data['Day_of_week_Wednesday'] = 1.0
    
    if load_type == "Maximum_Load": input_data['Load_Type_Maximum_Load'] = 1.0
    elif load_type == "Medium_Load": input_data['Load_Type_Medium_Load'] = 1.0
    
    # Enforce exact matrix column sequence matching the model signature
    df_features = pd.DataFrame([input_data])[EXPECTED_FEATURES]
    
    # Process prediction
    raw_prediction = model_pipeline.predict(df_features)[0]
    formatted_prediction = round(float(raw_prediction), 2)
    
    return templates.TemplateResponse(
        request=request, 
        name="predict.html", 
        context={"prediction": formatted_prediction}
    )
