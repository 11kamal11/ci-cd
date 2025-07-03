from odoo import models, fields
import base64
import pandas as pd
from io import StringIO
from prophet import Prophet
import traceback

class ForecastingInput(models.Model):
    _name = 'forecasting.input'
    _description = 'Forecast Input File'

    name = fields.Char('Filename')
    csv_file = fields.Binary('CSV File', required=True)
    forecast_result = fields.Text('Forecast Output', readonly=True)

    def run_forecast(self):
        for rec in self:
            try:
                if not rec.csv_file:
                    rec.forecast_result = "No CSV file found!"
                    continue

                # Decode base64 CSV and convert to DataFrame
                decoded_data = base64.b64decode(rec.csv_file)
                df = pd.read_csv(StringIO(decoded_data.decode('utf-8')))

                # Clean column names and assume first = date, second = value
                df.columns = df.columns.str.strip()
                if len(df.columns) < 2:
                    rec.forecast_result = "CSV must contain at least 2 columns (Date and Value)."
                    continue

                df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                df = df.dropna(subset=['ds', 'y'])

                if len(df) < 10:
                    rec.forecast_result = "Not enough data to forecast. Need at least 10 rows."
                    continue

                # Train Prophet model
                model = Prophet()
                model.fit(df)

                # Forecast next 6 months
                future = model.make_future_dataframe(periods=6, freq='M')
                forecast = model.predict(future)

                # Extract last 6 forecasted points
                forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
                forecast_text = forecast_summary.to_string(index=False)

                rec.forecast_result = f"Forecast Output (Next 6 Months):\n\n{forecast_text}"

            except Exception as e:
                rec.forecast_result = f"Error:\n{str(e)}\n\n{traceback.format_exc()}"
