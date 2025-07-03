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
                    rec.forecast_result = "No CSV file uploaded."
                    continue

                decoded_data = base64.b64decode(rec.csv_file)
                csv_text = decoded_data.decode('utf-8', errors='ignore')
                df = pd.read_csv(StringIO(csv_text))

                # Basic validation
                if df.shape[1] < 2:
                    rec.forecast_result = "CSV must have at least 2 columns: date and value."
                    continue

                # Auto-assign column names
                df.columns = [col.strip().lower() for col in df.columns]
                df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})

                # Convert types safely
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                df.dropna(subset=['ds', 'y'], inplace=True)

                if len(df) < 10:
                    rec.forecast_result = "Not enough data after cleaning. Need at least 10 valid rows."
                    continue

                # Train model
                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=6, freq='M')
                forecast = model.predict(future)

                output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
                rec.forecast_result = "Forecast (Next 6 Months):\n\n" + output.to_string(index=False)

            except Exception as e:
                rec.forecast_result = f"Error:\n{str(e)}\n\n{traceback.format_exc()}"
