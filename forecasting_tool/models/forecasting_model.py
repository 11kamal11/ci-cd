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
                    rec.forecast_result = "No file uploaded"
                    continue

                # Decode and load CSV
                decoded = base64.b64decode(rec.csv_file)
                text = decoded.decode('utf-8', errors='ignore')
                df = pd.read_csv(StringIO(text))

                # Basic structure check
                if df.shape[1] < 2:
                    rec.forecast_result = "CSV must have at least 2 columns (date + value)."
                    continue

                # Rename first two columns
                df.columns = [col.lower().strip() for col in df.columns]
                df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})

                # Parse columns
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                df.dropna(inplace=True)

                if df.empty:
                    rec.forecast_result = "No valid rows after cleanup."
                    continue

                # Prophet Forecasting
                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=6, freq='M')
                forecast = model.predict(future)

                output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
                rec.forecast_result = "Forecast:\n" + output.to_string(index=False)

            except Exception as e:
                rec.forecast_result = f"Error occurred:\n{str(e)}\n{traceback.format_exc()}"
