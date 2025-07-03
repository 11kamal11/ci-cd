from odoo import models, fields
import base64
import tempfile
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

class ForecastingInput(models.Model):
    _name = 'forecasting.input'
    _description = 'Forecast Input File'

    name = fields.Char('Filename')
    csv_file = fields.Binary('CSV File', required=True)
    forecast_result = fields.Text('Forecast Output', readonly=True)
    forecast_chart = fields.Binary('Forecast Chart', readonly=True, attachment=True)

    def run_forecast(self):
        for rec in self:
            if not rec.csv_file:
                continue
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(base64.b64decode(rec.csv_file))
                    tmp.close()
                    df = pd.read_csv(tmp.name)

                # Automatically detect columns
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower() or 'year' in col.lower()]
                value_cols = [col for col in df.select_dtypes(include=['number']).columns if col.lower() not in ['year', 'month', 'id']]

                if not date_cols or not value_cols:
                    rec.forecast_result = "Error: Couldn't detect a date or value column in the CSV."
                    return

                df = df[[date_cols[0], value_cols[0]]]
                df.columns = ['ds', 'y']
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df = df.dropna()

                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string()

                fig = model.plot(forecast)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                    fig.savefig(image_file.name)
                    image_file.seek(0)
                    rec.forecast_chart = base64.b64encode(image_file.read())

                plt.close(fig)

            except Exception as e:
                rec.forecast_result = f"Error during forecast: {str(e)}"
