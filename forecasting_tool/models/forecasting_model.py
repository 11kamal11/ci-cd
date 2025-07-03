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
        rec.forecast_result = False
        rec.forecast_chart = False

        if not rec.csv_file:
            rec.forecast_result = "No CSV file uploaded."
            return

        try:
            # Save uploaded binary CSV file to a temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(base64.b64decode(rec.csv_file))
                tmp.close()
                df = pd.read_csv(tmp.name)

            # Ensure proper column names
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'], errors='coerce')

            df = df.dropna()
            if df.empty:
                rec.forecast_result = "Uploaded file has no valid data after cleaning."
                return

            # Run Prophet
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)

            # Save results
            rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string()

            # Save plot
            fig = model.plot(forecast)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                fig.savefig(image_file.name)
                image_file.seek(0)
                rec.forecast_chart = base64.b64encode(image_file.read())

            plt.close(fig)

        except Exception as e:
            rec.forecast_result = f"Error during forecast: {str(e)}"

