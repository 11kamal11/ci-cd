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

                # Minimal Prophet forecasting
                df.columns = ['ds', 'y']  # Rename if your CSV uses different headers
                df['ds'] = pd.to_datetime(df['ds'])
                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                # Save the forecast output
                rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string()

                # Plot and store the chart
                fig = model.plot(forecast)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                    fig.savefig(image_file.name)
                    image_file.seek(0)
                    rec.forecast_chart = base64.b64encode(image_file.read())

                plt.close(fig)

            except Exception as e:
                rec.forecast_result = f"Error during forecast: {str(e)}"
