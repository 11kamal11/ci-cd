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
    import io
    for rec in self:
        rec.forecast_result = False
        rec.forecast_chart = False

        if not rec.csv_file:
            rec.forecast_result = "No CSV file uploaded."
            return

        try:
            # Read file
            file_content = base64.b64decode(rec.csv_file)
            df = pd.read_csv(io.BytesIO(file_content))

            # Check if we have at least 2 columns
            if df.shape[1] < 2:
                rec.forecast_result = "CSV must contain at least 2 columns: Date and Value."
                return

            # Try to automatically assign date and value columns
            df.columns = df.columns.str.strip()
            df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna()

            if df.empty:
                rec.forecast_result = "No valid data after parsing dates and values."
                return

            # Forecast
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)

            rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string()

            # Save chart
            fig = model.plot(forecast)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                fig.savefig(image_file.name)
                image_file.seek(0)
                rec.forecast_chart = base64.b64encode(image_file.read())
            plt.close(fig)

        except Exception as e:
            import traceback
            rec.forecast_result = f"Error during forecast:\n{str(e)}\n{traceback.format_exc()}"
