from odoo import models, fields
import base64
import tempfile
import pandas as pd
import io
from prophet import Prophet
import plotly.graph_objs as go

class ForecastingInput(models.Model):
    _name = 'forecasting.input'
    _description = 'Forecast Input File'

    name = fields.Char('Filename')
    csv_file = fields.Binary('CSV File', required=True)
    forecast_result = fields.Text('Forecast Output', readonly=True)
    forecast_chart = fields.Binary('Forecast Chart', readonly=True)

    def run_forecast(self):
        for rec in self:
            if not rec.csv_file:
                rec.forecast_result = "No CSV file uploaded."
                continue

            try:
                # Save uploaded CSV to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(base64.b64decode(rec.csv_file))
                    tmp.flush()

                    # Read CSV
                    df = pd.read_csv(tmp.name)

                # Check basic requirements
                if 'Year' not in df.columns or 'Month' not in df.columns:
                    rec.forecast_result = "CSV must have 'Year' and 'Month' columns."
                    continue

                if 'Pax From Origin' not in df.columns:
                    rec.forecast_result = "CSV must have 'Pax From Origin' column to forecast."
                    continue

                # Create 'ds' and 'y' for Prophet
                df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
                df = df.sort_values('ds')
                df = df[['ds', 'Pax From Origin']].rename(columns={'Pax From Origin': 'y'})

                # Prophet Forecast
                model = Prophet()
                model.fit(df)

                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                # Convert forecast head to string for display
                rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_string(index=False)

                # Plot forecast chart with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

                # Save plot to image buffer
                img_buf = io.BytesIO()
                fig.write_image(img_buf, format="png")
                img_buf.seek(0)

                # Store image as base64 binary
                rec.forecast_chart = base64.b64encode(img_buf.read())

            except Exception as e:
                rec.forecast_result = f"Error during forecast: {str(e)}"
