from odoo import models, fields
import base64
import tempfile
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import logging

_logger = logging.getLogger(__name__)

class ForecastingInput(models.Model):
    _name = 'forecasting.input'
    _description = 'Forecast Input File'

    name = fields.Char('Filename')
    csv_file = fields.Binary('CSV File', required=True)
    forecast_result = fields.Text('Forecast Output', readonly=True)
    forecast_chart = fields.Binary('Forecast Chart', readonly=True, attachment=True)
    bar_chart = fields.Binary('Bar Chart', readonly=True, attachment=True)
    histogram_chart = fields.Binary('Histogram Chart', readonly=True, attachment=True)
    pie_chart = fields.Binary('Pie Chart', readonly=True, attachment=True)

    def run_forecast(self):
        for rec in self:
            try:
                if not rec.csv_file:
                    rec.forecast_result = "No CSV file uploaded."
                    continue

                # Save and load CSV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(base64.b64decode(rec.csv_file))
                    tmp.close()
                    df = pd.read_csv(tmp.name)

                # Try to detect date and numeric columns
                date_col = None
                target_col = None

                for col in df.columns:
                    if pd.to_datetime(df[col], errors='coerce').notnull().sum() > len(df) * 0.6:
                        date_col = col
                        break

                for col in df.columns:
                    if col != date_col and pd.to_numeric(df[col], errors='coerce').notnull().sum() > len(df) * 0.6:
                        target_col = col
                        break

                if not date_col or not target_col:
                    rec.forecast_result = "Unable to detect valid date and numeric column."
                    continue

                df = df[[date_col, target_col]].dropna()
                df.columns = ['ds', 'y']
                df['ds'] = pd.to_datetime(df['ds'])
                df['y'] = pd.to_numeric(df['y'], errors='coerce')

                df = df.dropna()
                if len(df) < 10:
                    rec.forecast_result = "Not enough data after cleaning for forecasting."
                    continue

                # Prophet forecast
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string()
                3

                # Forecast chart
                fig1 = model.plot(forecast)
                rec.forecast_chart = self._save_figure_as_binary(fig1)
                plt.close(fig1)

                # Bar chart
                fig2, ax2 = plt.subplots()
                ax2.bar(df['ds'].dt.strftime('%Y-%m'), df['y'], color='skyblue')
                ax2.set_title('Bar Chart of Y over Time')
                ax2.tick_params(axis='x', rotation=45)
                rec.bar_chart = self._save_figure_as_binary(fig2)
                plt.close(fig2)

                # Histogram
                fig3, ax3 = plt.subplots()
                ax3.hist(df['y'], bins=20, color='orange', edgecolor='black')
                ax3.set_title('Histogram of Y')
                rec.histogram_chart = self._save_figure_as_binary(fig3)
                plt.close(fig3)

                # Pie chart
                df_grouped = df.groupby(df['ds'].dt.strftime('%Y-%m'))['y'].sum().sort_values(ascending=False).head(5)
                fig4, ax4 = plt.subplots()
                ax4.pie(df_grouped.values, labels=df_grouped.index, autopct='%1.1f%%')
                ax4.set_title('Top 5 Periods by Y')
                rec.pie_chart = self._save_figure_as_binary(fig4)
                plt.close(fig4)

            except Exception as e:
                _logger.error("Forecast error: %s", str(e))
                rec.forecast_result = f"Forecast error: {str(e)}"

    def _save_figure_as_binary(self, fig):
        import io
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return base64.b64encode(buffer.read())
