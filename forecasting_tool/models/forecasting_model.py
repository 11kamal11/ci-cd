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

    bar_chart = fields.Binary('Bar Chart', readonly=True, attachment=True)
    histogram_chart = fields.Binary('Histogram Chart', readonly=True, attachment=True)
    pie_chart = fields.Binary('Pie Chart', readonly=True, attachment=True)

    def run_forecast(self):
        for rec in self:
            if not rec.csv_file:
                continue
            try:
                # Read and decode CSV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(base64.b64decode(rec.csv_file))
                    tmp.close()
                    df = pd.read_csv(tmp.name)

                df.columns = ['ds', 'y']
                df['ds'] = pd.to_datetime(df['ds'])

                # Prophet Forecast
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                rec.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string()

                # Forecast Line Chart
                fig1 = model.plot(forecast)
                rec.forecast_chart = self._save_figure_as_binary(fig1)
                plt.close(fig1)

                # Bar Chart
                fig2, ax2 = plt.subplots()
                ax2.bar(df['ds'].dt.strftime('%Y-%m'), df['y'], color='skyblue')
                ax2.set_title('Bar Chart of Y values over Time')
                ax2.tick_params(axis='x', rotation=90)
                rec.bar_chart = self._save_figure_as_binary(fig2)
                plt.close(fig2)

                # Histogram
                fig3, ax3 = plt.subplots()
                ax3.hist(df['y'], bins=20, color='orange', edgecolor='black')
                ax3.set_title('Histogram of Y values')
                rec.histogram_chart = self._save_figure_as_binary(fig3)
                plt.close(fig3)

                # Pie Chart (top 5 summed periods)
                df_grouped = df.groupby(df['ds'].dt.strftime('%Y-%m'))['y'].sum().sort_values(ascending=False).head(5)
                fig4, ax4 = plt.subplots()
                ax4.pie(df_grouped.values, labels=df_grouped.index, autopct='%1.1f%%')
                ax4.set_title('Top 5 Periods by Y (Pie Chart)')
                rec.pie_chart = self._save_figure_as_binary(fig4)
                plt.close(fig4)

            except Exception as e:
                rec.forecast_result = f"Error during forecast: {str(e)}"

    def _save_figure_as_binary(self, fig):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
            fig.savefig(image_file.name)
            image_file.seek(0)
            return base64.b64encode(image_file.read())
