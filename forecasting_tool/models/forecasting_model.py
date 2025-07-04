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
                ax1 = fig1.gca()
                ax1.set_title('Forecast: Future Sales Prediction', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Date', fontsize=12)
                ax1.set_ylabel('Sales (y)', fontsize=12)
                ax1.grid(True)
                rec.forecast_chart = self._save_figure_as_binary(fig1)
                plt.close(fig1)

                # Bar chart
               fig2, ax2 = plt.subplots(figsize=(10, 5))
                bars = ax2.bar(df['ds'].dt.strftime('%Y-%m'), df['y'], color='dodgerblue')
                ax2.set_title('Monthly Sales Overview', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Month')
                ax2.set_ylabel('Sales')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
                for bar in bars:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.0f}', ha='center', va='bottom', fontsize=8)
                rec.bar_chart = self._save_figure_as_binary(fig2)
                plt.close(fig2)


                # Histogram
                fig3, ax3 = plt.subplots()
                ax3.hist(df['y'], bins=20, color='coral', edgecolor='black')
                ax3.set_title('Distribution of Sales (Histogram)', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Sales Amount')
                ax3.set_ylabel('Frequency')
                ax3.grid(True)
                rec.histogram_chart = self._save_figure_as_binary(fig3)
                plt.close(fig3)


                # Pie chart
                df_grouped = df.groupby(df['ds'].dt.strftime('%Y-%m'))['y'].sum().sort_values(ascending=False).head(5)
                fig4, ax4 = plt.subplots()
                colors = plt.cm.Paired.colors
                ax4.pie(df_grouped.values, labels=df_grouped.index, autopct='%1.1f%%', startangle=140, colors=colors)
                ax4.set_title('Top 5 Sales Months (Pie Chart)', fontsize=14, fontweight='bold')
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
