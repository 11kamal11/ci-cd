from odoo import models, fields, api
import base64
import tempfile
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import logging
import io

_logger = logging.getLogger(__name__)

class ForecastingInput(models.Model):
    _name = 'forecasting.input'
    _description = 'Forecast Input File'

    name = fields.Char('Filename', required=True)
    csv_file = fields.Binary('CSV File', required=True)
    forecast_result = fields.Text('Forecast Output', readonly=True)
    forecast_chart = fields.Binary('Forecast Chart', readonly=True, attachment=True)
    bar_chart = fields.Binary('Bar Chart', readonly=True, attachment=True)
    histogram_chart = fields.Binary('Histogram Chart', readonly=True, attachment=True)
    pie_chart = fields.Binary('Pie Chart', readonly=True, attachment=True)
    period = fields.Selection([
        ('quarterly', 'Quarterly'),
        ('half_yearly', 'Half-Yearly'),
        ('yearly', 'Yearly')
    ], string='Period', default='quarterly', required=True)

    def _detect_date_column(self, df):
        """Detect date column with multiple format attempts."""
        date_formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y', '%Y%m%d',
            '%Y-%m', '%Y-%m', '%m/%Y', '%b %Y', '%d %b %Y', '%Y'
        ]
        for col in df.columns:
            for fmt in date_formats:
                try:
                    parsed = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    if parsed.notnull().sum() > len(df) * 0.8:
                        _logger.info(f"Detected date column '{col}' with format '{fmt}'")
                        return col, fmt
                except:
                    continue
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notnull().sum() > len(df) * 0.8:
                    _logger.info(f"Detected date column '{col}' via auto-parsing")
                    return col, None
            except:
                continue
        _logger.warning("No valid date column found")
        return None, None

    def run_forecast(self):
        """Generate forecast and visualizations from uploaded CSV."""
        for rec in self:
            try:
                if not rec.csv_file:
                    rec.forecast_result = "No CSV file uploaded."
                    continue

                # Save and load CSV
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(base64.b64decode(rec.csv_file))
                    tmp.close()
                    df = pd.read_csv(tmp.name, encoding='utf-8', sep=None, engine='python')

                # Detect date and numeric columns
                date_col, date_format = self._detect_date_column(df)
                target_col = None
                for col in df.columns:
                    if col != date_col and pd.to_numeric(df[col], errors='coerce').notnull().sum() > len(df) * 0.8:
                        target_col = col
                        break

                if not date_col or not target_col:
                    rec.forecast_result = "Unable to detect valid date and numeric columns."
                    continue

                # Preprocess data
                df = df[[date_col, target_col]].dropna()
                df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce') if date_format else pd.to_datetime(df[date_col], errors='coerce')
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                df = df.dropna()

                # Filter for 2023-2025 data
                df = df[(df[date_col].dt.year >= 2023) & (df[date_col].dt.year <= 2025)]
                if len(df) < 10:
                    rec.forecast_result = "Not enough valid data points for 2023-2025."
                    continue

                # Resample based on period
                freq_map = {'quarterly': 'Q', 'half_yearly': '6M', 'yearly': 'Y'}
                df = df.set_index(date_col).resample(freq_map[rec.period]).sum().reset_index()
                df = df.rename(columns={date_col: 'ds', target_col: 'y'})

                if len(df) < 2:
                    rec.forecast_result = "Insufficient data points after resampling."
                    continue

                # Prophet forecast
                model = Prophet(
                    yearly_seasonality=len(df) >= 4 if rec.period != 'yearly' else False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    growth='linear'
                )
                model.fit(df)
                horizon = 4 if rec.period == 'quarterly' else 2 if rec.period == 'half_yearly' else 1
                future = model.make_future_dataframe(periods=horizon, freq=freq_map[rec.period])
                forecast = model.predict(future)

                # Store forecast output
                future_forecast = forecast[forecast['ds'] > df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                future_forecast['ds'] = future_forecast['ds'].dt.strftime('%Y-%m-%d')
                rec.forecast_result = future_forecast.to_string(index=False)

                # Forecast chart (line plot)
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(df['ds'], df['y'], label='Historical Sales', color='blue')
                ax1.plot(future['ds'], forecast['yhat'], label='Forecast', color='red')
                ax1.fill_between(future['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
                ax1.set_title(f'{rec.period.capitalize()} Sales Forecast (2023-2025)')
                ax1.legend()
                ax1.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                rec.forecast_chart = self._save_figure_as_binary(fig1)
                plt.close(fig1)

                # Bar chart
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.bar(df['ds'].dt.strftime('%Y-%m'), df['y'], color='skyblue')
                ax2.set_title(f'{rec.period.capitalize()} Sales Bar Chart (2023-2025)')
                ax2.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                rec.bar_chart = self._save_figure_as_binary(fig2)
                plt.close(fig2)

                # Histogram
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.hist(df['y'], bins=20, color='orange', edgecolor='black')
                ax3.set_title('Sales Distribution Histogram (2023-2025)')
                plt.tight_layout()
                rec.histogram_chart = self._save_figure_as_binary(fig3)
                plt.close(fig3)

                # Pie chart (fixed to avoid 100% issue)
                df_grouped = df.groupby(df['ds'].dt.strftime('%Y-%m'))['y'].sum().sort_values(ascending=False).head(5)
                total = df_grouped.sum()
                if total > 0:
                    percentages = (df_grouped / total * 100).round(1)
                    fig4, ax4 = plt.subplots(figsize=(8, 8))
                    ax4.pie(percentages, labels=df_grouped.index, autopct='%1.1f%%')
                    ax4.set_title(f'Top 5 {rec.period.capitalize()} Periods by Sales (2023-2025)')
                    plt.tight_layout()
                    rec.pie_chart = self._save_figure_as_binary(fig4)
                    plt.close(fig4)
                else:
                    rec.pie_chart = None
                    rec.forecast_result += "\nNo valid data for pie chart."

            except Exception as e:
                _logger.error(f"Forecast error: {str(e)}")
                rec.forecast_result = f"Forecast error: {str(e)}"

    def _save_figure_as_binary(self, fig):
        """Save matplotlib figure as binary."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            _logger.error(f"Error saving figure: {str(e)}")
            return None
        finally:
            plt.close(fig)