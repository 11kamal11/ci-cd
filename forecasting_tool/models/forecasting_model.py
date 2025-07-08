from odoo import models, fields, api
import base64
import tempfile
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import logging
import io

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

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

    def _detect_date_column(self, df):
        """Detect date column with multiple format attempts."""
        date_formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y', '%Y%m%d',
            '%Y-%m', '%Y%m', '%m/%Y', '%b %Y', '%d %b %Y', '%Y'
        ]
        for col in df.columns:
            for fmt in date_formats:
                try:
                    parsed = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    if parsed.notnull().sum() > len(df) * 0.8:
                        _logger.info(f"Detected date column '{col}' with format '{fmt}'")
                        return col, fmt
                except Exception as e:
                    _logger.debug(f"Failed to parse column '{col}' with format '{fmt}': {str(e)}")
                    continue
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notnull().sum() > len(df) * 0.8:
                    _logger.info(f"Detected date column '{col}' via auto-parsing")
                    return col, None
            except Exception as e:
                _logger.debug(f"Failed to auto-parse column '{col}': {str(e)}")
                continue
        _logger.warning("No valid date column found")
        return None, None

    def run_forecast(self):
        """Generate forecast and enhanced visualizations from uploaded CSV."""
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

                if len(df) < 10:
                    rec.forecast_result = "Not enough valid data points for forecasting."
                    continue

                # Rename columns for Prophet
                df = df.rename(columns={date_col: 'ds', target_col: 'y'})

                # Prophet forecast
                model = Prophet(
                    yearly_seasonality=len(df) >= 12,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    growth='linear'
                )
                model.fit(df)
                future = model.make_future_dataframe(periods=12, freq='M')
                forecast = model.predict(future)

                # Store forecast output (future predictions only)
                future_forecast = forecast[forecast['ds'] > df['ds'].max()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                future_forecast['ds'] = future_forecast['ds'].dt.strftime('%Y-%m-%d')
                rec.forecast_result = future_forecast.to_string(index=False)

                # Dynamic date range for chart titles
                min_date = df['ds'].min().strftime('%Y-%m')
                max_date = future['ds'].max().strftime('%Y-%m')

                # Enhanced Forecast chart (line plot)
                if SEABORN_AVAILABLE:
                    plt.style.use('seaborn')
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(df['ds'], df['y'], label='Historical Sales', color='#1f77b4', linewidth=2)
                ax1.plot(future['ds'], forecast['yhat'], label='Forecast', color='#ff7f0e', linewidth=2)
                ax1.fill_between(future['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='#ff7f0e', alpha=0.2)
                ax1.set_title(f'Sales Forecast ({min_date} to {max_date})', fontsize=14, pad=10)
                ax1.set_xlabel('Date', fontsize=12)
                ax1.set_ylabel('Sales', fontsize=12)
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend(loc='upper left', fontsize=10)
                ax1.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                rec.forecast_chart = self._save_figure_as_binary(fig1)
                plt.close(fig1)

                # Enhanced Bar chart
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                if SEABORN_AVAILABLE:
                    colors = sns.color_palette("Blues", len(df))
                else:
                    colors = ['#1f77b4'] * len(df)
                bars = ax2.bar(df['ds'].dt.strftime('%Y-%m'), df['y'], color=colors)
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
                ax2.set_title(f'Sales by Month ({min_date} to {df["ds"].max().strftime("%Y-%m")})', fontsize=14)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.set_ylabel('Sales', fontsize=12)
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                rec.bar_chart = self._save_figure_as_binary(fig2)
                plt.close(fig2)

                # Enhanced Histogram
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                if SCIPY_AVAILABLE and len(df['y']) > 1:
                    iqr = np.percentile(df['y'], 75) - np.percentile(df['y'], 25)
                    bin_width = 2 * iqr * len(df['y']) ** (-1/3) if iqr > 0 else (df['y'].max() - df['y'].min()) / 20
                    bins = int((df['y'].max() - df['y'].min()) / bin_width) if bin_width > 0 else 20
                    ax3.hist(df['y'], bins=bins, color='#ff7f0e', edgecolor='black', alpha=0.7)
                    kde = stats.gaussian_kde(df['y'])
                    x_range = np.linspace(df['y'].min(), df['y'].max(), 100)
                    ax3.plot(x_range, kde(x_range) * len(df['y']) * bin_width, color='#1f77b4', linewidth=2, label='Density')
                    ax3.legend(loc='upper right', fontsize=10)
                else:
                    ax3.hist(df['y'], bins=20, color='#ff7f0e', edgecolor='black', alpha=0.7)
                ax3.set_title('Sales Distribution', fontsize=14)
                ax3.set_xlabel('Sales', fontsize=12)
                ax3.set_ylabel('Frequency', fontsize=12)
                ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                rec.histogram_chart = self._save_figure_as_binary(fig3)
                plt.close(fig3)

                # Enhanced Pie chart (fixed 100% issue)
                df_grouped = df.groupby(df['ds'].dt.strftime('%Y-%m'))['y'].sum().sort_values(ascending=False).head(5)
                total = df_grouped.sum()
                if total > 0 and len(df_grouped) > 1:
                    percentages = (df_grouped / total * 100).round(1)
                    fig4, ax4 = plt.subplots(figsize=(10, 10))
                    if SEABORN_AVAILABLE:
                        colors = sns.color_palette("Set2", len(df_grouped))
                    else:
                        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
                    wedges, texts, autotexts = ax4.pie(percentages, labels=df_grouped.index, autopct='%1.1f%%', colors=colors, startangle=90)
                    for text in autotexts:
                        text.set_fontsize(10)
                    ax4.legend(wedges, df_grouped.index, title="Periods", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
                    ax4.set_title('Top 5 Periods by Sales', fontsize=14)
                    plt.tight_layout()
                    rec.pie_chart = self._save_figure_as_binary(fig4)
                    plt.close(fig4)
                else:
                    rec.pie_chart = None
                    rec.forecast_result += "\nInsufficient data for pie chart (need at least 2 periods with non-zero sales)."

            except Exception as e:
                _logger.error(f"Forecast error: {str(e)}")
                rec.forecast_result = f"Forecast error: {str(e)}"

    def _save_figure_as_binary(self, fig):
        """Save matplotlib figure as binary."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            _logger.error(f"Error saving figure: {str(e)}")
            return None
        finally:
            plt.close(fig)