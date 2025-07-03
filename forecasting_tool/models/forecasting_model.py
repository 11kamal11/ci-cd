from odoo import models, fields
import base64
import tempfile
import pandas as pd

class ForecastingInput(models.Model):
    _name = 'forecasting.input'
    _description = 'Forecast Input File'

    name = fields.Char('Filename')
    csv_file = fields.Binary('CSV File', required=True)
    forecast_result = fields.Text('Forecast Output', readonly=True)

    def run_forecast(self):
        for rec in self:
            if not rec.csv_file:
                continue
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(base64.b64decode(rec.csv_file))
                    tmp.close()
                    df = pd.read_csv(tmp.name)
                    # Simple preview (you can extend this later)
                    rec.forecast_result = str(df.head().to_string())
            except Exception as e:
                rec.forecast_result = f"Error reading CSV: {str(e)}"
