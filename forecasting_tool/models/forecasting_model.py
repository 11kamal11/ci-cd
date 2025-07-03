from prophet import Prophet
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def run_forecast(self):
    for rec in self:
        if not rec.csv_file:
            continue
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(base64.b64decode(rec.csv_file))
                tmp.close()
                df = pd.read_csv(tmp.name)

                # Make sure your CSV has columns 'Year', 'Month', 'Pax From Origin'
                df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
                df['y'] = df['Pax From Origin']  # or any other target column

                df = df[['ds', 'y']].dropna()

                m = Prophet()
                m.fit(df)

                future = m.make_future_dataframe(periods=6, freq='M')
                forecast = m.predict(future)

                # Save plot to image
                fig = m.plot(forecast)
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_bytes = buf.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode()

                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
                rec.forecast_result = forecast_data.to_string(index=False) + f"\n\n![Forecast](data:image/png;base64,{img_base64})"
        except Exception as e:
            rec.forecast_result = f"Error reading CSV: {str(e)}"
