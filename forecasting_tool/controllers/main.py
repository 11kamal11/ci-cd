# Basic controller setup (even if empty)
from odoo import http

class ForecastingToolController(http.Controller):
    @http.route('/forecasting_tool/hello', auth='public')
    def hello(self, **kwargs):
        return "Forecasting Tool is active."
