{
    'name': 'Forecasting Tool',
    'version': '1.0',
    'summary': 'Upload CSV and generate trend forecasts using Prophet',
    'author': 'Kamal',
    'category': 'Tools',
    'depends': ['base', 'web'],
    'data': [
        'security/ir.model.access.csv',
        'views/forecasting_views.xml',
    ],
    'installable': True,
    'application': True,
}
