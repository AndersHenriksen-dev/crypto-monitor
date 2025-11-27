from setuptools import setup, find_packages


setup(
    name="crypto_monitor",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # runtime dependencies
        'requests',
        'pandas',
        'numpy',
        'scikit-learn',
        'mlflow',
        'fastapi',
        'uvicorn',
    ],

    entry_points={
        'console_scripts': [
            'crypto-ingest=crypto_monitor.jobs.ingest_job:main',
            'crypto-train=crypto_monitor.jobs.train_job:main',
        ],
    }
)
