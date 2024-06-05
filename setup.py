from setuptools import setup, find_packages

setup(
    name='PyFLOptimizer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'flwr',
    ],
    entry_points={
        'console_scripts': [
            'pyfloptimizer-server=pyfloptimizer.server:main',
            'pyfloptimizer-client=pyfloptimizer.client:main',
        ],
    },
    description='A federated learning optimizer using Flower framework.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ELHAMDANI Mohamed Oussama',
    author_email='oussamahamdani1718@gmail.com',
    url='https://github.com/OussamaElhamdani/FL-Improvement-Game-Theoretic-Approach',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
