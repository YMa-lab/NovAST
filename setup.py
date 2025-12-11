from setuptools import setup, find_packages

setup(
    name='NovAST',
    version='0.1.0', 
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.3.2",
        "scanpy>=1.11.4",
        "anndata>=0.12.2",
        "scipy>=1.16.2",
        "scikit-learn>=1.7.2",
        "matplotlib>=3.10.6",
        "seaborn>=0.13.2",
        "PyYAML>=6.0.2",
        "torch>=2.8.0",
    ],
    author='Yu Zhu',
    description='NovAST is a deep learning framework for automated label transfer and novel cell type discovery in spatial transcriptomics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/YMa-lab/NovAST',
    classifiers=[
        'Programming Language :: Python :: 3.11',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)