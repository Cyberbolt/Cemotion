import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Cemotion",
    version="2.2.1",
    author="Cyberbolt",
    author_email="dtconlyone@gmail.com",
    description="Cemotion 是 Python 下的中文 NLP 库，可以进行中文情感倾向分析、通用领域中文分词。",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tqdm>=4.64.1',
        'joblib>=1.0.0',
        'requests>=2.25.1',
        'numpy>=1.19.5',
        'torch>=2.0.0',
        'transformers>=4.24.0',
        'modelscope>=1.14.0',
    ]    
)