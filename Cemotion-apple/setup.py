import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Cemotion-apple",
    version="0.3.2",
    author="Cyberbolt",
    author_email="735245473@qq.com",
    description="基于NLP的中文情感倾向分析库",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'tqdm>=4.56.0',
        'requests>=2.25.1',
        'jieba>=0.42.1',
    ]    
)