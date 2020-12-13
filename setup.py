from setuptools import setup

setup(
    name='EasyTransformer',
    version='1.0.1',
    author='kevinpro',
    author_email='3121416933@qq.com',
    url='https://github.com/Ricardokevins/EasyTransformer',
    description=('Simple implement of BERT and Transformer extracted from other repo'),
    packages=['EasyTransformer'],
    install_requires=['numpy','torch'],
)
#python setup.py sdist build
#twine upload dist/*