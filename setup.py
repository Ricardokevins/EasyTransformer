from setuptools import setup

setup(
    name='EasyTransformer',
    version='0.0.4',
    author='kevinpro',
    author_email='3121416933@qq.com',
    url='https://github.com/Ricardokevins',
    description=('Simple implement of BERT and Transformer extracted from other repo'),
    packages=['EasyTransformer'],
    install_requires=['numpy','torch'],
)
#python setup.py sdist build
#twine upload dist/*