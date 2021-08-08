from setuptools import setup

setup(
    name='EasyTransformer',
    version='1.2.3',
    author='kevinpro',
    author_email='3121416933@qq.com',
    url='https://github.com/Ricardokevins/EasyTransformer',
    description=('Simple implement of BERT and Transformer extracted from other repo'),
    packages=['EasyTransformer'],
    install_requires=['numpy','torch','tqdm'],
)
#python setup.py sdist build
#twine upload dist/*