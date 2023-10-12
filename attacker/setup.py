from setuptools import setup, find_packages

setup(
    name='attacker',
    version='0.0.1',
    author='Miao Li',
    author_email='limiao6321@gmail.com',
    description='This is a package used to attack on the gradient in federated RL',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
    ],
)