from setuptools import setup
from spaths.version import __version__

setup(
    name='spaths',
    version=__version__,
    description='Ensemble simulation of stochastic processes',
    url='',
    author='Przemyslaw Zielinski',
    author_email='zielinski.przemek@gmail.com',
    license='CC',
    packages=['spaths'],
    install_requires=[
        "numpy",
        "scipy",
        "jax"
    ],
    zip_safe=False
)
