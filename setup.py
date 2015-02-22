from setuptools import setup, find_packages    

setup(
    name='passage', 
    version='0.2.1',
    packages=find_packages(),
    description="""
        A little library for text analysis with RNNs.
    """,
    license="MIT License (See LICENSE)",
    url="https://github.com/IndicoDataSolutions/Passage",
    author="Alec Radford, Madison May",
    author_email="""
        Alec Radford <madison@indico.io>,
        Madison May <madison@indico.io>
    """,
    install_requires=[
        "numpy >= 1.8.1",
        "Theano >= 0.6.0",
    ],
)
