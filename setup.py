from distutils.core import setup

setup(
    name='sci_analysis',
    version='1.3.1',
    packages=['sci_analysis', 'sci_analysis.data', 'sci_analysis.test', 'sci_analysis.graphs', 'sci_analysis.analysis'],
    url='https://github.com/cmmorrow/sci-analysis',
    license='MIT License',
    author='chris morrow',
    author_email='cmmorrow@gmail.com',
    description='A light weight python data exploration and analysis tool',
    install_requires=['numpy', 'scipy', 'matplotlib']
)
