from distutils.core import setup

setup(
    name='sci_analysis',
    version='1.3.1',
    packages=[
        'data',
        'graphs',
        'analysis',
        'operations'
    ],
    url='https://github.com/cmmorrow/sci-analysis',
    license='MIT License',
    author='chris morrow',
    author_email='cmmorrow@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='statistics data EDA graphing',
    description='A light weight python data exploration and analysis tool',
    install_requires=['numpy', 'scipy', 'matplotlib']
)
