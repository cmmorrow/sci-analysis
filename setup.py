from setuptools import setup, find_packages

setup(
    name='sci_analysis',
    version='1.4.4',
    packages=find_packages(),
    url='https://github.com/cmmorrow/sci-analysis',
    license='MIT License',
    author='chris morrow',
    author_email='cmmorrow@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
	'Intended Audience :: Science/Research',
	'Intended Audience :: Manufacturing',
	'Intended Audience :: Financial and Insurance Industry',
	'Intended Audience :: Healthcare Industry',
	'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
	'Natural Language :: English',
	'Topic :: Scientific/Engineering :: Information Analysis',
	'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
	'Programming Language :: Python :: 3',
	'Programming Language :: Python :: 3.5'
    ],
    keywords='statistics data EDA graphing visualization analysis scientific',
    description='A light weight python data exploration and analysis tool',
    install_requires=['numpy', 'scipy', 'matplotlib', 'six'],
    test_suite='nosetests'
)
