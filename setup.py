from setuptools import setup, find_packages

with open('README.rst', 'r') as readme:
    long_description = readme.read()

setup(
    name='sci_analysis',
    version='2.1.0',
    packages=find_packages(),
    url='https://github.com/cmmorrow/sci-analysis',
    license='MIT License',
    author='chris morrow',
    author_email='cmmorrow@gmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
	    'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='statistics data EDA graphing visualization analysis scientific',
    description='An easy to use and powerful python-based data exploration and analysis tool',
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'six'],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, >=3.5',
    long_description=long_description,
)
