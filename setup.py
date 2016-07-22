from setuptools import setup
import os

# include data files
loki_data = []
for dirpath, dirnames, files in os.walk('loki/resources'):
    for f in files:
        if f.endswith('.dat'):
            loki_data.append(os.path.join(dirpath[6:], f))

setup(
    name = 'loki',
    version = '0.5',
    description = 'Simple model for galactic kinematics of low-mass stars',
    author = 'Christohper Theissen',
    author_email = 'ctheissen@gmail.com',
    license = 'MIT',
    url = 'https://github.com/ctheissen/LoKi',
    download_url = 'https://github.com/ctheissen/LoKi/tarball/0.11',
    packages = ['loki'],
    package_data = {
	'': loki_data,
    },
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is the project?
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        # should match "license" above
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
