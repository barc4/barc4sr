from setuptools import find_packages, setup

setup(
    name='barc4sr',
    version='2025.06.13',
    author='Rafael Celestre',
    author_email='rafael.celestre@synchrotron-soleil.fr',
    description='A Python package for Synchrotron Radiation calculations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/barc4/barc4sr',
    license='CC BY-NC-SA 4.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'numba>=0.53.1',
        'scipy>=1.5.0',
        'joblib>=0.14.0',
        'h5py>=3.0.0',
        'matplotlib>=3.3.0',
        'Pillow>=7.0.0',
        'imageio>=2.9.0',
        'scikit-image>=0.17.2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
